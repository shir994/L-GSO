from comet_ml import Experiment
import sys
import os
import click
import torch
import numpy as np
from typing import Callable
from torch import nn
from torch.nn import functional as F
from torch import optim
sys.path.append('../')
from typing import List, Union
from logger import SimpleLogger, CometLogger
from base_model import BaseConditionalGenerationOracle
sys.path.append('../..')
from model import YModel, LearningToSimGaussianModel, GaussianMixtureHumpModel, RosenbrockModel, \
    RosenbrockModelDegenerateInstrict, RosenbrockModelDegenerate, RosenbrockModelNoisless, BostonNNTuning, \
    GaussianMixtureHumpModelDeepDegenerate
from optimizer import BaseOptimizer
from typing import Callable
import time
import pyro
from torch.autograd import grad
from pyro import distributions as dist
from optimizer import SUCCESS, ITER_ESCEEDED, COMP_ERROR
from void_model import VoidModel
from tqdm import tqdm


def get_freer_gpu():
    """
    Function to get the freest GPU available in the system
    :return:
    """
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(get_freer_gpu()))
else:
    device = torch.device('cpu')
print("Using device = {}".format(device))


def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


class VoidOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 logger: Callable,
                 num_repetitions: int,
                 n_samples: int,
                 *args, **kwargs):
        super().__init__(oracle, x, trace=False, *args, **kwargs)
        self._x.requires_grad_(True)
        self._logger = logger
        self._num_repetitions = num_repetitions
        self._n_samples = n_samples

    def _step(self):
        # for comet throtling problemm...
        time.sleep(0.1)
        init_time = time.time()

        self._oracle.step()
        f_k = torch.tensor(0.)
        x_k = self._oracle._psi.clone().detach()
        d_k = self._oracle.grad(self._oracle._psi, num_repetitions=self._num_repetitions)
        self._x = x_k
        if self._num_iter % 10 == 0:
            f_k = self._oracle.func(self._oracle._psi, num_repetitions=self._num_repetitions)
            print("sigma", (1. + self._oracle._sigma.exp()).log())
            print(self._num_iter, f_k)
            self._logger.log_performance(y_sampler=self._oracle._y_model,
                                         current_psi=x_k,
                                         n_samples=self._n_samples, upload_pickle=False)
            # self._logger.log_grads(self._oracle,
            #                       y_sampler=self._oracle._y_model,
            #                       current_psi=x_k, num_repetitions=30000, n_samples=100, log_grad_diff=True)

            # estimate average grads from policy
            grad_void = []
            grad_void_true = []
            for _ in tqdm(range(2)):
                grad_void.append(self._oracle.grad(x_k, num_repetitions=5000))
            for _ in tqdm(range(2)):
                grad_void_true.append(self._oracle.policy_grad_true(x_k, num_repetitions=5000))
            grad_void = torch.stack(grad_void)
            grad_void_true = torch.stack(grad_void_true)
            grad_true = self._oracle._y_model.grad(x_k, num_repetitions=5000)
            grad_diffs = (grad_void.mean(dim=0) / grad_void.mean(dim=0).norm()) * ((grad_true / grad_void_true.norm()))
            print("Grad var", torch.norm(torch.var(grad_void, dim=0, keepdim=True),
                                                                            dim=1).item())
            self._logger._experiment.log_metric('Mean grad var', torch.norm(torch.var(grad_void, dim=0, keepdim=True),
                                                                            dim=1).item(), step=self._logger._epoch)
            self._logger._experiment.log_metric('Grad diff void psi cosine',
                                        grad_diffs.sum().item(),
                                        step=self._logger._epoch)
            grad_diffs = (grad_void.mean(dim=0) / grad_void.mean(dim=0).norm()) * (grad_void_true.mean(dim=0) / grad_void_true.mean(dim=0).norm())
            self._logger._experiment.log_metric('Grad diff void true cosine',
                                        grad_diffs.sum().item(),
                                        step=self._logger._epoch)
            print("Grad diffs void - policy", grad_diffs.sum())

            """
                model_grad_value_saved = model_grad_value
                model_grad_value_saved = model_grad_value_saved / model_grad_value_saved.norm(keepdim=True, dim=1)
                model_grad_value = model_grad_value.mean(dim=0)
                model_grad_value /= model_grad_value.norm()

                if log_grad_diff:
                    true_grad_value = y_sampler.grad(_current_psi, num_repetitions=num_repetitions)
                    # print("2", true_grad_value.shape)
                    true_grad_value = true_grad_value.mean(dim=0)
                    true_grad_value /= true_grad_value.norm()
                    # print("3", model_grad_value.shape, true_grad_value.shape)
                    self._experiment.log_metric('Mean grad diff',
                                                torch.norm(model_grad_value - true_grad_value).item(), step=self._epoch)

                    self._experiment.log_metric('Mean grad diff cos',
                                                (model_grad_value_saved * true_grad_value).sum(dim=1).mean().item(),
                                                step=self._epoch)
                """
        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


@click.command()
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--model_config_file', type=str, default='void_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--init_psi', type=str, default="0., 0.")
@click.option('--n_samples_per_dim', type=int, default=3000)
def main(
        logger,
        optimized_function,
        optimizer_config_file,
        model_config_file,
        project_name,
        work_space,
        tags,
        init_psi,
        n_samples_per_dim,
):
    model_config = getattr(__import__(model_config_file), 'model_config')
    model_config["num_repetitions"] = n_samples_per_dim
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)

    optimized_function_cls = str_to_class(optimized_function)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )
    experiment.log_parameters(
        {"model_{}".format(key): value for key, value in model_config.items()}
    )


    logger = str_to_class(logger)(experiment)
    y_model = optimized_function_cls(device=device, psi_init=init_psi)
    model = VoidModel(y_model=y_model,
                      psi=init_psi,
                      **model_config)

    optimizer = VoidOptimizer(
        oracle=model,
        x=init_psi,
        logger=logger,
        n_samples=model_config["K"],
        **optimizer_config)

    current_psi, status, history = optimizer.optimize()

    try:
        logger.log_optimizer(optimizer)
        logger.log_grads(model, y_sampler=y_model, current_psi=x_k, num_repetitions=30000,
                         n_samples=100, log_grad_diff=True)
        logger.log_performance(y_sampler=y_model,
                               current_psi=current_psi,
                               n_samples=5000)

    except Exception as e:
        print(e)
        raise
if __name__ == "__main__":
    main()
