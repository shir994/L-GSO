from comet_ml import Experiment
import sys
import os
import click
import torch
import numpy as np
sys.path.append('../')
sys.path.append('../..')
from typing import List, Union
from model import YModel, RosenbrockModel, MultimodalSingularityModel, GaussianMixtureHumpModel, \
                  LearningToSimGaussianModel, BernoulliModel, RosenbrockModelDegenerate
from optimizer import *
from lts_model import LearnToSimModel
from logger import SimpleLogger, CometLogger, GANLogger
from base_model import BaseConditionalGenerationOracle, ShiftedOracle


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
torch.autograd.set_detect_anomaly(True)

def str_to_class(classname: str):
    """
    Function to get class object by its name signature
    :param classname: str
        name of the class
    :return: class object with the same name signature as classname
    """
    return getattr(sys.modules[__name__], classname)


def end_to_end_training(epochs: int,
                        model_cls: BaseConditionalGenerationOracle,
                        optimizer_cls: BaseOptimizer,
                        optimized_function_cls: BaseConditionalGenerationOracle,
                        logger: BaseLogger,
                        model_config: dict,
                        optimizer_config: dict,
                        n_samples_per_dim: int,
                        step_data_gen: float,
                        n_samples: int,
                        current_psi: Union[List[float], torch.tensor],
                        reuse_optimizer: bool = False,
                        reuse_model: bool = False,
                        shift_model: bool = False,
                        finetune_model: bool = False,
                        experiment = None
                        ):
    """

    :param epochs: int
        number of local training steps to perfomr
    :param model_cls: BaseConditionalGenerationOracle
        model that is able to generate samples and calculate loss function
    :param optimizer_cls: BaseOptimizer
    :param logger: BaseLogger
    :param model_config: dict
    :param optimizer_config: dict
    :param n_samples_per_dim: int
    :param step_data_gen: float
    :param n_samples: int
    :param current_psi:
    :param reuse_model:
    :param reuse_optimizer:
    :param finetune_model:
    :param shift_model:

    :return:
    """
    y_sampler = optimized_function_cls(device=device, psi_init=current_psi)
    model = model_cls(y_model=y_sampler, **model_config).to(device)
    optimizer = optimizer_cls(oracle=model,
                              x=current_psi,
                              **optimizer_config)

    for epoch in range(epochs):
        if reuse_optimizer:
            optimizer.update(oracle=model,
                             x=current_psi)
        else:
            # find new psi
            optimizer = optimizer_cls(oracle=model,
                                      x=current_psi,
                                      **optimizer_config)

        current_psi, status, history = optimizer.optimize()
        # with torch.no_grad(): current_psi += shift

        print(current_psi, status)
        try:
            # logging optimization, i.e. statistics of psi
            # logger.log_grads(model, y_sampler, current_psi, n_samples_per_dim)
            logger.log_performance(y_sampler=y_sampler,
                                   current_psi=current_psi,
                                   n_samples=n_samples)
            logger.log_optimizer(optimizer)
            # logger.log_gan_samples(model, y_sampler, current_psi)
            # logger.log_oracle(oracle=model,
            #                   y_sampler=y_sampler,
            #                   current_psi=current_psi,
            #                   step_data_gen=step_data_gen,
            #                   num_samples=50)
        except Exception as e:
            print(e)
            raise
        torch.cuda.empty_cache()

    return


@click.command()
@click.option('--model', type=str, default='GANModel')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--logger', type=str, default='CometLogger')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--model_config_file', type=str, default='gan_config')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--epochs', type=int, default=500)
@click.option('--n_samples', type=int, default=10)
@click.option('--step_data_gen', type=float, default=1.)
@click.option('--n_samples_per_dim', type=int, default=3000)
@click.option('--reuse_optimizer', type=bool, default=False)
@click.option('--reuse_model', type=bool, default=False)
@click.option('--shift_model', type=bool, default=False)
@click.option('--finetune_model', type=bool, default=False)
@click.option('--init_psi', type=str, default="0., 0.")
def main(model,
         optimizer,
         logger,
         optimized_function,
         project_name,
         work_space,
         tags,
         model_config_file,
         optimizer_config_file,
         epochs,
         n_samples,
         step_data_gen,
         n_samples_per_dim,
         reuse_optimizer,
         reuse_model,
         shift_model,
         finetune_model,
         init_psi
         ):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    print(optimizer_config)
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    model_config['psi_dim'] = psi_dim
    model_config['n_samples'] = n_samples
    model_config['n_samples_per_dim'] = n_samples_per_dim
    optimizer_config['x_step'] = step_data_gen

    optimized_function_cls = str_to_class(optimized_function)
    model_cls = str_to_class(model)
    optimizer_cls = str_to_class(optimizer)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameter('model_type', model)
    experiment.log_parameter('optimizer_type', optimizer)
    experiment.log_parameters(
        {"model_{}".format(key): value for key, value in model_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )

    logger = str_to_class(logger)(experiment)

    end_to_end_training(
        epochs=epochs,
        model_cls=model_cls,
        optimizer_cls=optimizer_cls,
        optimized_function_cls=optimized_function_cls,
        logger=logger,
        model_config=model_config,
        optimizer_config=optimizer_config,
        current_psi=init_psi,
        n_samples_per_dim=n_samples_per_dim,
        step_data_gen=step_data_gen,
        n_samples=n_samples,
        reuse_optimizer=reuse_optimizer,
        reuse_model=reuse_model,
        shift_model=shift_model,
        finetune_model=finetune_model,
        experiment=experiment
    )


if __name__ == "__main__":
    main()

