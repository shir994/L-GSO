from comet_ml import Experiment
import traceback
import sys
import os
import click
import torch
import numpy as np
sys.path.append('../')
sys.path.append('./RegressionNN')
from typing import List, Union
from model import YModel, RosenbrockModel, MultimodalSingularityModel, GaussianMixtureHumpModel, \
                  LearningToSimGaussianModel, \
                  ModelDegenerate, ModelInstrict, Hartmann6, \
                  RosenbrockModelInstrict, RosenbrockModelDegenerate, RosenbrockModelDegenerateInstrict, BOCKModel, \
                  RosenbrockModelDeepDegenerate, GaussianMixtureHumpModelDeepDegenerate, \
                  GaussianMixtureHumpModelDegenerate, RosenbrockModelDeepDegenerate, BostonNNTuning
#from ffjord_model import FFJORDModel
from gan_model import GANModel
from optimizer import *
from logger import SimpleLogger, CometLogger, GANLogger, RegressionLogger
from base_model import BaseConditionalGenerationOracle, ShiftedOracle
from constraints_utils import make_box_barriers, add_barriers_to_oracle
from experience_replay import ExperienceReplay, ExperienceReplayAdaptive
REWEIGHT = False
if REWEIGHT:
    from hep_ml import reweight
from base_model import average_block_wise


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
                        use_experience_replay: bool =True,
                        add_box_constraints: bool=False,
                        experiment=None,
                        scale_psi=False
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
    gan_logger = GANLogger(experiment)
    # gan_logger = RegressionLogger(experiment)
    # gan_logger = None

    y_sampler = optimized_function_cls(device=device, psi_init=current_psi)
    model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)

    optimizer = optimizer_cls(
        oracle=model,
        x=current_psi,
        **optimizer_config
    )
    print(model_config)
    exp_replay = ExperienceReplay(
        psi_dim=model_config['psi_dim'],
        y_dim=model_config['y_dim'],
        x_dim=model_config['x_dim'],
        device=device
    )
    weights = None

    logger.log_performance(y_sampler=y_sampler,
                           current_psi=current_psi,
                           n_samples=n_samples)
    for epoch in range(epochs):
        # generate new data sample
        x, condition = y_sampler.generate_local_data_lhs(
            n_samples_per_dim=n_samples_per_dim,
            step=step_data_gen,
            current_psi=current_psi,
            n_samples=n_samples)
        if x is None and condition is None:
            print("Empty training set, continue")
            continue
        x_exp_replay, condition_exp_replay = exp_replay.extract(psi=current_psi, step=step_data_gen)
        exp_replay.add(y=x, condition=condition)
        x = torch.cat([x, x_exp_replay], dim=0)
        condition = torch.cat([condition, condition_exp_replay], dim=0)
        used_samples = n_samples

        # breaking things
        if model_config.get("predict_risk", False):
            condition = condition[::n_samples_per_dim, :current_psi.shape[0]]
            x = y_sampler.func(condition, num_repetitions=n_samples_per_dim).reshape(-1, x.shape[1])
        print(x.shape, condition.shape)
        ## Scale train set
        if scale_psi:
            scale_factor = 10
            feature_max = condition[:, :model_config['psi_dim']].max(axis=0)[0]
            y_sampler.scale_factor = scale_factor
            y_sampler.feature_max = feature_max
            y_sampler.scale_psi = True
            print("MAX FEATURES", feature_max)
            condition[:, :model_config['psi_dim']] /= feature_max * scale_factor
            current_psi = current_psi / feature_max * scale_factor
            print(feature_max.shape, current_psi.shape)
            print("MAX PSI", current_psi)

        model.train()
        if reuse_model:
            if shift_model:
                if isinstance(model, ShiftedOracle):
                    model.set_shift(current_psi.clone().detach())
                else:
                    model = ShiftedOracle(oracle=model, shift=current_psi.clone().detach())
                model.fit(x, condition=condition, weights=weights)
            else:
                model.fit(x, condition=condition, weights=weights)
        else:
            # if not reusing model
            # then at each epoch re-initialize and re-fit
            model = model_cls(y_model=y_sampler, **model_config, logger=gan_logger).to(device)
            print("y_shape: {}, cond: {}".format(x.shape, condition.shape))
            model.fit(x, condition=condition, weights=weights)

        model.eval()

        if reuse_optimizer:
            optimizer.update(oracle=model,
                             x=current_psi)
        else:
            # find new psi
            optimizer = optimizer_cls(oracle=model,
                                      x=current_psi,
                                      **optimizer_config)

        if add_box_constraints:
            box_barriers = make_box_barriers(current_psi, step_data_gen)
            add_barriers_to_oracle(oracle=model, barriers=box_barriers)

        previous_psi = current_psi.clone()
        current_psi, status, history = optimizer.optimize()
        if scale_psi:
            current_psi, status, history = optimizer.optimize()
            current_psi = current_psi / scale_factor * feature_max
            y_sampler.scale_psi = False
            print("NEW_PSI: ", current_psi)

        try:
            # logging optimization, i.e. statistics of psi
            logger.log_grads(model, y_sampler, current_psi, n_samples_per_dim, log_grad_diff=False)
            logger.log_optimizer(optimizer)
            logger.log_performance(y_sampler=y_sampler,
                                   current_psi=current_psi,
                                   n_samples=n_samples)
            experiment.log_metric("used_samples_per_step", used_samples)
            experiment.log_metric("sample_size", len(x))

        except Exception as e:
            print(e)
            print(traceback.format_exc())
            # raise
        torch.cuda.empty_cache()
    logger.func_saver.join()
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
@click.option('--epochs', type=int, default=10000)
@click.option('--n_samples', type=int, default=10)
@click.option('--lr', type=float, default=1e-1)
@click.option('--step_data_gen', type=float, default=0.1)
@click.option('--n_samples_per_dim', type=int, default=3000)
@click.option('--reuse_optimizer', type=bool, default=False)
@click.option('--reuse_model', type=bool, default=False)
@click.option('--shift_model', type=bool, default=False)
@click.option('--finetune_model', type=bool, default=False)
@click.option('--add_box_constraints', type=bool, default=False)
@click.option('--use_experience_replay', type=bool, default=True)
@click.option('--init_psi', type=str, default="0., 0.")
@click.option('--scale_psi', type=bool, default=False)
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
         lr,
         finetune_model,
         use_experience_replay,
         add_box_constraints,
         init_psi,
         scale_psi
         ):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    init_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    model_config['psi_dim'] = psi_dim
    optimizer_config['x_step'] = step_data_gen
    optimizer_config['lr'] = lr

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
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('optim_params', {}).items()}
    )

    logger = str_to_class(logger)(experiment)
    print("Using device = {}".format(device))

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
        add_box_constraints=add_box_constraints,
        use_experience_replay=use_experience_replay,
        experiment=experiment,
        scale_psi=scale_psi
    )


if __name__ == "__main__":
    main()

