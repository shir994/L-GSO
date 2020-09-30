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
from model import YModel, LearningToSimGaussianModel, GaussianMixtureHumpModel
from optimizer import *
from typing import Callable
import time
import pyro
from torch.autograd import grad
from pyro import distributions as dist
from void_model import VoidModel
from optimizer import SUCCESS, ITER_ESCEEDED, COMP_ERROR


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


@click.command()
@click.option('--logger', type=str, default='CometLogger')
@click.option('--model_config_file', type=str, default='void_config')
@click.option('--optimizer', type=str, default='GradientDescentOptimizer')
@click.option('--optimized_function', type=str, default='YModel')
@click.option('--optimizer_config_file', type=str, default='optimizer_config')
@click.option('--project_name', type=str, prompt='Enter project name')
@click.option('--work_space', type=str, prompt='Enter workspace name')
@click.option('--tags', type=str, prompt='Enter tags comma separated')
@click.option('--epochs', type=int, default=500)
@click.option('--reuse_optimizer', type=bool, default=True)
@click.option('--init_psi', type=str, default="0., 0.")
def main(
        logger,
        optimizer,
        optimized_function,
        optimizer_config_file,
        model_config_file,
        project_name,
        work_space,
        tags,
        reuse_optimizer,
        init_psi,
        epochs
):
    model_config = getattr(__import__(model_config_file), 'model_config')
    optimizer_config = getattr(__import__(optimizer_config_file), 'optimizer_config')
    current_psi = torch.tensor([float(x.strip()) for x in init_psi.split(',')]).float().to(device)
    psi_dim = len(init_psi)
    print(psi_dim, current_psi)
    optimized_function_cls = str_to_class(optimized_function)
    optimizer_cls = str_to_class(optimizer)

    experiment = Experiment(project_name=project_name, workspace=work_space)
    experiment.add_tags([x.strip() for x in tags.split(',')])
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.items()}
    )
    experiment.log_parameters(
        {"optimizer_{}".format(key): value for key, value in optimizer_config.get('line_search_options', {}).items()}
    )

    logger = str_to_class(logger)(experiment)
    y_model = optimized_function_cls(device=device, psi_init=current_psi)


    model = VoidModel(y_model=y_model, **model_config)
    optimizer = optimizer_cls(
        oracle=model,
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
        try:
            logger.log_optimizer(optimizer)
            logger.log_grads(model, y_sampler=y_model, current_psi=current_psi, num_repetitions=5000)
            logger.log_performance(y_sampler=y_model,
                                   current_psi=current_psi,
                                   n_samples=5000)

        except Exception as e:
            print(e)
            raise

if __name__ == "__main__":
    main()