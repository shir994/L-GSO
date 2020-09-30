from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
import sys


class BaseConditionalGeneratorModel(nn.Module, ABC):
    """
    Base class for implementation of conditional generation model.
    In our case condition is concatenation of psi and x,
    i.e. condition = torch.cat([psi, x], dim=1)
    """

    @property
    def device(self):
        """
        Just a nice class to get current device of the model
        Might be error-prone thou
        :return: torch.device
        """
        return next(self.parameters()).device

    @abstractmethod
    def fit(self, y, condition):
        """
        Computes the value of function at point x.
        :param y: target variable
        :param condition: torch.Tensor
            Concatenation of [psi, x]
        """
        raise NotImplementedError('fit is not implemented.')

    @abstractmethod
    def loss(self, y, condition):
        """
        Computes model loss for given y and condition.
        """
        raise NotImplementedError('loss is not implemented.')

    @abstractmethod
    def generate(self, condition):
        """
        Generates samples for given conditions
        """
        raise NotImplementedError('predict is not implemented.')

    @abstractmethod
    def log_density(self, y, condition):
        """
        Computes log density for given conditions and y
        """
        raise NotImplementedError('log_density is not implemented.')


def average_block_wise(x, num_repetitions):
    n = x.shape[0]
    if len(x.shape) == 1:
        return F.avg_pool1d(x.view(1, 1, n),
                            kernel_size=num_repetitions,
                            stride=num_repetitions).view(-1)
    elif len(x.shape) == 2:
        cols = x.shape[1]
        return F.avg_pool1d(x.view(1, cols, n),
                            kernel_size=num_repetitions,
                            stride=num_repetitions).view(-1, cols)
    else:
        NotImplementedError("average_block_wise do not support >2D tensors")


class BaseConditionalGenerationOracle(BaseConditionalGeneratorModel, ABC):
    """
    Base class for implementation of loss oracle.
    """
    def __init__(self, y_model, psi_dim, x_dim, y_dim):
        super(BaseConditionalGenerationOracle, self).__init__()
        self.__y_model = y_model
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._n_calls = 0

    @property
    def _y_model(self):
        return self.__y_model

    def func(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the value of function with specified condition.
        :param condition: torch.Tensor
            condition of models, i.e. psi
        :param num_repetitions:
        :return:
        """
        self._n_calls += 1
        condition = condition.to(self.device)
        if isinstance(num_repetitions, int):
            if len(condition.size()) == 1:
                conditions = condition.repeat(num_repetitions, 1)
            else:
                n = len(condition)
                conditions = condition.repeat(1, num_repetitions).view(num_repetitions * n, -1)
            conditions = torch.cat([
                conditions,
                self._y_model.sample_x(len(conditions)).to(self.device)
            ], dim=1)
            y = self.generate(conditions)
            # loss = self._y_model.loss(y=y)
            loss = self._y_model.loss(y=y, conditions=conditions)
            if len(condition.size()) == 1:
                return loss.mean()
            else:
                loss = average_block_wise(loss, num_repetitions=num_repetitions)
                return loss
        else:
            condition = torch.cat([
                condition,
                self._y_model.sample_x(len(condition)).to(self.device)
            ], dim=1)
            y = self.generate(condition)
            loss = self._y_model.loss(y=y, conditions=condition)
            return loss

    def grad(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the gradient of function with specified condition.
        If num_repetitions is not None then condition assumed
        to be 1-d tensor, which would be repeated num_repetitions times
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            1D torch tensor
        """
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            return grad([self.func(condition, num_repetitions=num_repetitions).sum()], [condition])[0]
        else:
            return grad([self.func(condition).sum()], [condition])[0]

    def hessian(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
        """
        Computes the Hessian matrix at point x.
        :param condition: torch.Tensor
            2D or 1D array on conditions for generator
        :param num_repetitions:
        :return: torch.Tensor
            2D torch tensor with second derivatives
        """
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        if isinstance(num_repetitions, int):
            return hessian_calc(self.func(condition, num_repetitions=num_repetitions).sum(), condition)
        else:
            return hessian_calc(self.func(condition).sum(), condition)


class ShiftedOracle:
    def __init__(self, oracle: BaseConditionalGenerationOracle, shift: torch.Tensor):
        self._oracle = oracle
        self._shift = shift.detach().clone()

    def set_shift(self, shift):
        self._shift = shift.detach().clone()

    def __getattr__(self, attr):
        orig_attr = self._oracle.__getattribute__(attr)
        if not hasattr(orig_attr, '__name__'):
            return orig_attr
        if orig_attr.__name__ in [
            'func',
            'grad',
            'hessian']:
            def hooked(*args, **kwargs):
                with torch.no_grad():
                    if 'condition' in kwargs:
                        condition = kwargs['condition']
                    else:
                        condition = args[0]
                        args = args[1:]
                    condition = condition.clone().detach()
                    condition = condition - self._shift
                    kwargs['condition'] = condition
                result = orig_attr(*args, **kwargs)
                if result is self._oracle:
                    return self
                return result
            return hooked
        elif orig_attr.__name__ in [
            'loss',
            'fit',
            'generate']:
            def hooked(*args, **kwargs):
                with torch.no_grad():
                    if 'condition' in kwargs:
                        condition = kwargs['condition']
                    else:
                        condition = args[0]
                        args = args[1:]
                    condition = condition.clone().detach()
                    kwargs['condition'] = condition
                    kwargs['condition'][:, :len(self._shift)] = kwargs['condition'][:, :len(self._shift)] - self._shift
                result = orig_attr(*args, **kwargs)
                if result is self._oracle:
                    return self
                return result
            return hooked
        elif callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                if result is self._oracle:
                    return self
                return result
            return hooked
        else:
            return orig_attr
