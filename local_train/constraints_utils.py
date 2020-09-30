from base_model import BaseConditionalGenerationOracle
from typing import Callable
import torch
import types


class LinearConstraint:
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __call__(self, x):
        if x.ndimension() == 1:
            return (self._a * x).sum() - self._b
        elif x.ndimension() == 2:
            return torch.mv(x, self._a) - self._b


class LogBarrier:
    def __init__(self, g: Callable):
        self._g = g

    def __call__(self, x):
        barrier = - (- self._g(x)).log()
        return barrier


def func_with_barriers(self, condition: torch.Tensor, num_repetitions: int = None) -> torch.Tensor:
    func = self.__func(condition=condition, num_repetitions=num_repetitions)
    bounds_regularization = 0.
    if hasattr(self, '_barrier_lr'):
        barrier_lr = self._barrier_lr
    else:
        barrier_lr = 1.
    for bound in self._barriers:
        bounds_regularization += barrier_lr * bound(condition)
    return func + bounds_regularization


def add_barriers_to_oracle(oracle, barriers):
    """
    """
    setattr(oracle, '__func', oracle.__getattribute__('func'))
    setattr(oracle, 'func', types.MethodType(func_with_barriers, oracle))
    setattr(oracle, '_barriers', barriers)


def make_box_barriers(psis, step):
    constraints = []
    for i, psi in enumerate(psis):
        a = torch.zeros_like(psis).to(psis.device)
        a[i] = 1.
        b = psi + step
        constraints.append(LogBarrier(LinearConstraint(a=a, b=b)))

        a = torch.zeros_like(psis).to(psis.device)
        a[i] = -1.
        b = -(psi - step)
        constraints.append(LogBarrier(LinearConstraint(a=a, b=b)))
    return constraints
