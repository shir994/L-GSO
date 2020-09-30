import numpy as np
from numpy.linalg import LinAlgError
from base_model import BaseConditionalGenerationOracle
import scipy
from datetime import datetime
from collections import defaultdict
import torch
import time


class OracleWrapper:
    def __init__(self, oracle: BaseConditionalGenerationOracle, num_repetitions):
        self._oracle = oracle
        self._num_repetitions = num_repetitions

    def func(self, x):
        x = torch.tensor(x).float().to(self._oracle.device)
        return self._oracle.func(x, num_repetitions=self._num_repetitions).item()

    def grad(self, x):
        x = torch.tensor(x).float().to(self._oracle.device)
        return self._oracle.grad(x, num_repetitions=self._num_repetitions).detach().cpu().numpy()

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        return np.squeeze(self.grad(x + alpha * d)).dot(d)


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Armijo', **kwargs):
        self._method = method
        self.alpha_0 = None
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None, num_repetitions=1000):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : torch.Tensor
            Starting point
        d_k : torch.Tensor
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if isinstance(x_k, torch.Tensor):
            x_k = x_k.detach().cpu().numpy()
        if isinstance(d_k, torch.Tensor):
            d_k = d_k.detach().cpu().numpy()
        oracle = OracleWrapper(oracle, num_repetitions=num_repetitions)
        if previous_alpha is None:
            previous_alpha = self.alpha_0
        else:
            previous_alpha = previous_alpha
        method = self._method
        # new_alpha = None
        if method == 'Wolfe':
            new_alpha = scipy.optimize.line_search(oracle.func, oracle.grad, x_k, d_k, c1=self.c1, c2=self.c2)[0]
            if not (new_alpha is None):
                return new_alpha
            else:
                method = 'Armijo'
        elif method == 'Armijo':
            while oracle.func_directional(x_k, d_k, previous_alpha) > oracle.func_directional(x_k, d_k, 0) + self.c1 * previous_alpha * oracle.grad_directional(x_k, d_k, 0):
                previous_alpha = previous_alpha / 2
            return previous_alpha
        elif method == 'Constant':
            return self.c
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()
