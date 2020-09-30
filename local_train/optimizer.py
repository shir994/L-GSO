from abc import ABC, abstractmethod
import numpy as np
from base_model import BaseConditionalGenerationOracle
from numpy.linalg import LinAlgError
from line_search_tool import LineSearchTool, get_line_search_tool
from pyro import distributions as dist
from torch import optim
from torch import nn
from logger import BaseLogger
from collections import defaultdict
import copy
import scipy
import matplotlib.pyplot as plt
import torch
import time
import sys
sys.path.append('../')
from lbfgs import LBFGS, FullBatchLBFGS
import copy
from scipy.stats import chi2
from sobol import sobol_generate

SUCCESS = 'success'
ITER_ESCEEDED = 'iterations_exceeded'
COMP_ERROR = 'computational_error'
SPHERE = True


class BaseOptimizer(ABC):
    """
    Base class for optimization of some function with logging
    functionality spread by all classes
    """
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 x_step: float = np.inf,  # step_data_gen
                 correct: bool = True,
                 tolerance: torch.Tensor = torch.tensor(1e-4),
                 trace: bool = True,
                 num_repetitions: int = 1000,
                 max_iters: int = 1000,
                 *args, **kwargs):
        self._oracle = oracle
        self._oracle.eval()
        self._history = defaultdict(list)
        self._x = x.clone().detach()
        self._x_init = copy.deepcopy(x)
        self._x_step = x_step
        self._tolerance = tolerance
        self._trace = trace
        self._max_iters = max_iters
        self._correct = correct
        self._num_repetitions = num_repetitions
        self._num_iter = 0.
        self._alpha_k = 0.
        self._previous_n_calls = 0

    def _update_history(self, init_time):
        self._history['time'].append(
            time.time() - init_time
        )
        self._history['func_evals'].append(
            self._oracle._n_calls - self._previous_n_calls
        )
        self._history['func'].append(
            self._oracle.func(self._x,
                              num_repetitions=self._num_repetitions).detach().cpu().numpy()
        )

        if not (
                (
                        type(self._oracle).__name__ in [
                    "BOCKModel",
                    "BostonNNTuning", "RosenbrockModelDegenerate",
                    "GaussianMixtureHumpModelDeepDegenerate", "NumericalDifferencesModel"]
                )
        ):
            self._history['grad'].append(
                self._oracle.grad(self._x,
                                  num_repetitions=self._num_repetitions).detach().cpu().numpy()
            )
        else:
            self._history['grad'].append(np.zeros_like(self._x.detach().cpu().numpy()))

        self._history['x'].append(
            self._x.detach().cpu().numpy()
        )
        self._history['alpha'].append(
            self._alpha_k
        )
        self._previous_n_calls = self._oracle._n_calls

    def optimize(self):
        """
        Run optimization procedure
        :return:
            torch.Tensor:
                x optim
            str:
                status_message
            defaultdict(list):
                optimization history
        """
        for i in range(self._max_iters):
            status = self._step()
            if status == COMP_ERROR:
                return self._x.detach().clone(), status, self._history
            elif status == SUCCESS:
                return self._x.detach().clone(), status, self._history
        return self._x.detach().clone(), ITER_ESCEEDED, self._history

    def update(self, oracle: BaseConditionalGenerationOracle, x: torch.Tensor, step=None):
        self._oracle = oracle
        self._x.data = x.data
        if step:
            self._x_step = step
        self._x_init = copy.deepcopy(x.detach().clone())
        self._history = defaultdict(list)

    @abstractmethod
    def _step(self):
        """
        Compute update of optimized parameter
        :return:
        """
        raise NotImplementedError('_step is not implemented.')

    def _post_step(self, init_time):
        """
        This function saves stats in history and forces
        :param init_time:
        :return:
        """
        if self._correct:
            if not SPHERE:
                self._x.data = torch.max(torch.min(self._x, self._x_init + self._x_step), self._x_init - self._x_step)
            else:
                # sphere cut
                x_corrected = self._x.data - self._x_init.data
                if x_corrected.norm() > self._x_step:
                    x_corrected = self._x_step * x_corrected / (x_corrected.norm())
                    x_corrected.data = x_corrected.data + self._x_init.data
                    self._x.data = x_corrected.data

        self._num_iter += 1
        if self._trace:
            self._update_history(init_time=init_time)

    def update_optimizer(self):
        pass


class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr
        with torch.no_grad():
            x_k = x_k + d_k * self._alpha_k
        grad_norm = torch.norm(d_k).item()
        self._x = x_k

        super()._post_step(init_time)
        # seems not cool to call super method in the middle of function...

        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class NewtonOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr  # in newton method learning rate used to initialize line search tool
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)
        self._alpha_k = None

    def _step(self):
        # seems like a bad dependence...
        init_time = time.time()
        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        d_k = -self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        h_d = self._oracle.hessian(x_k, num_repetitions=self._num_repetitions)
        try:
            c_and_lower = scipy.linalg.cho_factor(h_d.detach().cpu().numpy())
            d_k = scipy.linalg.cho_solve(c_and_lower, d_k.detach().cpu().numpy())
            d_k = torch.tensor(d_k).float().to(self._oracle.device)
        except LinAlgError:
            pass
        self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                           x_k,
                                                           d_k,
                                                           previous_alpha=self._lr,
                                                           num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr

        with torch.no_grad():
            x_k = x_k + d_k * self._alpha_k
        self._x = x_k
        super()._post_step(init_time)

        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


def d_computation_in_lbfgs(d, history):
    l = len(history)
    mu = list()
    for i in range(l)[::-1]:
        s = history[i][0]
        y = history[i][1]
        mu.append(s.dot(d) / s.dot(y))
        d -= y * mu[-1]
    mu = mu[::-1]
    s = history[-1][0]
    y = history[-1][1]
    d = d * s.dot(y) / y.dot(y)
    for i in range(l):
        s = history[i][0]
        y = history[i][1]
        beta = y.dot(d) / s.dot(y)
        d += (mu[i] - beta) * s
    return d


class LBFGSOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 memory_size: int = 20,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._sy_history = list()
        self._alpha_k = None
        self._memory_size = memory_size
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)

        if len(self._sy_history) > 0:
            d_k = d_computation_in_lbfgs(-g_k.clone().detach(), self._sy_history)
        else:
            d_k = - g_k.clone().detach()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)

        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr
        x_k = x_k + d_k * self._alpha_k
        self._x = x_k.clone().detach()
        g_k_new = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        self._sy_history.append((self._alpha_k * d_k, g_k_new - g_k))
        if len(self._sy_history) > self._memory_size:
            self._sy_history.pop(0)

        super()._post_step(init_time=init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class LBFGSNoisyOptimizer(BaseOptimizer):
    def __init__(
            self,
            oracle: BaseConditionalGenerationOracle,
            x: torch.Tensor,
            lr: float = 1e-1,
            memory_size: int = 5,
            line_search='Wolfe',
            lr_algo='None',
            *args, **kwargs
    ):
        super().__init__(oracle, x, *args, **kwargs)
        self._line_search = line_search
        self._lr = lr
        self._alpha_k = None
        self._lr_algo = lr_algo  # None, grad, dim
        if not (lr_algo in ["None", "Grad", "Dim"]):
            ValueError("lr_algo is not right")
        if self._x_step:
            self._optimizer = LBFGS(params=[self._x], lr=self._x_step / 10., line_search=line_search, history_size=memory_size)
        else:
            self._optimizer = LBFGS(params=[self._x], lr=self._lr, line_search=line_search, history_size=memory_size)

    def _step(self):
        x_k = self._x.detach().clone()
        x_k.requires_grad_(True)
        self._optimizer.param_groups[0]['params'][0] = x_k
        init_time = time.time()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        grad_normed = g_k  # (g_k / g_k.norm())
        self._state_dict = copy.deepcopy(self._optimizer.state_dict())

        if self._lr_algo == "None":
            self._optimizer.param_groups[0]['lr'] = self._x_step
        elif self._lr_algo == "Grad":
            self._optimizer.param_groups[0]['lr'] = self._x_step / g_k.norm().item()
        elif self._lr_algo == "Dim":
            self._optimizer.param_groups[0]['lr'] = self._x_step / np.sqrt(chi2.ppf(0.95, df=len(g_k)))
        # define closure for line search
        def closure():
            self._optimizer.zero_grad()
            loss = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
            return loss
        # two-loop recursion to compute search direction
        p = self._optimizer.two_loop_recursion(-grad_normed)
        options = {
            'closure': closure,
            'current_loss': f_k,
            'interpolate': False
        }
        if self._line_search == 'Wolfe':
            lbfg_opt = self._optimizer.step(p, grad_normed, options=options)
            f_k, d_k, lr = lbfg_opt[0], lbfg_opt[1], lbfg_opt[2]
        elif self._line_search == 'Armijo':
            lbfg_opt = self._optimizer.step(p, grad_normed, options=options)
            f_k, lr = lbfg_opt[0], lbfg_opt[1]
            d_k = -g_k
        elif self._line_search == 'None':
            # self._optimizer.param_groups[0]['lr'] = 1.
            d_k = -g_k
            lbfg_opt = self._optimizer.step(p, grad_normed, options=options)
            lr = lbfg_opt
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        grad_normed = g_k  # (g_k / g_k.norm())
        self._optimizer.curvature_update(grad_normed, eps=0.2, damping=False)
        self._lbfg_opt = lbfg_opt
        grad_norm = d_k.norm().item()
        self._x = x_k

        super()._post_step(init_time=init_time)

        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR

    def reverse_optimizer(self, **kwargs):
        self._optimizer.load_state_dict(self._state_dict)

class ConjugateGradientsOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 line_search_options: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._alpha_k = None
        self._d_k = None
        if not line_search_options:
            line_search_options = {
                'alpha_0': self._lr,
                'c':  self._lr
            }
        self._line_search_tool = get_line_search_tool(line_search_options)

    def _step(self):
        init_time = time.time()

        x_k = self._x.clone().detach()
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)
        g_k = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        if self._d_k is None:
            self._d_k = -g_k.clone().detach()

        norm_squared = g_k.pow(2).sum()

        if self._alpha_k is None:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=None,
                                                               num_repetitions=self._num_repetitions)
        else:
            self._alpha_k = self._line_search_tool.line_search(self._oracle,
                                                               x_k,
                                                               self._d_k,
                                                               previous_alpha=2 * self._alpha_k,
                                                               num_repetitions=self._num_repetitions)
        if self._alpha_k is None:
            print('alpha_k is None!')
            self._alpha_k = self._lr

        x_k = x_k + self._d_k * self._alpha_k
        g_k_next = self._oracle.grad(x_k, num_repetitions=self._num_repetitions)
        beta_k = g_k_next.dot((g_k_next - g_k)) / norm_squared
        self._d_k = -g_k_next + beta_k * self._d_k
        self._x = x_k.clone().detach()

        super()._post_step(init_time)
        grad_norm = torch.norm(g_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(self._d_k).all()):
            return COMP_ERROR


class TorchOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 torch_model: str = 'Adam',
                 optim_params: dict = {},
                 lr_algo: str = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._lr_algo = lr_algo
        self._alpha_k = self._lr
        self._torch_model = torch_model
        self._optim_params = optim_params
        self._base_optimizer = getattr(optim, self._torch_model)(
            params=[self._x], lr=lr, **self._optim_params
        )
        self._state_dict = copy.deepcopy(self._base_optimizer.state_dict())
        print(self._base_optimizer)

    def _step(self):
        init_time = time.time()
        self._base_optimizer.zero_grad()
        d_k = self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach()

        if self._lr_algo == "None":
            self._base_optimizer.param_groups[0]['lr'] = self._x_step
        elif self._lr_algo == "Grad":
            self._base_optimizer.param_groups[0]['lr'] = self._x_step / d_k.norm().item()
        elif self._lr_algo == "Dim":
            self._base_optimizer.param_groups[0]['lr'] = self._x_step / np.sqrt(chi2.ppf(0.95, df=len(d_k)))
        else:
            pass

        print("Grad: ", d_k)
        self._x.grad = d_k.detach().clone()
        self._state_dict = copy.deepcopy(self._base_optimizer.state_dict())
        self._base_optimizer.step()
        print("PSI", self._x)
        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(self._x).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR

    def reverse_optimizer(self, **kwargs):
        self._base_optimizer.load_state_dict(self._state_dict)


class GPOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 base_estimator="gp",
                 acq_func='gp_hedge',
                 acq_optimizer="sampling",
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        from skopt import Optimizer
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._opt_result = None
        borders = []
        x_step = self._x_step
        if x_step is None:
            x_step = self._lr
        for xi in x.detach().cpu().numpy():
            borders.append((xi - x_step, xi + x_step))
        self._base_optimizer = Optimizer(borders,
                                         base_estimator=base_estimator,
                                         acq_func=acq_func,
                                         acq_optimizer=acq_optimizer,
                                         acq_optimizer_kwargs={"n_jobs": -1})

    def optimize(self):
        f_k = self._oracle.func(self._x, num_repetitions=self._num_repetitions).item()
        self._base_optimizer.tell(
            self.bound_x(self._x.detach().cpu().numpy().tolist()),
            f_k
        )
        x, status, history = super().optimize()
        self._x = torch.tensor(self._opt_result.x).float().to(self._oracle.device)
        return self._x.detach().clone(), status, history

    def bound_x(self, x):
        x_new = []
        for xi, space in zip(x, self._base_optimizer.space):
            if xi in space:
                pass
            else:
                xi = np.clip(xi, space.low + 1e-3, space.high - 1e-3)
            x_new.append(xi)
        return x_new

    def _step(self):
        init_time = time.time()

        x_k = self._base_optimizer.ask()
        x_k = torch.tensor(x_k).float().to(self._oracle.device)
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)

        print(x_k, f_k)

        self._opt_result = self._base_optimizer.tell(
            self.bound_x(x_k.detach().cpu().numpy().tolist()),
            f_k.item()
        )
        self._x = torch.tensor(self._opt_result['x']).float().to(self._oracle.device) # x_k.detach().clone()
        print(self._opt_result['x'], self._opt_result['fun'])
        super()._post_step(init_time)
        # grad_norm = torch.norm(d_k).item()
        # if grad_norm < self._tolerance:
        #     return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all()):
            return COMP_ERROR


class HybridMC(nn.Module):
    def __init__(self,
                 model: BaseConditionalGenerationOracle,
                 l: int,
                 epsilon: float,
                 q: torch.Tensor,
                 num_repetitions: int,
                 covariance='unit',
                 t=1.):
        super(HybridMC, self).__init__()
        self._model = model
        self._epsilon = epsilon
        self._q = torch.tensor(q).float()
        self._dim_q = len(self._q)
        self._p = torch.zeros_like(self._q).to(model.device)
        self._num_repetitions = num_repetitions
        self._l = l
        self._t = t
        if covariance == 'unit':
            self._mean_p = torch.zeros(self._dim_q).to(model.device)
            self._cov_p = torch.eye(self._dim_q).to(model.device)

    def step(self, t=1.):
        self._t = t
        self._p = dist.MultivariateNormal(self._mean_p, covariance_matrix=self._cov_p).sample()
        q_old = self._q.clone().detach()
        p_old = self._p.clone().detach()
        # leap frogs steps
        for i in range(self._l):
            self._p = self._p - (self._epsilon / 2) * self._model.grad(self._q, num_repetitions=self._num_repetitions)
            self._q = self._q + self._epsilon * self._p
            self._p = self._p - (self._epsilon / 2) * self._model.grad(self._q, num_repetitions=self._num_repetitions)

        # metropolis acceptance step
        with torch.no_grad():
            H_end = (self._model.func(self._q, num_repetitions=self._num_repetitions) / self._t + self._p.pow(2).sum()).item()
            H_start = (self._model.func(q_old, num_repetitions=self._num_repetitions) / self._t + p_old.pow(2).sum()).item()

        acc_prob = min(1, np.exp(H_start - H_end))
        if not np.random.binomial(1, acc_prob):
            self._q = q_old.clone().detach()
        return self._q


class HMCOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 l: int = 20,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._base_optimizer = HybridMC(model=oracle,
                                        epsilon=lr,
                                        l=l,
                                        num_repetitions=self._num_repetitions,
                                        q=self._x.detach().clone())

    def _step(self):
        init_time = time.time()

        x_k = self._base_optimizer.step()
        self._x = x_k.clone().detach()
        d_k = self._oracle.grad(self._x, num_repetitions=self._num_repetitions).detach()
        f_k = self._oracle.func(self._x, num_repetitions=self._num_repetitions).detach()
        self._x.grad = d_k
        self._base_optimizer.step()
        self._base_optimizer.zero_grad()

        super()._post_step(init_time)
        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class LTSOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr

    def _step(self):
        init_time = time.time()
        x_k = self._x.clone().detach()
        d_k = -self._oracle.grad(x_k, update_baselines=True).detach()
        with torch.no_grad():
            x_k = x_k + d_k * self._lr
        grad_norm = torch.norm(d_k).item()

        self._x = x_k
        super()._post_step(init_time)

        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class CMAGES(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1e-1,
                 p: int = 10,
                 beta: float = 2.,
                 alpha: float = 0.5,
                 k: int = 20,
                 sigma2: float = (0.1)**2,
                 torch_model: str = 'Adam',
                 optim_params: dict = None,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._lr = lr
        self._grads = []
        self._p = p
        self._k = k
        self._beta = beta
        self._alpha = alpha
        self._sigma2 = sigma2
        self._torch_model = torch_model
        if not optim_params:
            optim_params = dict()
        self._optim_params = optim_params
        self._base_optimizer = getattr(optim, self._torch_model)(
            params=[self._x], lr=lr, **self._optim_params
        )

    def _step(self):
        init_time = time.time()
        x_k = self._x.clone().detach()
        d_k = -self._oracle.grad(x_k).detach()
        self._grads.append(d_k)
        if len(self._grads) > self._k:
            self._grads.pop(0)
        U, *_ = torch.svd(torch.stack(self._grads, dim=1))

        dim = len(self._x)
        covariance = (self._alpha / dim) * torch.eye(dim) + (1 - self._alpha) / self._k * torch.mm(U, U.t())
        es_grad = torch.zeros_like(self._x)
        for _ in range(self._p):
            noise = dist.MultivariateNormal(loc=torch.zeros(dim), covariance_matrix=self._sigma2 * covariance).sample()
            es_grad += noise * (
                    self._oracle.func(self._x + noise, num_repetitions=self._num_repetitions) -
                    self._oracle.func(self._x - noise, num_repetitions=self._num_repetitions)
            )

        d_k = (self._beta / (2 * self._sigma2 * self._p)) * es_grad

        self._base_optimizer.zero_grad()
        self._x.grad = d_k.detach().clone()
        self._base_optimizer.step()

        super()._post_step(init_time)

        grad_norm = torch.norm(d_k).item()
        if grad_norm < self._tolerance:
            return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(d_k).all()):
            return COMP_ERROR


class BOCKOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 num_init: int = 20,
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._opt_result = None
        borders = []
        x_step = self._x_step
        if x_step is None:
            x_step = self._lr
        for xi in x.detach().cpu().numpy():
            borders.append((xi - x_step, xi + x_step))
        self._borders = torch.tensor(borders).float().to(self._x).t()
        borders = self._borders.t()
        soboleng = torch.quasirandom.SobolEngine(dimension=len(self._x))
        x_tmp = soboleng.draw(num_init).float().to(self._x)
        x_tmp = x_tmp * (borders[:, 1] - borders[:, 0]) + borders[:, 0]
        self._X_dataset = torch.zeros(0, len(self._x)).to(self._x)
        self._y_dataset = torch.zeros(0, 1).to(self._x)
        self._y_noise = torch.zeros(0, 1).to(self._x)

        self._X_dataset = torch.cat(
            [
                self._X_dataset,
                self._x.detach().view(-1, len(self._x)),
                x_tmp.to(x).detach().view(-1, len(self._x))
            ],
            dim=0
        )
        func_x_, conditions_ = self._oracle._y_model.generate_data_at_point(n_samples_per_dim=self._num_repetitions, current_psi=self._x)
        func_x_ = self._oracle._y_model.loss(func_x_, conditions=conditions_)
        func_x_t = [
            self._oracle._y_model.loss(*self._oracle._y_model.generate_data_at_point(n_samples_per_dim=self._num_repetitions, current_psi=x_t)).detach()
            for x_t in x_tmp
        ]
        self._y_dataset = torch.cat(
            [
                self._y_dataset,
                func_x_.mean().detach().view(1, 1)
            ] + [func_x_t_.mean().detach().view(1, 1) for func_x_t_ in func_x_t]
        )
        self._y_noise = torch.cat(
            [
                self._y_noise,
                func_x_.std().detach().view(1, 1) / np.sqrt(len(func_x_))
            ] + [func_x_t_.std().detach().view(1, 1) / np.sqrt(len(func_x_t_)) for func_x_t_ in func_x_t]
        )
        self._state_dict = None

    def bound_x(self, x):
        x_new = []
        for xi, space in zip(x, self._base_optimizer.space):
            if xi in space:
                pass
            else:
                xi = np.clip(xi, space.low + 1e-3, space.high - 1e-3)
            x_new.append(xi)
        return x_new

    def _step(self):
        from gp_botorch import SingleTaskGP, ExpectedImprovement, bo_step, CustomCylindricalGP
        init_time = time.time()
        print(self._X_dataset.shape, self._y_dataset.shape)
        GP = lambda X, y, noise, borders: CustomCylindricalGP(X, y.view(-1, 1), noise, borders)
        acquisition = lambda gp, y: ExpectedImprovement(gp, y.min(), maximize=False)
        objective = lambda x: self._oracle._y_model.func(x, num_repetitions=self._num_repetitions)  # .view(-1, 1)
        print("_y_dataset", self._y_dataset[-1], self._X_dataset[-1], self._y_noise[-1])
        X, y, gp = bo_step(
            self._X_dataset,
            self._y_dataset.view(-1, 1),
            noise=self._y_noise.view(-1, 1),
            objective=objective,
            bounds=self._borders,
            GP=GP,
            acquisition=acquisition,
            q=1,
            state_dict=self._state_dict
        )
        self._state_dict = gp.state_dict()
        print(self._state_dict)
        self._X_dataset = X
        self._y_dataset = y
        f_k = self._y_dataset[-1]  # or best?
        x_k = self._X_dataset[-1]
        func_x_, conditions_ = self._oracle._y_model.generate_data_at_point(n_samples_per_dim=self._num_repetitions, current_psi=self._x)
        func_x_ = self._oracle._y_model.loss(func_x_, conditions=conditions_).detach()
        self._y_noise = torch.cat([self._y_noise, func_x_.std().view(1, 1) / np.sqrt(len(func_x_))])
        self._x = self._X_dataset[self._y_dataset.argmin()].clone().detach()
        super()._post_step(init_time)
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all()):
            return COMP_ERROR


"""
class GPBOOptimizer(BaseOptimizer):
    def __init__(self,
                 oracle: BaseConditionalGenerationOracle,
                 x: torch.Tensor,
                 lr: float = 1.,
                 base_estimator="gp",
                 acq_func='gp_hedge',
                 acq_optimizer="sampling",
                 *args, **kwargs):
        super().__init__(oracle, x, *args, **kwargs)
        from bayes_opt import BayesianOptimization, UtilityFunction

        self._x.requires_grad_(True)
        self._lr = lr
        self._alpha_k = self._lr
        self._opt_result = None
        borders = {}
        x_step = self._x_step
        if x_step is None:
            x_step = self._lr
        for i, xi in enumerate(x.detach().cpu().numpy()):
            borders["{}".format(i)] = (xi - x_step, xi + x_step)
        self._optimizer = BayesianOptimization(
            f=None,
            pbounds=borders,
            verbose=2,
            random_state=1,
        )
        self._utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    def x_to_dict(self, x):
        d = {}
        for i, xi in enumerate(x.detach().cpu().numpy()):
            d["{}".format(i)] = xi
        return d

    def dict_to_x(self, d):
        keys = sorted(d.keys())
        x = []
        for k in keys:
            x.append(d[k])
        return torch.tensor(x).float()

    def optimize(self):
        f_k = self._oracle.func(self._x, num_repetitions=self._num_repetitions).item()
        # self._optimizer.register(
        #     params=self.x_to_dict(self._x),
        #     target=f_k,
        # )

        x, status, history = super().optimize()
        self._x = self.dict_to_x(self._optimizer.max["params"]).float().to(self._oracle.device)
        return self._x.detach().clone(), status, history

    def bound_x(self, x):
        x_new = []
        for xi, space in zip(x, self._base_optimizer.space):
            if xi in space:
                pass
            else:
                xi = np.clip(xi, space.low + 1e-3, space.high - 1e-3)
            x_new.append(xi)
        return x_new

    def _step(self):
        init_time = time.time()

        x_k = self._optimizer.suggest(self._utility)
        x_k = self.dict_to_x(x_k).float().to(self._oracle.device)
        f_k = self._oracle.func(x_k, num_repetitions=self._num_repetitions)

        print(x_k, f_k)

        self._optimizer.register(
            params=self.x_to_dict(x_k),
            target=-f_k.item(),
        )

        self._x = self.dict_to_x(self._optimizer.max["params"]).float().to(self._oracle.device)
        # torch.tensor(self._opt_result['x']).float().to(self._oracle.device) # x_k.detach().clone()
        print(self._optimizer.max)
        super()._post_step(init_time)
        # grad_norm = torch.norm(d_k).item()
        # if grad_norm < self._tolerance:
        #     return SUCCESS
        if not (torch.isfinite(x_k).all() and
                torch.isfinite(f_k).all()):
            return COMP_ERROR
"""