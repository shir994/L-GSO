import torch
from torch import nn
from torch import optim
import sys
sys.path.append('../')
from base_model import BaseConditionalGenerationOracle
import pyro
from pyro import distributions as dist
from torch.nn import functional as F
from torch.autograd import grad


class ControlVariate(nn.Module):
    def __init__(self, psi_dim):
        super(ControlVariate, self).__init__()
        self.lin1 = nn.Linear(psi_dim, 10)
        self.lin2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = self.lin2(x)
        return x


class NormalPolicy:
    def __init__(self):
        pass

    def __call__(self, mu, sigma, N=1):
        return pyro.sample("psi", dist.Normal(mu.repeat(N, 1), (1. + sigma.repeat(N, 1).exp()).log()))

    def log_prob(self, mu, sigma, x):
        return dist.Normal(mu, (1. + sigma.exp()).log()).log_prob(x)


class VoidModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 psi,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 K: int,
                 num_repetitions: int,
                 lr: float,
                 cv_lr: float,
                 fill_value_std: float = 0.,
                 ):
        super().__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        # self._psi = init_psi.clone()
        self._device = y_model.device
        self._psi = psi.clone().detach().to(self._device)
        self._psi.requires_grad_(True)
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._K = K
        self._lr = lr
        self._cv_lr = cv_lr
        self._policy = NormalPolicy()
        self._control_variate = ControlVariate(psi_dim).to(self._device)
        self._control_variate_parameters = list(self._control_variate.parameters())
        self._sigma = torch.full((psi_dim, ), fill_value=fill_value_std, requires_grad=True, device=self._device)
        self._num_repetitions = num_repetitions
        self._optimizer = optim.Adam(params=[self._psi, self._sigma] + self._control_variate_parameters, lr=cv_lr)

    def sample_conditions(self, n: int) -> torch.Tensor:
        conditions = self._policy(self._psi, self._sigma, N=n).to(self._device)
        return conditions

    def policy_grad_true(self, condition, **kwargs):
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        x_grad_total = torch.zeros_like(condition)
        # for k in range(self._K):
        action = self._policy(condition, self._sigma, N=self._K)
        # print(action.shape)
        r = self._y_model.func(action, num_repetitions=self._num_repetitions).view(-1)  # no .detach()!
        x_grad_1 = grad([r.sum()], [condition], retain_graph=True)[0]
        x_grad = x_grad_1
        x_grad_total += x_grad / self._K
        return x_grad_total.clone().detach()

    def grad(self, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)
        x_grad_total = torch.zeros_like(condition)
        # for k in range(self._K):
        action = self._policy(condition, self._sigma, N=self._K)
        r = self._y_model.func(action, num_repetitions=self._num_repetitions).detach().view(-1)
        c = self._control_variate(action).view(-1)
        log_prob_condition = condition.detach().clone().to(self.device)
        log_prob_condition.requires_grad_(True)
        log_prob = self._policy.log_prob(mu=log_prob_condition, sigma=self._sigma, x=action.detach())
        x_grad_1 = grad([(log_prob * ((r - c).view(-1, 1)).detach()).sum()], [log_prob_condition], retain_graph=True)[0]
        x_grad_2 = grad([c.sum()], [condition], retain_graph=True)[0]
        x_grad = x_grad_1 + x_grad_2
        x_grad_total += x_grad / self._K

        return x_grad_total.clone().detach()

    def step(self):
        self._optimizer.zero_grad()
        self._psi.grad = torch.zeros_like(self._psi)
        self._sigma.grad = torch.zeros_like(self._sigma)
        for parameter in self._control_variate_parameters:
            parameter.grad = torch.zeros_like(parameter)

        for k in range(self._K):
            action = self._policy(self._psi, self._sigma, N=1)
            r = self._y_model.func(action, num_repetitions=self._num_repetitions).detach().view(1)
            c = self._control_variate(action).view(1)
            log_prob_condition = self._psi.detach().clone().to(self.device)
            log_prob_condition.requires_grad_(True)
            log_prob = self._policy.log_prob(mu=log_prob_condition, sigma=self._sigma, x=action.detach())
            x_grad_1, sigma_grad_1 = grad([log_prob.mean()], [log_prob_condition, self._sigma], retain_graph=True, create_graph=True)
            x_grad_2, sigma_grad_2 = grad([c.sum()], [self._psi, self._sigma], retain_graph=True, create_graph=True)
            x_grad = x_grad_1 * (r - c) + x_grad_2
            sigma_grad = sigma_grad_1 * (r - c) + sigma_grad_2

            parameters_grad = grad(
                [(x_grad.pow(2).mean() + sigma_grad.pow(2).mean())],
                self._control_variate_parameters
            )

            with torch.no_grad():
                # print("x_grad.clone().detach() ", x_grad.clone().detach().shape)
                self._psi.grad += x_grad.clone().detach() / self._K
                self._sigma.grad += sigma_grad.clone().detach() / self._K
                for parameter, parameter_grad in zip(self._control_variate_parameters, parameters_grad):
                    parameter.grad += parameter_grad.clone().detach() / self._K
        self._optimizer.step()

    def fit(self, x, current_psi):
        pass

    def generate(self, condition):
        return self._y_model.generate(condition)

    def log_density(self, y, condition):
        pass

    def loss(self, y, condition):
        pass
