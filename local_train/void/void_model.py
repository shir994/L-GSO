import torch
from torch import nn
from torch import optim
from base_model import BaseConditionalGenerationOracle
import pyro
from pyro import distributions as dist
from torch.nn import functional as F
from torch.autograd import grad


class ControlVariate(nn.Module):
    def __init__(self, psi_dim):
        super(ControlVariate, self).__init__()
        self.lin1 = nn.Linear(psi_dim, 5)
        self.lin2 = nn.Linear(5, 1)

    def forward(self, x):
        x = F.tanh(self.lin1(x))
        x = self.lin2(x)
        return x


class NormalPolicy:
    def __init__(self):
        pass

    def __call__(self, mu, sigma, N=1):
        return pyro.sample("psi", dist.Normal(mu.repeat(N, 1), (1 + sigma.repeat(N, 1).exp()).log()))

    def log_prob(self, mu, sigma, x):
        return dist.Normal(mu, (1 + sigma.exp()).log()).log_prob(x)


class VoidModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 psi,
                 psi_dim: int,
                 y_dim: int,
                 x_dim: int,
                 K: int = 1,
                 num_repetitions: int = 3000,
                 lr: float = 1.):
        super().__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        # self._psi = init_psi.clone()
        self._device = y_model.device
        self._psi = psi.clone().detach().to(self._device)
        self._psi.requires_grad_(True)
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._K = K
        self._lr = lr
        self._alpha_k = self._lr
        self._policy = NormalPolicy()
        self._control_variate = ControlVariate(psi_dim).to(self._device)
        self._control_variate_parameters = list(self._control_variate.parameters())
        self._sigma = torch.zeros(psi_dim, requires_grad=True, device=self._device)
        self._num_repetitions = num_repetitions
        self._optimizer = optim.Adam(params=[self._psi, self._sigma] + self._control_variate_parameters)

    def grad(self, condition: torch.Tensor, **kwargs) -> torch.Tensor:
        condition = condition.detach().clone().to(self.device)
        condition.requires_grad_(True)

        action = self._policy(condition, self._sigma, N=self._K)
        r = self._y_model.func(action, num_repetitions=self._num_repetitions).view(-1, 1)
        c = self._control_variate(action)
        log_prob = self._policy.log_prob(mu=condition, sigma=self._sigma, x=action.detach()).mean()

        x_grad_1, sigma_grad_1 = grad([log_prob], [condition, self._sigma], retain_graph=True, create_graph=True)
        x_grad_2, sigma_grad_2 = grad([c.mean()], [condition, self._sigma], retain_graph=True, create_graph=True)

        x_grad = x_grad_1 * (r - c) + x_grad_2
        sigma_grad = sigma_grad_1 * (r - c) + sigma_grad_2

        return x_grad.mean(dim=0).clone().detach()

    def step(self):
        self._optimizer.zero_grad()
        self._psi.grad = torch.zeros_like(self._psi)
        self._sigma.grad = torch.zeros_like(self._sigma)
        for parameter in self._control_variate_parameters:
            parameter.grad = torch.zeros_like(parameter)

        action = self._policy(self._psi, self._sigma, N=self._K)
        r = self._y_model.func(action, num_repetitions=self._num_repetitions).view(-1, 1)
        c = self._control_variate(action)
        log_prob = self._policy.log_prob(mu=self._psi, sigma=self._sigma, x=action.detach()).mean()

        x_grad_1, sigma_grad_1 = grad([log_prob], [self._psi, self._sigma], retain_graph=True, create_graph=True)
        x_grad_2, sigma_grad_2 = grad([c.mean()], [self._psi, self._sigma], retain_graph=True, create_graph=True)

        x_grad = x_grad_1 * (r - c) + x_grad_2
        sigma_grad = sigma_grad_1 * (r - c) + sigma_grad_2

        parameters_grad = grad([x_grad.mean(dim=0).pow(2).mean() + sigma_grad.mean(dim=0).pow(2).mean()],
                                self._control_variate_parameters)

        with torch.no_grad():
            self._psi.grad = x_grad.mean(dim=0).clone().detach()
            self._sigma.grad = sigma_grad.mean(dim=0).clone().detach()
            for parameter, parameter_grad in zip(self._control_variate_parameters, parameters_grad):
                parameter.grad = parameter_grad.clone().detach()
        self._optimizer.step()

    def fit(self, x, current_psi):
        pass

    def generate(self, condition):
        return self._y_model.generate(condition)

    def log_density(self, y, condition):
        pass

    def loss(self, y, condition):
        pass