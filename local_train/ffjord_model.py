import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as dataset_utils
import copy
from base_model import BaseConditionalGenerationOracle
import sys
sys.path.append('./ffjord/')
import ffjord
import ffjord.lib
import ffjord.lib.utils as utils
from ffjord.lib.visualize_flow import visualize_transform
import ffjord.lib.layers.odefunc as odefunc
from ffjord.train_misc import standard_normal_logprob, create_regularization_fns
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint
from ffjord.custom_model import build_model_tabular, get_transforms, compute_loss
import lib.layers as layers
from tqdm import tqdm, trange
from typing import Tuple
import swats
import warnings
import numpy as np
warnings.filterwarnings("ignore")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, improvement=1e-4):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self._improvement = improvement
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self._improvement:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class FFJORDModel(BaseConditionalGenerationOracle):
    def __init__(self,
                 y_model: BaseConditionalGenerationOracle,
                 x_dim: int,
                 psi_dim: int,
                 y_dim: int,
                 num_blocks: int = 1,
                 lr: float = 1e-3,
                 epochs: int = 10,
                 bn_lag: float = 1e-3,
                 instance_noise_std: float = None,
                 log_prob_grad_penalty: float = None,
                 batch_norm: bool = True,
                 solver='fixed_adams',  # dopri5 fixed_adams
                 hidden_dims: Tuple[int] = (32, 32),
                 logger=None,
                 **kwargs):
        super(FFJORDModel, self).__init__(y_model=y_model, x_dim=x_dim, psi_dim=psi_dim, y_dim=y_dim)
        self._x_dim = x_dim
        self._y_dim =y_dim
        self._psi_dim = psi_dim
        self._model = build_model_tabular(dims=self._y_dim,
                                          condition_dim=self._psi_dim + self._x_dim,
                                          layer_type='concat_v2',
                                          num_blocks=num_blocks,
                                          rademacher=False,
                                          nonlinearity='tanh',
                                          solver=solver,
                                          hidden_dims=hidden_dims,
                                          bn_lag=bn_lag,
                                          batch_norm=batch_norm,
                                          regularization_fns=None)
        self._sample_fn, self._density_fn = get_transforms(self._model)
        self._epochs = epochs
        self._lr = lr
        self.logger = logger
        self._cond_mean = torch.zeros(self._x_dim + self._psi_dim).float().to(y_model._device)
        self._cond_std = torch.ones(self._x_dim + self._psi_dim).float().to(y_model._device)
        self._y_mean = torch.zeros(self._y_dim).float().to(y_model._device)
        self._y_std = torch.ones(self._y_dim).float().to(y_model._device)
        self._instance_noise_std = instance_noise_std
        self._log_prob_grad_penalty = log_prob_grad_penalty

    def loss(self, y, condition, weights=None):
        return compute_loss(self._model, data=y.detach(), condition=condition.detach())

    @staticmethod
    def instance_noise(data, std):
        return data + torch.randn_like(data) * data.std(dim=0) * std

    def log_prob_grad_penalty(self, condition, y):
        alpha = torch.rand(len(condition), 1).expand(condition.size()).to(condition)
        condition_hat = Variable(
            alpha * condition.data +
            (1 - alpha) * (
                    condition.data +
                    self._log_prob_grad_penalty * condition.data.std(dim=0) * torch.rand(condition.size()).to(condition)
            ),
            requires_grad=True
        )
        _, density_fn = get_transforms(self._model)
        density = density_fn(y, condition_hat)
        gradients = torch.autograd.grad(
            outputs=density,
            inputs=condition_hat,
            grad_outputs=torch.ones(density.size()).to(condition),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1.) ** 2).mean()
        print('gradient_penalty', gradient_penalty.item())
        return gradient_penalty

    def fit(self, y, condition, weights=None):
        self._cond_mean = condition.mean(0).detach().clone()
        self._cond_std = condition.std(0).detach().clone()
        self._y_mean = y.mean(0).detach().clone()
        self._y_std = y.std(0).detach().clone()

        y = (y - self._y_mean) / self._y_std
        condition = (condition - self._cond_mean) / self._cond_std

        self.train()
        print(self.device)
        trainable_parameters = list(self._model.parameters())
        optimizer = swats.SWATS(trainable_parameters, lr=self._lr, verbose=True)
        best_params = self._model.state_dict()
        best_loss = 1e6
        early_stopping = EarlyStopping(patience=200, verbose=True)
        """
        dataset = dataset_utils.TensorDataset(condition, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=262144, shuffle=True, pin_memory=True)
        for epoch in range(self._epochs):
            loss_sum = 0.
            for condition_batch, y_batch in train_loader:
                condition_batch = condition_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = self.loss(y_batch, condition_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            if loss_sum < best_loss:
                best_params = copy.deepcopy(self._model.state_dict())
                best_loss = loss_sum
            early_stopping(loss_sum)
            if early_stopping.early_stop:
                break
        """
        dataset = dataset_utils.TensorDataset(condition, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128000, shuffle=True)
        for _ in tqdm(range(self._epochs)):
            loss_sum = 0.
            for condition_batch, y_batch in train_loader:
                if self._instance_noise_std:
                    y_batch = self.instance_noise(y_batch, self._instance_noise_std)
                optimizer.zero_grad()
                loss = self.loss(y_batch, condition_batch)
                print("log prob", loss.item())
                if self._log_prob_grad_penalty:
                    loss = loss + self.log_prob_grad_penalty(condition_batch, y_batch)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            if loss_sum < best_loss:
                best_params = copy.deepcopy(self._model.state_dict())
                best_loss = loss_sum
            early_stopping(loss_sum)
            if early_stopping.early_stop:
                break
        self._model.load_state_dict(best_params)
        self.eval()
        self._sample_fn, self._density_fn = get_transforms(self._model)
        if self.logger:
            self.logger.log_validation_metrics(self._y_model, y, condition, self,
                                               (condition[:, :self._psi_dim].min(dim=0)[0].view(-1),
                                                condition[:, :self._psi_dim].max(dim=0)[0].view(-1)),
                                               batch_size=1000, calculate_validation_set=False)
            self.logger.add_up_epoch()
        return self

    def generate(self, condition):
        condition = (condition - self._cond_mean) / self._cond_std
        n = len(condition)
        z = torch.randn(n, self._y_dim).to(self.device)
        y = self._sample_fn(z, condition)
        y = y * self._y_std + self._y_mean
        return y

    def log_density(self, y, condition):
        return self._density_fn(y, condition)

    def train(self):
        super().train(True)
        for module in self._model.modules():
            if hasattr(module, 'odeint'):
                module.__setattr__('odeint', odeint_adjoint)

    def eval(self):
        super().train(False)
        # for module in self._model.modules():
        #   if hasattr(module, 'odeint'):
        #        module.__setattr__('odeint', odeint)
