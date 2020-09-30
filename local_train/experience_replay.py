import torch
from scipy.stats import chi2
import numpy as np

class ExperienceReplay:
    def __init__(self, psi_dim, x_dim, y_dim, device, sphere_cut=False):
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._device = device
        self._sphere_cut = sphere_cut
        self._y = torch.zeros(0, self._y_dim).float().to('cpu')
        self._condition = torch.zeros(0, self._x_dim + self._psi_dim).float().to('cpu')

    def add(self, y, condition):
        if y is None and condition is None:
            y = torch.zeros(0, self._y_dim).float()
            condition = torch.zeros(0, self._x_dim + self._psi_dim).float()
        self._y = torch.cat([self._y, y.to('cpu').detach().clone()], dim=0)
        self._condition = torch.cat([self._condition, condition.to('cpu').detach().clone()], dim=0)
        return self

    def extract(self, psi, step):
        psi = psi.float().to('cpu').detach().clone()

        if self._sphere_cut:
            mask = ((self._condition[:, :self._psi_dim] - psi).pow(2).sum(dim=1).sqrt() < step)  # sphere
        else:
            mask = ((self._condition[:, :self._psi_dim] - psi).abs() < step).all(dim=1)

        if mask.sum() > 1000000:  # memory issues
            idx = np.random.choice(np.where(mask.detach().cpu().numpy())[0], size=1000000, replace=False)
            new_mask = np.zeros(len(mask))
            new_mask[idx] = 1.
            mask = torch.tensor(new_mask).bool()
        y = (self._y[mask]).to(self._device)
        condition = (self._condition[mask]).to(self._device)
        return y, condition


class ExperienceReplayClample:
    def __init__(self, psi_dim, x_dim, y_dim, device, sphere_cut=False):
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._device = device
        self._sphere_cut = sphere_cut
        self._y = torch.zeros(0, self._y_dim).float().to('cpu')
        self._condition = torch.zeros(0, self._x_dim + selsf._psi_dim).float().to('cpu')

    def add(self, y, condition):
        if y is None and condition is None:
            y = torch.zeros(0, self._y_dim).float()
            condition = torch.zeros(0, self._x_dim + self._psi_dim).float()
        self._y = torch.cat([self._y, y.to('cpu')], dim=0)
        self._condition = torch.cat([self._condition, condition.to('cpu')], dim=0)
        return self

    def extract(self, psi, step):
        psi = psi.float().to('cpu')
        if self._sphere_cut:
            mask = ((self._condition[:, :self._psi_dim] - psi).pow(2).sum(dim=1).sqrt() < step)  # sphere
        else:
            mask = ((self._condition[:, :self._psi_dim] - psi).abs() < step).all(dim=1)
        y = (self._y[mask]).to(self._device)
        condition = (self._condition[mask]).to(self._device)
        return y, condition


# TODO: add capping and resampling
class ExperienceReplayAdaptive:
    def __init__(self, psi_dim, x_dim, y_dim, device, sphere_cut=False):
        self._psi_dim = psi_dim
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._device = device
        self._sphere_cut = sphere_cut
        self._y = torch.zeros(0, self._y_dim).float().to('cpu')
        self._condition = torch.zeros(0, self._x_dim + self._psi_dim).float().to('cpu')

    def add(self, y, condition):
        self._y = torch.cat([self._y, y.to('cpu')], dim=0)
        self._condition = torch.cat([self._condition, condition.to('cpu')], dim=0)
        return self

    def extract(self, psi, sigma, q=0.95):
        psi = psi.float().to('cpu')
        mask = (((self._condition[:, :self._psi_dim] - psi).pow(2) / sigma.pow(2)).sum(dim=1) < chi2(df=self._psi_dim).ppf(q))
        y = (self._y[mask]).to(self._device)
        condition = (self._condition[mask]).to(self._device)
        return y, condition
