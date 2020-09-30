"""
code partially taken from
https://github.com/bayesgroup/deepbayes-2019/blob/master/seminars/day4/gp/BayesOpt/bayesopt_solution.ipynb
"""
import torch
import gpytorch
import botorch
from botorch.models import HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import joint_optimize, sequential_optimize, optimize_acqf
from botorch.acquisition import ExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.kernels import CylindricalKernel, MaternKernel, ScaleKernel
from botorch.optim.fit import fit_gpytorch_torch
import traceback
import numpy as np
USE_SCIPY = False


def initialize_model(X, y, GP, noise, bounds, state_dict=None, *GP_args, **GP_kwargs):
    """
    Create GP model and fit it. The function also accepts
    state_dict which is used as an initialization for the GP model.

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        Input values

    y : torch.tensor, shape=(n_samples,)
        Output values

    GP : botorch.models.Model
        GP model class

    state_dict : dict
        GP model state dict

    Returns
    -------
    mll : gpytorch.mlls.MarginalLoglikelihood
        Marginal loglikelihood

    gp :
    """
    model = GP(X, y, noise, bounds, *GP_args, **GP_kwargs).to(X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def bo_step(X,
            y,
            noise,
            objective,
            bounds,
            GP=None,
            acquisition=None,
            q=1,
            state_dict=None,
            plot=False):
    """
    One iteration of Bayesian optimization:
        1. Fit GP model using (X, y)
        2. Create acquisition function
        3. Optimize acquisition function to obtain candidate point
        4. Evaluate objective at candidate point
        5. Add new point to the data set

    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        Input values

    y : torch.tensor, shape=(n_samples,)
        Objective values

    objective : callable, argument=torch.tensor
        Objective black-box function, accepting as an argument torch.tensor

    bounds : torch.tensor, shape=(2, dim)
        Box-constraints

    GP : callable
        GP model class constructor. It is a function that takes as input
        2 tensors - X, y - and returns an instance of botorch.models.Model.

    acquisition : callable
        Acquisition function construction. It is a function that receives
        one argument - GP model - and returns an instance of
        botorch.acquisition.AcquisitionFunction

    q : int
        Number of candidate points to find

    state_dict : dict
        GP model state dict

    plot : bool
        Flag indicating whether to plot the result

    Returns
    -------
    X : torch.tensor
        Tensor of input values with new point

    y : torch.tensor
        Tensor of output values with new point

    gp : botorch.models.Model
        Constructed GP model
    """
    attempts = 0
    while attempts < 10:
        try:
            options = {'lr': 1e-1 / (1 + 10**attempts), 'maxiter': 100}
            # Create GP model
            mll, gp = initialize_model(X, y, noise=noise, bounds=bounds, GP=GP, state_dict=state_dict)
            if USE_SCIPY:
                fit_gpytorch_model(mll)
            else:
                fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch, options=options)

            # Create acquisition function
            acquisition_ = acquisition(gp, y)

            # Optimize acquisition function
            candidate, acq_value_list = joint_optimize(
                acquisition_, bounds=bounds, q=q, num_restarts=20, raw_samples=20000,
            )
            break
        except RuntimeError:
            # state_dict = None
            attempts += 1
            if attempts > 1:
                state_dict = None
            print("Attempt #{}".format(attempts), traceback.print_exc())

    X = torch.cat([X, candidate])
    y = torch.cat([y, objective(candidate).detach().view(1, 1)], dim=0)
    if plot:
        utils.plot_acquisition(acquisition, X, y, candidate)

    return X, y, gp


def map_box_ball(x, borders):
    dim = len(borders)
    # from borders to [-1, 1]^d
    x = (x - borders.mean(dim=1)) / ((borders[:, 1] - borders[:, 0]) / 2)
    # from [-1, 1]^d to Ball(0, 1)
    x = x / np.sqrt(dim)
    return x


def map_ball_box(x, borders):
    dim = len(borders)
    # from Ball(0, 1) to [-1, 1]^d
    x = np.sqrt(dim) * x
    # from [-1, 1]^d to borders
    x = x * ((borders[:, 1] - borders[:, 0]) / 2) + borders.mean(dim=1)
    return x


class KumaAlphaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(KumaAlphaPrior, self).__init__()
        self.log_a_max = np.log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_a_max
        ))


class KumaBetaPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(KumaBetaPrior, self).__init__()
        self.log_b_max = np.log(2)
        pass

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(0.01).to(x)
        return torch.sum(torch.log(
            torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).exp() + 0.5 / self.log_b_max
        ))


class AngularWeightsPrior(gpytorch.priors.Prior):
    def __init__(self):
        super(AngularWeightsPrior, self).__init__()

    def log_prob(self, x):
        x = torch.log(x)
        loc = torch.tensor(0.).to(x)
        scale = torch.tensor(2.).to(x)
        return torch.distributions.Normal(loc=loc, scale=scale).log_prob(x).sum()


class CustomCylindricalGP(SingleTaskGP, GPyTorchModel):  # FixedNoiseGP SingleTaskGP
    def __init__(self, train_X, train_Y, noise, borders):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y)  # GaussianLikelihood())  # GaussianLikelihood() noise.squeeze(-1)
        self.borders = borders.t()
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(CylindricalKernel(
            num_angular_weights=4,
            alpha_prior=KumaAlphaPrior(),
            alpha_constraint=gpytorch.constraints.constraints.Interval(lower_bound=0.5, upper_bound=1.),
            beta_prior=KumaBetaPrior(),
            beta_constraint=gpytorch.constraints.constraints.Interval(lower_bound=1., upper_bound=2.),
            radial_base_kernel=MaternKernel(),
            # angular_weights_constraint=gpytorch.constraints.constraints.Interval(lower_bound=np.exp(-12.),
            #                                                                      upper_bound=np.exp(20.)),
            angular_weights_prior=AngularWeightsPrior()
        ))
        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        x = map_box_ball(x, borders=self.borders)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
