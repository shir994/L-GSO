import six
import math
import torch

import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx

def set_cnf_options(model, solver, rademacher, residual, atol=1e-3, rtol=1e-3):
    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = solver
            module.atol = atol
            module.rtol = rtol
            # If using fixed-grid adams, restrict order to not be too high.
            if solver in ['fixed_adams', 'explicit_adams']:
                module.solver_options['max_order'] = 4
        if isinstance(module, layers.ODEfunc):
            module.rademacher = rademacher
            module.residual = residual
    model.apply(_set)
    
# layer_type - ["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
def build_model_tabular(dims=2,
                        condition_dim=2,
                        layer_type='concatsquash', 
                        nonlinearity='relu', 
                        residual=False, 
                        rademacher=False,
                        train_T=True,
                        solver='dopri5',
                        time_length=0.1,
                        divergence_fn='brute_force', # ["brute_force", "approximate"]
                        hidden_dims=(32, 32), 
                        num_blocks=1, batch_norm=False, 
                        bn_lag=0, regularization_fns=None):

    def build_cnf():
        diffeq = layers.ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            condition_dim=condition_dim,
            strides=None,
            conv=False,
            layer_type=layer_type,
            nonlinearity=nonlinearity,
        )
        odefunc = layers.ODEfunc(
            diffeq=diffeq,
            divergence_fn=divergence_fn,
            residual=residual,
            rademacher=rademacher,
        )
        cnf = layers.CNF(
            odefunc=odefunc,
            T=time_length,
            train_T=train_T,
            regularization_fns=regularization_fns,
            solver=solver,
        )
        return cnf
    chain = [build_cnf() for _ in range(num_blocks)]
    if batch_norm:
        bn_layers = [layers.MovingBatchNorm1d(dims, bn_lag=bn_lag) for _ in range(num_blocks)]
        bn_chain = [layers.MovingBatchNorm1d(dims, bn_lag=bn_lag)]
        for a, b in zip(chain, bn_layers):
            bn_chain.append(a)
            bn_chain.append(b)
        chain = bn_chain
    model = layers.SequentialFlow(chain)
    set_cnf_options(model, solver, rademacher, residual)
    return model


def standard_normal_logprob(z, data):
    logZ = -0.5 * math.log(2 * math.pi)
    data_ref = torch.zeros_like(z)
    zpow = (z - data_ref).pow(2)
    return logZ - zpow / 2


def compute_loss(model, data, condition, weights=None):
    zero = torch.zeros(data.shape[0], 1).to(data.device)
    z, delta_logp = model(data, zero, condition=condition)
    logpz = standard_normal_logprob(z, data).sum(1, keepdim=True)
    logpx = logpz - delta_logp
    if weights is None:
        loss = -torch.mean(logpx)
    else:
        weights = torch.tensor(weights).float().to(logpx)
        loss = -torch.mean(logpx * weights)
    return loss


def get_transforms(model):
    def sample_fn(z, condition, logpz=None):
        if logpz is not None:
            return model(z, condition=condition, logpz=logpz, reverse=True)
        else:
            return model(z, condition=condition, reverse=True)

    def density_fn(x, condition, logpx=None):
        if logpx is not None:
            return model(x, condition=condition, logpz=logpx, reverse=False)
        else:
            return model(x, condition=condition, reverse=False)
    return sample_fn, density_fn
