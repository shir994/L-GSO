import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from scipy.misc import central_diff_weights
from tqdm import tqdm
from typing import Callable
from joblib import Parallel, delayed
PARALLEL = False


def richardson(f, x, n, h):
    """
    Richardson's Extrapolation
    """
    d = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        d[i, 0] = 0.5 * (f(y=x + h) - f(y=x - h)) / h
        p4 = 1  # values of 4^j
        for j in range(1, i + 1):
            p4 = 4 * p4
            d[i, j] = d[i, j-1] + (d[i, j-1] - d[i-1, j-1]) / (p4 - 1)
        h = 0.5 * h
    return d[n, n]


def n_order_scheme(f, x, n, h):
    """
    n-order differentiable scheme
    """
    weights = central_diff_weights(n)
    grad_f = 0
    i_s, j_s = list(zip(*list(enumerate(np.arange(-n // 2, n // 2) + 1))))
    if PARALLEL:
        f_js = Parallel(n_jobs=len(j_s), prefer="threads")(delayed(f)(y=x + j * h) for j in j_s)
    else:
        f_js = [f(y=x + j * h) for j in j_s]
    for i, f_j in zip(i_s, f_js):
        if weights[i] != 0:
            grad_f += weights[i] * f_j
    return grad_f / h


def partial_function(f, x, i, y):
    return f((x[:i] + [y] + x[i + 1:]))


def compute_gradient_of_vector_function(f: Callable,
                                        x: list,
                                        n: int,
                                        h: float,
                                        scheme: Callable):
    dim = len(x)
    if PARALLEL:
        partial_derivatives = Parallel(n_jobs=dim, prefer="threads")(delayed(scheme)(f=partial(partial_function, f=f, x=x, i=i), x=x[i], n=n, h=h) for i in range(dim))
    else:
        partial_derivatives = [scheme(f=partial(partial_function, f=f, x=x, i=i), x=x[i], n=n, h=h) for i in range(dim)]
    # partial_derivatives = []
    # for i in range(dim):
    #    partial_derivatives.append(
    #         scheme(f=partial(partial_function, f=f, x=x, i=i), x[i], n=n, h=h)
    #     )
    return partial_derivatives

