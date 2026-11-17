
from pathlib import Path as _PathForBootstrap
import sys as _sysForBootstrap

_CODE_DIR = _PathForBootstrap(__file__).resolve().parents[1]
if str(_CODE_DIR) not in _sysForBootstrap.path:
    _sysForBootstrap.path.insert(0, str(_CODE_DIR))

import torch


def pairwise_squared_distances(x, y):
    """Compute squared Euclidean cost matrix between point clouds x and y."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True).T
    cost = x2 + y2 - 2.0 * (x @ y.T)
    return cost.clamp_min(0.0)


def sinkhorn_cost(x, y, epsilon=0.05, n_iters=80, x_weights=None, y_weights=None):
    """
    Log-domain entropic OT with squared Euclidean costs and optional particle weights.

    Missing weights represent a uniform empirical distribution.
    """
    cost_matrix = pairwise_squared_distances(x, y)

    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be positive.")

    n, m = cost_matrix.shape
    tiny = torch.finfo(cost_matrix.dtype).tiny
    log_a = (
        (-torch.log(x.new_tensor(float(n)))).expand(n)
        if x_weights is None else x_weights.clamp_min(tiny).log()
    )
    log_b = (
        (-torch.log(x.new_tensor(float(m)))).expand(m)
        if y_weights is None else y_weights.clamp_min(tiny).log()
    )
    log_K = -cost_matrix / eps

    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(int(n_iters)):
        u = log_a - torch.logsumexp(log_K + v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_K.T + u[None, :], dim=1)

    plan = torch.exp(u[:, None] + log_K + v[None, :])
    return (plan * cost_matrix).sum()


def sinkhorn_divergence(x, y, epsilon=0.05, n_iters=80, x_weights=None, y_weights=None):
    """
    Debiased Sinkhorn divergence.

    It is usually better behaved than raw entropic OT because it subtracts
    self-similarity terms from both point clouds.
    """
    cost_xy = sinkhorn_cost(x, y, epsilon, n_iters, x_weights, y_weights)
    cost_xx = sinkhorn_cost(x, x, epsilon, n_iters, x_weights, x_weights)
    cost_yy = sinkhorn_cost(y, y, epsilon, n_iters, y_weights, y_weights)
    return cost_xy - 0.5 * cost_xx - 0.5 * cost_yy
