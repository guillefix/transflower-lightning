"""Logistic distribution functions."""
import torch
import torch.nn.functional as F

from models.util import safe_log


def _log_pdf(x, mean, log_scale):
    """Element-wise log density of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = z - log_scale - 2 * F.softplus(z)

    return log_p


def _log_cdf(x, mean, log_scale):
    """Element-wise log CDF of the logistic distribution."""
    z = (x - mean) * torch.exp(-log_scale)
    log_p = F.logsigmoid(z)

    return log_p


def mixture_log_pdf(x, prior_logits, means, log_scales):
    """Log PDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_pdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_log_cdf(x, prior_logits, means, log_scales):
    """Log CDF of a mixture of logistic distributions."""
    log_ps = F.log_softmax(prior_logits, dim=1) \
        + _log_cdf(x.unsqueeze(1), means, log_scales)
    log_p = torch.logsumexp(log_ps, dim=1)

    return log_p


def mixture_inv_cdf(y, prior_logits, means, log_scales,
                    eps=1e-10, max_iters=100):
    """Inverse CDF of a mixture of logisitics. Iterative algorithm."""
    if y.min() <= 0 or y.max() >= 1:
        raise RuntimeError('Inverse logisitic CDF got y outside (0, 1)')

    def body(x_, lb_, ub_):
        cur_y = torch.exp(mixture_log_cdf(x_, prior_logits, means,
                                          log_scales))
        gt = (cur_y > y).type(y.dtype)
        lt = 1 - gt
        new_x_ = gt * (x_ + lb_) / 2. + lt * (x_ + ub_) / 2.
        new_lb = gt * lb_ + lt * x_
        new_ub = gt * x_ + lt * ub_
        return new_x_, new_lb, new_ub

    x = torch.zeros_like(y)
    max_scales = torch.sum(torch.exp(log_scales), dim=1, keepdim=True)
    lb, _ = (means - 20 * max_scales).min(dim=1)
    ub, _ = (means + 20 * max_scales).max(dim=1)
    diff = float('inf')

    i = 0
    while diff > eps and i < max_iters:
        new_x, lb, ub = body(x, lb, ub)
        diff = (new_x - x).abs().max()
        x = new_x
        i += 1

    return x


def inverse(x, reverse=False):
    """Inverse logistic function."""
    if reverse:
        z = torch.sigmoid(x)
        ldj = F.softplus(x) + F.softplus(-x)
    else:
        z = -safe_log(x.reciprocal() - 1.)
        ldj = -safe_log(x) - safe_log(1. - x)

    return z, ldj
