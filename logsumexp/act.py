import torch
from logsumexp_cuda_ext import act
from torch.autograd.function import once_differentiable

__all__ = ['activation']


class LogSumExpPotAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, mu, sigma):
        ctx.save_for_backward(x, weight, sigma)
        ctx.mu = mu
        f, f_prime, max_exp = act.lse_forward(x, weight, mu, sigma)
        ctx.max_exp = max_exp
        return f, f_prime

    @once_differentiable
    @staticmethod
    def backward(ctx, grad_out, grad_out_prime):
        x, weight, sigma = ctx.saved_tensors
        grad_x, grad_w, grad_sigma = act.lse_backward(
            x,
            weight,
            grad_out,
            grad_out_prime,
            ctx.mu,
            sigma,
            ctx.max_exp,
        )
        return grad_x, grad_w, None, grad_sigma


def pot_act(
    x: torch.Tensor,
    weight: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> list[torch.Tensor]:
    return LogSumExpPotAct.apply(x, weight, mu, sigma)
