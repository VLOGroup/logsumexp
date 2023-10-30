import math

import logsumexp
import torch as th


def proj_simplex_simul(
    x: th.Tensor,  # 2-D array of weights,
    # projection is performed along the 0-th axis
    s: float = 1.,  # axis interesection
):
    K = x.shape[1]
    k = th.linspace(1, K, K, device=x.device)
    x_s = th.sort(x, dim=1, descending=True)[0]
    t = (th.cumsum(x_s, dim=1) - s) / k[None]
    mu = th.max(t, dim=1, keepdim=True)[0]
    return th.clamp(x - mu, 0, s)


def weight_init(
    vmin: float,
    vmax: float,
    n_w: int,
    scale: float,
    mode: str,
) -> th.Tensor:
    x = th.linspace(vmin, vmax, n_w, dtype=th.float32)
    match mode:
        case "constant":
            w = th.ones_like(x) * scale
        case "linear":
            w = x * scale
        case "quadratic":
            w = x**2 * scale
        case "abs":
            w = th.abs(x) * scale
            w -= w.max()
            w = w.abs()
        case "student-t":
            alpha = 100
            w = scale * math.sqrt(alpha) * x / (1 + 0.5 * alpha * x**2)
        case "Student-T":
            a_ = 0.1 * 78
            b_ = 0.1 * 78**2
            denom = 1 + (a_ * x)**2
            w = b_ / (2 * a_**2) * th.log(denom)

    return w


def f_for(x, ws, mus, sigma):
    accumulated = th.zeros_like(x)
    max_exponent = th.tensor([-1e20]).to(x)
    for mu in mus:
        d = x - mu
        exponent = -d**2 / (2 * sigma**2)
        max_exponent = th.maximum(max_exponent, th.max(exponent))
    for w, mu in zip(ws.T, mus):
        d = x - mu
        exponent = -d**2 / (2 * sigma**2)
        accumulated += w[:, None] * th.exp(exponent - max_exponent)
    return -(th.log(accumulated) + max_exponent)


def f(x, ws, mus, sigma):
    d = x[:, :, None] - mus[None, None, :]
    exponent = -d**2 / (2 * sigma**2)
    max_exponent = th.max(exponent)
    return -(
        th.log((ws[:, None] * th.exp(exponent - max_exponent)).sum(2)) +
        max_exponent
    )


def f_prime(x, ws, mus, sigma):
    d = x[:, :, None] - mus[None, None, :]
    exponent = -d**2 / (2 * sigma**2)
    max_exponent = th.max(exponent)
    return (ws[:, None] * d * th.exp(exponent - max_exponent)).sum(2) / (
        ws[:, None] * th.exp(exponent - max_exponent) * sigma**2
    ).sum(2)


class TorchForAutogradNet(th.nn.Module):
    def __init__(
        self,
        ws: th.Tensor,
        mus: th.Tensor,
        sigma: float,
    ):
        super().__init__()
        self.ws = ws
        self.mus = mus
        self.sigma = sigma

    def grad_(self, x):
        f_ = f_for(x, self.ws, self.mus, self.sigma)
        return f_, th.autograd.grad(f_.sum(), x)


class TorchAutogradNet(th.nn.Module):
    def __init__(
        self,
        ws: th.Tensor,
        mus: th.Tensor,
        sigma: float,
    ):
        super().__init__()
        self.ws = ws
        self.mus = mus
        self.sigma = sigma

    def grad_(self, x):
        f_ = f(x, self.ws, self.mus, self.sigma)
        return f_, th.autograd.grad(f_.sum(), x)


class TorchExplicitNet(th.nn.Module):
    def __init__(
        self,
        ws: th.Tensor,
        mus: th.Tensor,
        sigma: float,
    ):
        super().__init__()
        self.ws = ws
        self.mus = mus
        self.sigma = sigma

    def grad_(self, x):
        return (
            f(x, self.ws, self.mus, self.sigma),
            f_prime(x, self.ws, self.mus, self.sigma),
        )


class CppAutogradNet(th.nn.Module):
    def __init__(
        self,
        ws: th.Tensor,
        mus: th.Tensor,
        sigma: float,
    ):
        super().__init__()
        self.ws = ws
        self.mus = mus
        self.sigma = sigma

    def grad_(self, x):
        f_ = logsumexp.pot_act(x, self.ws, self.mus, self.sigma)[0]
        return f_, th.autograd.grad(f_.sum(), x)


class CppExplicitNet(th.nn.Module):
    def __init__(
        self,
        ws: th.Tensor,
        mus: th.Tensor,
        sigma: float,
    ):
        super().__init__()
        self.ws = ws
        self.mus = mus
        self.sigma = sigma

    def grad_(self, x):
        return logsumexp.pot_act(x, self.ws, self.mus, self.sigma)
