import math

import logsumexp
import torch as th
from torch.cuda import Event

vmin = -1.2
vmax = 1.2

n_f = 4
n_w = 5
sigma = th.tensor([(vmax - vmin) / (n_w)]).cuda().double().repeat(n_f)[:, None,
                                                                       None]
sigma.requires_grad_(True)


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
        exponent = -d**2 / (2 * sigma[:, 0, 0][:, None]**2)
        max_exponent = th.maximum(max_exponent, th.max(exponent))
    for w, mu in zip(ws.T, mus):
        d = x - mu
        exponent = -d**2 / (2 * sigma[:, 0, 0][:, None]**2)
        accumulated += (
            w[:, None] / (math.sqrt(2 * math.pi) * sigma[:, 0, 0][:, None]) *
            th.exp(exponent - max_exponent)
        )
    return -(th.log(accumulated) + max_exponent)


def f(x, ws, mus, sigma):
    d = x[:, :, None] - mus[None, None, :]
    exponent = -d**2 / (2 * sigma**2)
    max_exponent = th.max(exponent)
    return -(
        th.log((
            ws[:, None] /
            (math.sqrt(2 * math.pi) * sigma) * th.exp(exponent - max_exponent)
        ).sum(2)) + max_exponent
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
        start = Event(True)
        stop = Event(True)

        start.record()
        a = logsumexp.pot_act(x, self.ws, self.mus,
                                        self.sigma)[0].mean().item()
        stop.record()
        th.cuda.synchronize()
        print(start.elapsed_time(stop))
        return a


sz = 3
for dtype in [th.float64]:
    mus = th.linspace(vmin, vmax, n_w, device='cuda', dtype=dtype)
    w = weight_init(vmin, vmax, n_w, 0.001,
                    'abs')[None].repeat(n_f, 1).cuda().to(dtype)
    # w = th.rand_like(w)
    w = proj_simplex_simul(w)
    w.requires_grad_(True)
    batch_size = 3
    x = th.linspace(
        vmin,
        vmax,
        sz**2,
        dtype=dtype,
        device='cuda',
    )[None].repeat(n_f, 1).view(1, n_f, sz, sz)
    x = x.repeat(batch_size, 1, 1, 1).clone()
    x = th.randn_like(x)
    x.requires_grad_(True)

    # psi
    f_c, dxf_c = logsumexp.pot_act(x, w, mus, sigma[:, 0, 0])
    f_th = f(
        x.permute(1, 0, 2, 3).reshape(n_f, batch_size * sz * sz), w, mus, sigma
    ).reshape(n_f, batch_size, sz, sz).permute(1, 0, 2, 3)
    f_for_th = f_for(
        x.permute(1, 0, 2, 3).reshape(n_f, batch_size * sz * sz), w, mus, sigma
    ).reshape(n_f, batch_size, sz, sz).permute(1, 0, 2, 3)
    assert th.allclose(f_c, f_th)
    assert th.allclose(f_th, f_for_th)

    grad_input = th.ones_like(x)
    # first derivatives
    dxf_th = f_prime(
        x.permute(1, 0, 2, 3).reshape(n_f, batch_size * sz * sz), w, mus, sigma
    ).reshape(n_f, batch_size, sz, sz).permute(1, 0, 2, 3)
    dxf_auto = th.autograd.grad(
        f_th,
        x,
        grad_outputs=th.ones_like(x),
        create_graph=True,
        retain_graph=True
    )[0]
    dsf_auto = th.autograd.grad(f_th, sigma, grad_input, retain_graph=True)[0]
    dsf_c = th.autograd.grad(f_c, sigma, grad_input, retain_graph=True)[0]
    assert th.allclose(dxf_c, dxf_th)
    assert th.allclose(dxf_c, dxf_auto)

    # second derivatives
    dwdxf_c = th.autograd.grad(dxf_c, w, grad_input, retain_graph=True)[0]
    dwdxf_th = th.autograd.grad(dxf_th, w, grad_input, retain_graph=True)[0]
    dwdxf_auto = th.autograd.grad(dxf_auto, w, grad_input,
                                  retain_graph=True)[0]
    dsdxf_auto = th.autograd.grad(
        dxf_th, sigma, grad_input, retain_graph=True
    )[0]
    dsdxf_c = th.autograd.grad(dxf_c, sigma, grad_input, retain_graph=True)[0]
    assert th.allclose(dwdxf_c, dwdxf_th)
    assert th.allclose(dwdxf_c, dwdxf_auto)
    assert th.allclose(dsf_auto, dsf_c)
    assert th.allclose(dsdxf_auto, dsdxf_c)

    # unneeded second derivatives
    dxdxf_c = th.autograd.grad(dxf_c, x, grad_input)[0]
    dxdxf_th = th.autograd.grad(dxf_th, x, grad_input)[0]
    assert th.allclose(dxdxf_c, dxdxf_th)
    print('all done')
