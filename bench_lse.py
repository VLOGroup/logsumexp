import torch as th
import torch.utils.benchmark as benchmark
import test_logsumexp as lse
from torch.profiler import profile, record_function, ProfilerActivity

vmin = -1.2
vmax = 1.2
n_w = 30
dtype = th.float32
n_f = 32
N, C, H, W = 50, n_f, 128, 128
sigma = (vmax - vmin) / (n_w - 1)
mus = th.linspace(vmin, vmax, n_w, device='cuda', dtype=dtype)

inp = th.randn((N, C, H, W)).cuda()
w = lse.proj_simplex_simul(lse.weight_init(
    vmin, vmax, n_w, 0.001, 'abs'
)[None].repeat(n_f, 1).cuda().to(dtype))

num_threads = th.get_num_threads()

models = [
    (lse.TorchForAutogradNet, 'TorchForAutogradNet'),
    (lse.TorchAutogradNet, 'TorchAutogradNet'),
    (lse.TorchExplicitNet, 'TorchExplicitNet'),
    (lse.CppAutogradNet, 'CppAutogradNet'),
    (lse.CppExplicitNet, 'CppExplicitNet'),
]

model, str_repr = models[0]
print(str_repr)
x = inp.clone()
if str_repr.startswith('Torch'):
    x = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
if 'Autograd' in str_repr:
    x.requires_grad = True

# timer = benchmark.Timer(
#     stmt=f'{str_repr}(w, mus, sigma).grad_(x)',
#     setup=f'from test_logsumexp import {str_repr}',
#     globals=globals(),
#     num_threads=num_threads,
# )
# print(timer.timeit(100))

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True, record_shapes=True, with_stack=True
) as prof:
    model(w, mus, sigma).grad_(x)
prof.export_stacks(f"./profiler_stacks_{str_repr}.txt", "self_cuda_time_total")
print(prof.key_averages()
      .table(sort_by="self_cpu_memory_usage", row_limit=10))
del x
th.cuda.empty_cache()
