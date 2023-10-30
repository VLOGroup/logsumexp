import torch

from logsumexp import pot_act

x = torch.randn(2,16,11,11).cuda()
weight = torch.ones(16,21).cuda()

x.requires_grad_(True)
weight.requires_grad_(True)

out = activation(x,weight, vmin=-2, vmax=1)
print(out)

loss = torch.sum(out**2)
loss.backward()

print(x.grad)
print(weight.grad)
