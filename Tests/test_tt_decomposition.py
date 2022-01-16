import torch
import numpy as np
from qtensor import Info, MPS

info = Info()
mps = MPS(info)

tensor = torch.randn([2] * 5, dtype=info.data_type, device=info.device)

print(tensor.size())

mps.tt_decomposition(tensor)

print(mps.r)

print(torch.norm(mps.return_full_tensor() - tensor))
print(torch.sqrt(torch.sum(torch.abs((mps.return_full_tensor() - tensor)) ** 2)))
