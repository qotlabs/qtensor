import torch
from qtensor import MPS, Info

# N = 2
# info = Info()
# state = MPS(info)
# state.all_zeros_state(N)
# state.full_tensor_calculate()
# print(list(state.full_tensor.size()))
# print(state.r)
# print(state.phys_ind)
# print(state.tt_cores)
# print(torch.reshape(state.full_tensor, [1] + state.phys_ind + [1]).size())

info = Info()
A = torch.randn([2] * 5, dtype=info.data_type, device=info.device)
print(A.size())
train = MPS(info)
train.tt_decomposition(A)
print(train.r)
train.full_tensor_calculate()
print(torch.sum((train.full_tensor - A) ** 2))
