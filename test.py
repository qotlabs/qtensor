import torch
from qtensor import MPS, Info

N = 2
info = Info()
state = MPS(info)
state.all_zeros_state(N)
state.full_tensor_calculate()
# print(list(state.full_tensor.size()))
# print(state.r)
# print(state.phys_ind)
# print(state.tt_cores)
# print(torch.reshape(state.full_tensor, [1] + state.phys_ind + [1]).size())

print(state.r)
A = state.full_tensor

print(A.size())
train = MPS(info)
train.tt_decomposition(A)
print(train.r)
train.full_tensor_calculate()
print(torch.sum((train.full_tensor - A) ** 2))
print()
print(train.tt_cores[0].size())

print('New testing')
print(A.size())
print(state.phys_ind)
state.full_tensor_calculate()
print(state.full_tensor)
print(state.return_full_tensor())
print(state.full_tensor)

# U = torch.tensor([[0, 1], [1, 0]], dtype=info.data_type, device=info.device)
# print(U)
# state.one_qubit_gate(U, 0)
# print(state.tt_cores[0].size())
# state.full_tensor_calculate()
# print(state.full_tensor)
# print(state.return_full_tensor())

print()
print('Testing of two-qubit gates')
U1 = torch.tensor([[1, 0, 0, 0],
                  [-1, 1, 0, 0],
                  [2, 0, 0, 1],
                  [3, 0, 1, 0]], dtype=info.data_type, device=info.device)
print(U1)
state.full_tensor_calculate()
print(state.r)
print(state.return_full_tensor())
U2 = torch.tensor([[0, 1],
                  [1, 0]], dtype=info.data_type, device=info.device)
print(U2)
state.one_qubit_gate(U2, 0)
state.one_qubit_gate(U2, 1)
state.two_qubit_gate(U1, 0)
state.full_tensor_calculate()
print(state.return_full_tensor())
print(state.r)
print(state.phys_ind)
state.one_qubit_gate(U2, 1)
state.two_qubit_gate(U1, 0)
print(state.r)
