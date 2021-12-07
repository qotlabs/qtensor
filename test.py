import numpy as np
import torch
from qtensor import MPS, State, Info
import copy

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

# print(state.r)
# A = state.full_tensor
#
# print(A.size())
# train = MPS(info)
# train.tt_decomposition(A)
# print(train.r)
# train.full_tensor_calculate()
# print(torch.sum((train.full_tensor - A) ** 2))
# print()
# print(train.tt_cores[0].size())
#
# print('New testing')
# print(A.size())
# print(state.phys_ind)
# state.full_tensor_calculate()
# print(state.full_tensor)
# print(state.return_full_tensor())
# print(state.full_tensor)

# U = torch.tensor([[0, 1], [1, 0]], dtype=info.data_type, device=info.device)
# print(U)
# state.one_qubit_gate(U, 0)
# print(state.tt_cores[0].size())
# state.full_tensor_calculate()
# print(state.full_tensor)
# print(state.return_full_tensor())

# print()
# print('Testing of two-qubit gates')
# U1 = torch.tensor([[1, 0, 0, 0],
#                    [0, 1, 0, 0],
#                    [0, 0, 0, 1],
#                    [0, 0, 1, 0]], dtype=info.data_type, device=info.device)
# print(U1)
# state.full_tensor_calculate()
# print(state.r)
# print(state.return_full_tensor())
# U2 = torch.tensor([[0, 1],
#                   [1, 0]], dtype=info.data_type, device=info.device)
# print(U2)
# state.one_qubit_gate(U2, 0)
# state.one_qubit_gate(U2, 1)
# state.two_qubit_gate(U1, 0)
# state.two_qubit_gate(U1, 0)
# state.full_tensor_calculate()
# print(state.return_full_tensor())
# print(state.r)
# print(state.phys_ind)
# state.one_qubit_gate(U2, 1)
# state.two_qubit_gate(U1, 0)
# print(state.r)

# N = 3
# swap = torch.tensor([[1, 0, 0, 0],
#                      [0, 0, 1, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 0, 1]], dtype=info.data_type, device=info.device)
# print(swap)
# state.all_zeros_state(N)
# state.one_qubit_gate(U2, 0)
# state.full_tensor_calculate()
# print(state.return_full_tensor())
# print(state.r)
# state.two_qubit_gate(swap, 0)
# state.two_qubit_gate(swap, 1)
# state.two_qubit_gate(swap, 0)
# state.two_qubit_gate(swap, 1)
# state.two_qubit_gate(swap, 0)
# state.full_tensor_calculate()
# print(state.return_full_tensor())
# print(state.r)
#
# print(state.tt_cores[2].size())
# print(state.get_norm())
#
# info = Info()
# A = torch.randn([2, 2, 2, 2, 2], dtype=info.data_type, device=info.device)
# mps = MPS(info)
# mps.tt_decomposition(A, max_rank=2)
# print(mps.r)
# # print(A.norm())
# # print(mps.get_norm())
#
# print()
# print(A[1][0][0][1][1])
# print(mps.tt_cores[0].size(), mps.tt_cores[1].size(), mps.tt_cores[2].size(), mps.tt_cores[3].size(),
#       mps.tt_cores[4].size())
# print(mps.get_element([1, 0, 0, 1, 1]))
# print('Hello', mps.tt_cores[0][:, 0, :].size())
# print()
#
# mps.all_zeros_state(5)
# mps_other = copy.deepcopy(mps)
# mps.one_qubit_gate(U2, 0)
# print(mps.scalar_product(mps_other))
#
# print(torch.tensor([1, 0]).size())

# print('Testing State')
# N = 3
# info = Info()
# state = State(info)
# state.all_zeros_state(N)
# print(state.phys_ind)
# print(state.full_vector)
# print(state.return_full_tensor().size())
# print(state.return_full_vector().size())
# print(state.get_element([1, 1, 1]))

# A1 = torch.tensor([[1, 0], [0, 1]], dtype=info.data_type, device=info.device)
# A2 = torch.tensor([[1, 2], [3, 4]], dtype=info.data_type, device=info.device)
# print(torch.kron(A1, A2))
#
# X = torch.tensor([1, 1], dtype=info.data_type, device=info.device)
# print(A2.size())
# print(A2.mv(X))
# print(torch.mv(A2, X))

# # Testing State
# print('Testing State')
# N = 3
# info = Info()
# H = (1 / np.sqrt(2)) * torch.tensor([[1, 1],
#                                      [1, -1]], dtype=info.data_type, device=info.device)
# CNOT = torch.tensor([[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0]], dtype=info.data_type, device=info.device)
# state = State(Info())
# state.all_zeros_state(N)
# state.one_qubit_gate(H, 0)
# state.two_qubit_gate(CNOT, 0)
# print('Hello')
# print(state.return_full_vector())
# phi = State(Info())
# phi.all_zeros_state(N)
# phi.one_qubit_gate(H, 0)
# print(state.scalar_product(phi))
# print(state.get_norm())
# print(state.get_element([0, 0, 0]))
# print(state.scalar_product(state))
#
# print()
#
# # Testing MPS
# print('Testing MPS')
# N = 3
# info = Info()
# H = (1 / np.sqrt(2)) * torch.tensor([[1, 1],
#                                      [1, -1]], dtype=info.data_type, device=info.device)
# CNOT = torch.tensor([[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0]], dtype=info.data_type, device=info.device)
# state = MPS(Info())
# state.all_zeros_state(N)
# state.one_qubit_gate(H, 0)
# state.two_qubit_gate(CNOT, 0)
# print(state.return_full_vector())
# phi = MPS(Info())
# phi.all_zeros_state(N)
# phi.one_qubit_gate(H, 0)
# print(state.scalar_product(phi))
# print(state.get_norm())
# print(state.get_element([0, 0, 0]))
# print(state.r)
# print(state.scalar_product(state))

import numpy as np
from qtensor import IsingHam
from qtensor import Info, MPS, Gates, CircuitCZ
from qtensor import Load
import matplotlib.pyplot as plt
import time

N = 20
D = 13

info = Info()
mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)
circuit = CircuitCZ(gates)

fid_result = []

circuit.evolution([mps], N, D, max_rank=None, ort=False)
