import numpy as np
import copy
import torch
from qtensor import Info, State, MPS, Circuit, Gates, Load

# N = 40
# D = 3
# max_rank = 4
#
# info = Info()
#
# mps = MPS(info)
# mps.all_zeros_state(N)
# # state = State(info)
# # state.all_zeros_state(N)
# gates = Gates(info)
# circuit = Circuit(gates)
# circuit.evolution([mps], N, D, max_rank=max_rank)
# mps_copy = copy.deepcopy(mps)
# mps.sequence_qr(22)
# print(mps.fidelity(mps_copy))
# print(mps_copy.r)
# print(mps.r)
# print([mps.check_ort_left(i) for i in range(N)])
# print([mps.check_ort_right(i) for i in range(N)])


# A = torch.rand(5, 2, 3) + 1j * torch.rand(5, 2, 3)
# B = torch.rand(3, 2, 7) + 1j * torch.rand(3, 2, 7)
# print(A.shape, B.shape)
# print(A, B)
#
# C = torch.tensordot(A, B, dims=([2], [0]))
# print(C.shape)
# C = torch.reshape(C, (10, 14))
# print(C.shape)
# u, s, v = torch.linalg.svd(C, full_matrices=False)
# max_rank = 3
# u = u[:, 0:max_rank]
# s = s[0:max_rank]
# v = v[0:max_rank, :]
# print(u.shape, s.shape, v.shape)
# print(np.sum(np.abs((np.array(u) @ np.diag(np.array(s)) @ np.array(v)) - np.array(C)) ** 2))
#
# print('Test QR')
#
# q, r = torch.linalg.qr(C)
# print(q.shape, r.shape)
#
# q = q[:, 0:max_rank]
# r = r[0:max_rank, :]
# print(q.shape, r.shape)
# print(np.sum(np.abs((np.array(q) @ np.array(r)) - np.array(C)) ** 2))
# print(r)

# N = 5
# D = 5
# max_rank = 5
# info = Info()
#
# mps = MPS(info)
# mps.all_zeros_state(N)
# state = State(info)
# state.all_zeros_state(N)
# gates = Gates(info)
# circuit = Circuit(gates)
# circuit.evolution([mps, state], N, D, max_rank=max_rank, ort=True)
# print(np.sum(np.abs(np.array(mps.return_full_tensor() - state.return_full_tensor())) ** 2))
# mps_copy = copy.deepcopy(mps)
# state_new = State(info)
# state_new.all_zeros_state(N)
# state_new.full_vector = mps.return_full_vector()
# print(state.fidelity(state_new))
# print(mps_copy.r)
# print(mps.r)
# print([mps.check_ort_left(i) for i in range(N)])
# print([mps.check_ort_right(i) for i in range(N)])

# Check fidelity function

N = 5
D = 5
max_rank = 5
info = Info()

mps_1 = MPS(info)
mps_1.all_zeros_state(N)
mps_2 = MPS(info)
mps_2.all_zeros_state(N)
state_1 = State(info)
state_1.all_zeros_state(N)
state_2 = State(info)
state_2.all_zeros_state(N)
gates = Gates(info)
circuit = Circuit(gates)
circuit.evolution([mps_1, state_1], N, D, max_rank=max_rank, ort=True)
circuit.evolution([mps_2, state_2], N, D, max_rank=max_rank, ort=True)
print(mps_1.fidelity(mps_2))
print(state_1.fidelity(state_2))
