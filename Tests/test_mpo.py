import torch
import numpy as np
from qtensor import Info, MPS, MPO, CircuitCXFix, Gates

N = 5
D = 3

info = Info()
mps = MPS(info)
mpo = MPO(info)

mps.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitCXFix(gates)
list_of_parameters = 2 * np.pi * np.random.rand(3 * N * D)
circuit.evolution(list_of_parameters, mps, N, D)
print(mps.r)

# 1
vec = mps.return_full_vector()
rho = np.dot(np.array(vec).reshape(2 ** N, 1), np.conjugate(np.array(vec).reshape(1, 2 ** N)))
print(rho)
rho_ideal = np.array(rho, dtype=complex)

# 2
tensor = torch.tensor(rho, dtype=info.data_type, device=info.device)
tensor = torch.reshape(tensor, [2] * (2 * N))
print(tensor.size())
mpo.tt_decomposition(tensor)
print(mpo.r)
print(torch.reshape(mpo.return_full_tensor(), (2 ** N, 2 ** N)))
rho_tt_decomposition = np.array(torch.reshape(mpo.return_full_tensor(), (2 ** N, 2 ** N)), dtype=complex)

# 3
mpo.gen_mpo_from_mps(mps)
print(mpo.r)
print(mpo.tt_cores[0].size())
print(mpo.get_trace())

print(mpo.return_full_tensor().size())
print(torch.reshape(mpo.return_full_tensor(), (2 ** N, 2 ** N)))
rho_gen_mpo_from_mps = np.array(torch.reshape(mpo.return_full_tensor(), (2 ** N, 2 ** N)), dtype=complex)

# 4
mps_test = MPS(info)
mps_test.all_zeros_state(N)
mpo_test = MPO(info)
mpo_test.gen_mpo_from_mps(mps_test)
print(mpo_test.r)
print(mpo_test.phys_ind_i, mpo_test.phys_ind_j)
circuit.evolution(list_of_parameters, mpo_test, N, D)
print(torch.reshape(mpo_test.return_full_tensor(), (2 ** N, 2 ** N)))
rho_mps_transformations = np.array(torch.reshape(mpo_test.return_full_tensor(), (2 ** N, 2 ** N)), dtype=complex)

print(np.sum(np.abs(rho_ideal - rho_tt_decomposition) ** 2), np.sum(np.abs(rho_ideal - rho_gen_mpo_from_mps) ** 2),
      np.sum(np.abs(rho_ideal - rho_mps_transformations) ** 2))

# 5
test_matrix = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
test_tensor = torch.reshape(torch.tensor(test_matrix, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo.tt_decomposition(test_tensor)
print(np.trace(test_matrix), mpo.get_trace())
