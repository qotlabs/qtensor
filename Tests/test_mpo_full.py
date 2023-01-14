import torch
import numpy as np
from qtensor import Info, MPS, MPO, CircuitCXFix, Gates

N = 5
D = 3

info = Info()
mps = MPS(info)
mpo = MPO(info)

print('1. Test tt_decomposition and get_full_matrix')
test_matrix = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
test_tensor = torch.reshape(torch.tensor(test_matrix, dtype=info.data_type, device=info.device), [2] * 2 * N)
print(test_tensor.size())
mpo.tt_decomposition(test_tensor)
print(mpo.r)
# print(test_matrix)
# print(mpo.get_full_matrix())
print('Main test: ', np.sum(np.abs((test_matrix - np.array(mpo.get_full_matrix()))) ** 2))

print('2. Test get_product_matrix')
A_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
B_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
A_test_tensor = torch.reshape(torch.tensor(A_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
B_test_tensor = torch.reshape(torch.tensor(B_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo_A = MPO(info)
mpo_B = MPO(info)
mpo_A.tt_decomposition(A_test_tensor)
mpo_B.tt_decomposition(B_test_tensor)

result_1 = np.dot(A_test, B_test)
result_2 = mpo_A.get_product_matrix(mpo_B).get_full_matrix()
print('Main test: ', np.sum(np.abs((result_1 - np.array(result_2))) ** 2))

print('3. Test star')
A_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
A_test_tensor = torch.reshape(torch.tensor(A_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo_A = MPO(info)
mpo_A.tt_decomposition(A_test_tensor)
print('Main test: ', np.sum(np.abs((A_test.T.conjugate() - np.array(mpo_A.star().get_full_matrix()))) ** 2))

print('4. Test get_element')
A_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
A_test_tensor = torch.reshape(torch.tensor(A_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo_A = MPO(info)
mpo_A.tt_decomposition(A_test_tensor)
print('Main test: ', A_test_tensor[0, 1, 1, 0, 1, 1, 0, 1, 1, 0].item(), mpo_A.get_element([0, 1, 1, 0, 1, 1, 0, 1, 1, 0]))

print('5. Test get_trace')
A_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
B_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
A_test_tensor = torch.reshape(torch.tensor(A_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
B_test_tensor = torch.reshape(torch.tensor(B_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo_A = MPO(info)
mpo_B = MPO(info)
mpo_A.tt_decomposition(A_test_tensor)
mpo_B.tt_decomposition(B_test_tensor)

trace_1 = np.trace(np.dot(A_test, B_test))
product = mpo_A.get_product_matrix(mpo_B)
trace_2 = product.get_trace()
print('Main test: ', trace_1, trace_2)

print('6. Test get_tensor_for_trace')
A_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
B_test = np.random.randn(2 ** N, 2 ** N) + 1j * np.random.randn(2 ** N, 2 ** N)
A_test_tensor = torch.reshape(torch.tensor(A_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
B_test_tensor = torch.reshape(torch.tensor(B_test, dtype=info.data_type, device=info.device), [2] * 2 * N)
mpo_A = MPO(info)
mpo_B = MPO(info)
mpo_A.tt_decomposition(A_test_tensor)
mpo_B.tt_decomposition(B_test_tensor)

trace_ideal = np.trace(np.dot(A_test, B_test))

product_matrix = mpo_A.get_product_matrix(mpo_B)
trace_check_get_trace = product_matrix.get_trace()

trace_check_product_trace = mpo_A.get_trace_product_matrix(mpo_B)

print('Main test: ', trace_ideal, trace_check_get_trace, trace_check_product_trace)

trace_test = mpo_B.get_tensor_for_trace(mpo_A, 0)
print(torch.tensordot(trace_test, mpo_B.tt_cores[0], dims=([0, 2, 1, 3], [0, 1, 2, 3])).item())
trace_test = mpo_B.get_tensor_for_trace(mpo_A, 1)
print(torch.tensordot(trace_test, mpo_B.tt_cores[1], dims=([0, 2, 1, 3], [0, 1, 2, 3])).item())
trace_test = mpo_B.get_tensor_for_trace(mpo_A, 2)
print(torch.tensordot(trace_test, mpo_B.tt_cores[2], dims=([0, 2, 1, 3], [0, 1, 2, 3])).item())
trace_test = mpo_B.get_tensor_for_trace(mpo_A, 3)
print(torch.tensordot(trace_test, mpo_B.tt_cores[3], dims=([0, 2, 1, 3], [0, 1, 2, 3])).item())
trace_test = mpo_B.get_tensor_for_trace(mpo_A, 4)
print(torch.tensordot(trace_test, mpo_B.tt_cores[4], dims=([0, 2, 1, 3], [0, 1, 2, 3])).item())

print('7. Test all_zeros_state, gen_mpo_from_mps, one_qubit_gate, two_qubit_gate')
mps_psi = MPS(info)
mpo = MPO(info)
mpo_psi = MPO(info)

mps_psi.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitCXFix(gates)
list_of_parameters = 2 * np.pi * np.random.rand(3 * N * D)
circuit.evolution(list_of_parameters, mps_psi, N, D)
print(mps_psi.r)

mpo.all_zeros_state(N)
circuit.evolution(list_of_parameters, mpo, N, D)
mpo_psi.gen_mpo_from_mps(mps_psi)

print(mpo.get_trace())

print(mpo.get_product_matrix(mpo_psi).get_trace())
print('Main test: ', np.sum(np.abs(np.array(mpo.get_full_matrix()) - np.array(mpo_psi.get_full_matrix())) ** 2))

print('8. Test tt_decomposition')
matrix = mpo.get_full_matrix()
tensor = torch.reshape(mpo.get_full_matrix(), [2] * 2 * N)
mpo.tt_decomposition(tensor)
print(mpo.r)
print('Main test: ', np.sum(np.abs(np.array(matrix) - np.array(mpo.get_full_matrix()))) ** 2)

print('9. Test gen_random_mpo')
mpo = MPO(info)
mpo.gen_random_mpo(N, 5)
print(mpo.r)
print('Main test: ', mpo.get_trace())
print('Main test: ', np.linalg.eigvals(np.array(mpo.get_full_matrix())))
print('Main test: ',
      np.sum(np.abs(np.array(mpo.get_full_matrix()) - np.array(mpo.get_full_matrix()).T.conjugate()) ** 2))
