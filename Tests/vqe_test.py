import numpy as np
import copy
from qtensor import Info, MPS, IsingHam, VQECircuitCX, Gates

# N = 10
# D = 5
#
# info = Info()
#
# mps_1 = MPS(info)
# mps_1.all_zeros_state(N)
# mps_2 = MPS(info)
# mps_2.all_zeros_state(N)
#
# gates = Gates(info)
# vqe_circuit = VQECircuitCXError(gates)
#
# list_of_parameters = 2 * np.pi * np.random.rand(4 * N * D)
# vqe_circuit.evolution(list_of_parameters, [mps_1, mps_2], N, D, [None, 4], ort=True)
# ising_ham = IsingHam(N, gates)
# print(ising_ham.list_mean_ham([mps_1, mps_2]))
# print(len(list_of_parameters))
# print(mps_1.r, mps_2.r)

N = 10
D = 5

info = Info()

mps_0 = MPS(info)
mps_0.all_zeros_state(N)

gates = Gates(info)
vqe_circuit = VQECircuitCX(gates)

list_of_parameters = 2 * np.pi * np.random.rand(4 * N * D)

vqe_circuit.evolution(list_of_parameters, [mps_0], N, D, [None], ort=False)
ising_ham = IsingHam(N, gates, info)
print(ising_ham.list_mean_ham([mps_0])[0])

grad = []
for i in range(len(list_of_parameters)):
    print(i)
    mps = MPS(info)
    mps.all_zeros_state(N)
    vqe_circuit.gradient_evolution(i, list_of_parameters, [mps], N, D, [None], ort=False)
    grad.append(ising_ham.list_grad_mean_ham([mps_0], [mps])[0])

print(grad)
