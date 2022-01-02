from qtensor import Info, MPS, Gates, CircuitCX
import torch
import numpy as np
from qtensor import Load
import matplotlib.pyplot as plt
import time

# N = 5
# D = 5
#
# info = Info()
# mps = MPS(info)
# mps.all_zeros_state(N)
#
# gates = Gates(info)
# circuit = CircuitCX(gates)
#
# circuit.evolution([mps], N, D, max_rank=None, ort=False)
#
# print(mps.r)
# print(mps.scalar_product(mps))
# print(mps.get_density_matrix(0))
# print(torch.trace(mps.get_density_matrix(0)))

N = 10
D = 10

info = Info()
mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)
circuit = CircuitCX(gates)

circuit.evolution([mps], N, D, max_rank=None, ort=False)

for i in range(N):
    rho_i = mps.get_density_matrix(i)
    purity = np.trace(np.dot(np.array(rho_i, dtype=complex), np.array(rho_i, dtype=complex))).real
    print(i, 'qubit: ', purity)
