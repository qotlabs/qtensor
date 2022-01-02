from qtensor import Info, MPS, Gates, CircuitCX
import torch
import numpy as np
from qtensor import Load
import matplotlib.pyplot as plt
import time

N = 5

info = Info()
mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)

gates_stochastic = [gates.Rn_random() for i in range(N)]

for i in range(N):
    mps.one_qubit_gate(gates_stochastic[i], i)

u = gates.Rn_random()

rho_0 = mps.get_density_matrix(4)

rho_1 = np.dot(np.array(u, dtype=complex), np.dot(np.array(rho_0, dtype=complex),
                                                  np.array(u, dtype=complex).T.conjugate()))

mps.one_qubit_gate(u, 4)
rho_2 = np.array(mps.get_density_matrix(4))

print(mps.r)
print(np.sum(np.abs(rho_1 - rho_2) ** 2))
