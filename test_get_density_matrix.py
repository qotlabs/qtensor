from qtensor import Info, MPS, Gates, CircuitCX
import torch
from qtensor import Load
import matplotlib.pyplot as plt
import time

N = 5
D = 5

info = Info()
mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)
circuit = CircuitCX(gates)

circuit.evolution([mps], N, D, max_rank=None, ort=False)

# print(torch.ones([1, 1, 1, 1]))
print(mps.r)
print(mps.scalar_product(mps))
print(mps.get_density_matrix(0))
print(torch.trace(mps.get_density_matrix(0)))
# print(mps.get_norm_test(2))
# print(mps.get_norm_simple_test())
# print(mps.get_norm_test_complex(2))
