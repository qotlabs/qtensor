import numpy as np
from qtensor import VQEOptimizer, VQECircuitCX
from qtensor import Info, Gates, IsingHam, MPS, MPSGrad, IsingHamAnalytical

N = 5
D = 1

info = Info()
gates = Gates(info)
ham = IsingHam(N, gates, info)
# ham = IsingHamAnalytical(N, gates, info)

print('Min energy = ', ham.get_min_energy())
# print(ham.get_min_energy_analytical())

vqe_circuit = VQECircuitCX(gates)

vqe_optimizer = VQEOptimizer(MPS, MPSGrad, info, N, D, ham, gates, vqe_circuit, max_rank=None)

number_of_iterations = 1000

list_of_parameters = 2 * np.pi * np.random.rand(4 * N * D)

result = vqe_optimizer.optimize(list_of_parameters, number_of_iterations)
# result = vqe_optimizer.global_optimize(list_of_parameters)
# result = vqe_optimizer.cobyla_optimize(list_of_parameters, number_of_iterations)

print(result)
