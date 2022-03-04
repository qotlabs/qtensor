import numpy as np
from qtensor import MitigationCircuitCX, MitigationOptimizer
from qtensor import Info, Gates, MPS

N = 10
D = 5

info = Info()
gates = Gates(info)

mitigation_circuit = MitigationCircuitCX(gates)

mitigation_optimizer = MitigationOptimizer(MPS, info, N, D, gates, mitigation_circuit, max_rank=5, ort=False)

number_of_iterations = 10

list_of_parameters_fix = 2 * np.pi * np.random.rand(4 * N * D)
list_of_parameters = 0 * 2 * np.pi * np.random.rand(4 * N * D)

mitigation_optimizer.set_parameters_circuit(list_of_parameters_fix)

result = mitigation_optimizer.optimize(list_of_parameters, number_of_iterations)

print(mitigation_optimizer.infidelity_truncation)

print(result)
