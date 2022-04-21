import numpy as np
from qtensor import MitigationCircuitCX, MitigationOptimizer
from qtensor import Info, Gates, MPS

N = 10
D = 10

info = Info()
gates = Gates(info)

mitigation_circuit = MitigationCircuitCX(info, gates)

# number_of_iterations = 10

# list_of_parameters_fix = 2 * np.pi * np.random.rand(4 * N * D)
# list_of_parameters = 0.0 * 2 * np.pi * np.random.rand(4 * N)

# state = MPS(info)
# state.all_zeros_state(N)
# mitigation_circuit.evolution(list_of_parameters_fix, state, N, D, max_rank=2, ort=True)

results = []
for k in range(50):
    state = MPS(info)
    state.all_zeros_state(N)
    list_of_parameters_fix = 2 * np.pi * np.random.rand(4 * N * D)
    results.append(mitigation_circuit.evolution(list_of_parameters_fix, state, N, D, max_rank=2, ort=True))
print(results)