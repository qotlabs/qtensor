import numpy as np
from qtensor import VQEOptimizer, VQECircuitCXError
from qtensor import Info, Gates, IsingHam, MPS, Load

N = 50
D = 1

info = Info()
gates = Gates(info)
ham = IsingHam(N, gates, info)

# print('Min energy = ', ham.get_min_energy())

# load = Load('Results.xlsx')
# sheet_name = 'VQE'
# load.write_data(sheet_name, 'F', 22, 22, [ham.get_min_energy()])

vqe_circuit = VQECircuitCXError(gates)

vqe_optimizer = VQEOptimizer(MPS, info, N, D, ham, gates, vqe_circuit, max_rank=None)

number_of_iterations = 20

list_of_parameters = 2 * np.pi * np.random.rand(4 * N * D)

result = vqe_optimizer.optimize(list_of_parameters, number_of_iterations)

load = Load('Results.xlsx')
sheet_name = 'VQE'
load.write_data(sheet_name, 'J', 1, 20, result)
