import numpy as np
from qtensor import Info, Gates, VQECircuitCX, MPS, IsingHamAnalytical
import time

N = 15
D = 10

info = Info()
state = MPS(info)
state.all_zeros_state(N)

gates = Gates(info)
circuit = VQECircuitCX(gates)

ising_ham = IsingHamAnalytical(N, gates, info)

list_of_parameters = 2 * np.pi * np.random.rand(4 * N * D)

time_start = time.time()
for i in range(1):
    print('i = ', i)
    state.all_zeros_state(N)
    circuit.evolution(list_of_parameters, state, N, D, max_rank=15, ort=False)
    print(ising_ham.list_mean_ham([state]))

    state.all_zeros_state(N)
    circuit.evolution(list_of_parameters, state, N, D, max_rank=15, ort=True)
    print(ising_ham.list_mean_ham([state]))

    state.all_zeros_state(N)
    circuit.evolution(list_of_parameters, state, N, D, max_rank=None, ort=False)
    print(ising_ham.list_mean_ham([state]))

    state.all_zeros_state(N)
    circuit.evolution(list_of_parameters, state, N, D, max_rank=None, ort=True)
    print(ising_ham.list_mean_ham([state]))
print(state.r)
print(time.time() - time_start)
