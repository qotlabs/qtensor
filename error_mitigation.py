import numpy as np
from qtensor import IsingHam
from qtensor import Info, MPS, Gates, CircuitCXError
from qtensor import Load
import matplotlib.pyplot as plt
import time

N = 50
D = 5
# list_max_rank = list(map(int, 2 ** np.linspace(0, 5, 6)))
list_max_rank = list(map(int, np.linspace(1, 16, 16)))
list_max_rank.append(None)
list_max_rank += list(map(int, np.linspace(1, 16, 16)))
list_max_rank.append(None)
print(list_max_rank)

time_start = time.time()

info = Info()

list_of_mps = []
for max_rank in list_max_rank:
    mps = MPS(info)
    mps.all_zeros_state(N)
    list_of_mps.append(mps)

gates = Gates(info)
circuit = CircuitCXError(gates)

fid_result = []

circuit.evolution(list_of_mps, N, D, list_max_rank, ort=False)

ising_ham = IsingHam(N, gates)
list_mean_ham = ising_ham.list_mean_ham(list_of_mps)
print(list_mean_ham)

print(time.time() - time_start)

fig, ax = plt.subplots()
plt.scatter(list_max_rank, list_mean_ham)
plt.xscale('log')
plt.show()

load = Load('Results.xlsx')
sheet_name = 'Error_mitigation'
load.write_data(sheet_name, 'S', 1, 34, list_max_rank)
load.write_data(sheet_name, 'T', 1, 34, list_mean_ham)
