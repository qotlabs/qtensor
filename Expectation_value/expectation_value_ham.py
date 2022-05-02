import numpy as np
from qtensor import Info, Gates, MPS
from qtensor import CircuitCXFix, CircuitCZFix
from qtensor import IsingHam, IsingHamAnalytical
from qtensor import Loader
import copy
from tqdm import tqdm

N = 50
D = 5
info = Info()
gates = Gates(info)
ham = IsingHamAnalytical(N, gates, info)

nums_of_sample = 100

list_ranks = np.arange(1, 33, 1)
# print(list_ranks)

results_mean_exact = []
results_mean = []
results_mean_ort = []

circuit = CircuitCXFix(gates)

for k in tqdm(range(nums_of_sample)):
    state_exact = MPS(info)
    state_exact.all_zeros_state(N)
    params_fix = 2 * np.pi * np.random.rand(3 * N * D)
    circuit.evolution(params_fix, state_exact, N, D, max_rank=None, ort=False)
    mean_exact = ham.mean_ham(state_exact)

    for j in range(len(list_ranks)):
        state_tmp = MPS(info)
        state_tmp.all_zeros_state(N)
        state = copy.deepcopy(state_tmp)
        state_ort = copy.deepcopy(state_tmp)

        circuit.evolution(params_fix, state, N, D, max_rank=list_ranks[j], ort=False)
        circuit.evolution(params_fix, state_ort, N, D, max_rank=list_ranks[j], ort=True)

        mean_without_ort = ham.mean_ham(state)
        mean_ort = ham.mean_ham(state_ort)

        results_mean_exact.append(mean_exact)
        results_mean.append(mean_without_ort)
        results_mean_ort.append(mean_ort)

results_mean_exact = np.array(results_mean_exact)
results_mean = np.array(results_mean)
results_mean_ort = np.array(results_mean_ort)

loader = Loader('Results.xlsx')
sheet_name = 'CX'

loader.write_data(sheet_name, 'A', 1, nums_of_sample * len(list_ranks), results_mean_exact)
loader.write_data(sheet_name, 'B', 1, nums_of_sample * len(list_ranks), results_mean)
loader.write_data(sheet_name, 'C', 1, nums_of_sample * len(list_ranks), results_mean_ort)
