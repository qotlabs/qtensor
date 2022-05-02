import numpy as np
from qtensor import Info, Gates, MPS
from qtensor import CircuitCXFix, CircuitCZFix
from qtensor import Loader
import copy
from tqdm import tqdm

N = 10
D = 10
info = Info()
gates = Gates(info)

nums_of_sample = 10000

results_fid = []
results_fid_ort = []

for k in tqdm(range(nums_of_sample)):
    state = MPS(info)
    state.all_zeros_state(N)

    state_ort = copy.deepcopy(state)
    state_exact = copy.deepcopy(state)

    circuit = CircuitCXFix(gates)

    params_fix = 2 * np.pi * np.random.rand(3 * N * D)

    circuit.evolution(params_fix, state_exact, N, D, max_rank=None, ort=False)
    circuit.evolution(params_fix, state, N, D, max_rank=2, ort=False)
    circuit.evolution(params_fix, state_ort, N, D, max_rank=2, ort=True)

    results_fid.append(state.fidelity(state_exact))
    results_fid_ort.append(state_ort.fidelity(state_exact))

results_fid = np.array(results_fid)
results_fid_ort = np.array(results_fid_ort)

loader = Loader('Results.xlsx')
sheet_name = 'CX'

loader.write_data(sheet_name, 'A', 1, nums_of_sample, results_fid)
loader.write_data(sheet_name, 'B', 1, nums_of_sample, results_fid_ort)

N = 10
D = 10
info = Info()
gates = Gates(info)

nums_of_sample = 10000

results_fid = []
results_fid_ort = []

for k in tqdm(range(nums_of_sample)):
    state = MPS(info)
    state.all_zeros_state(N)

    state_ort = copy.deepcopy(state)
    state_exact = copy.deepcopy(state)

    circuit = CircuitCZFix(gates)

    params_fix = 2 * np.pi * np.random.rand(3 * N * D)

    circuit.evolution(params_fix, state_exact, N, D, max_rank=None, ort=False)
    circuit.evolution(params_fix, state, N, D, max_rank=2, ort=False)
    circuit.evolution(params_fix, state_ort, N, D, max_rank=2, ort=True)

    results_fid.append(state.fidelity(state_exact))
    results_fid_ort.append(state_ort.fidelity(state_exact))

results_fid = np.array(results_fid)
results_fid_ort = np.array(results_fid_ort)

loader = Loader('Results.xlsx')
sheet_name = 'CZ'

loader.write_data(sheet_name, 'A', 1, nums_of_sample, results_fid)
loader.write_data(sheet_name, 'B', 1, nums_of_sample, results_fid_ort)
