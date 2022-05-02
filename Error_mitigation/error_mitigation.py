import numpy as np
from qtensor import MitigationStartCircuitCX, MitigationFinishCircuitCX, MitigationFullCircuitCX
from qtensor import MitigationAllOneLayerCircuitCX, MitigationAllTwoLayerCircuitCX
from qtensor import MitigationWithoutCircuitCX
from qtensor import Info, Gates, MPS
from tqdm import tqdm
from qtensor import Loader

N = 5
D = 10

info = Info()
gates = Gates(info)

number_of_iterations = 1
nums_of_samples = 100
params_fix_matrix = 2 * np.pi * np.random.rand(nums_of_samples, 4 * N * D)


# mitigation_circuit = MitigationStartCircuitCX(info, gates, number_of_iterations)
#
# fid_start_results = []
# fid_finish_results = []
# fid_start_ort_results = []
# fid_finish_ort_results = []
# fid_results = []
# fid_ort_results = []
#
# for k in tqdm(range(nums_of_samples)):
#     params_fix = params_fix_matrix[k]
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start, fid_finish = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=False)
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start_ort, fid_finish_ort = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=True)
#
#     fid_start_results.append(fid_start)
#     fid_finish_results.append(fid_finish)
#     fid_start_ort_results.append(fid_start_ort)
#     fid_finish_ort_results.append(fid_finish_ort)
#
# fid_start_results = np.array(fid_start_results)
# fid_finish_results = np.array(fid_finish_results)
# fid_start_ort_results = np.array(fid_start_ort_results)
# fid_finish_ort_results = np.array(fid_finish_ort_results)
#
# loader = Loader('Results.xlsx')
# sheet_name = 'Start'
# loader.write_data(sheet_name, 'A', 1, nums_of_samples, fid_start_results)
# loader.write_data(sheet_name, 'B', 1, nums_of_samples, fid_finish_results)
# loader.write_data(sheet_name, 'C', 1, nums_of_samples, fid_start_ort_results)
# loader.write_data(sheet_name, 'D', 1, nums_of_samples, fid_finish_ort_results)
#
#
# mitigation_circuit = MitigationFinishCircuitCX(info, gates, number_of_iterations)
#
# fid_start_results = []
# fid_finish_results = []
# fid_start_ort_results = []
# fid_finish_ort_results = []
# fid_results = []
# fid_ort_results = []
#
# for k in tqdm(range(nums_of_samples)):
#     params_fix = params_fix_matrix[k]
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start, fid_finish = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=False)
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start_ort, fid_finish_ort = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=True)
#
#     fid_start_results.append(fid_start)
#     fid_finish_results.append(fid_finish)
#     fid_start_ort_results.append(fid_start_ort)
#     fid_finish_ort_results.append(fid_finish_ort)
#
# fid_start_results = np.array(fid_start_results)
# fid_finish_results = np.array(fid_finish_results)
# fid_start_ort_results = np.array(fid_start_ort_results)
# fid_finish_ort_results = np.array(fid_finish_ort_results)
#
# loader = Loader('Results.xlsx')
# sheet_name = 'Finish'
# loader.write_data(sheet_name, 'A', 1, nums_of_samples, fid_start_results)
# loader.write_data(sheet_name, 'B', 1, nums_of_samples, fid_finish_results)
# loader.write_data(sheet_name, 'C', 1, nums_of_samples, fid_start_ort_results)
# loader.write_data(sheet_name, 'D', 1, nums_of_samples, fid_finish_ort_results)


mitigation_circuit = MitigationAllOneLayerCircuitCX(info, gates, number_of_iterations)

fid_start_results = []
fid_finish_results = []
fid_start_ort_results = []
fid_finish_ort_results = []
fid_start_results_layer = []
fid_finish_results_layer = []
fid_start_ort_results_layer = []
fid_finish_ort_results_layer = []

for k in tqdm(range(nums_of_samples)):
    params_fix = params_fix_matrix[k]

    state = MPS(info)
    state.all_zeros_state(N)
    fid_start, fid_finish, fid_start_layer, fid_finish_layer = \
        mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=False)

    state = MPS(info)
    state.all_zeros_state(N)
    fid_start_ort, fid_finish_ort, fid_start_ort_layer, fid_finish_ort_layer = \
        mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=True)

    fid_start_results.append(fid_start)
    fid_finish_results.append(fid_finish)
    fid_start_ort_results.append(fid_start_ort)
    fid_finish_ort_results.append(fid_finish_ort)
    fid_start_results_layer.extend(list(fid_start_layer))
    fid_finish_results_layer.extend(list(fid_finish_layer))
    fid_start_ort_results_layer.extend(list(fid_start_ort_layer))
    fid_finish_ort_results_layer.extend(list(fid_finish_ort_layer))

fid_start_results = np.array(fid_start_results)
fid_finish_results = np.array(fid_finish_results)
fid_start_ort_results = np.array(fid_start_ort_results)
fid_finish_ort_results = np.array(fid_finish_ort_results)
fid_start_results_layer = np.array(fid_start_results_layer)
fid_finish_results_layer = np.array(fid_finish_results_layer)
fid_start_ort_results_layer = np.array(fid_start_ort_results_layer)
fid_finish_ort_results_layer = np.array(fid_finish_ort_results_layer)

loader = Loader('Results.xlsx')
sheet_name = 'AllOneLayer'
loader.write_data(sheet_name, 'A', 1, nums_of_samples, fid_start_results)
loader.write_data(sheet_name, 'B', 1, nums_of_samples, fid_finish_results)
loader.write_data(sheet_name, 'C', 1, nums_of_samples, fid_start_ort_results)
loader.write_data(sheet_name, 'D', 1, nums_of_samples, fid_finish_ort_results)

loader = Loader('Results.xlsx')
sheet_name = 'AllOneLayerMany'
loader.write_data(sheet_name, 'A', 1, nums_of_samples * D, fid_start_results_layer)
loader.write_data(sheet_name, 'B', 1, nums_of_samples * D, fid_finish_results_layer)
loader.write_data(sheet_name, 'C', 1, nums_of_samples * D, fid_start_ort_results_layer)
loader.write_data(sheet_name, 'D', 1, nums_of_samples * D, fid_finish_ort_results_layer)


mitigation_circuit = MitigationAllTwoLayerCircuitCX(info, gates, number_of_iterations)

fid_start_results = []
fid_finish_results = []
fid_start_ort_results = []
fid_finish_ort_results = []
fid_start_results_layer = []
fid_finish_results_layer = []
fid_start_ort_results_layer = []
fid_finish_ort_results_layer = []

for k in tqdm(range(nums_of_samples)):
    params_fix = params_fix_matrix[k]

    state = MPS(info)
    state.all_zeros_state(N)
    fid_start, fid_finish, fid_start_layer, fid_finish_layer = \
        mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=False)

    state = MPS(info)
    state.all_zeros_state(N)
    fid_start_ort, fid_finish_ort, fid_start_ort_layer, fid_finish_ort_layer = \
        mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=True)

    fid_start_results.append(fid_start)
    fid_finish_results.append(fid_finish)
    fid_start_ort_results.append(fid_start_ort)
    fid_finish_ort_results.append(fid_finish_ort)
    fid_start_results_layer.extend(list(fid_start_layer))
    fid_finish_results_layer.extend(list(fid_finish_layer))
    fid_start_ort_results_layer.extend(list(fid_start_ort_layer))
    fid_finish_ort_results_layer.extend(list(fid_finish_ort_layer))

fid_start_results = np.array(fid_start_results)
fid_finish_results = np.array(fid_finish_results)
fid_start_ort_results = np.array(fid_start_ort_results)
fid_finish_ort_results = np.array(fid_finish_ort_results)
fid_start_results_layer = np.array(fid_start_results_layer)
fid_finish_results_layer = np.array(fid_finish_results_layer)
fid_start_ort_results_layer = np.array(fid_start_ort_results_layer)
fid_finish_ort_results_layer = np.array(fid_finish_ort_results_layer)

loader = Loader('Results.xlsx')
sheet_name = 'AllTwoLayer'
loader.write_data(sheet_name, 'A', 1, nums_of_samples, fid_start_results)
loader.write_data(sheet_name, 'B', 1, nums_of_samples, fid_finish_results)
loader.write_data(sheet_name, 'C', 1, nums_of_samples, fid_start_ort_results)
loader.write_data(sheet_name, 'D', 1, nums_of_samples, fid_finish_ort_results)

loader = Loader('Results.xlsx')
sheet_name = 'AllTwoLayerMany'
loader.write_data(sheet_name, 'A', 1, nums_of_samples * (D // 2), fid_start_results_layer)
loader.write_data(sheet_name, 'B', 1, nums_of_samples * (D // 2), fid_finish_results_layer)
loader.write_data(sheet_name, 'C', 1, nums_of_samples * (D // 2), fid_start_ort_results_layer)
loader.write_data(sheet_name, 'D', 1, nums_of_samples * (D // 2), fid_finish_ort_results_layer)


# mitigation_circuit = MitigationFullCircuitCX(info, gates, number_of_iterations)
#
# fid_start_results = []
# fid_finish_results = []
# fid_start_ort_results = []
# fid_finish_ort_results = []
# fid_results = []
# fid_ort_results = []
#
# for k in tqdm(range(nums_of_samples)):
#     params_fix = params_fix_matrix[k]
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start, fid_finish = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=False)
#
#     state = MPS(info)
#     state.all_zeros_state(N)
#     fid_start_ort, fid_finish_ort = mitigation_circuit.evolution(state, params_fix, N, D, max_rank=2, ort=True)
#
#     fid_start_results.append(fid_start)
#     fid_finish_results.append(fid_finish)
#     fid_start_ort_results.append(fid_start_ort)
#     fid_finish_ort_results.append(fid_finish_ort)
#
# fid_start_results = np.array(fid_start_results)
# fid_finish_results = np.array(fid_finish_results)
# fid_start_ort_results = np.array(fid_start_ort_results)
# fid_finish_ort_results = np.array(fid_finish_ort_results)
#
# loader = Loader('Results.xlsx')
# sheet_name = 'Full'
# loader.write_data(sheet_name, 'A', 1, nums_of_samples, fid_start_results)
# loader.write_data(sheet_name, 'B', 1, nums_of_samples, fid_finish_results)
# loader.write_data(sheet_name, 'C', 1, nums_of_samples, fid_start_ort_results)
# loader.write_data(sheet_name, 'D', 1, nums_of_samples, fid_finish_ort_results)
