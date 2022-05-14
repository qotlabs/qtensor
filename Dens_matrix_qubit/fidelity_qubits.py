import numpy as np
from qtensor import Info, Gates, MPS, CircuitCXRanking, CircuitCXRankingFull, Loader, fidelity, purity
from tqdm import tqdm

N = 10
D = 10

info = Info()
gates = Gates(info)

mps = MPS(info)
circuit = CircuitCXRanking(gates)

max_rank_list = [1, 2, 4, 8, 16, 32, None]

k = 1000

fidelity_array = np.zeros((len(max_rank_list), N))

for l in tqdm(range(k)):
    gates_random = [gates.Rn_random() for _ in range(N * D)]
    list_rho = []
    for i, max_rank in enumerate(max_rank_list):
        mps.all_zeros_state(N)
        circuit.evolution(gates_random, mps, N, D, max_rank, ort=True)
        list_rho.append([mps.get_density_matrix(j) for j in range(N)])
    for i in range(len(max_rank_list)):
        for j in range(N):
            fidelity_array[i, j] += fidelity(list_rho[-1][j], list_rho[i][j])

fidelity_array /= k

print(fidelity_array)

loader = Loader('Results.xlsx')
sheet_name = 'Fidelity_qubits'

loader.write_data(sheet_name, 'A', 1, 10, fidelity_array[0])
loader.write_data(sheet_name, 'B', 1, 10, fidelity_array[1])
loader.write_data(sheet_name, 'C', 1, 10, fidelity_array[2])
loader.write_data(sheet_name, 'D', 1, 10, fidelity_array[3])
loader.write_data(sheet_name, 'E', 1, 10, fidelity_array[4])
loader.write_data(sheet_name, 'F', 1, 10, fidelity_array[5])
loader.write_data(sheet_name, 'J', 1, 10, fidelity_array[6])
