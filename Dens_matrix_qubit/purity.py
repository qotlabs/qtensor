import numpy as np
from qtensor import Info, Gates, MPS, CircuitCX, Loader, fidelity, purity

N = 10
D = 10

info = Info()
gates = Gates(info)

mps = MPS(info)
circuit = CircuitCX(gates)

D_list = [i for i in range(0, 11, 2)]

print(D_list)

k = 1000

purity_array = np.zeros((len(D_list), N))

for l in range(k):
    for i, D in enumerate(D_list):
        mps.all_zeros_state(N)
        circuit.evolution([mps], N, D, max_rank=None, ort=False)
        for j in range(N):
            purity_array[i, j] += purity(mps.get_density_matrix(j))
purity_array /= k

print(purity_array)

loader = Loader('Results.xlsx')
sheet_name = 'Purity'

loader.write_data(sheet_name, 'A', 1, 10, purity_array[0])
loader.write_data(sheet_name, 'B', 1, 10, purity_array[1])
loader.write_data(sheet_name, 'C', 1, 10, purity_array[2])
loader.write_data(sheet_name, 'D', 1, 10, purity_array[3])
loader.write_data(sheet_name, 'E', 1, 10, purity_array[4])
loader.write_data(sheet_name, 'F', 1, 10, purity_array[5])
