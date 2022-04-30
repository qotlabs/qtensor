from qtensor import Info, State, MPS, CircuitCZFid, Gates, Loader
import time

N = 60
D = 150
max_rank = 64

time_start = time.time()

info = Info()

mps = MPS(info)
mps.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitCZFid(gates)

fid_result = []

circuit.evolution(mps, N, D, fid_result, max_rank=max_rank, ort=False)

print(fid_result)
print(len(fid_result))

load = Loader('../Results.xlsx')
sheet_name = 'Two_qubit_fidelity_ort'
load.write_data(sheet_name, 'D', 1, D, fid_result)

print(time.time() - time_start)
