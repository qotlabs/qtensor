from qtensor import Info, State, MPS, CircuitCXFid, CircuitCXMultiFid, Gates, Loader
import time

N = 20
D = 40
max_rank = 50

time_start = time.time()

info = Info()

mps = MPS(info)
mps.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitCXFid(gates)

fid_result_two = []

circuit.evolution(mps, N, D, fid_result_two, max_rank=max_rank, ort=False)

print(fid_result_two)
print(len(fid_result_two))

load = Loader('../Results.xlsx')
sheet_name = 'Multi_qubit_fidelity_ort'
load.write_data(sheet_name, 'K', 1, D, fid_result_two)

fid_result_multi = []

mps = MPS(info)
mps.all_zeros_state(N)
mps_exact = MPS(info)
mps_exact.all_zeros_state(N)
circuit = CircuitCXMultiFid(gates)
circuit.evolution(mps, mps_exact, N, D, fid_result_multi, max_rank=max_rank, ort=False)

print(fid_result_multi)
print(len(fid_result_multi))

load = Loader('../Results.xlsx')
sheet_name = 'Multi_qubit_fidelity_ort'
load.write_data(sheet_name, 'L', 1, D, fid_result_multi)

print(time.time() - time_start)
