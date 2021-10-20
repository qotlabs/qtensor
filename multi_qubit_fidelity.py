from qtensor import Info, State, MPS, CircuitCXFid, CircuitCXMultiFid, Gates, Load

N = 20
D = 25
max_rank = 50

info = Info()

mps = MPS(info)
mps.all_zeros_state(N)
gates = Gates(info)
circuit = CircuitCXFid(gates)

fid_result_two = []

circuit.evolution(mps, N, D, fid_result_two, max_rank=max_rank)

print(fid_result_two)
print(len(fid_result_two))

load = Load('Results.xlsx')
sheet_name = 'Multi_qubit_fidelity'
load.write_data(sheet_name, 'E', 1, 25, fid_result_two)

fid_result_multi = []

mps = MPS(info)
mps.all_zeros_state(N)
mps_exact = MPS(info)
mps_exact.all_zeros_state(N)
circuit = CircuitCXMultiFid(gates)
circuit.evolution(mps, mps_exact, N, D, fid_result_multi, max_rank=max_rank)

print(fid_result_multi)
print(len(fid_result_multi))

load = Load('Results.xlsx')
sheet_name = 'Multi_qubit_fidelity'
load.write_data(sheet_name, 'F', 1, 25, fid_result_multi)
