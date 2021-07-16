from qtensor import Info, State, MPS, Circuit, Gates

N = 10000
D = 10

info = Info()

# state = State(info)
# state.all_zeros_state(N)

mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)
circuit = Circuit(gates)

circuit.evolution([mps], N, D, max_rank=8)

# print(mps.return_full_vector())
# print(state.return_full_vector())

print(mps.r)
for i in range(N):
    print(mps.tt_cores[i].size())
