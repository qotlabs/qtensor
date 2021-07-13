from qtensor import Info, State, MPS, Circuit, Gates

N = 10
D = 5

info = Info()

state = State(info)
state.all_zeros_state(N)

mps = MPS(info)
mps.all_zeros_state(N)

gates = Gates(info)
circuit = Circuit(gates)

circuit.evolution([state, mps], N, D)

print(mps.return_full_vector())
print(state.return_full_vector())

print(mps.r)
for i in range(N):
    print(mps.tt_cores[i].size())