from qtensor import Info, State, MPS, Circuit, Gates

N = 20
D = 10

info = Info()

# state = State(info)
# state.all_zeros_state(N)

mps = MPS(info)
mps.all_zeros_state(N)
print('Norma = ', mps.get_norm())
gates = Gates(info)
circuit = Circuit(gates)

circuit.evolution([mps], N, D, max_rank=1024)

# print(mps.return_full_vector())
# print(state.return_full_vector())

print(mps.r)
for i in range(N):
    print(mps.tt_cores[i].size())

print('Norma = ', mps.get_norm())

mps_start = MPS(info)
mps_start.all_zeros_state(N)
print('p_x = ', mps.scalar_product(mps_start))
