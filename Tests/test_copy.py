import copy
from qtensor import MPS, Gates, Info

N = 10

info = Info()
gates = Gates(info)
state = MPS(info)

state.all_zeros_state(N)

state_copy = copy.deepcopy(state)

state_copy.one_qubit_gate(gates.X(), 5)

print(state.fidelity(state_copy))
