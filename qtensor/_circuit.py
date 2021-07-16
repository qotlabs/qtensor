import numpy as np
import torch


class Circuit(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, states_list, N, D, max_rank=None):
        parity = False
        for d in range(D):
            for i in range(N):
                Rn = self.gates.Rn_random()
                for state in states_list:
                    state.one_qubit_gate(Rn, i)
            if not parity:
                for i in range(0, N - 1, 2):
                    for state in states_list:
                        state.two_qubit_gate(self.gates.CX(), i, max_rank)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    for state in states_list:
                        state.two_qubit_gate(self.gates.CX(), i, max_rank)
                parity = False
