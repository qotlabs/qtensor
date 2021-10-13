import copy
import numpy as np
import torch


class Circuit(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, states_list, N, D, max_rank=None, ort=False):
        parity = False
        for d in range(D):
            for i in range(N):
                Rn = self.gates.Rn_random()
                for state in states_list:
                    state.one_qubit_gate(Rn, i)
            if not parity:
                for i in range(0, N - 1, 2):
                    for state in states_list:
                        state.two_qubit_gate(self.gates.CZ(), i, max_rank, ort)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    for state in states_list:
                        state.two_qubit_gate(self.gates.CZ(), i, max_rank, ort)
                parity = False


class CircuitFid(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, state, N, D, fid_result, max_rank=None, ort=False):
        parity = False
        for d in range(D):
            print('d = ', d)
            for i in range(N):
                Rn = self.gates.Rn_random()
                state.one_qubit_gate(Rn, i)
            if not parity:
                fid_result_point = []
                for i in range(0, N - 1, 2):
                    state_exact = copy.deepcopy(state)
                    state.two_qubit_gate(self.gates.CZ(), i, max_rank=max_rank, ort=ort)
                    state_exact.two_qubit_gate(self.gates.CZ(), i, ort=ort)
                    fid_result_point.append(state.fidelity(state_exact))
                fid_result.append(np.exp(np.log(np.array(fid_result_point)).mean()))
                parity = True
            else:
                fid_result_point = []
                for i in range(1, N - 1, 2):
                    state_exact = copy.deepcopy(state)
                    state.two_qubit_gate(self.gates.CZ(), i, max_rank=max_rank, ort=ort)
                    state_exact.two_qubit_gate(self.gates.CZ(), i, ort=ort)
                    fid_result_point.append(state.fidelity(state_exact))
                fid_result.append(np.exp(np.log(np.array(fid_result_point)).mean()))
                parity = False


class CircuitMultiFid(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, state, state_exact, N, D, fid_result, max_rank=None, ort=False):
        parity = False
        for d in range(D):
            print('d = ', d)
            for i in range(N):
                Rn = self.gates.Rn_random()
                state.one_qubit_gate(Rn, i)
                state_exact.one_qubit_gate(Rn, i)
            if not parity:
                for i in range(0, N - 1, 2):
                    state.two_qubit_gate(self.gates.CZ(), i, max_rank=max_rank, ort=ort)
                    state_exact.two_qubit_gate(self.gates.CZ(), i, ort=ort)
                fid_result.append(state.fidelity(state_exact))
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    state.two_qubit_gate(self.gates.CZ(), i, max_rank=max_rank, ort=ort)
                    state_exact.two_qubit_gate(self.gates.CZ(), i, ort=ort)
                fid_result.append(state.fidelity(state_exact))
                parity = False
