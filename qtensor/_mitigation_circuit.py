import copy
import numpy as np
import torch


class MitigationCircuitCX(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, list_of_parameters_fix, state, N, D, max_rank=None, ort=False):
        index_of_parameter_fix = 0
        parity = False
        for d in range(D):
            for i in range(N):
                GPhase_1 = self.gates.GPhase(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Rx_2 = self.gates.Rx(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Ry_3 = self.gates.Ry(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Rx_4 = self.gates.Rx(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1

                state.one_qubit_gate(GPhase_1, i)
                state.one_qubit_gate(Rx_2, i)
                state.one_qubit_gate(Ry_3, i)
                state.one_qubit_gate(Rx_4, i)

            if not parity:
                for i in range(0, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                parity = False

    def evolution_mitigation(self, list_of_parameters_fix, list_of_parameters, state, N, D, max_rank=None, ort=False):
        index_of_parameter_fix = 0
        index_of_parameter = 0
        parity = False
        for d in range(D):
            for i in range(N):
                GPhase_1 = self.gates.GPhase(list_of_parameters[index_of_parameter])
                index_of_parameter += 1
                Rx_2 = self.gates.Rx(list_of_parameters[index_of_parameter])
                index_of_parameter += 1
                Ry_3 = self.gates.Ry(list_of_parameters[index_of_parameter])
                index_of_parameter += 1
                Rx_4 = self.gates.Rx(list_of_parameters[index_of_parameter])
                index_of_parameter += 1

                state.one_qubit_gate(GPhase_1, i)
                state.one_qubit_gate(Rx_2, i)
                state.one_qubit_gate(Ry_3, i)
                state.one_qubit_gate(Rx_4, i)

            for i in range(N):
                GPhase_1 = self.gates.GPhase(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Rx_2 = self.gates.Rx(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Ry_3 = self.gates.Ry(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1
                Rx_4 = self.gates.Rx(list_of_parameters_fix[index_of_parameter_fix])
                index_of_parameter_fix += 1

                state.one_qubit_gate(GPhase_1, i)
                state.one_qubit_gate(Rx_2, i)
                state.one_qubit_gate(Ry_3, i)
                state.one_qubit_gate(Rx_4, i)

            if not parity:
                for i in range(0, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                parity = False
