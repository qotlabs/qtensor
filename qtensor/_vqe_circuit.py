import copy
import numpy as np
import torch


class VQECircuitCXError(object):
    def __init__(self, gates):
        self.gates = gates

    def evolution(self, list_of_parameters, states_list, N, D, max_rank_list, ort=False):
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

                for state in states_list:
                    state.one_qubit_gate(GPhase_1, i)
                    state.one_qubit_gate(Rx_2, i)
                    state.one_qubit_gate(Ry_3, i)
                    state.one_qubit_gate(Rx_4, i)

            if not parity:
                for i in range(0, N - 1, 2):
                    for j, state in enumerate(states_list):
                        state.two_qubit_gate(self.gates.CX(), i, max_rank_list[j], ort)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    for j, state in enumerate(states_list):
                        state.two_qubit_gate(self.gates.CX(), i, max_rank_list[j], ort)
                parity = False

    def gradient_evolution(self, index_grad, list_of_parameters, states_list, N, D, max_rank_list, ort=False):
        index_of_parameter = 0
        parity = False
        for d in range(D):
            for i in range(N):
                if index_of_parameter == index_grad:
                    GPhase_1 = self.gates.GPhase_der(list_of_parameters[index_of_parameter])
                else:
                    GPhase_1 = self.gates.GPhase(list_of_parameters[index_of_parameter])
                index_of_parameter += 1

                if index_of_parameter == index_grad:
                    Rx_2 = self.gates.Rx_der(list_of_parameters[index_of_parameter])
                else:
                    Rx_2 = self.gates.Rx(list_of_parameters[index_of_parameter])
                index_of_parameter += 1

                if index_of_parameter == index_grad:
                    Ry_3 = self.gates.Ry_der(list_of_parameters[index_of_parameter])
                else:
                    Ry_3 = self.gates.Ry(list_of_parameters[index_of_parameter])
                index_of_parameter += 1

                if index_of_parameter == index_grad:
                    Rx_4 = self.gates.Rx_der(list_of_parameters[index_of_parameter])
                else:
                    Rx_4 = self.gates.Rx(list_of_parameters[index_of_parameter])
                index_of_parameter += 1

                for state in states_list:
                    state.one_qubit_gate(GPhase_1, i)
                    state.one_qubit_gate(Rx_2, i)
                    state.one_qubit_gate(Ry_3, i)
                    state.one_qubit_gate(Rx_4, i)

            if not parity:
                for i in range(0, N - 1, 2):
                    for j, state in enumerate(states_list):
                        state.two_qubit_gate(self.gates.CX(), i, max_rank_list[j], ort)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    for j, state in enumerate(states_list):
                        state.two_qubit_gate(self.gates.CX(), i, max_rank_list[j], ort)
                parity = False
