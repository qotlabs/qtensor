import copy
import numpy as np
from qtensor import MPS, MPSGrad
from scipy.optimize import minimize
from scipy import optimize


class MitigationCircuitCX(object):
    def __init__(self, info, gates):
        self.info = info
        self.gates = gates

    def evolution(self, list_of_parameters_fix, state, N, D, max_rank=None, ort=False):
        index_of_parameter_fix = 0
        parity = False
        state_check = copy.deepcopy(state)
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

                state_check.one_qubit_gate(GPhase_1, i)
                state_check.one_qubit_gate(Rx_2, i)
                state_check.one_qubit_gate(Ry_3, i)
                state_check.one_qubit_gate(Rx_4, i)

            if not parity:
                for i in range(0, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                    state_check.two_qubit_gate(self.gates.CX(), i, max_rank=None, ort=False)
                parity = True
            else:
                for i in range(1, N - 1, 2):
                    state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
                    state_check.two_qubit_gate(self.gates.CX(), i, max_rank=None, ort=False)
                parity = False

            # print('Fidelity_before_mitigation = ', state.fidelity(state_check))
            fidelity_before = state.fidelity(state_check)

        # Mitigation block

        list_of_parameters = 0.00 * 2 * np.pi * np.random.rand(4 * N)
        number_of_iterations = 2
        list_of_parameters = self.mitigation_optimizer(list_of_parameters, copy.deepcopy(state),
                                                       copy.deepcopy(state_check), number_of_iterations, N)
        index_of_parameter = 0
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

        # print('Fidelity_after_mitigation = ', state.fidelity(state_check))
        fidelity_after = state.fidelity(state_check)
        print('Fidelity up:', (fidelity_after - fidelity_before) / fidelity_before)
        return (fidelity_after - fidelity_before) / fidelity_before

    def mitigation_optimizer(self, list_of_parameters, state, state_check, number_of_iterations, N):
        x0 = list_of_parameters
        print('Number of iterations: ', 0, ' ', 'Infidelity = ', self.infidelity(x0, copy.deepcopy(state),
                                                                                 copy.deepcopy(state_check), N))
        for i in range(0, number_of_iterations, 1):
            res = minimize(self.infidelity, x0, args=(copy.deepcopy(state), copy.deepcopy(state_check), N),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            print('Number of iterations: ', i + 1, ' ', 'Infidelity = ', self.infidelity(x0, copy.deepcopy(state),
                                                                                         copy.deepcopy(state_check), N))
            # print('list_of_parameters:', x0)
        return x0

    def infidelity(self, list_of_parameters, state, state_check, N):
        index_of_parameter = 0
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

        return 1 - state.fidelity(state_check)
