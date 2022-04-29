import copy
import numpy as np
from scipy.optimize import minimize
import time


class MitigationBaseCircuitCX(object):
    def __init__(self, info, gates, iter_nums):
        self.info = info
        self.gates = gates
        self.iter_nums = iter_nums

    def evolution_two_qubits_layer(self, state, parity, N, max_rank=None, ort=False):
        if not parity:
            for i in range(0, N - 1, 2):
                state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)
        else:
            for i in range(1, N - 1, 2):
                state.two_qubit_gate(self.gates.CX(), i, max_rank, ort)

    def evolution_one_qubits_layer(self, state, N, parameters_fix_part):
        for i in range(0, N, 1):
            GPhase_1 = self.gates.GPhase(parameters_fix_part[4 * i])
            Rx_2 = self.gates.Rx(parameters_fix_part[4 * i + 1])
            Ry_3 = self.gates.Ry(parameters_fix_part[4 * i + 2])
            Rx_4 = self.gates.Rx(parameters_fix_part[4 * i + 3])

            state.one_qubit_gate(GPhase_1, i)
            state.one_qubit_gate(Rx_2, i)
            state.one_qubit_gate(Ry_3, i)
            state.one_qubit_gate(Rx_4, i)

    def evolution_mitigation(self, state, N, parameters_opt_part):
        self.evolution_one_qubits_layer(state, N, parameters_opt_part)

    def evolution(self, state, parameters_fix, N, D, max_rank=None, ort=False):
        parity = False
        for d in range(D):
            self.evolution_one_qubits_layer(state, N, parameters_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            if not parity:
                parity = True
            else:
                parity = False

    @staticmethod
    def print_iterations(number_of_iterations, infidelity_value):
        print('Number of iterations: ', number_of_iterations, ' ', 'Infidelity = ', infidelity_value)


class MitigationStartCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        # Mitigation block
        iteration_nums = self.iter_nums
        params_opt = 0.00 * 2 * np.pi * np.random.rand(4 * N)
        params_opt, fid_start, fid_finish = self.mitigation_optimizer(params_opt, params_fix, copy.deepcopy(state),
                                                                      copy.deepcopy(state), iteration_nums, N, D,
                                                                      max_rank, ort)
        self.evolution_mitigation(state, N, params_opt)
        super().evolution(state, params_fix, N, D, max_rank, ort)
        return fid_start, fid_finish

    # Use copy.deepcopy() with transfer state and state_check!
    def infidelity(self, parameters_opt, parameters_fix, state, state_check, N, D, max_rank, ort):
        state = copy.deepcopy(state)
        state_check = copy.deepcopy(state_check)
        self.evolution_mitigation(state, N, parameters_opt)
        super().evolution(state, parameters_fix, N, D, max_rank, ort)
        super().evolution(state_check, parameters_fix, N, D, max_rank=None, ort=False)
        return 1 - state.fidelity(state_check)

    # Use copy.deepcopy() with transfer state and state_check!
    def mitigation_optimizer(self, params_opt, params_fix, state, state_check, iteration_nums, N, D, max_rank, ort):
        x0 = params_opt
        inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort)
        # self.print_iterations(0, inf_val)
        fid_start = 1 - inf_val
        for i in range(0, iteration_nums, 1):
            res = minimize(self.infidelity, x0,
                           args=(params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank,
                                      ort)
            # self.print_iterations(i + 1, inf_val)
        fid_finish = 1 - inf_val
        return x0, fid_start, fid_finish


class MitigationFinishCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        state_check = copy.deepcopy(state)
        super().evolution(state, params_fix, N, D, max_rank, ort)
        super().evolution(state_check, params_fix, N, D, max_rank=None, ort=False)
        # Mitigation block
        iteration_nums = self.iter_nums
        params_opt = 0.00 * 2 * np.pi * np.random.rand(4 * N)
        params_opt, fid_start, fid_finish = self.mitigation_optimizer(params_opt, params_fix, copy.deepcopy(state),
                                                                      copy.deepcopy(state_check), iteration_nums, N, D,
                                                                      max_rank, ort)
        self.evolution_mitigation(state, N, params_opt)
        return fid_start, fid_finish

    # Use copy.deepcopy() with transfer state and state_check!
    def infidelity(self, parameters_opt, parameters_fix, state, state_check, N, D, max_rank, ort):
        state = copy.deepcopy(state)
        state_check = copy.deepcopy(state_check)
        self.evolution_mitigation(state, N, parameters_opt)
        return 1 - state.fidelity(state_check)

    # Use copy.deepcopy() with transfer state and state_check!
    def mitigation_optimizer(self, params_opt, params_fix, state, state_check, iteration_nums, N, D, max_rank, ort):
        x0 = params_opt
        inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort)
        # self.print_iterations(0, inf_val)
        fid_start = 1 - inf_val
        for i in range(0, iteration_nums, 1):
            res = minimize(self.infidelity, x0,
                           args=(params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank,
                                      ort)
            # self.print_iterations(i + 1, inf_val)
        fid_finish = 1 - inf_val
        return x0, fid_start, fid_finish


class MitigationAllOneLayerCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        state_exact = copy.deepcopy(state)
        state_check = copy.deepcopy(state)
        fid_start_list = []
        fid_finish_list = []
        iteration_nums = self.iter_nums
        parity = False
        for d in range(D):
            params_opt = 0.00 * 2 * np.pi * np.random.rand(4 * N)
            params_opt, fid_start, fid_finish = self.mitigation_optimizer(params_opt,
                                                                          params_fix[(4 * N * d):(4 * N * (d + 1))],
                                                                          copy.deepcopy(state), copy.deepcopy(state),
                                                                          iteration_nums, N, parity, max_rank, ort)
            fid_start_list.append(fid_start)
            fid_finish_list.append(fid_finish)

            self.evolution_one_qubits_layer(state, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            self.evolution_mitigation(state, N, params_opt[0:(4 * N)])

            self.evolution_one_qubits_layer(state_check, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state_check, parity, N, max_rank, ort=False)

            self.evolution_one_qubits_layer(state_exact, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state_exact, parity, N, max_rank=None, ort=False)
            if not parity:
                parity = True
            else:
                parity = False
        return state_check.fidelity(state_exact), state.fidelity(state_exact)

    # Use copy.deepcopy() with transfer state and state_check!
    def infidelity(self, parameters_opt, parameters_fix, state, state_check, N, p, max_rank, ort):
        state = copy.deepcopy(state)
        state_check = copy.deepcopy(state_check)
        self.evolution_one_qubits_layer(state, N, parameters_fix)
        self.evolution_two_qubits_layer(state, p, N, max_rank, ort)
        self.evolution_mitigation(state, N, parameters_opt[0:(4 * N)])

        self.evolution_one_qubits_layer(state_check, N, parameters_fix)
        self.evolution_two_qubits_layer(state_check, p, N, max_rank=None, ort=False)
        return 1 - state.fidelity(state_check)

    # Use copy.deepcopy() with transfer state and state_check!
    def mitigation_optimizer(self, params_opt, params_fix, state, state_check, iteration_nums, N, p, max_rank, ort):
        x0 = params_opt
        inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, p, max_rank, ort)
        # self.print_iterations(0, inf_val)
        fid_start = 1 - inf_val
        for i in range(0, iteration_nums, 1):
            res = minimize(self.infidelity, x0,
                           args=(params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, p, max_rank, ort),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, p, max_rank,
                                      ort)
            # self.print_iterations(i + 1, inf_val)
        fid_finish = 1 - inf_val
        return x0, fid_start, fid_finish


class MitigationAllTwoLayerCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        state_exact = copy.deepcopy(state)
        state_check = copy.deepcopy(state)
        fid_start_list = []
        fid_finish_list = []
        iteration_nums = self.iter_nums
        for d in range(D // 2):
            params_opt = 0.00 * 2 * np.pi * np.random.rand(4 * N)
            params_opt, fid_start, fid_finish = self.mitigation_optimizer(params_opt,
                                                                          params_fix[(4 * N * 2 * d):
                                                                                     (4 * N * 2 * (d + 1))],
                                                                          copy.deepcopy(state), copy.deepcopy(state),
                                                                          iteration_nums, N, max_rank, ort)
            fid_start_list.append(fid_start)
            fid_finish_list.append(fid_finish)

            self.evolution_one_qubits_layer(state, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state, False, N, max_rank, ort)
            self.evolution_one_qubits_layer(state, N, params_fix[(4 * N):(8 * N)])
            self.evolution_two_qubits_layer(state, True, N, max_rank, ort)
            self.evolution_mitigation(state, N, params_opt[0:(4 * N)])

            self.evolution_one_qubits_layer(state_check, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state_check, False, N, max_rank, ort)
            self.evolution_one_qubits_layer(state_check, N, params_fix[(4 * N):(8 * N)])
            self.evolution_two_qubits_layer(state_check, True, N, max_rank, ort)

            self.evolution_one_qubits_layer(state_exact, N, params_fix[0:(4 * N)])
            self.evolution_two_qubits_layer(state_exact, False, N, max_rank=None, ort=False)
            self.evolution_one_qubits_layer(state_exact, N, params_fix[(4 * N):(8 * N)])
            self.evolution_two_qubits_layer(state_exact, True, N, max_rank=None, ort=False)
        return state_check.fidelity(state_exact), state.fidelity(state_exact)

    # Use copy.deepcopy() with transfer state and state_check!
    def infidelity(self, parameters_opt, parameters_fix, state, state_check, N, max_rank, ort):
        state = copy.deepcopy(state)
        state_check = copy.deepcopy(state_check)
        self.evolution_one_qubits_layer(state, N, parameters_fix[0:(4 * N)])
        self.evolution_two_qubits_layer(state, False, N, max_rank, ort)
        self.evolution_one_qubits_layer(state, N, parameters_fix[(4 * N):(8 * N)])
        self.evolution_two_qubits_layer(state, True, N, max_rank, ort)
        self.evolution_mitigation(state, N, parameters_opt[0:(4 * N)])

        self.evolution_one_qubits_layer(state_check, N, parameters_fix[0:(4 * N)])
        self.evolution_two_qubits_layer(state_check, False, N, max_rank=None, ort=False)
        self.evolution_one_qubits_layer(state_check, N, parameters_fix[(4 * N):(8 * N)])
        self.evolution_two_qubits_layer(state_check, True, N, max_rank=None, ort=False)
        return 1 - state.fidelity(state_check)

    # Use copy.deepcopy() with transfer state and state_check!
    def mitigation_optimizer(self, params_opt, params_fix, state, state_check, iteration_nums, N, max_rank, ort):
        x0 = params_opt
        inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, max_rank, ort)
        # self.print_iterations(0, inf_val)
        fid_start = 1 - inf_val
        for i in range(0, iteration_nums, 1):
            res = minimize(self.infidelity, x0,
                           args=(params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, max_rank, ort),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, max_rank,
                                      ort)
            # self.print_iterations(i + 1, inf_val)
        fid_finish = 1 - inf_val
        return x0, fid_start, fid_finish


class MitigationFullCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        # Mitigation block
        iteration_nums = self.iter_nums
        params_opt = 0.00 * 2 * np.pi * np.random.rand(4 * N * D)
        params_opt, fid_start, fid_finish = self.mitigation_optimizer(params_opt, params_fix, copy.deepcopy(state),
                                                                      copy.deepcopy(state), iteration_nums, N, D,
                                                                      max_rank, ort)
        parity = False
        for d in range(D):
            self.evolution_mitigation(state, N, params_opt[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_one_qubits_layer(state, N, params_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            if not parity:
                parity = True
            else:
                parity = False
        return fid_start, fid_finish

    # Use copy.deepcopy() with transfer state and state_check!
    def infidelity(self, parameters_opt, parameters_fix, state, state_check, N, D, max_rank, ort):
        state = copy.deepcopy(state)
        state_check = copy.deepcopy(state_check)
        parity = False
        for d in range(D):
            self.evolution_mitigation(state, N, parameters_opt[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_one_qubits_layer(state, N, parameters_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            if not parity:
                parity = True
            else:
                parity = False

        parity = False
        for d in range(D):
            self.evolution_one_qubits_layer(state_check, N, parameters_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state_check, parity, N, max_rank=None, ort=False)
            if not parity:
                parity = True
            else:
                parity = False
        return 1 - state.fidelity(state_check)

    # Use copy.deepcopy() with transfer state and state_check!
    def mitigation_optimizer(self, params_opt, params_fix, state, state_check, iteration_nums, N, D, max_rank, ort):
        x0 = params_opt
        inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort)
        # self.print_iterations(0, inf_val)
        fid_start = 1 - inf_val
        for i in range(0, iteration_nums, 1):
            res = minimize(self.infidelity, x0,
                           args=(params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank, ort),
                           method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            inf_val = self.infidelity(x0, params_fix, copy.deepcopy(state), copy.deepcopy(state_check), N, D, max_rank,
                                      ort)
            # self.print_iterations(i + 1, inf_val)
        fid_finish = 1 - inf_val
        return x0, fid_start, fid_finish


class MitigationWithoutCircuitCX(MitigationBaseCircuitCX):
    def __init__(self, info, gates, iter_nums):
        super().__init__(info, gates, iter_nums)

    def evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        state_check = copy.deepcopy(state)
        parity = False
        for d in range(D):
            self.evolution_one_qubits_layer(state, N, params_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            self.evolution_one_qubits_layer(state_check, N, params_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state_check, parity, N, max_rank=None, ort=False)
            if not parity:
                parity = True
            else:
                parity = False
        return state.fidelity(state_check)

    def full_evolution(self, state, params_fix, N, D, max_rank=None, ort=False):
        state_check = copy.deepcopy(state)
        fidelity_list = []
        parity = False
        for d in range(D):
            self.evolution_one_qubits_layer(state, N, params_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state, parity, N, max_rank, ort)
            self.evolution_one_qubits_layer(state_check, N, params_fix[(4 * N * d):(4 * N * (d + 1))])
            self.evolution_two_qubits_layer(state_check, parity, N, max_rank=None, ort=False)
            fidelity_list.append(state.fidelity(state_check))
            if not parity:
                parity = True
            else:
                parity = False
        return np.array(fidelity_list)
