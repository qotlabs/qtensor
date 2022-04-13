import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import optimize
import copy


class MitigationOptimizer(object):
    def __init__(self, state, info, N, D, gates, mitigation_circuit, max_rank=None, ort=False):
        self.state = state
        self.info = info
        self.N = N
        self.D = D
        self.gates = gates
        self.mitigation_circuit = mitigation_circuit
        self.max_rank = max_rank
        self.ort = ort
        self.count_mean_energy_calling = 0
        self.list_of_parameters_fix = None
        self.state_exact = None
        self.infidelity_truncation = None

    def set_parameters_circuit(self, list_of_parameters_fix):
        self.list_of_parameters_fix = list_of_parameters_fix

    def calculate_state_exact(self):
        state_exact = self.state(self.info)
        state_exact.all_zeros_state(self.N)
        self.mitigation_circuit.evolution(self.list_of_parameters_fix, state_exact, self.N, self.D)
        self.state_exact = state_exact

    def get_infidelity_truncation(self):
        return self.infidelity_truncation

    def infidelity(self, list_of_parameters):
        self.count_mean_energy_calling += 1
        # print('mean energy calling: ', self.count_mean_energy_calling)
        state_rank_truncation = self.state(self.info)
        state_rank_truncation.all_zeros_state(self.N)
        self.mitigation_circuit.evolution(self.list_of_parameters_fix, state_rank_truncation, self.N, self.D,
                                          max_rank=self.max_rank, ort=self.ort)
        state_mitigation = self.state(self.info)
        state_mitigation.all_zeros_state(self.N)
        self.mitigation_circuit.evolution_mitigation(self.list_of_parameters_fix, list_of_parameters, state_mitigation,
                                                     self.N, self.D, max_rank=self.max_rank, ort=self.ort)
        self.infidelity_truncation = 1 - self.state_exact.fidelity(state_rank_truncation)
        # print('Infidelity_truncation = ', self.infidelity_truncation)
        # print(state_exact.r, state_rank_truncation.r, state_mitigation.r)
        return 1 - self.state_exact.fidelity(state_mitigation)

    def gradient_infidelity(self, list_of_parameters):
        print('Gradient calculating')
        grad = []
        delta = 1e-05
        for i in range(len(list_of_parameters)):
            list_of_parameters_i_new = copy.deepcopy(list_of_parameters)
            list_of_parameters_i_new[i] = list_of_parameters[i] + delta
            grad_i = (self.infidelity(list_of_parameters_i_new) - self.infidelity(list_of_parameters)) / delta
            grad.append(grad_i)
        # print(grad)
        return grad

    def optimize(self, list_of_parameters, number_of_iterations):
        list_results = []
        x0 = list_of_parameters
        list_results.append(self.infidelity(x0))
        print('Number of iterations: ', 0, ' ', 'Infidelity = ', self.infidelity(x0))
        for i in range(1, number_of_iterations, 1):
            res = minimize(self.infidelity, x0, method='L-BFGS-B', jac=self.gradient_infidelity,
                           options={'disp': False, 'maxiter': 1})
            # res = minimize(self.infidelity, x0, method='L-BFGS-B', options={'disp': False, 'maxiter': 1})
            # res = minimize(self.infidelity, x0, method='Nelder-Mead', options={'disp': False, 'maxiter': 1})
            x0 = res.x
            list_results.append(self.infidelity(x0))
            print('Number of iterations: ', i, ' ', 'Infidelity = ', self.infidelity(x0))
        return list_results
