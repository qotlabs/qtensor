import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import optimize
import copy


class VQEOptimizer(object):
    def __init__(self, state, state_grad, info, N, D, ham, gates, vqe_circuit, max_rank=None, ort=False):
        self.state = state
        self.state_grad = state_grad
        self.info = info
        self.N = N
        self.D = D
        self.ham = ham
        self.gates = gates
        self.vqe_circuit = vqe_circuit
        self.max_rank = max_rank
        self.ort = ort
        self.count_mean_energy_calling = 0

    def mean_energy(self, list_of_parameters):
        self.count_mean_energy_calling += 1
        # print('mean energy calling: ', self.count_mean_energy_calling)
        mps_0 = self.state(self.info)
        mps_0.all_zeros_state(self.N)
        self.vqe_circuit.evolution(list_of_parameters, mps_0, self.N, self.D, self.max_rank, ort=self.ort)
        return self.ham.list_mean_ham([mps_0])[0]

    def gradient_mean_energy(self, list_of_parameters):
        mps_0 = self.state(self.info)
        mps_0.all_zeros_state(self.N)
        self.vqe_circuit.evolution(list_of_parameters, mps_0, self.N, self.D, self.max_rank, ort=self.ort)
        grad = []
        if self.max_rank is None:
            for i in range(len(list_of_parameters)):
                mps = self.state_grad(self.info)
                mps.all_zeros_state(self.N)
                self.vqe_circuit.gradient_evolution(i, list_of_parameters, mps, self.N, self.D, self.max_rank, ort=self.ort)
                grad.append(self.ham.list_grad_mean_ham([mps_0], [mps])[0])
        else:
            delta = 1e-05
            for i in range(len(list_of_parameters)):
                list_of_parameters_i_new = copy.deepcopy(list_of_parameters)
                list_of_parameters_i_new[i] = list_of_parameters[i] + delta
                grad_i = (self.mean_energy(list_of_parameters_i_new) - self.mean_energy(list_of_parameters)) / delta
                grad.append(grad_i)
        # grad_test = []
        # delta = 1e-05
        # for i in range(len(list_of_parameters)):
        #     list_of_parameters_i_new = copy.deepcopy(list_of_parameters)
        #     list_of_parameters_i_new[i] = list_of_parameters[i] + delta
        #     grad_i_test = (self.mean_energy(list_of_parameters_i_new) - self.mean_energy(list_of_parameters)) / delta
        #     grad_test.append(grad_i_test)
        # print('|Grad - Grad_test|^2: ', np.sum((np.array(grad) - np.array(grad_test)) ** 2))
        return grad

    def optimize(self, list_of_parameters, number_of_iterations):
        list_results = []
        x0 = list_of_parameters
        list_results.append(self.mean_energy(x0))
        print('Number of iterations: ', 0, ' ', 'Mean energy = ', self.mean_energy(x0))
        for i in range(1, number_of_iterations, 1):
            res = minimize(self.mean_energy, x0, method='L-BFGS-B', jac=self.gradient_mean_energy,
                           options={'disp': False, 'maxiter': 1})
            x0 = res.x
            list_results.append(self.mean_energy(x0))
            print('Number of iterations: ', i, ' ', 'Mean energy = ', self.mean_energy(x0))
        return list_results
