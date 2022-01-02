import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import optimize


class VQEOptimizer(object):
    def __init__(self, state, info, N, D, ham, gates, vqe_circuit, max_rank=None, ort=False):
        self.state = state
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
        self.vqe_circuit.evolution(list_of_parameters, [mps_0], self.N, self.D, [self.max_rank], ort=self.ort)
        return self.ham.list_mean_ham([mps_0])[0]

    def gradient_mean_energy(self, list_of_parameters):
        mps_0 = self.state(self.info)
        mps_0.all_zeros_state(self.N)
        self.vqe_circuit.evolution(list_of_parameters, [mps_0], self.N, self.D, [self.max_rank], ort=self.ort)
        grad = []
        for i in range(len(list_of_parameters)):
            # print(i)
            mps = self.state(self.info)
            mps.all_zeros_state(self.N)
            self.vqe_circuit.gradient_evolution(i, list_of_parameters, [mps], self.N, self.D, [self.max_rank],
                                                ort=self.ort)
            grad.append(self.ham.list_grad_mean_ham([mps_0], [mps])[0])
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

    def global_optimize(self, list_of_parameters):
        x0 = list_of_parameters
        bounds = [(0, 2 * np.pi) for x in list_of_parameters]
        results = optimize.dual_annealing(self.mean_energy, bounds, maxfun=5000, x0=x0)
        return results

    def cobyla_optimize(self, list_of_parameters, number_of_iterations):
        list_results = []
        x0 = list_of_parameters
        list_results.append(self.mean_energy(x0))
        print('Number of iterations: ', 0, ' ', 'Mean energy = ', self.mean_energy(x0))
        for i in range(1, number_of_iterations, 1):
            res = minimize(self.mean_energy, x0, method='COBYLA',
                           options={'disp': False, 'maxiter': 2000})
            x0 = res.x
            list_results.append(self.mean_energy(x0))
            print('Number of iterations: ', i, ' ', 'Mean energy = ', self.mean_energy(x0))
        return list_results
