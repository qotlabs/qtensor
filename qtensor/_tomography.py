import torch
import numpy as np
from scipy.optimize import minimize
import copy
import random
from qtensor import MPS, MPO, CircuitCXFix, Gates


class DataModel(object):
    def __init__(self, info):
        self.info = info
        self.N = None
        self.state = None
        self.data_train = []
        self.data_test = []

    def gen_pure_state(self, N, max_rank):
        """
            Generates pure state |\psi><\psi| in MPO format with max rank equal to max_rank ** 2
        """
        self.N = N
        gates = Gates(self.info)
        circuit = CircuitCXFix(gates)
        mps = MPS(self.info)
        mps.all_zeros_state(N)
        D = int(np.log2(max_rank)) + 1
        list_of_parameters = 2 * np.pi * np.random.rand(3 * N * D)
        circuit.evolution(list_of_parameters, mps, N, D, max_rank=max_rank)
        mpo = MPO(self.info)
        mpo.gen_mpo_from_mps(mps)
        self.state = mpo

    def gen_mixed_state(self, N, max_rank):
        """
            Generates mixed state in MPO format with max max rank equal to max_rank ** 2
        """
        self.N = N
        mpo = MPO(self.info)
        mpo.gen_random_mpo(N, max_rank)
        self.state = mpo

    def gen_data(self, m_train, max_rank_train, m_test, max_rank_test):
        self.data_train = []
        self.data_test = []

        gates = Gates(self.info)
        circuit = CircuitCXFix(gates)
        D_train = int(np.log2(max_rank_train)) + 1
        D_test = int(np.log2(max_rank_test)) + 1

        for _ in range(m_train):
            mps = MPS(self.info)
            mps.all_zeros_state(self.N)
            list_of_parameters = 2 * np.pi * np.random.rand(3 * self.N * D_train)
            if max_rank_train == 1:
                for i in range(0, self.N, 1):
                    Rn = gates.Rn(list_of_parameters[3 * i], list_of_parameters[3 * i + 1],
                                  list_of_parameters[3 * i + 2])
                    mps.one_qubit_gate(Rn, i)
            else:
                circuit.evolution(list_of_parameters, mps, self.N, D_train, max_rank=max_rank_train)
            mpo = MPO(self.info)
            mpo.gen_mpo_from_mps(mps)
            self.data_train.append((mpo, self.get_prob(mpo)))

        for _ in range(m_test):
            mps = MPS(self.info)
            mps.all_zeros_state(self.N)
            list_of_parameters = 2 * np.pi * np.random.rand(3 * self.N * D_test)
            if max_rank_test == 1:
                for i in range(0, self.N, 1):
                    Rn = gates.Rn(list_of_parameters[3 * i], list_of_parameters[3 * i + 1],
                                  list_of_parameters[3 * i + 2])
                    mps.one_qubit_gate(Rn, i)
            else:
                circuit.evolution(list_of_parameters, mps, self.N, D_test, max_rank=max_rank_test)
            mpo = MPO(self.info)
            mpo.gen_mpo_from_mps(mps)
            self.data_test.append((mpo, self.get_prob(mpo)))

    def get_prob(self, E):
        if self.state is None:
            raise RuntimeError('State is not defined')
        else:
            p = self.state.get_trace_product_matrix(E)
            return p.real

    def get_mini_batch(self, mini_batch_size):
        x = np.random.randint(0, len(self.data_train), mini_batch_size)
        mini_batch_train = [self.data_train[x[i]] for i in range(mini_batch_size)]
        return mini_batch_train


class LearnModel(object):
    def __init__(self, info):
        self.info = info
        self.N = None
        self.omega = None
        self.model = None

    def gen_start_state(self, N, max_rank):
        """
            Generates mixed state in MPO format with max max rank equal to max_rank ** 2
        """
        self.N = N
        mpo = MPO(self.info)
        mpo.gen_random_cores(N, max_rank)
        self.omega = mpo
        self.model = self.omega.get_product_matrix(self.omega.star())
        trace = np.abs(self.model.get_trace())
        coeff = np.exp(np.log(trace) / self.N)
        for i in range(self.N):
            self.omega.tt_cores[i] /= np.sqrt(coeff)
        for i in range(self.N):
            self.model.tt_cores[i] /= coeff

    def get_prob(self, E):
        if self.model is None:
            raise RuntimeError('State is not defined')
        else:
            p = self.model.get_trace_product_matrix(E)
        return p.real

    def get_params(self):
        params = []
        for i in range(0, self.N, 1):
            core = np.array(self.omega.tt_cores[i], dtype=complex).reshape(-1)
            for j in range(0, len(core), 1):
                params += [core[j].real, core[j].imag]
        return params

    def set_params(self, params):
        index = 0
        for i in range(0, self.N, 1):
            size = 2 * self.omega.r[i] * self.omega.phys_ind_i[i] * self.omega.phys_ind_j[i] * self.omega.r[i + 1]
            shape = (self.omega.r[i], self.omega.phys_ind_i[i], self.omega.phys_ind_j[i], self.omega.r[i + 1])
            core_real = np.array(params[index:(index + size):2]).reshape(shape)
            core_imag = np.array(params[(index + 1):(index + size):2]).reshape(shape)
            core = core_real + 1j * core_imag
            index += size
            self.omega.tt_cores[i] = torch.tensor(core, dtype=self.info.data_type, device=self.info.device)
        self.model = self.omega.get_product_matrix(self.omega.star())

    def func_loss(self, params, mini_batch, coeff):
        mini_batch_size = len(mini_batch)
        self.set_params(params)
        p_exact_list = np.array([mini_batch[i][1] for i in range(mini_batch_size)])
        p_model_list = np.array([self.get_prob(mini_batch[i][0]) for i in range(mini_batch_size)])
        loss = coeff * np.sum((p_model_list - p_exact_list) ** 2) / mini_batch_size
        return loss

    def grad_func_loss(self, params, mini_batch, coeff):
        mini_batch_size = len(mini_batch)
        self.set_params(params)
        p_exact_list = np.array([mini_batch[i][1] for i in range(mini_batch_size)])
        p_model_list = np.array([self.get_prob(mini_batch[i][0]) for i in range(mini_batch_size)])
        grad_trace_loss = []
        for k in range(0, mini_batch_size, 1):
            E = mini_batch[k][0]
            omega_E = self.omega.star().get_product_matrix(E)
            for i in range(0, self.N, 1):
                size = 2 * self.omega.r[i] * self.omega.phys_ind_i[i] * self.omega.phys_ind_j[i] * self.omega.r[i + 1]
                shape = (self.omega.r[i], self.omega.phys_ind_i[i], self.omega.phys_ind_j[i], self.omega.r[i + 1])
                aux_tensor = self.omega.get_tensor_for_trace(omega_E, i)
                for j in range(0, size, 1):
                    core = torch.tensor(np.zeros(size // 2, dtype=complex), dtype=self.info.data_type,
                                        device=self.info.device)
                    if j % 2 == 0:
                        core[j // 2] = 1.0
                    else:
                        core[j // 2] = 1j
                    core = torch.reshape(core, list(shape))
                    trace = torch.tensordot(aux_tensor, core, dims=([0, 2, 1, 3], [0, 1, 2, 3])).item()
                    grad_trace_loss.append(2 * trace.real)
        grad_trace_loss = np.array(grad_trace_loss)
        grad_trace_loss = grad_trace_loss.reshape((mini_batch_size, len(params)))
        grad_loss = coeff * np.tensordot((2 * (p_model_list - p_exact_list)), grad_trace_loss, axes=([0], [0]))
        return grad_loss / mini_batch_size

    def grad_func_loss_test(self, params, mini_batch, coeff):
        mini_batch_size = len(mini_batch)
        self.set_params(params)
        delta = 0.0001
        grad_loss = []
        for i in range(len(params)):
            func = self.func_loss(params, mini_batch, coeff)
            params_new = copy.deepcopy(params)
            params_new[i] = params_new[i] + delta
            func_new = self.func_loss(params_new, mini_batch, coeff)
            der = (func_new - func) / delta
            grad_loss.append(der)
        return np.array(grad_loss)

    def optimize(self, data_model, mini_batch_size, num_of_iters, coeff):
        x0 = self.get_params()
        for iterations in range(num_of_iters):
            mini_batch = data_model.get_mini_batch(mini_batch_size)
            res = minimize(self.func_loss, x0, args=(mini_batch, coeff), method='BFGS',
                           jac=self.grad_func_loss, options={'disp': False, 'maxiter': 1})
            # res = minimize(self.func_loss, x0, args=(mini_batch, coeff), method='BFGS',
            #                options={'disp': False, 'maxiter': 1})
            x0 = res.x
            self.set_params(x0)
            print('Iterations: ', iterations, ' Infidelity: ', self.func_loss(x0, mini_batch, coeff))
