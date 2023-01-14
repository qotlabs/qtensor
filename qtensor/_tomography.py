import torch
import numpy as np
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
