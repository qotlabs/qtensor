import torch
import numpy as np
import copy
from qtensor import MPS


class MPO(MPS):
    def __init__(self, info):
        super().__init__(info)
        self.phys_ind_i = []
        self.phys_ind_j = []

    def all_zeros_state(self, n):
        self.tt_cores = []
        self.r = [1]
        self.phys_ind = []
        self.phys_ind_i = []
        self.phys_ind_j = []
        for i in range(n):
            self.tt_cores.append(torch.reshape(torch.tensor([[1, 0], [0, 0]], dtype=self.info.data_type,
                                                            device=self.info.device), (1, 2, 2, 1)))
            self.r.append(1)
            self.phys_ind.append(4)
            self.phys_ind_i.append(2)
            self.phys_ind_j.append(2)
        self.N = n

    def mpo_to_mps(self):
        self.phys_ind = []
        for i in range(self.N):
            self.phys_ind.append(self.phys_ind_i[i] * self.phys_ind_j[i])
            self.tt_cores[i] = torch.reshape(self.tt_cores[i], (self.r[i], self.phys_ind_i[i] * self.phys_ind_j[i],
                                                                self.r[i + 1]))

    def mpo_from_mps(self):
        for i in range(self.N):
            self.tt_cores[i] = torch.reshape(self.tt_cores[i], (self.r[i], self.phys_ind_i[i], self.phys_ind_j[i],
                                                                self.r[i + 1]))

    def gen_mpo_from_mps(self, mps: MPS):
        """
            If given MPS is psi, makes MPO equal to |psi><psi|
        """
        self.N = mps.N
        self.r = []
        self.phys_ind_i = []
        self.phys_ind_j = []
        self.tt_cores = []
        for i in range(self.N):
            up_core_curr = torch.reshape(mps.tt_cores[i], (mps.r[i], mps.phys_ind[i], 1, mps.r[i + 1]))
            down_core_curr = torch.reshape(torch.conj(mps.tt_cores[i]), (mps.r[i], mps.phys_ind[i], 1, mps.r[i + 1]))
            mpo_core_curr = torch.tensordot(up_core_curr, down_core_curr, dims=([2], [2]))
            mpo_core_curr = torch.transpose(mpo_core_curr, 1, 3)
            mpo_core_curr = torch.transpose(mpo_core_curr, 2, 4)
            mpo_core_curr = torch.transpose(mpo_core_curr, 2, 3)
            mpo_core_curr = torch.reshape(mpo_core_curr, (mps.r[i] * mps.r[i], mps.phys_ind[i], mps.phys_ind[i],
                                                          mps.r[i + 1] * mps.r[i + 1]))
            self.tt_cores.append(mpo_core_curr)
            self.r.append(mps.r[i] * mps.r[i])
            self.phys_ind_i.append(mps.phys_ind[i])
            self.phys_ind_j.append(mps.phys_ind[i])
        self.r.append(mps.r[self.N] * mps.r[self.N])

    def get_product_matrix(self, other):
        """
            return self * other
        """
        result = MPO(self.info)
        result.N = self.N
        result.tt_cores = []
        for i in range(self.N):
            mpo_core_curr = torch.tensordot(self.tt_cores[i], other.tt_cores[i], dims=([2], [1]))
            mpo_core_curr = torch.transpose(mpo_core_curr, 1, 3)
            mpo_core_curr = torch.transpose(mpo_core_curr, 2, 4)
            mpo_core_curr = torch.transpose(mpo_core_curr, 2, 3)
            mpo_core_curr = torch.reshape(mpo_core_curr, (self.r[i] * other.r[i], self.phys_ind_i[i],
                                                          other.phys_ind_j[i], self.r[i + 1] * other.r[i + 1]))
            result.tt_cores.append(mpo_core_curr)
            result.r.append(self.r[i] * other.r[i])
            result.phys_ind_i.append(self.phys_ind_i[i])
            result.phys_ind_j.append(other.phys_ind_j[i])
        result.r.append(self.r[self.N] * other.r[self.N])
        return result

    def get_trace_product_matrix(self, other):
        """
            return trace E * \rho, other = E, self = \rho Tr(other * this)
        """
        core_prev = torch.tensordot(other.tt_cores[0], self.tt_cores[0], dims=([2, 1], [1, 2]))
        for i in range(1, len(other.tt_cores), 1):
            core_prev = torch.tensordot(core_prev, other.tt_cores[i], dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([2, 4, 3], [0, 1, 2]))
            core_prev = torch.transpose(core_prev, 1, 2)
        trace = core_prev[0][0][0][0]
        return trace.item()

    def get_tensor_for_trace(self, other, n):
        """
            return partial trace E * \rho, other = E, self = \rho Tr(other * this)
        """
        if n == 0:
            core_base = torch.tensordot(other.tt_cores[0], other.tt_cores[1], dims=([3], [0]))
            core_base = torch.tensordot(core_base, self.tt_cores[1], dims=([4, 3], [1, 2]))
            for i in range(2, self.N, 1):
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([5], [0]))
                core_base = torch.tensordot(core_base, other.tt_cores[i], dims=([3, 5, 6], [0, 2, 1]))
                core_base = torch.transpose(core_base, 3, 5)
                core_base = torch.transpose(core_base, 4, 5)
            F = core_base[:, :, :, 0, :, 0]
            """          
                         1     
                        _|____________
                   0---|              |  
                F =    |_____         |  
                        2|   |        |  
                         3---|        | 
                             |________| 
            """

        elif n == (self.N - 1):
            core_base = torch.tensordot(other.tt_cores[0], self.tt_cores[0], dims=([2, 1], [1, 2]))
            for i in range(1, self.N - 1, 1):
                core_base = torch.tensordot(core_base, other.tt_cores[i], dims=([1], [0]))
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([2, 4, 3], [0, 1, 2]))
                core_base = torch.transpose(core_base, 1, 2)
            core_base = torch.tensordot(core_base, other.tt_cores[self.N - 1], dims=([1], [0]))
            F = core_base[0, 0, :, :, :, :]
            """          
                                    1
                         ___________|_
                        |             |---3
                        |        _____|  
                F =     |       |   |2
                        |       |---0
                        |_______|
            """
        else:
            core_base = torch.tensordot(other.tt_cores[0], self.tt_cores[0], dims=([2, 1], [1, 2]))
            for i in range(1, n, 1):
                core_base = torch.tensordot(core_base, other.tt_cores[i], dims=([1], [0]))
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([2, 4, 3], [0, 1, 2]))
                core_base = torch.transpose(core_base, 1, 2)
            core_base = torch.tensordot(core_base, other.tt_cores[n], dims=([1], [0]))
            core_next = torch.tensordot(other.tt_cores[n + 1], self.tt_cores[n + 1], dims=([2, 1], [1, 2]))
            for i in range(n + 2, self.N, 1):
                core_next = torch.tensordot(core_next, other.tt_cores[i], dims=([1], [0]))
                core_next = torch.tensordot(core_next, self.tt_cores[i], dims=([2, 4, 3], [0, 1, 2]))
                core_next = torch.transpose(core_next, 1, 2)
            core_finish = torch.tensordot(core_base, core_next, dims=([5], [0]))
            F = core_finish[0, 0, :, :, :, 0, :, 0]
            """     
                               1
                     ___________|____________
                    |      ____________      |
                F = |     |     |2     |     |
                    |     |---0    3---|     |
                    |_____|            |_____| 
            """
        return F

    def one_qubit_gate(self, u, n):
        core = self.tt_cores[n]
        up_core = torch.tensordot(u, core, dims=([1], [1]))
        up_core = torch.transpose(up_core, 0, 1)
        down_core = torch.tensordot(up_core, torch.conj(u), dims=([2], [1]))
        down_core = torch.transpose(down_core, 2, 3)
        self.tt_cores[n] = down_core

    def two_qubit_gate(self, u, n, max_rank=None, ort=False):
        self.mpo_to_mps()
        if ort:
            self.sequence_qr(n)
        self.mpo_from_mps()
        u = torch.reshape(u, [2, 2, 2, 2])
        phi = torch.tensordot(self.tt_cores[n], self.tt_cores[n + 1], dims=([3], [0]))
        phi = torch.tensordot(u, phi, dims=([2, 3], [1, 3]))
        phi = torch.tensordot(phi, torch.conj(u), dims=([3, 4], [2, 3]))
        phi = torch.transpose(phi, 0, 2)
        phi = torch.transpose(phi, 1, 2)
        phi = torch.transpose(phi, 2, 4)
        phi = torch.transpose(phi, 3, 5)
        phi = torch.transpose(phi, 3, 4)
        unfolding = torch.reshape(phi, (self.r[n] * self.phys_ind_i[n] * self.phys_ind_j[n],
                                  self.phys_ind_i[n + 1] * self.phys_ind_j[n + 1] * self.r[n + 2]))
        compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
        self.r[n + 1] = compressive_left.size()[1]
        self.tt_cores[n] = torch.reshape(compressive_left, [self.r[n], self.phys_ind_i[n], self.phys_ind_j[n],
                                                            self.r[n + 1]])
        self.tt_cores[n + 1] = torch.reshape(compressive_right, [self.r[n + 1], self.phys_ind_i[n + 1],
                                                                 self.phys_ind_j[n + 1], self.r[n + 2]])
        self.normalization(n)

    def get_trace(self):
        matrix_list = [torch.einsum('ijjk', self.tt_cores[i]) for i in range(self.N)]
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0].item()

    def normalization(self, n):
        self.tt_cores[n] = self.tt_cores[n] / self.get_trace()

    def tt_decomposition(self, full_tensor, max_rank=None):
        self.N = len(full_tensor.size()) // 2
        self.phys_ind_i = list(full_tensor.size())[0:self.N]
        self.phys_ind_j = list(full_tensor.size())[self.N:]
        new_index = []
        for i in range(self.N):
            new_index.append(self.phys_ind_i[i])
            new_index.append(1)
        new_index += self.phys_ind_j
        full_tensor = torch.reshape(full_tensor, tuple(new_index))
        for i in range(self.N):
            full_tensor = torch.transpose(full_tensor, 2 * i + 1, i + 2 * self.N)
        aux_index = []
        for i in range(self.N):
            aux_index.append(self.phys_ind_i[i] * self.phys_ind_j[i])
        full_tensor = torch.reshape(full_tensor, tuple(aux_index))
        super().tt_decomposition(full_tensor, max_rank=max_rank)
        self.mpo_from_mps()

    def return_full_tensor(self):
        full_tensor = self.tt_cores[0]
        for i in range(1, len(self.tt_cores), 1):
            full_tensor = torch.tensordot(full_tensor, self.tt_cores[i], dims=([-1], [0]))
        full_tensor = torch.reshape(full_tensor, list(full_tensor.size())[1:-1])
        full_tensor = torch.reshape(full_tensor, list(full_tensor.size()) + [1] * self.N)
        for i in range(self.N):
            full_tensor = torch.transpose(full_tensor, 2 * i + 1, i + 2 * self.N)
        full_tensor = torch.reshape(full_tensor, tuple(self.phys_ind_i + self.phys_ind_j))
        return full_tensor
        
    def get_element(self, list_of_index):
        list_of_index_i = list_of_index[0:self.N]
        list_of_index_j = list_of_index[self.N:]
        matrix_list = [self.tt_cores[i][:, list_of_index_i[i], list_of_index_j[i], :] for i in range(self.N)]
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0].item()
    
    def pure_state(self, mps: MPS):
        """
        If given MPS is phi, makes MPO equal to |phi><phi|.
        """
        self.N = mps.N
        self.r = []
        for i in range(self.N + 1):
            self.r.append(mps.r[i] * mps.r[i])
        
        self.phys_ind = []
        self.phys_ind_i = []
        self.phys_ind_j = []
        for i in range(self.N):
            self.phys_ind.append(mps.phys_ind[i] * mps.phys_ind[i])
            self.phys_ind_i.append(mps.phys_ind[i])
            self.phys_ind_j.append(mps.phys_ind[i])
        
        self.tt_cores = []
        for i in range(self.N):
            core1 = torch.conj(mps.tt_cores[i])
            core2 = mps.tt_cores[i]
            self.tt_cores.append(torch.reshape(torch.kron(core1, core2),
                                               (self.r[i], self.phys_ind_i[i], self.phys_ind_j[i], self.r[i + 1])))

    def get_product_trace(self, B):
        """
        Trace of product self * B
        """
        assert self.N == B.N
        matrix_list = []
        for i in range(self.N):
            core1 = self.tt_cores[i]
            core2 = B.tt_cores[i]
            kernel = torch.einsum('aijb,cjid->acbd', core1, core2)
            matrix_list.append(torch.reshape(kernel, (self.r[i] * B.r[i], self.r[i + 1] * B.r[i + 1])))
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0]

    def random_full(self, N: int, R: int):
        """
        Make kernels with random values, ranks equal to R
        """
        
        self.N = N
        
        self.r = [1]
        for i in range(N - 1):
            self.r.append(R)
        self.r.append(1)
        self.phys_ind_i = [2 for i in range(N)]
        self.phys_ind_j = [2 for i in range(N)]
        
        self.tt_cores = []
        for i in range(self.N):
            self.tt_cores.append(2 * torch.rand((self.r[i], self.phys_ind_i[i], self.phys_ind_j[i], self.r[i + 1]),
                                 dtype=self.info.data_type, device=self.info.device) - 1)

    def transpose(self):
        mpo = MPO(self.info)
        mpo.N = self.N
        mpo.r = copy.deepcopy(self.r)
        mpo.phys_ind_i = copy.deepcopy(self.phys_ind_j)
        mpo.phys_ind_j = copy.deepcopy(self.phys_ind_i)
        mpo.tt_cores = copy.deepcopy(self.tt_cores)
        for i in range(mpo.N):
            mpo.tt_cores[i] = torch.transpose(mpo.tt_cores[i], 1, 2)
        return mpo

    def star(self):
        """
            transpose + conj
        """
        mpo = self.transpose()
        for i in range(mpo.N):
            mpo.tt_cores[i] = torch.conj(mpo.tt_cores[i])
        return mpo
    
    def get_product(self, B):
        """
        return MPO: result of product self * B
        """
        assert self.phys_ind_j == B.phys_ind_i
        
        result = MPO(self.info)
        
        result.N = self.N
        result.phys_ind_i = self.phys_ind_i
        result.phys_ind_j = B.phys_ind_j
        
        result.r = [self.phys_ind_j[0]]
        for i in range(self.N - 1):
            result.r.append(self.r[i + 1] * B.r[i + 1] * self.phys_ind_j[i] * self.phys_ind_j[i + 1])
        result.r.append(self.phys_ind_j[-1])
        
        result.tt_cores = []
        
        for i in range(self.N):
            j_prev = 1
            if i != 0:
                j_prev = self.phys_ind_j[i - 1]
            j_next = 1
            if i + 1 < self.N:
                j_next = self.phys_ind_j[i + 1]
            
            core1 = torch.unsqueeze(self.tt_cores[i], 0)
            core1 = core1.expand(j_prev, core1.size(1), core1.size(2), core1.size(3), core1.size(4))
            core2 = torch.unsqueeze(B.tt_cores[i], 0)
            core2 = core2.expand(j_next, core2.size(1), core2.size(2), core2.size(3), core2.size(4))
            
            kernel = torch.einsum('abcde,fghij->bgadciejhf', core1, core2)
            for x in range(self.phys_ind_j[i]):
                for y in range(self.phys_ind_j[i]):
                    if x != y:
                        kernel[:, :, :, x, :, :, :, :, y, :] = 0
            result.tt_cores.append(torch.reshape(kernel, (result.r[i], result.phys_ind_i[i], result.phys_ind_j[i], result.r[i + 1])))
    
        result.r[0] = 1
        result.r[-1] = 1
        result.tt_cores[0] = torch.sum(result.tt_cores[0], dim=0)[None, :, :, :]
        result.tt_cores[-1] = torch.sum(result.tt_cores[-1], dim=3)[:, :, :, None]
    
        return result

    def get_full_matrix(self):
        full_tensor = self.return_full_tensor()
        n = len(full_tensor.size()) // 2
        rows_num = np.prod(np.array(full_tensor.size())[0:n])
        cols_num = np.prod(np.array(full_tensor.size())[n:])
        return torch.reshape(full_tensor, (rows_num, cols_num))

    def random_rho(self, N: int, R: int):
        """
        Generate random N-qubit MPO with max rank R
        """
        L = MPO(self.info)
        L.random_full(N, R)
        mpo = L.get_product(L.star())
        trace = torch.abs(mpo.get_trace()).item()
        C = np.exp(np.log(trace) / mpo.N)
        for i in range(mpo.N):
            mpo.tt_cores[i] /= C
        self.N = mpo.N
        self.tt_cores = mpo.tt_cores
        self.r = mpo.r
        self.phys_ind = mpo.phys_ind
        self.phys_ind_i = mpo.phys_ind_i
        self.phys_ind_j = mpo.phys_ind_j
