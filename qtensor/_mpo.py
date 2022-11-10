import torch
from qtensor import MPS


class MPO(MPS):
    def __init__(self, info):
        super().__init__(info)
        self.phys_ind_i = []
        self.phys_ind_j = []

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
        return element[0][0]

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
