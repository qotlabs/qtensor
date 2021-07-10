import numpy as np
import torch


class MPS(object):
    def __init__(self, info):
        self.tt_cores = []
        self.info = info
        self.N = None
        self.r = []
        self.phys_ind = []
        self.full_tensor = None

    def all_zeros_state(self, n):
        self.r = [1]
        self.phys_ind = []
        for i in range(n):
            self.tt_cores.append(torch.reshape(torch.tensor([1, 0], dtype=self.info.data_type, device=self.info.device),
                                               (1, 2, 1)))
            self.r.append(1)
            self.phys_ind.append(2)
        self.N = n

    def full_tensor_calculate(self):
        full_tensor = self.tt_cores[0]
        for i in range(1, len(self.tt_cores), 1):
            full_tensor = torch.tensordot(full_tensor, self.tt_cores[i], dims=([-1], [0]))
        self.full_tensor = full_tensor.reshape(self.phys_ind)

    def tt_decomposition(self, full_tensor):
        self.phys_ind = list(full_tensor.size())
        self.N = len(full_tensor.size())
        print(self.phys_ind)
        full_tensor = torch.reshape(full_tensor, [1] + self.phys_ind + [1])
        self.tt_cores = []
        print(self.N)
        # 0 tt_core
        unfolding = torch.reshape(full_tensor, (self.phys_ind[0], np.prod(np.array(self.phys_ind[1:]))))
        compressive_left, compressive_right = MPS.tt_svd(unfolding, self.info)
        self.r.append(compressive_left.size()[1])
        self.tt_cores.append(torch.reshape(compressive_left, (1, self.phys_ind[0], self.r[0])))
        compressive_right = torch.reshape(compressive_right, [self.r[0]] + self.phys_ind[1:])
        # 1, ..., N - 2 tt_cores
        for i in range(1, self.N - 1, 1):
            unfolding = torch.reshape(compressive_right, (self.phys_ind[i] * self.r[i - 1],
                                                          int(np.prod(np.array(self.phys_ind[i + 1:])))))
            compressive_left, compressive_right = MPS.tt_svd(unfolding, self.info)
            self.r.append(compressive_left.size()[1])
            self.tt_cores.append(torch.reshape(compressive_left, (self.r[i - 1], self.phys_ind[i], self.r[i])))
            compressive_right = torch.reshape(compressive_right, [self.r[i]] + self.phys_ind[i + 1:])
        # N - 1 tt_core
        self.r.append(1)
        print(compressive_right.size())
        self.tt_cores.append(torch.reshape(compressive_right, (self.r[self.N - 2], self.phys_ind[self.N - 1], 1)))
        self.r = [1] + self.r

    @staticmethod
    def tt_svd(unfolding, info):
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = torch.tensor(s, dtype=info.data_type, device=info.device)
        compressive_core_left = torch.tensordot(u, torch.diag(s), dims=([1], [0]))
        compressive_core_right = v
        return compressive_core_left, compressive_core_right
