import numpy as np
import torch


class MPS(object):
    def __init__(self, info):
        self.tt_cores = []
        self.info = info
        self.N = None
        self.r = []
        self.phys_ind = []

    def all_zeros_state(self, n):
        self.tt_cores = []
        self.r = [1]
        self.phys_ind = []
        for i in range(n):
            self.tt_cores.append(torch.reshape(torch.tensor([1, 0], dtype=self.info.data_type, device=self.info.device),
                                               (1, 2, 1)))
            self.r.append(1)
            self.phys_ind.append(2)
        self.N = n

    def one_qubit_gate(self, u, n):
        core = self.tt_cores[n]
        self.tt_cores[n] = torch.transpose(torch.tensordot(u, core, dims=([1], [1])), 0, 1)

    def two_qubit_gate(self, u, n, max_rank=None, ort=False):
        if ort:
            self.sequence_qr(n)
        u = torch.reshape(u, [2, 2, 2, 2])
        phi = torch.tensordot(self.tt_cores[n], self.tt_cores[n + 1], dims=([2], [0]))
        phi = torch.tensordot(u, phi, dims=([2, 3], [1, 2]))
        phi = torch.transpose(phi, 0, 2)
        phi = torch.transpose(phi, 1, 2)
        unfolding = phi.reshape([self.r[n] * self.phys_ind[n], self.r[n + 2] * self.phys_ind[n + 1]])
        compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
        self.r[n + 1] = compressive_left.size()[1]
        self.tt_cores[n] = torch.reshape(compressive_left, [self.r[n], self.phys_ind[n], self.r[n + 1]])
        self.tt_cores[n + 1] = torch.reshape(compressive_right, [self.r[n + 1], self.phys_ind[n + 1], self.r[n + 2]])
        self.normalization(n)

    def normalization(self, n):
        self.tt_cores[n] = self.tt_cores[n] / self.get_norm()

    def return_full_tensor(self):
        full_tensor = self.tt_cores[0]
        for i in range(1, len(self.tt_cores), 1):
            full_tensor = torch.tensordot(full_tensor, self.tt_cores[i], dims=([-1], [0]))
        full_tensor = full_tensor.reshape(self.phys_ind)
        return full_tensor

    def return_full_vector(self):
        full_tensor = self.return_full_tensor()
        return torch.reshape(full_tensor, (-1, ))

    def tt_decomposition(self, full_tensor, max_rank=None):
        self.phys_ind = list(full_tensor.size())
        self.N = len(full_tensor.size())
        full_tensor = torch.reshape(full_tensor, [1] + self.phys_ind + [1])
        self.tt_cores = []
        # 0 tt_core
        unfolding = torch.reshape(full_tensor, (self.phys_ind[0], np.prod(np.array(self.phys_ind[1:]))))
        compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
        self.r.append(compressive_left.size()[1])
        self.tt_cores.append(torch.reshape(compressive_left, (1, self.phys_ind[0], self.r[0])))
        compressive_right = torch.reshape(compressive_right, [self.r[0]] + self.phys_ind[1:])
        # 1, ..., N - 2 tt_cores
        for i in range(1, self.N - 1, 1):
            unfolding = torch.reshape(compressive_right, (self.phys_ind[i] * self.r[i - 1],
                                                          int(np.prod(np.array(self.phys_ind[i + 1:])))))
            compressive_left, compressive_right = MPS.tt_svd(unfolding, max_rank)
            self.r.append(compressive_left.size()[1])
            self.tt_cores.append(torch.reshape(compressive_left, (self.r[i - 1], self.phys_ind[i], self.r[i])))
            compressive_right = torch.reshape(compressive_right, [self.r[i]] + self.phys_ind[i + 1:])
        # N - 1 tt_core
        self.r.append(1)
        self.tt_cores.append(torch.reshape(compressive_right, (self.r[self.N - 2], self.phys_ind[self.N - 1], 1)))
        self.r = [1] + self.r

    @staticmethod
    def tt_svd(unfolding, max_rank=None):
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        if max_rank is not None:
            u = u[:, 0:max_rank]
            s = s[0:max_rank]
            v = v[0:max_rank, :]
        compressive_left = torch.tensordot(u, torch.diag(s), dims=([1], [0]))
        compressive_right = v
        return compressive_left, compressive_right

    # Mistake!
    @staticmethod
    def tt_qr(unfolding, rank=None):
        print('Unfolding_size QR = ', unfolding.size())
        q, r = torch.linalg.qr(unfolding)
        if rank is not None:
            q = q[:, 0:rank]
            r = r[0:rank, :]
        compressive_left = q
        compressive_right = r
        # print('Test QR: ', np.sum(np.abs(np.array(unfolding - np.dot(np.array(q, dtype=complex),
        #                                                              np.array(r, dtype=complex)))) ** 2))
        return compressive_left, compressive_right

    # Mistake!
    @staticmethod
    def tt_lq(unfolding, rank=None):
        print('Unfolding_size LQ = ', unfolding.size())
        q, r = torch.linalg.qr(torch.transpose(torch.conj(unfolding), 0, 1))
        l = torch.transpose(torch.conj(r), 0, 1)
        q = torch.transpose(torch.conj(q), 0, 1)
        if rank is not None:
            l = l[:, 0:rank]
            q = q[0:rank, :]
        compressive_left = l
        compressive_right = q
        # print('Test LQ: ', np.sum(np.abs(np.array(unfolding - np.dot(np.array(l, dtype=complex),
        #                                                              np.array(q, dtype=complex)))) ** 2))
        return compressive_left, compressive_right

    @staticmethod
    def tt_svd_left(unfolding, rank=None):
        u, s, v = torch.linalg.svd(unfolding, full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        q = u
        r = torch.tensordot(torch.diag(s), v, dims=([1], [0]))
        if rank is not None:
            q = q[:, 0:rank]
            r = r[0:rank, :]
        compressive_left = q
        compressive_right = r
        return compressive_left, compressive_right

    @staticmethod
    def tt_svd_right(unfolding, rank=None):
        u, s, v = torch.linalg.svd(torch.transpose(torch.conj(unfolding), 0, 1), full_matrices=False)
        s = s * (1.0 + 0.0 * 1j)
        q = u
        r = torch.tensordot(torch.diag(s), v, dims=([1], [0]))
        l = torch.transpose(torch.conj(r), 0, 1)
        q = torch.transpose(torch.conj(q), 0, 1)
        if rank is not None:
            l = l[:, 0:rank]
            q = q[0:rank, :]
        compressive_left = l
        compressive_right = q
        return compressive_left, compressive_right

    def sequence_qr_left(self, n):
        for i in range(0, n, 1):
            phi = torch.tensordot(self.tt_cores[i], self.tt_cores[i + 1], dims=([2], [0]))
            unfolding = torch.reshape(phi, (self.r[i] * self.phys_ind[i], self.phys_ind[i + 1] * self.r[i + 2]))
            # compressive_left, compressive_right = MPS.tt_qr(unfolding, rank=self.r[i + 1])
            compressive_left, compressive_right = MPS.tt_svd_left(unfolding, rank=self.r[i + 1])
            self.tt_cores[i] = torch.reshape(compressive_left, (self.r[i], self.phys_ind[i], self.r[i + 1]))
            self.tt_cores[i + 1] = torch.reshape(compressive_right, (self.r[i + 1], self.phys_ind[i + 1],
                                                                     self.r[i + 2]))

    def sequence_qr_right(self, n):
        for i in range(self.N - 1, n + 1, -1):
            phi = torch.tensordot(self.tt_cores[i - 1], self.tt_cores[i], dims=([2], [0]))
            unfolding = torch.reshape(phi, (self.r[i - 1] * self.phys_ind[i - 1], self.phys_ind[i] * self.r[i + 1]))
            # compressive_left, compressive_right = MPS.tt_lq(unfolding, rank=self.r[i])
            compressive_left, compressive_right = MPS.tt_svd_right(unfolding, rank=self.r[i])
            self.tt_cores[i - 1] = torch.reshape(compressive_left, (self.r[i - 1], self.phys_ind[i - 1], self.r[i]))
            self.tt_cores[i] = torch.reshape(compressive_right, (self.r[i], self.phys_ind[i], self.r[i + 1]))

    def sequence_qr(self, n):
        if n == 0:
            self.sequence_qr_right(n)
            pass
        elif n == (self.N - 2):
            self.sequence_qr_left(n)
        else:
            self.sequence_qr_left(n)
            self.sequence_qr_right(n)

    def check_ort_left(self, n):
        psi = self.tt_cores[n]
        psi_0 = np.array(psi[:, 0, :])
        psi_1 = np.array(psi[:, 1, :])
        matrix = np.dot(psi_0.T.conjugate(), psi_0) + np.dot(psi_1.T.conjugate(), psi_1)
        if np.sum(np.abs(matrix - np.eye(self.r[n + 1], dtype=complex)) ** 2) < 10 ** (-10):
            return 1
        else:
            return 0

    def check_ort_right(self, n):
        psi = self.tt_cores[n]
        psi_0 = np.array(psi[:, 0, :])
        psi_1 = np.array(psi[:, 1, :])
        matrix = np.dot(psi_0, psi_0.T.conjugate()) + np.dot(psi_1, psi_1.T.conjugate())
        if np.sum(np.abs(matrix - np.eye(self.r[n], dtype=complex)) ** 2) < 10 ** (-10):
            return 1
        else:
            return 0

    def get_norm(self):
        core_prev = torch.tensordot(self.tt_cores[0], torch.conj(self.tt_cores[0]), dims=([1], [1]))
        for i in range(1, len(self.tt_cores), 1):
            core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
            core_prev = torch.tensordot(core_prev, torch.conj(self.tt_cores[i]), dims=([2], [0]))
            core_prev = torch.einsum('ijklkn', core_prev)
            core_prev = torch.transpose(core_prev, 1, 2)
        norm_square = core_prev[0][0][0][0]
        norm = torch.abs(torch.sqrt(norm_square))
        return norm

    def scalar_product(self, phi):
        """
            Calculating <phi|psi>
        """
        if len(self.tt_cores) != len(phi.tt_cores):
            raise RuntimeError('Different size of tensors')
        else:
            core_prev = torch.tensordot(self.tt_cores[0], torch.conj(phi.tt_cores[0]), dims=([1], [1]))
            for i in range(1, len(self.tt_cores), 1):
                core_prev = torch.tensordot(core_prev, self.tt_cores[i], dims=([1], [0]))
                core_prev = torch.tensordot(core_prev, torch.conj(phi.tt_cores[i]), dims=([2], [0]))
                core_prev = torch.einsum('ijklkn', core_prev)
                core_prev = torch.transpose(core_prev, 1, 2)
            scalar_product = core_prev[0][0][0][0]
            return scalar_product

    def get_element(self, list_of_index):
        matrix_list = [self.tt_cores[i][:, index, :] for i, index in enumerate(list_of_index)]
        element = matrix_list[0]
        for matrix in matrix_list[1:]:
            element = torch.tensordot(element, matrix, dims=([1], [0]))
        return element[0][0]

    def fidelity(self, phi):
        """
            Calculating |<phi|psi>|^2
        """
        overlap = self.scalar_product(phi)
        fid = overlap * torch.conj(overlap)
        return float(fid)
