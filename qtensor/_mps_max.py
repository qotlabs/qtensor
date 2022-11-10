import torch
import copy
from qtensor import MPS


class MPSMax(MPS):
    def __init__(self, info):
        super().__init__(info)
        self.mps_trunc = None

    def gen_stoch_state(self, n, max_rank):
        d = 2 ** n
        random_vec = torch.randn(d, dtype=self.info.data_type, device=self.info.device)
        random_state = random_vec / torch.sqrt(torch.tensordot(random_vec, torch.conj(random_vec), dims=([0], [0])))
        random_state_tensor = torch.reshape(random_state, [2] * n)
        self.tt_decomposition(random_state_tensor)
        self.mps_trunc = MPSMax(self.info)
        self.mps_trunc.tt_decomposition(random_state_tensor, max_rank=max_rank)
        self.mps_trunc.normalization(0)

    def sequence_qr_calc_norm(self, n):
        if n == 0:
            self.sequence_qr_right(n - 1)
        elif n == (self.N - 1):
            self.sequence_qr_left(n)
        else:
            self.sequence_qr_left(n)
            self.sequence_qr_right(n - 1)

    def get_tensor_F(self, n):
        if n == 0:
            core_base = torch.tensordot(self.tt_cores[0], self.tt_cores[1], dims=([2], [0]))
            core_base = torch.tensordot(core_base, torch.conj(self.mps_trunc.tt_cores[1]), dims=([2], [1]))
            for i in range(2, self.N, 1):
                core_base = torch.tensordot(core_base, torch.conj(self.mps_trunc.tt_cores[i]), dims=([4], [0]))
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([2], [0]))
                core_base = torch.einsum('ijklmlo', core_base)
                core_base = torch.transpose(core_base, 3, 4)
                core_base = torch.transpose(core_base, 2, 3)
            F = core_base[:, :, 0, :, 0]
        elif n == (self.N - 1):
            core_base = torch.tensordot(self.tt_cores[0], torch.conj(self.mps_trunc.tt_cores[0]), dims=([1], [1]))
            for i in range(1, self.N - 1, 1):
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([1], [0]))
                core_base = torch.tensordot(core_base, torch.conj(self.mps_trunc.tt_cores[i]), dims=([2], [0]))
                core_base = torch.einsum('ijklkn', core_base)
                core_base = torch.transpose(core_base, 1, 2)
            core_base = torch.tensordot(core_base, self.tt_cores[self.N - 1], dims=([1], [0]))
            F = core_base[0, 0, :, :, :]
        else:
            core_base = torch.tensordot(self.tt_cores[0], torch.conj(self.mps_trunc.tt_cores[0]), dims=([1], [1]))
            for i in range(1, n, 1):
                core_base = torch.tensordot(core_base, self.tt_cores[i], dims=([1], [0]))
                core_base = torch.tensordot(core_base, torch.conj(self.mps_trunc.tt_cores[i]), dims=([2], [0]))
                core_base = torch.einsum('ijklkn', core_base)
                core_base = torch.transpose(core_base, 1, 2)
            core_base = torch.tensordot(core_base, self.tt_cores[n], dims=([1], [0]))
            core_next = torch.tensordot(self.tt_cores[n + 1], torch.conj(self.mps_trunc.tt_cores[n + 1]),
                                        dims=([1], [1]))
            for i in range(n + 2, self.N, 1):
                core_next = torch.tensordot(core_next, self.tt_cores[i], dims=([1], [0]))
                core_next = torch.tensordot(core_next, torch.conj(self.mps_trunc.tt_cores[i]), dims=([2], [0]))
                core_next = torch.einsum('ijklkn', core_next)
                core_next = torch.transpose(core_next, 1, 2)
            core_finish = torch.tensordot(core_base, core_next, dims=([4], [0]))
            F = core_finish[0, 0, :, :, 0, :, 0]
        return F

    def get_fidelity(self):
        return self.fidelity(self.mps_trunc)

    def get_best_approximate(self, num_passages=100):
        print('0 iterations: fidelity = {}'.format(self.fidelity(self.mps_trunc)))
        for i in range(num_passages):
            for n in range(0, self.N, 1):
                self.mps_trunc.sequence_qr_calc_norm(n)
                F = self.get_tensor_F(n)
                f = torch.tensordot(F, torch.conj(F), dims=([0, 1, 2], [0, 1, 2]))
                self.mps_trunc.tt_cores[n] = copy.deepcopy(F / torch.sqrt(f))
                print(n, end=' ')
            print('{} iterations: fidelity = {}'.format(i + 1, self.fidelity(self.mps_trunc)))
