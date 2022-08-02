import torch
from qtensor import Info, MPSMax

N = 5

info = Info()
state = MPSMax(info)

state.gen_stoch_state(N, max_rank=3)

print(state.r)
print(state.mps_trunc.r)
print(state.fidelity(state.mps_trunc))

print(state.get_norm())
print(state.mps_trunc.get_norm())

print(state.phys_ind, state.mps_trunc.phys_ind)

n = 4
state.mps_trunc.sequence_qr_calc_norm(n)
print(state.mps_trunc.get_norm() ** 2, torch.tensordot(state.mps_trunc.tt_cores[n], torch.conj(state.mps_trunc.tt_cores[n]),
                                                  dims=([0, 1, 2], [0, 1, 2])))

print(state.get_tensor_F(0).size())
print(state.get_tensor_F(4).size())
print(state.get_tensor_F(3).size())

print(state.scalar_product(state.mps_trunc))
print(torch.tensordot(state.get_tensor_F(0), torch.conj(state.mps_trunc.tt_cores[0]), dims=([0, 1, 2], [0, 1, 2])))
print(torch.tensordot(state.get_tensor_F(1), torch.conj(state.mps_trunc.tt_cores[1]), dims=([0, 1, 2], [0, 1, 2])))
print(torch.tensordot(state.get_tensor_F(2), torch.conj(state.mps_trunc.tt_cores[2]), dims=([0, 1, 2], [0, 1, 2])))
print(torch.tensordot(state.get_tensor_F(3), torch.conj(state.mps_trunc.tt_cores[3]), dims=([0, 1, 2], [0, 1, 2])))
print(torch.tensordot(state.get_tensor_F(4), torch.conj(state.mps_trunc.tt_cores[4]), dims=([0, 1, 2], [0, 1, 2])))
print(torch.tensordot(state.return_full_vector(), torch.conj(state.mps_trunc.return_full_vector()), dims=([0], [0])))
