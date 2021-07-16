import numpy as np
import torch
from qtensor import Info

# x = torch.randn(5, 3, dtype=torch.complex128)
# print(x)
# print(torch.cuda.device_count())
# print(torch.cuda.is_available())
# gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(gpu)
# A = torch.tensor([1 + 1j, 2, 3])
# B = torch.tensor([2, 2 - 2j, 2])
# A1 = np.array([1 + 1j, 2, 3])
# B1 = np.array([2, 2 - 2j, 2])
# print(np.dot(np.conj(A1), B1))
# print(A.vdot(B))
# print(A.size())
A = torch.rand(2, 5, 1)
B = torch.rand(7, 5, 3)
print(A.size())
print(B.size())
print(torch.tensordot(A, B, dims=([1], [1])).size())
# torch.tensordot(A, B, dims=([1, 0], [0, 1]))
A = torch.rand(7, 5)
A = torch.tensordot(torch.rand(7, 1), torch.rand(1, 5), dims=([1], [0]))
print(A)
print(torch.matrix_rank(A))
U, S, V = torch.linalg.svd(A, full_matrices=False)
print(U)
print(S)
print(V)
A_reshape = torch.reshape(A, (-1,))
print(A_reshape.size())

print("""
           -----------                  --------       -------
    m     |           |      k    m    |        |  l  |       |    k
    ------|           |------- => -----|        |-----|       |----- 
          |           |                |        |     |       |
           -----------                  --------       -------
             |     |                       |              | 
             |i    |j                      |i             |j
""")

info = Info()
A = torch.randn((3, 3), dtype=info.data_type, device=info.device)
print(A)

A = torch.tensor([[2, 1],
                  [3, 4]], dtype=info.data_type, device=info.device)
A = A[0:1, :]
print(A)
