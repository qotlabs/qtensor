from mpi4py import MPI
import numpy as np

from qtensor import VQEOptimizer, VQECircuitCX
from qtensor import Info, Gates, MPS, MPSGrad, IsingHamAnalytical
from qtensor import Loader

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

D = 5
number_of_iterations = 50
nums_of_qubits_on_process = 1

info = Info()
gates = Gates(info)

print('rank = ', rank)

if rank != 0:
    list_N_part = [n for n in range(rank * nums_of_qubits_on_process + 0,
                                    (rank + 1) * nums_of_qubits_on_process + 0, 1)]
    print('n_qubits = ', list_N_part)
    result_part = []
    for n in list_N_part:
        ham = IsingHamAnalytical(n, gates, info)
        vqe_circuit = VQECircuitCX(gates)
        vqe_optimizer = VQEOptimizer(MPS, MPSGrad, info, n, D, ham, gates, vqe_circuit, max_rank=None, ort=False)
        list_of_parameters = 2 * np.pi * np.random.rand(4 * n * D)
        result_part += list(vqe_optimizer.optimize(list_of_parameters, number_of_iterations))
    result_part = np.array(result_part, dtype=np.float64)
    print('len(result_part) = ', len(result_part))
    comm.Send([result_part, len(result_part), MPI.DOUBLE], dest=0, tag=0)
else:
    result = np.empty((numprocs - 1) * nums_of_qubits_on_process * number_of_iterations, dtype=np.float64)
    for k in range(1, numprocs, 1):
        left = (k - 1) * nums_of_qubits_on_process * number_of_iterations
        right = k * nums_of_qubits_on_process * number_of_iterations
        print('buf = ', left, right, len(result[left:right]), len(result[0:10]))
        comm.Recv([result[left:right], len(result[left:right]), MPI.DOUBLE], source=k, tag=0)
    loader = Loader('Results.xlsx')
    sheet_name = 'VQE_Ising_Analytical'
    loader.write_data(sheet_name, 'D', 1 + 750, len(result) + 750, result)
