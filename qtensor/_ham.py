import copy
import torch


class IsingHam(object):
    def __init__(self, n, gates, info):
        self.N = n
        self.gates = gates
        self.info = info

    def mean_ham(self, psi):
        """
        Model of Ising Hamiltonian:
        H = -\dfrac{1}{2} \sum_{i = 1}^{N - 1} \sigma^{(i)}_z \sigma^{(i + 1)}_z -
        \sum_{i = 1}^{N} \sigma^{(i)}_x
        """
        mean = 0
        for i in range(0, self.N - 1, 1):
            psi_tmp = copy.deepcopy(psi)
            psi_tmp.one_qubit_gate(-self.gates.Z(), i)
            psi_tmp.one_qubit_gate(self.gates.Z(), i + 1)
            mean += 0.5 * psi_tmp.scalar_product(psi).real
        for i in range(0, self.N, 1):
            psi_tmp = copy.deepcopy(psi)
            psi_tmp.one_qubit_gate(self.gates.X(), i)
            mean += (- psi_tmp.scalar_product(psi).real)
        return float(mean)

    def grad_mean_ham(self, psi, grad_psi):
        """
        Model of Ising Hamiltonian:
        H = -\dfrac{1}{2} \sum_{i = 1}^{N - 1} \sigma^{(i)}_z \sigma^{(i + 1)}_z -
        \sum_{i = 1}^{N} \sigma^{(i)}_x
        """
        mean = 0
        for i in range(0, self.N - 1, 1):
            psi_tmp = copy.deepcopy(grad_psi)
            psi_tmp.one_qubit_gate(-self.gates.Z(), i)
            psi_tmp.one_qubit_gate(self.gates.Z(), i + 1)
            mean += 0.5 * psi_tmp.scalar_product(psi).real
        for i in range(0, self.N, 1):
            psi_tmp = copy.deepcopy(grad_psi)
            psi_tmp.one_qubit_gate(self.gates.X(), i)
            mean += (- psi_tmp.scalar_product(psi).real)
        return 2 * float(mean).real

    def list_mean_ham(self, list_psi):
        list_of_means = []
        for psi in list_psi:
            list_of_means.append(self.mean_ham(psi))
        return list_of_means

    def list_grad_mean_ham(self, list_psi, list_grad_psi):
        list_of_grad_means = []
        for i, psi in enumerate(list_psi):
            list_of_grad_means.append(self.grad_mean_ham(psi, list_grad_psi[i]))
        return list_of_grad_means

    def get_z_z(self, n):
        u = torch.kron(self.gates.Z(), self.gates.Z())
        identity_gate = torch.tensor([[1, 0], [0, 1]], dtype=self.info.data_type, device=self.info.device)
        if n == 0:
            ham = u
        else:
            ham = identity_gate
            for i in range(1, n, 1):
                ham = torch.kron(ham, identity_gate)
            ham = torch.kron(ham, u)
        for i in range(n + 2, self.N, 1):
            ham = torch.kron(ham, identity_gate)
        return ham

    def get_x(self, n):
        u = self.gates.X()
        identity_gate = torch.tensor([[1, 0], [0, 1]], dtype=self.info.data_type, device=self.info.device)
        if n == 0:
            ham = u
        else:
            ham = identity_gate
            for i in range(1, n, 1):
                ham = torch.kron(ham, identity_gate)
            ham = torch.kron(ham, u)
        for i in range(n + 1, self.N, 1):
            ham = torch.kron(ham, identity_gate)
        return ham

    def get_ham_matrix(self):
        ham = torch.zeros((2 ** self.N, 2 ** self.N), device=self.info.device, dtype=self.info.data_type)
        for i in range(0, self.N - 1, 1):
            ham += (-0.5 * self.get_z_z(i))
        for i in range(self.N):
            ham += (-self.get_x(i))
        return ham

    def get_min_energy(self):
        eigenvalues = list(map(float, list(torch.linalg.eig(self.get_ham_matrix())[0])))
        min_energy = min(eigenvalues)
        return min_energy
