import copy


class IsingHam(object):
    def __init__(self, n, gates):
        self.N = n
        self.gates = gates

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

    def list_mean_ham(self, list_psi):
        list_of_means = []
        for psi in list_psi:
            list_of_means.append(self.mean_ham(psi))
        return list_of_means
