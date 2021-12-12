import numpy as np
import torch


class Gates(object):
    def __init__(self, info):
        self.info = info

    def I(self):
        return torch.tensor([[1, 0],
                             [0, 1]], dtype=self.info.data_type, device=self.info.device)

    def X(self):
        return torch.tensor([[0, 1],
                             [1, 0]], dtype=self.info.data_type, device=self.info.device)

    def Y(self):
        return torch.tensor([[0, -1j],
                             [1j, 0]], dtype=self.info.data_type, device=self.info.device)

    def Z(self):
        return torch.tensor([[1, 0],
                             [0, -1]], dtype=self.info.data_type, device=self.info.device)

    def H(self):
        return (1 / np.sqrt(2)) * torch.tensor([[1, 1],
                                                [1, -1]], dtype=self.info.data_type, device=self.info.device)

    def GPhase(self, alpha):
        return np.exp(1j * alpha) * self.I()

    def GPhase_der(self, alpha):
        return 1j * np.exp(1j * alpha) * self.I()

    def Rx(self, theta):
        return np.cos(theta / 2) * self.I() - 1j * np.sin(theta / 2) * self.X()

    def Rx_der(self, theta):
        return -0.5 * np.sin(theta / 2) * self.I() - 1j * 0.5 * np.cos(theta / 2) * self.X()

    def Ry(self, theta):
        return np.cos(theta / 2) * self.I() - 1j * np.sin(theta / 2) * self.Y()

    def Ry_der(self, theta):
        return -0.5 * np.sin(theta / 2) * self.I() - 1j * 0.5 * np.cos(theta / 2) * self.Y()

    def Rz(self, theta):
        return np.cos(theta / 2) * self.I() - 1j * np.sin(theta / 2) * self.Z()

    def Rz_der(self, theta):
        return -0.5 * np.sin(theta / 2) * self.I() - 1j * 0.5 * np.cos(theta / 2) * self.Z()

    def Rn(self, alpha, phi, theta):
        nx = np.sin(alpha) * np.cos(phi)
        ny = np.sin(alpha) * np.sin(phi)
        nz = np.cos(alpha)
        return np.cos(theta / 2) * self.I() - 1j * np.sin(theta / 2) * (nx * self.X() + ny * self.Y() + nz * self.Z())

    def Rn_random(self):
        alpha = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        return self.Rn(alpha, phi, theta)

    def CX(self):
        return torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0]], dtype=self.info.data_type, device=self.info.device)

    def CY(self):
        return torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, -1j],
                             [0, 0, 1j, 0]], dtype=self.info.data_type, device=self.info.device)

    def CZ(self):
        return torch.tensor([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, -1]], dtype=self.info.data_type, device=self.info.device)

    def SWAP(self):
        return torch.tensor([[1, 0, 0, 0],
                             [0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [0, 0, 0, 1]], dtype=self.info.data_type, device=self.info.device)
