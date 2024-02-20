import copy

from typing import List

import numpy as np


class BasisTauToRho:


    @staticmethod
    def _a(X: np.ndarray):
        return 1 + np.sqrt(1 + X)
    
    @staticmethod
    def _b(X: np.ndarray, Z: np.ndarray):
        return 1 + np.sqrt(1 + X) + (X * (1 - Z)) / 2.0
    
    @staticmethod
    def _R(X_array: np.ndarray, Z_array: np.ndarray):
        # TODO for Z = 1, we get a = b and thus a divide by 0

        R = np.zeros((5, 5, len(X_array), len(Z_array)))

        for X_idx, X in enumerate(X_array):
            for Z_idx, Z in enumerate(Z_array):
                a = BasisTauToRho._a(X)
                b = BasisTauToRho._b(X, Z)

                R[0, 0, X_idx, Z_idx] = np.power(b, 2)/np.power(a, 2)
                R[0, 3, X_idx, Z_idx] = -(2.0 * b)/np.power(a, 2)
                R[0, 4, X_idx, Z_idx] = -1.0/np.power(a, 2)

                R[1, 2, X_idx, Z_idx] = -1.0

                R[2, 0, X_idx, Z_idx] = 2.0 * np.power(a, 2) - a - 3.0 * b + np.power(b, 2)/np.power(a, 2)
                R[2, 1, X_idx, Z_idx] = a - b
                R[2, 2, X_idx, Z_idx] = 1.0
                R[2, 3, X_idx, Z_idx] = 2.0 * (2.0 - b/np.power(a, 2))
                R[2, 4, X_idx, Z_idx] = -1.0/np.power(a, 2)

                R[3, 0, X_idx, Z_idx] = 1.0 - a + b - np.power(b, 2)/np.power(a, 2)
                R[3, 1, X_idx, Z_idx] = -(2.0 * np.power(a, 2) - 3.0 * a - b + 1.0)
                R[3, 2, X_idx, Z_idx] = -(2.0 * np.power(a, 2) - 3.0 * a - b) / (a - b)
                R[3, 3, X_idx, Z_idx] = 2.0 * b / np.power(a, 2)
                R[3, 4, X_idx, Z_idx] = 1.0/np.power(a, 2) - 2.0/(a - b)

                R[4, 0, X_idx, Z_idx] = 1.0 - 2.0 * a + 2.0 * b - np.power(b, 2)/np.power(a, 2)
                R[4, 1, X_idx, Z_idx] = -1.0
                R[4, 2, X_idx, Z_idx] = (2.0 * np.power(a, 2) - 3.0 * a - b) / (a - b)
                R[4, 3, X_idx, Z_idx] = -2.0 * (2.0 - b/np.power(a, 2))
                R[4, 4, X_idx, Z_idx] = 1.0/np.power(a, 2) + 2.0/(a - b)

        return R

    @staticmethod
    def build_rho_base_dressing_functions_from_tau(dataloader):
        R = BasisTauToRho._R(dataloader.X_unique, dataloader.Z_unique)

        V = np.zeros_like(dataloader.f)
        for X_idx in range(len(dataloader.X_unique)):
            for Z_idx in range(len(dataloader.Z_unique)):
                V[:, X_idx, Z_idx] = np.matmul(np.transpose(R[:, :, X_idx, Z_idx]), dataloader.f[:, X_idx, Z_idx])

        return dataloader.X_unique, dataloader.Z_unique, V