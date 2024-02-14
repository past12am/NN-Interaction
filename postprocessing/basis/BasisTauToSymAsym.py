import copy

from typing import List

import pandas as pd
import numpy as np


class BasisTauToSymAsym:

    @staticmethod
    def build_alternate_basis_pds(pd_ff_tau_basis_list: List):
        exit(1)
        pd_ff_T_basis_list = copy.deepcopy(pd_ff_tau_basis_list)

        for pd_ff_T in pd_ff_T_basis_list:
            #pd_ff_T.drop("h", axis=1, inplace=True)
            pd_ff_T["f"] = 0

        #base_conv_prefs = [[3/8, 1/4, 1/8, 1/4, 0], [3/8, -(1/4), 1/8, -(1/4), 0], [0, 1/2, 0, -(1/2), 1/2], [0, -(1/2), 0, 1/2, 1/2], [-6, 0, 6, 0, 0]]
        base_conv_prefs = [[3/8, 3/8, 0, 0, -6], [1/4, -(1/4), 1/2, -(1/2), 0], [1/8, 1/8, 0, 0, 6], [1/4, -(1/4), -(1/2), 1/2, 0], [0, 0, 1/2, 1/2, 0]]

        for i in range(len(pd_ff_T_basis_list)):
            for j in range(len(pd_ff_tau_basis_list)):
                pd_ff_T_basis_list[i]["f"] += base_conv_prefs[i][j] * pd_ff_tau_basis_list[j]["f"]

        
        #pd_ff_T_basis_list[0]["f"] = pd_ff_tau_basis_list[0]["f"] + pd_ff_tau_basis_list[1]["f"] - 1/3 * pd_ff_tau_basis_list[4]["f"]
        #pd_ff_T_basis_list[1]["f"] = pd_ff_tau_basis_list[0]["f"] - pd_ff_tau_basis_list[1]["f"] + 1/2 * (pd_ff_tau_basis_list[2]["f"] - pd_ff_tau_basis_list[3]["f"])
        #pd_ff_T_basis_list[2]["f"] = pd_ff_tau_basis_list[0]["f"] + pd_ff_tau_basis_list[1]["f"] + pd_ff_tau_basis_list[4]["f"]
        #pd_ff_T_basis_list[3]["f"] = pd_ff_tau_basis_list[0]["f"] - pd_ff_tau_basis_list[1]["f"] - 1/2 * (pd_ff_tau_basis_list[2]["f"] - pd_ff_tau_basis_list[3]["f"])
        #pd_ff_T_basis_list[4]["f"] = pd_ff_tau_basis_list[2]["f"] + pd_ff_tau_basis_list[3]["f"]


        return pd_ff_T_basis_list
    

    @staticmethod
    def build_alternate_basis_numpy(f: np.ndarray):
        base_conv_prefs = np.array([[3/8, 3/8, 0, 0, -6], [1/4, -(1/4), 1/2, -(1/2), 0], [1/8, 1/8, 0, 0, 6], [1/4, -(1/4), -(1/2), 1/2, 0], [0, 0, 1/2, 1/2, 0]])

        f_transformed = np.zeros_like(f)
        for X_idx in range(f.shape[0]):
            for Z_idx in range(f.shape[1]):
                f_transformed[X_idx, Z_idx, :] = np.matmul(base_conv_prefs, f[X_idx, Z_idx, :])

        return f_transformed


    @staticmethod
    def build_alternate_basis_inverse_numpy(F: np.ndarray):
        base_conv_prefs = np.array([[1, 1, 1, 1, 0], [1, -1, 1, -1, 0], [0, 1/2, 0, -(1/2), 1], [0, -(1/2), 0, 1/2, 1], [-(1/24), 0, 1/8, 0, 0]])

        f_transformed = np.zeros_like(F)
        for X_idx in range(F.shape[0]):
            for Z_idx in range(F.shape[1]):
                f_transformed[X_idx, Z_idx, :] = np.matmul(base_conv_prefs, F[X_idx, Z_idx, :])

        return f_transformed