import os
import json
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from data.Dataloader import Dataloader
from visualization.plotting import plot_full_amplitude_np, plot_form_factor_np


tensorBasisNamesTau = {
    0: "$1 \otimes 1$",
    1: "$\gamma_{5} \otimes \gamma_{5}$",
    2: "$\gamma^{\mu} \otimes \gamma^{\mu}$",
    3: "$\gamma_{5} \gamma^{\mu} \otimes \gamma_{5} \gamma^{\mu}$",
    4: "$\\frac{1}{8} [\gamma^{\mu}, \gamma^{\\nu}] \otimes [\gamma^{\mu}, \gamma^{\\nu}]$",
}

tensorBasisNamesT = {
    0: "$S_1$",
    1: "$S_2$",
    2: "$A_1$",
    3: "$A_2$",
    4: "$A_3$",
}




def plotAmplitudes(dataloader):
    for base_idx in range(5):
        plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.f[:, :, base_idx], "f", tensorBasisNamesTau[base_idx])

    for base_idx in range(5):
        plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.F[:, :, base_idx], "F", tensorBasisNamesT[base_idx])


def plotFullSymAmplitude(dataloader_qx, dataloader_dqx):
    # Plot Full (Anti-) Symmetric Amplitude
    F_complete = 1/2 * (dataloader_qx.F_flavored - dataloader_dqx.F_flavored)
    plot_full_amplitude_np(dataloader_qx.X, dataloader_qx.Z, F_complete, tensorBasisNamesT, None)


def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/"  #sys.argv[1]


    tensorbase_type = "tau"   #sys.argv[2]

    amplitude_isospin = 0   #int(sys.argv[3])

    dq_1_type = "scalar"    #sys.argv[2]
    dq_2_type = "scalar"    #sys.argv[3]

    Z_range = 1
    X_range_lower = 0


    # Construct directory string fitting specs
    data_path = data_base_path + f"/BASE-{tensorbase_type}_I-{amplitude_isospin}_DQ-{dq_1_type}-{dq_2_type}/"

    # Load data files
    dataloader_qx = Dataloader(data_base_path, "tau", amplitude_isospin, "quark_exchange", dq_1_type, dq_2_type, Z_range, X_range_lower)
    dataloader_dqx = Dataloader(data_base_path, "tau", amplitude_isospin, "diquark_exchange", dq_1_type, dq_2_type, Z_range, X_range_lower)

    plotAmplitudes(dataloader_qx)
    plotFullSymAmplitude(dataloader_qx, dataloader_dqx)





    # Legacy Prostprocessing
    # Note: qx run_13, dqx run_9 perfectly work for (anti) symmetric result
    #plot_dirac_space_dressing_functions(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", "quark_exchange", savefig=True)
    #plot_dirac_space_dressing_functions(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", "diquark_exchange", savefig=True)

    #full_amplitude_symmetric_basis(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", 0, dq_1_type, dq_2_type, Z_range, X_range_lower, savefig=True)

    # Plot tau basis result in symmetric basis
    #plot_dirac_space_dressing_functions_tau_to_T(data_path, "quark_exchange")


    #for base_idx, pd_ff in enumerate(dataloader_qx.pd_dirac_space_form_factors_list):
    #    plot_form_factor(pd_ff, tensorBasisNamesT[base_idx], base_idx, data_path + f"/{dataloader_qx.process_type}_F{base_idx + 1}", False)






if __name__ == "__main__":
    main()
    exit()