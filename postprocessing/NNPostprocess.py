import os
import json
import typing
import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from data.Dataloader import Dataloader

from FTHandler import *

from visualization.plotting import Plotter, PlotterFullAmplitude

plt.rcParams['text.usetex'] = True


tensorBasisNamesTau = {
    0: "$1 \\otimes 1$",
    1: "$\\gamma_{5} \\otimes \\gamma_{5}$",
    2: "$\\gamma^{\\mu} \\otimes \\gamma^{\\mu}$",
    3: "$\\gamma_{5} \\gamma^{\\mu} \\otimes \\gamma_{5} \\gamma^{\\mu}$",
    4: "$\\frac{1}{8} [\\gamma^{\\mu}, \\gamma^{\\nu}] \\otimes [\\gamma^{\\mu}, \\gamma^{\\nu}]$",
}

tensorBasisNamesTauPrime = {
    0: "$\\tau'_0$",
    1: "$\\tau'_1$",
    2: "$\\tau'_2$",
    3: "$\\tau'_3$",
    4: "$\\tau'_4$",
}

tensorBasisNamesT = {
    0: "$S_1$",
    1: "$S_2$",
    2: "$A_1$",
    3: "$A_2$",
    4: "$A_3$",
}

tensorBasisNamesRho = {
    0: "$\\rho_1 = 1 \\otimes 1$",
    1: "$\\rho_2 = \\vec{\\sigma} \\otimes \\vec{\\sigma}$",
    2: "$\\rho_3 = \\frac{1}{4M^2} (\\vec{\\sigma} \\cdot \\vec{q}) \\otimes (\\vec{\\sigma} \\cdot \\vec{q})$",
    3: "$\\rho_4 = \\frac{i}{4M^2} (\\vec{\\sigma} \\otimes 1 + 1 \\otimes \\vec{\\sigma}) (\\vec{q} \\times \\vec{p})$",
    4: "$\\rho_5 = \\frac{1}{4M^2} \\vec{\\sigma} \\cdot (\\vec{q} \\times \\vec{p}) \\otimes \\vec{\\sigma} \\cdot (\\vec{q} \\times \\vec{p})$",
}



tensorBasisNamesDict = {
    "tau": tensorBasisNamesTau,
    "tau_prime": tensorBasisNamesTauPrime,
    "T": tensorBasisNamesT,
    "rho": tensorBasisNamesRho
}




def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/data/"
    output_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/"

    tensorbase_type = "T"

    dq_1_type = "scalar"
    dq_2_type = "scalar"

    Z_range = 0.99
    X_range_lower = 0

    M_nucleon = 0.94    # GeV


    # Load data files
    # TODO error happens for S1 and A1 when doing tau --> T, seems correct for T --> tau
    qx_process_type = "quark_exchange"
    dqx_process_type = "diquark_exchange"
    dataloader_qx = Dataloader(data_base_path, tensorbase_type, qx_process_type, dq_1_type, dq_2_type, Z_range, X_range_lower)
    dataloader_dqx = Dataloader(data_base_path, tensorbase_type, dqx_process_type, dq_1_type, dq_2_type, Z_range, X_range_lower)


    # Instantiate Plotter
    qx_run_path_name = dataloader_qx.latest_run_dir_process
    dqx_run_path_name = dataloader_dqx.latest_run_dir_process
    
    plotter_qx = Plotter(output_base_path, tensorBasisNamesDict, qx_process_type, qx_run_path_name, dataloader_qx.process_spec, True, include_loglog_plots=False)
    plotter_dqx = Plotter(output_base_path, tensorBasisNamesDict, dqx_process_type, dqx_run_path_name, dataloader_dqx.process_spec, True, include_loglog_plots=False)

    plotter_combined = PlotterFullAmplitude(output_base_path, dataloader_qx, dataloader_dqx, True)

    plotter_qx.show_plots = False
    plotter_dqx.show_plots = False
    plotter_combined.show_plots = False






    # Plot amplitudes
    plotter_qx.plotAmplitudes_h(dataloader_qx, "AmplitudeQuarkExchange_h", dataloader_qx.process_spec["projection_basis"], 0, process_abbrev="q")
    plotter_qx.plotAmplitudes(dataloader_qx, "AmplitudeQuarkExchange_f", "AmplitudeQuarkExchange_F", 0, process_abbrev="q")

    plotter_dqx.plotAmplitudes_h(dataloader_dqx, "AmplitudeDiquarkExchange_h", dataloader_dqx.process_spec["projection_basis"], 0, process_abbrev="dq")
    plotter_dqx.plotAmplitudes(dataloader_dqx, "AmplitudeDiquarkExchange_f", "AmplitudeDiquarkExchange_F", 0, process_abbrev="dq")

    plotter_combined.plotFullSymAmplitudeIsospin0(tensorBasisNamesT, "FullSymAmplitudeI0")
    plotter_combined.plotFullSymAmplitudeIsospin1(tensorBasisNamesT, "FullSymAmplitudeI1")


    # Perform FT
    # Select quark or diquark exchange  # TODO fix plot names/scale names
    ampHandler_rho_qx, qx__f_l_r, qx__r_grid = perform_FT_of_amplitudes(dataloader_qx, plotter_qx, tensorBasisNamesRho, "quark_exchange")
    ampHandler_rho_dqx, dqx__f_l_r, dqx__r_grid = perform_FT_of_amplitudes(dataloader_dqx, plotter_dqx, tensorBasisNamesRho, "diquark_exchange")
    

    ############################################## Position Space ##############################################
    r_grid = qx__r_grid

    #       Isoscalar
    V_l_r__I0 = qx__f_l_r

    #       Isovector
    V_l_r__I1 = 2 * qx__f_l_r - dqx__f_l_r

    #       Correction for dimensionless qtys
    V_l_r__I0[2, ...] = V_l_r__I0[2, ...] / (4.0 * np.square(M_nucleon))
    V_l_r__I0[3, ...] = V_l_r__I0[3, ...] / (4.0 * np.square(M_nucleon))
    V_l_r__I0[4, ...] = V_l_r__I0[4, ...] / (4.0 * np.power(M_nucleon, 4))

    V_l_r__I1[2, ...] = V_l_r__I1[2, ...] / (4.0 * np.square(M_nucleon))
    V_l_r__I1[3, ...] = V_l_r__I1[3, ...] / (4.0 * np.square(M_nucleon))
    V_l_r__I1[4, ...] = V_l_r__I1[4, ...] / (4.0 * np.power(M_nucleon, 4))


    #       Plot final results
    ylabels_I0 = ["V_{\\mathrm{C}}", "V_{\\mathrm{S}}", "V_{\\mathrm{T}}", "V_{\\mathrm{SO}}", "V_{\\mathrm{Q}}"]
    ylabels_I1 = ["W_{\\mathrm{C}}", "W_{\\mathrm{S}}", "W_{\\mathrm{T}}", "W_{\\mathrm{SO}}", "W_{\\mathrm{Q}}"]

    plotter_combined.plot_final_res(V_l_r__I0, r_grid, "r", "1 / GeV", ylabels_I0, tensorBasisNamesDict, "isoscalar", 2)
    plotter_combined.plot_final_res(V_l_r__I1, r_grid, "r", "1 / GeV", ylabels_I1, tensorBasisNamesDict, "isovector", 2)


    # Total
    #plotter_combined.plot_final_res(V_l_r__I0 + V_l_r__I1, r_grid, "r", "1 / GeV", [yl1 + " + " + yl2 for yl1, yl2 in zip(ylabels_I0, ylabels_I1)])




    ############################################## Momentum Space ##############################################
    #   Construct momentum space potentials
    q_grid = ampHandler_rho_qx.q
    qx__f_l_q = ampHandler_rho_qx.f_l_q
    dqx__f_l_q = ampHandler_rho_dqx.f_l_q

    #       Isoscalar
    V_l_q__I0 = qx__f_l_q

    #       Isovector
    V_l_q__I1 = 2 * qx__f_l_q - dqx__f_l_q

    #       Correction for dimensionless qtys
    V_l_q__I0[2, ...] = V_l_q__I0[2, ...] / (4.0 * np.square(M_nucleon))
    V_l_q__I0[3, ...] = V_l_q__I0[3, ...] / (4.0 * np.square(M_nucleon))
    V_l_q__I0[4, ...] = V_l_q__I0[4, ...] / (4.0 * np.power(M_nucleon, 4))

    V_l_q__I1[2, ...] = V_l_q__I1[2, ...] / (4.0 * np.square(M_nucleon))
    V_l_q__I1[3, ...] = V_l_q__I1[3, ...] / (4.0 * np.square(M_nucleon))
    V_l_q__I1[4, ...] = V_l_q__I1[4, ...] / (4.0 * np.power(M_nucleon, 4))




    # export partial waves datafiles (csv)
    # r_grid
    # V_l_r__I0
    # V_l_r__I1
    #
    # q_grid
    # V_l_q__I0
    # V_l_q__I1

    export_results(plotter_combined.cur_proc_run_base_path, 5, "V_l_r_I0", r_grid, V_l_r__I0, "r")
    export_results(plotter_combined.cur_proc_run_base_path, 5, "V_l_r_I1", r_grid, V_l_r__I1, "r")
    export_results(plotter_combined.cur_proc_run_base_path, 5, "V_l_q_I0", q_grid, V_l_q__I0, "q")
    export_results(plotter_combined.cur_proc_run_base_path, 5, "V_l_q_I1", q_grid, V_l_q__I1, "q")



    exit()



def export_results(datapath, num_basis_el, fname_prefix, grid, f_l_var, varname):
    header = [varname, "s", "p", "d", "f", "g"]

    for base_idx in range(num_basis_el):
        with open(datapath + "/" + f"{fname_prefix}_rho_{base_idx + 1}.csv", "w") as csvfile:
            res_writer = csv.writer(csvfile, delimiter=";")

            # header
            res_writer.writerow(header)

            for grid_idx, grid_val in enumerate(grid):
                row = [grid_val, f_l_var[base_idx, 0, grid_idx], f_l_var[base_idx, 1, grid_idx], f_l_var[base_idx, 2, grid_idx], f_l_var[base_idx, 3, grid_idx], f_l_var[base_idx, 4, grid_idx]]
                res_writer.writerow(row)




if __name__ == "__main__":
    main()
    exit()