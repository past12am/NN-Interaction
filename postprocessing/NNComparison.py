import csv
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt


def flatten_dim1(xss):
    return [x for xs in xss for x in xs]

def import_results_lit_reid(datapath, process, dist_colname):
    return import_results_lit_nijm(datapath, process, "pwave", dist_colname)


def import_results_lit_nijm(datapath, process, tensor, dist_colname):
    process_files = [f for f in listdir(datapath) if isfile(join(datapath, f)) and f[0:2] == process]

    var_grids = list()
    LSJ_names = list()
    LSJ_grids = list()

    for process_file in process_files:
        with open(join(datapath, process_file), "r") as csv_process:
            res_reader = csv.DictReader(csv_process, delimiter=",")
            
            LSJ_names.append(process_file[3:-4])

            var_grids.append(list())
            LSJ_grids.append(list())

            for row in res_reader:
                var_grids[-1].append(float(row[dist_colname]))
                LSJ_grids[-1].append(float(row[tensor]))

    for idx in range(len(LSJ_grids)):
        var_grids[idx] = np.array(var_grids[idx])
        LSJ_grids[idx] = np.array(LSJ_grids[idx])

    return var_grids, LSJ_grids, LSJ_names


def import_results_own(datapath, isospin, tensor_name_abbrv):
    process_files = [f for f in listdir(datapath) if isfile(join(datapath, f)) and f[-4:] == ".csv" and f[6:7] == str(isospin) and f[8:10] == (tensor_name_abbrv if len(tensor_name_abbrv) == 2 else tensor_name_abbrv + "_")]

    var_grids = list()
    LSJ_names = list()
    LSJ_grids = list()
    

    for process_file in process_files:
        with open(join(datapath, process_file), "r") as csv_process:
            res_reader = csv.DictReader(csv_process, delimiter=";")

            LSJ_names.extend(res_reader.fieldnames[1:])
            
            var_grids.append([list() for i in range(len(res_reader.fieldnames[1:]))])
            LSJ_grids.append([list() for i in range(len(res_reader.fieldnames[1:]))])

            for row in res_reader:
                for idx, lsj in enumerate(res_reader.fieldnames[1:]):
                    var_grids[-1][idx].append(float(row["r"]))
                    LSJ_grids[-1][idx].append(float(row[lsj]))

    var_grids = flatten_dim1(var_grids)
    LSJ_grids = flatten_dim1(LSJ_grids)

    var_grids = [np.array(var_grid) for var_grid in var_grids]
    LSJ_grids = [np.array(LSJ_grid) for LSJ_grid in LSJ_grids]

    return var_grids, LSJ_grids, LSJ_names


def find_normalization(NN_LSJ_grid_list):
    scalar = np.nanmax(np.abs(NN_LSJ_grid_list[0]))

    for NN_LSJ_grid in NN_LSJ_grid_list:
        cur = np.nanmax(np.abs(NN_LSJ_grid))
        if(cur > scalar):
            scalar = cur

    return 1/scalar



def potential_comparison_plots(NN_tuple__own, NN_tuple__nijmI, NN_tuple__nijmII, NN_tuple__reid93, tensor):
    NN_r_grid_list__lit_nijmII, NN_LSJ_grid_list__lit_nijmII, NN_LSJ_names__lit_nijmII = NN_tuple__nijmII
    NN_r_grid_list__lit_nijmI, NN_LSJ_grid_list__lit_nijmI, NN_LSJ_names__lit_nijmI = NN_tuple__nijmI
    NN_r_grid_list__lit_reid93, NN_LSJ_grid_list__lit_reid93, NN_LSJ_names__lit_reid93 = NN_tuple__reid93
    NN_r_grid_list, NN_LSJ_grid_list, NN_LSJ_names = NN_tuple__own


    # Scale all potentials to maximim of 1
    own_scaler = find_normalization(NN_LSJ_grid_list)
    nijmI_scaler = find_normalization(NN_LSJ_grid_list__lit_nijmI)
    nijmII_scaler = find_normalization(NN_LSJ_grid_list__lit_nijmII)
    reid93_scaler = find_normalization(NN_LSJ_grid_list__lit_reid93)


    for (NN_r_grid__lit_nijmI, NN_LSJ_grid__lit_nijmI, NN_LSJ_name__lit_nijmI) in zip(NN_r_grid_list__lit_nijmI, NN_LSJ_grid_list__lit_nijmI, NN_LSJ_names__lit_nijmI):
        for (NN_r_grid__lit_nijmII, NN_LSJ_grid__lit_nijmII, NN_LSJ_name__lit_nijmII) in zip(NN_r_grid_list__lit_nijmII, NN_LSJ_grid_list__lit_nijmII, NN_LSJ_names__lit_nijmII):
            for (NN_r_grid__lit_reid93, NN_LSJ_grid__lit_reid93, NN_LSJ_name__lit_reid93) in zip(NN_r_grid_list__lit_reid93, NN_LSJ_grid_list__lit_reid93, NN_LSJ_names__lit_reid93):
                for (NN_r_grid, NN_LSJ_grid, NN_LSJ_name) in zip(NN_r_grid_list, NN_LSJ_grid_list, NN_LSJ_names):

                    if(NN_LSJ_name__lit_nijmI != NN_LSJ_name or  NN_LSJ_name__lit_nijmII != NN_LSJ_name or NN_LSJ_name__lit_reid93 != NN_LSJ_name):
                        continue                    


                    # Plotting
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

                    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.2, right=0.92, hspace=0.2, wspace=0.2)

                    ax.plot(NN_r_grid, own_scaler * NN_LSJ_grid, label=f"{NN_LSJ_name} Own")
                    ax.plot(NN_r_grid__lit_nijmI, -nijmI_scaler * NN_LSJ_grid__lit_nijmI, label=f"{NN_LSJ_name} Nijmegen I")
                    ax.plot(NN_r_grid__lit_nijmII, -nijmII_scaler * NN_LSJ_grid__lit_nijmII, label=f"{NN_LSJ_name} Nijmegen II")
                    ax.plot(NN_r_grid__lit_reid93, reid93_scaler * NN_LSJ_grid__lit_reid93, label=f"{NN_LSJ_name} Reid 93 - All Tensors")

                    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
                    fig.suptitle(tensor, x=mid, fontsize="xx-large")

                    ax.set_xlabel(f"$r$  [1/GeV]", fontsize="large")
                    ax.set_ylabel(f"$R_C^{{(NN)}}(r)$", fontsize="large")
                    ax.grid(color='lightgray', linestyle='dashed')
                    ax.legend()

                    plt.show()
                

def main():
    process = "NN"
    M_nucleon = 0.94    # GeV
    max_r = 4           # 1/GeV

    # literature results (reid93)   (currently only central)
    NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid = import_results_lit_reid("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/reid93/", process, "r")

    # Nijmegen results
    NN_C_r_grid_list__lit_nijmI, NN_C_LSJ_grid_list__lit_nijmI, NN_C_LSJ_names__lit_nijmI = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmI/", process, "VC", "R")
    NN_SS_r_grid_list__lit_nijmI, NN_SS_LSJ_grid_list__lit_nijmI, NN_SS_LSJ_names__lit_nijmI = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmI/", process, "VSS", "R")
    NN_T_r_grid_list__lit_nijmI, NN_T_LSJ_grid_list__lit_nijmI, NN_T_LSJ_names__lit_nijmI = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmI/", process, "VT", "R")
    NN_SO_r_grid_list__lit_nijmI, NN_SO_LSJ_grid_list__lit_nijmI, NN_SO_LSJ_names__lit_nijmI = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmI/", process, "VLSA", "R")
    NN_Q_r_grid_list__lit_nijmI, NN_Q_LSJ_grid_list__lit_nijmI, NN_Q_LSJ_names__lit_nijmI = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmI/", process, "VQ12", "R")

    NN_C_r_grid_list__lit_nijmII, NN_C_LSJ_grid_list__lit_nijmII, NN_C_LSJ_names__lit_nijmII = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmII/", process, "VC", "R")
    NN_SS_r_grid_list__lit_nijmII, NN_SS_LSJ_grid_list__lit_nijmII, NN_SS_LSJ_names__lit_nijmII = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmII/", process, "VSS", "R")
    NN_T_r_grid_list__lit_nijmII, NN_T_LSJ_grid_list__lit_nijmII, NN_T_LSJ_names__lit_nijmII = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmII/", process, "VT", "R")
    NN_SO_r_grid_list__lit_nijmII, NN_SO_LSJ_grid_list__lit_nijmII, NN_SO_LSJ_names__lit_nijmII = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmII/", process, "VLSA", "R")
    NN_Q_r_grid_list__lit_nijmII, NN_Q_LSJ_grid_list__lit_nijmII, NN_Q_LSJ_names__lit_nijmII = import_results_lit_nijm("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/cmake-build-debug/bin/nijmII/", process, "VQ12", "R")

    # our results   # Note: sum up tensor contributions, reid doesn't distinguish them
    NN_C_r_grid_list, NN_C_LSJ_grid_list, NN_C_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "C")
    NN_SS_r_grid_list, NN_SS_LSJ_grid_list, NN_SS_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "SS")
    NN_T_r_grid_list, NN_T_LSJ_grid_list, NN_T_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "T")
    NN_SO_r_grid_list, NN_SO_LSJ_grid_list, NN_SO_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "SO")
    NN_Q_r_grid_list, NN_Q_LSJ_grid_list, NN_Q_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "Q")
    
    # TODO: Do this? Move factor for dimensionless basis elements to amplitudes
    #NN_T_LSJ_grid_list = [NN_T_LSJ_grid / (4.0 * np.square(M_nucleon)) for NN_T_LSJ_grid in NN_T_LSJ_grid_list]
    #NN_SO_LSJ_grid_list = [NN_SO_LSJ_grid / (4.0 * np.square(M_nucleon)) for NN_SO_LSJ_grid in NN_SO_LSJ_grid_list]
    #NN_Q_LSJ_grid_list = [NN_Q_LSJ_grid / (4.0 * np.power(M_nucleon, 4)) for NN_Q_LSJ_grid in NN_Q_LSJ_grid_list]

    NN_full_grid_list = [NN_C_LSJ_grid + NN_SS_LSJ_grid + NN_T_LSJ_grid + NN_SO_LSJ_grid + NN_Q_LSJ_grid for NN_C_LSJ_grid, NN_SS_LSJ_grid, NN_T_LSJ_grid, NN_SO_LSJ_grid, NN_Q_LSJ_grid in zip(NN_C_LSJ_grid_list, NN_SS_LSJ_grid_list, NN_T_LSJ_grid_list, NN_SO_LSJ_grid_list, NN_Q_LSJ_grid_list)]
    NN_LSJ_names = NN_C_LSJ_names

    potential_comparison_plots((NN_C_r_grid_list, NN_C_LSJ_grid_list, NN_C_LSJ_names),
                               (NN_C_r_grid_list__lit_nijmI, NN_C_LSJ_grid_list__lit_nijmI, NN_C_LSJ_names__lit_nijmI),
                               (NN_C_r_grid_list__lit_nijmII, NN_C_LSJ_grid_list__lit_nijmII, NN_C_LSJ_names__lit_nijmII),
                               (NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid),
                               "Central")

    potential_comparison_plots((NN_SS_r_grid_list, NN_SS_LSJ_grid_list, NN_SS_LSJ_names),
                               (NN_SS_r_grid_list__lit_nijmI, NN_SS_LSJ_grid_list__lit_nijmI, NN_SS_LSJ_names__lit_nijmI),
                               (NN_SS_r_grid_list__lit_nijmII, NN_SS_LSJ_grid_list__lit_nijmII, NN_SS_LSJ_names__lit_nijmII),
                               (NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid),
                               "Spin-Spin")

    #potential_comparison_plots((NN_T_r_grid_list, NN_T_LSJ_grid_list, NN_T_LSJ_names),
    #                           (NN_T_r_grid_list__lit_nijmI, NN_T_LSJ_grid_list__lit_nijmI, NN_T_LSJ_names__lit_nijmI),
    #                           (NN_T_r_grid_list__lit_nijmII, NN_T_LSJ_grid_list__lit_nijmII, NN_T_LSJ_names__lit_nijmII),
    #                           (NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid),
    #                           "Tensor")    # TODO nijm Tensor is different from ours

    potential_comparison_plots((NN_SO_r_grid_list, NN_SO_LSJ_grid_list, NN_SO_LSJ_names),
                               (NN_SO_r_grid_list__lit_nijmI, NN_SO_LSJ_grid_list__lit_nijmI, NN_SO_LSJ_names__lit_nijmI),
                               (NN_SO_r_grid_list__lit_nijmII, NN_SO_LSJ_grid_list__lit_nijmII, NN_SO_LSJ_names__lit_nijmII),
                               (NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid),
                               "Spin-Orbit")
    
    potential_comparison_plots((NN_Q_r_grid_list, NN_Q_LSJ_grid_list, NN_Q_LSJ_names),
                               (NN_Q_r_grid_list__lit_nijmI, NN_Q_LSJ_grid_list__lit_nijmI, NN_Q_LSJ_names__lit_nijmI),
                               (NN_Q_r_grid_list__lit_nijmII, NN_Q_LSJ_grid_list__lit_nijmII, NN_Q_LSJ_names__lit_nijmII),
                               (NN_r_grid_list__lit_reid, NN_LSJ_grid_list__lit_reid, NN_LSJ_names__lit_reid),
                               "Quadratic Spin-Orbit")




if __name__ == "__main__":
    main()