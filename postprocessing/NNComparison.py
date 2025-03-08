import csv
from os import listdir
from os.path import isfile, join

import numpy as np
import matplotlib.pyplot as plt


def flatten_dim1(xss):
    return [x for xs in xss for x in xs]

def import_results_lit(datapath, process):
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
                var_grids[-1].append(float(row["r"]))
                LSJ_grids[-1].append(float(row["pwave"]))

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

    return flatten_dim1(var_grids), flatten_dim1(LSJ_grids), LSJ_names

def main():
    process = "NN"

    # literature results (reid93)   (currently only central)
    NN_C_r_grid_list__lit, NN_C_LSJ_grid_list__lit, NN_C_LSJ_names__lit = import_results_lit("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/reid93/", process)

    # our results   # TODO sum up tensor contributions, reid doesn't distinguish them
    NN_C_r_grid_list, NN_C_LSJ_grid_list, NN_C_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_35_dqx-run_35", 1, "C")
    


    for (NN_C_r_grid__lit, NN_C_LSJ_grid__lit, NN_C_LSJ_name__lit) in zip(NN_C_r_grid_list__lit, NN_C_LSJ_grid_list__lit, NN_C_LSJ_names__lit):
        for (NN_C_r_grid, NN_C_LSJ_grid, NN_C_LSJ_name) in zip(NN_C_r_grid_list, NN_C_LSJ_grid_list, NN_C_LSJ_names):
            if(NN_C_LSJ_name__lit != NN_C_LSJ_name):
                continue


            # Scale to have things match up
            reid_scaler = np.nanmax(np.abs(NN_C_LSJ_grid)) / np.nanmax(np.abs(NN_C_LSJ_grid__lit))
            if(reid_scaler == 0):
                reid_scaler = 1


            # Plotting
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

            fig.subplots_adjust(top=0.88, bottom=0.11, left=0.2, right=0.92, hspace=0.2, wspace=0.2)

            ax.plot(NN_C_r_grid, NN_C_LSJ_grid, label=f"{NN_C_LSJ_name} Own")
            ax.plot(NN_C_r_grid__lit, reid_scaler * NN_C_LSJ_grid__lit, label=f"{NN_C_LSJ_name} Reid93")

                
            #mid = (fig.subplotpars.right + fig.subplotpars.left)/2
            #fig.suptitle(title, x=mid, fontsize="xx-large")

            ax.set_xlabel(f"$r$  [1/GeV]", fontsize="large")
            ax.set_ylabel(f"$R_C^{{(NN)}}(r)$", fontsize="large")
            ax.grid(color='lightgray', linestyle='dashed')
            ax.legend()

            plt.show()




if __name__ == "__main__":
    main()