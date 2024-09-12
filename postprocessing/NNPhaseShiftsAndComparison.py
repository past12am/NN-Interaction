import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pwave_names = ["s", "p", "d", "f", "g"]


def import_results(datapath, num_basis_el, fname_prefix, varname, num_pwaves):
    var_grid = None
    f_l_var = [None for i in range(num_basis_el)]

    for base_idx in range(num_basis_el):
        with open(datapath + "/" + f"{fname_prefix}_rho_{base_idx + 1}.csv", "r") as csvfile:
            res_reader = csv.DictReader(csvfile, delimiter=";")

            f_l_var[base_idx] = [list() for j in range(num_pwaves)]

            var_grid = list()
            for row in res_reader:
                var_grid.append(float(row[varname]))

                f_l_var[base_idx][0].append(float(row[pwave_names[0]]))
                f_l_var[base_idx][1].append(float(row[pwave_names[1]]))
                f_l_var[base_idx][2].append(float(row[pwave_names[2]]))
                f_l_var[base_idx][3].append(float(row[pwave_names[3]]))
                f_l_var[base_idx][4].append(float(row[pwave_names[4]]))

    return np.array(f_l_var), np.array(var_grid)
                

def check_S_Matrix_unitarity(var_grid: np.ndarray, f_l_var: np.ndarray):
    max_diff_to_1 = 0

    for basis_idx in range(f_l_var.shape[0]):
        for l in range(f_l_var.shape[1]):
            for var_idx, x in enumerate(var_grid):
                diff_to_1 = np.abs(np.abs(1 + 2j * x * f_l_var[basis_idx, l, var_idx]) - 1)

                if(diff_to_1 > max_diff_to_1):
                    max_diff_to_1 = diff_to_1

    print(max_diff_to_1)


def calculate_phase_shifts(var_grid: np.ndarray, f_l_var: np.ndarray):
    delta_l_var = np.zeros_like(f_l_var)

    for basis_idx in range(f_l_var.shape[0]):
        for l in range(f_l_var.shape[1]):
            for var_idx, x in enumerate(var_grid):
                delta_l_var[basis_idx, l, var_idx] = 0.5 * np.arctan(2.0 * x * f_l_var[basis_idx, l, var_idx])

    return delta_l_var / (2 * np.pi) * 360


def plot_phase_shifts(delta_l_var, var_grid, cur_proc_run_base_path, xlabel, x_label_unit, base_type, process_isospin, fig_name, savefig: bool=True, show_plots: bool=False):
    for basis_idx in range(delta_l_var.shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        axs = [ax]

        fig.subplots_adjust(top=0.88, bottom=0.11, left=0.2, right=0.92, hspace=0.2, wspace=0.2)

        for l in range(delta_l_var.shape[1]):
            axs[0].plot(var_grid, delta_l_var[basis_idx, l, :], label=f"{pwave_names[l]}-wave")
            
            
        mid = (fig.subplotpars.right + fig.subplotpars.left)/2
        fig.suptitle(f"$\delta_{{{basis_idx + 1} l}}^{{(I={process_isospin})}}({xlabel})$", x=mid, fontsize="xx-large")

        axs[0].set_xlabel(f"${xlabel}$  [{x_label_unit}]", fontsize="large")
        axs[0].set_ylabel(f"$\delta_{{{basis_idx + 1} l}}^{{(I={process_isospin})}}$", fontsize="large")
        axs[0].grid(color='lightgray', linestyle='dashed')
        #axs[0].spines[['right', 'top']].set_visible(False)
        axs[0].legend()


        if(savefig):
            plt.savefig(cur_proc_run_base_path + "/" + f"{fig_name}_{"isovector" if process_isospin == 1 else "isoscalar"}_{base_type}{basis_idx + 1}.pdf", dpi=600)

        if(show_plots):
            plt.show()

        plt.close()



def plot_central_potentials(V_C_list, r_grid_list, potential_name_list, colours, cur_proc_run_base_path, xlabel, x_label_unit, fig_name, savefig: bool=True, show_plots: bool=False):

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    axs = [ax]

    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.2, right=0.92, hspace=0.2, wspace=0.2)
    for V_C, r_grid, potential_name, colour in zip(V_C_list, r_grid_list, potential_name_list, colours):

        axs[0].plot(r_grid, V_C, label=f"{potential_name}", c=colour)

        axs[0].set_xlabel(f"${xlabel}$  [{x_label_unit}]", fontsize="large")
        axs[0].set_ylabel(f"$V(r)$", fontsize="large")
        axs[0].grid(color='lightgray', linestyle='dashed')
        #axs[0].spines[['right', 'top']].set_visible(False)
        axs[0].legend()

        axs[0].set_xlim([0, 2])


    if(savefig):
        plt.savefig(cur_proc_run_base_path + "/" + f"{fig_name}.pdf", dpi=600)

    if(show_plots):
        plt.show()

    plt.close()



def main():
    base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/"
    qx_run_dir = f"run_{16}"
    dqx_run_dir = f"run_{12}"

    process_spec = dict()
    process_spec["basis"] = "T"
    process_spec["invert_strategy"] = "numeric_matrix_inverse"

    run_path_name = f"qx-{qx_run_dir}_dqx-{dqx_run_dir}"
    base_spec_dir_name = "qx_" + process_spec["basis"] + "_" + process_spec["invert_strategy"] + "-dq_" + process_spec["basis"] + "_" + process_spec["invert_strategy"]

    cur_proc_run_base_path = base_path + "/" + base_spec_dir_name + "/" + run_path_name + "/"

    num_pwaves = 5



    # Read results
    V_l_r__I0, r_grid_I0 = import_results(cur_proc_run_base_path, 5, "V_l_r_I0", "r", num_pwaves)
    V_l_r__I1, r_grid_I1 = import_results(cur_proc_run_base_path, 5, "V_l_r_I1", "r", num_pwaves)
    V_l_q__I0, q_grid_I0 = import_results(cur_proc_run_base_path, 5, "V_l_q_I0", "q", num_pwaves)
    V_l_q__I1, q_grid_I1 = import_results(cur_proc_run_base_path, 5, "V_l_q_I1", "q", num_pwaves)


    # Check vanishing complex part of phase shift due to S Matrix Unitarity
    check_S_Matrix_unitarity(r_grid_I0, V_l_r__I0)
    check_S_Matrix_unitarity(r_grid_I1, V_l_r__I1)
    check_S_Matrix_unitarity(q_grid_I0, V_l_q__I0)
    check_S_Matrix_unitarity(q_grid_I1, V_l_q__I1)


    # Calculate Phase shifts
    delta_l_q__I0 = calculate_phase_shifts(q_grid_I0, V_l_q__I0)
    delta_l_q__I1 = calculate_phase_shifts(q_grid_I1, V_l_q__I1)
    delta_l_r__I0 = calculate_phase_shifts(r_grid_I0, V_l_r__I0)
    delta_l_r__I1 = calculate_phase_shifts(r_grid_I1, V_l_r__I1)


    # Plot Phase Shifts
    #plot_phase_shifts(delta_l_q__I0, q_grid_I0, cur_proc_run_base_path, "q", "GeV", "rho", 0, "PhaseShift", False, True)
    #plot_phase_shifts(delta_l_q__I1, q_grid_I1, cur_proc_run_base_path, "q", "GeV", "rho", 0, "PhaseShift", False, True)



    ##################################################### Potential comparison #######################################################
    
    # Load Reid93 1S0 neutron neutron Central Potential
    reid93_csv = pd.read_csv("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/reid93/Reid93Potential.csv")
    V_1S0_reid93 = reid93_csv["v11"].to_numpy()
    r_grid_reid93 = reid93_csv["r"].to_numpy()

    nijmII_csv = pd.read_csv("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/ExperimentalPotentials/nijmegen/nijm.csv")
    V_C_nijmII = nijmII_csv["vc"].to_numpy()
    r_grid_nijmII = nijmII_csv["r"].to_numpy()

    reid_scaler = np.max(V_l_r__I0[0, 0, 1:]) / np.max(np.abs(V_1S0_reid93))
    nijm_scaler = np.max(V_l_r__I0[0, 0, 1:]) / np.max(np.abs(V_C_nijmII))

    potential_list = [-V_C_nijmII * nijm_scaler, V_1S0_reid93 * reid_scaler, V_l_r__I0[0, 0, :]]
    r_grid_list = [r_grid_nijmII, r_grid_reid93, r_grid_I0]
    potentail_names_list = ["NijmegenII $^1S_0$ Central Potential", "Reid93 $^1S_0$ Potential", "$V_\\text{C}$"]
    colours = ["C2", "C1", "C0"]

    plot_central_potentials(potential_list, 
                            r_grid_list, 
                            potentail_names_list, 
                            colours,
                            cur_proc_run_base_path, 
                            "r", 
                            "1 / GeV", 
                            "CentralPotentialComparison",
                            savefig=True, 
                            show_plots=True)





if __name__ == "__main__":
    main()