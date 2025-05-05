import csv
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from NNComparison import import_results_own, import_results_mixing


M_nucleon = 0.94    # GeV    

pwave_names = ["s", "p", "d", "f", "g", "h", "i", "j", "k"]
pwave_names_capital = ["S", "P", "D", "F", "G", "H", "I", "J", "K"]

def check_S_Matrix_unitarity(var_grid: np.ndarray, f_l_var: np.ndarray):
    pass


def calculate_phase_shifts(var_grid: np.ndarray, f_l_var: np.ndarray):
    pass


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



def r_of_TLab(TLab):
    r = np.sqrt(TLab * M_nucleon / 2.0)      # M_nucleon = 0.94    # GeV
    return r


def tau_js_Ll(T_matrix_element, TLab):
    r = r_of_TLab(TLab)
    return - np.pi / 2 * np.square(M_nucleon) / np.sqrt(np.square(M_nucleon) + np.square(r)) * r * T_matrix_element


def S_js_Ll(T_matrix_element, TLab, J, S, L, Lp):
    delta = 0
    if(L == Lp):
        delta += 1

    return delta + 2j * tau_js_Ll(T_matrix_element, TLab)


def LSJ_from_spectr(LSJ_name):
    s = (int(LSJ_name[0]) - 1) // 2
    l = pwave_names_capital.index(LSJ_name[1])
    j = int(LSJ_name[2])

    return (l, s, j)


def spectr_from_LSJ(l, s, j):
    res = "xxx"
    res[0] = str(int(2 * s + 1))
    res[1] = pwave_names_capital[l]
    res[2] = str(int(j))

    return res


def calc_phase_shift_uncoupled(tau):
    return 0.5 * np.arctan(2 * np.real(tau) / (1 - np.imag(tau)))

def calc_mixing_angle_epsilon(S_Jpm, S_Jpp, S_Jmm):
    return 0.5 * np.arctan(-1j * S_Jpm / np.sqrt(S_Jpp * S_Jmm))

def calc_phase_shift_coupled(S_Jx, epsilon_J):
    return 0.5 * np.arctan(np.imag(S_Jx / np.cos(2 * epsilon_J)) / np.real(S_Jx / np.cos(2 * epsilon_J)))



def calc_phase_shifts(T_grid_list, LSJ_grid_list, LSJ_names,
                      T_grid_list_mixing, mixing_grid_list, mixing_Jvals):
    max_l = 0
    max_j = 0
    for T_grid, LSJ_grid, LSJ_name in zip(T_grid_list, LSJ_grid_list, LSJ_names):
        l, s, j = LSJ_from_spectr(LSJ_name)

        if(l > max_l):
            max_l = l
        
        if(j > max_j):
            max_j = j


    lsj_T_Lsj = dict() # np.zeros((max_l + 1, max_l + 1, num_s, max_j + 1, len(T_grid)))    # Assuming all T_grids equal length
    delta = dict()

    # Uncoupled: L = l
    for T_grid, LSJ_grid, LSJ_name in zip(T_grid_list, LSJ_grid_list, LSJ_names):
        l, s, j = LSJ_from_spectr(LSJ_name)
        L = l

        if(l not in lsj_T_Lsj.keys()):
            lsj_T_Lsj[l] = dict()
            delta[l] = dict()
        
        if(L not in lsj_T_Lsj[l].keys()):
            lsj_T_Lsj[l][L] = dict()
            delta[l][L] = dict()
        
        if(s not in lsj_T_Lsj[l][L].keys()):
            lsj_T_Lsj[l][L][s] = dict()
            delta[l][L][s] = dict()

        if(j not in lsj_T_Lsj[l][L][s].keys()):
            lsj_T_Lsj[l][L][s][j] = LSJ_grid
            delta[l][L][s][j] = np.zeros_like(LSJ_grid)
        else:
            raise Exception("Twice same combination")

    # Mixing: l = j-1, L=j+1
    for T_grid_mixing, mixing_grid, J_val in zip(T_grid_list_mixing[:-1], mixing_grid_list[:-1], mixing_Jvals[:-1]):  # TODO remove the aritfact of the 0 (last element) in processing code
        j = int(J_val)
        l = j-1
        L = j+1
        s = 1

        if(l not in lsj_T_Lsj.keys()):
            lsj_T_Lsj[l] = dict()
            lsj_T_Lsj[L] = dict()
            delta[l] = dict()
        
        if(L not in lsj_T_Lsj[l].keys()):
            lsj_T_Lsj[l][L] = dict()
            lsj_T_Lsj[L][l] = dict()
            delta[l][L] = dict()
        
        if(s not in lsj_T_Lsj[l][L].keys()):
            lsj_T_Lsj[l][L][s] = dict()
            lsj_T_Lsj[L][l][s] = dict()
            delta[l][L][s] = dict()

        if(j not in lsj_T_Lsj[l][L][s].keys()):
            lsj_T_Lsj[l][L][s][j] = mixing_grid
            lsj_T_Lsj[L][l][s][j] = mixing_grid
            delta[l][L][s][j] = dict()
        else:
            raise Exception("Twice same combination")



    # mixing elements
    epsilon_j = dict()
    for j in range(1, max_j-1):

        #if(j+1 not in lsj_T_Lsj or j-1 not in lsj_T_Lsj[j+1] or 1 not in lsj_T_Lsj[j+1][j-1] or j not in lsj_T_Lsj[j+1][j-1][1]):
        #    continue

        epsilon_j[j] = calc_mixing_angle_epsilon(
            S_js_Ll(lsj_T_Lsj[j+1][j-1][1][j], T_grid, j, 1, j+1, j-1),
            S_js_Ll(lsj_T_Lsj[j+1][j+1][1][j], T_grid, j, 1, j+1, j+1),
            S_js_Ll(lsj_T_Lsj[j-1][j-1][1][j], T_grid, j, 1, j-1, j-1)
        )
    
    # Phase Shifts
    for LSJ_name in LSJ_names:
        l, s, j = LSJ_from_spectr(LSJ_name)
        L = l

        # uncoupled
        delta[l][L][s][j][:] = calc_phase_shift_uncoupled(tau_js_Ll(lsj_T_Lsj[l][L][s][j], T_grid))


        # we are missing the mixing angles beyond that point
        if (j >= max_j-1 or j==0):
            continue

        if(s == 1): # no spin 0 coupled channel
            if(l == j+1 or l == j-1):
                # Coupled: l = J+1
                delta[l][L][s][j][:] = calc_phase_shift_coupled(S_js_Ll(lsj_T_Lsj[l][L][s][j], T_grid, j, s, l, L), epsilon_j[j])

                # Coupled: l = J-1
                delta[l][L][s][j][:] = calc_phase_shift_coupled(S_js_Ll(lsj_T_Lsj[l][L][s][j], T_grid, j, s, l, L), epsilon_j[j])

    

    return delta
    


def sum_contribs(C_list, SS_list, T_list, SO_list, Q_list):
    return [C + SS + T + SO for C, SS, T, SO, Q in zip(C_list, SS_list, T_list, SO_list, Q_list)]


def main():
    #   Load Lab Energy Data
    C_T_grid_list, C_LSJ_grid_list, C_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "C", "T")
    SS_T_grid_list, SS_LSJ_grid_list, SS_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "SS", "T")
    T_T_grid_list, T_LSJ_grid_list, T_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "T", "T")
    SO_T_grid_list, SO_LSJ_grid_list, SO_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "SO", "T")
    Q_T_grid_list, Q_LSJ_grid_list, Q_LSJ_names = import_results_own("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "Q", "T")

    C_T_grid_list_mixing, C_mixing_grid_list, C_mixing_Jvals = import_results_mixing("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "C", "T")
    SS_T_grid_list_mixing, SS_mixing_grid_list, SS_mixing_Jvals = import_results_mixing("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "SS", "T")
    T_T_grid_list_mixing, T_mixing_grid_list, T_mixing_Jvals = import_results_mixing("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "T", "T")
    SO_T_grid_list_mixing, SO_mixing_grid_list, SO_mixing_Jvals = import_results_mixing("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "SO", "T")
    Q_T_grid_list_mixing, Q_mixing_grid_list, Q_mixing_Jvals = import_results_mixing("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/qx_tau_analytic-dq_tau_analytic/qx-run_87_dqx-run_87", 1, "Q", "T")



    # Calc Phase Shifts
    C_delta = calc_phase_shifts(C_T_grid_list, C_LSJ_grid_list, C_LSJ_names,
                                        C_T_grid_list_mixing, C_mixing_grid_list, C_mixing_Jvals)
    SS_delta = calc_phase_shifts(SS_T_grid_list, SS_LSJ_grid_list, SS_LSJ_names,
                                         SS_T_grid_list_mixing, SS_mixing_grid_list, SS_mixing_Jvals)
    T_delta = calc_phase_shifts(T_T_grid_list, T_LSJ_grid_list, T_LSJ_names,
                                        T_T_grid_list_mixing, T_mixing_grid_list, T_mixing_Jvals)
    SO_delta = calc_phase_shifts(SO_T_grid_list, SO_LSJ_grid_list, SO_LSJ_names,
                                         SO_T_grid_list_mixing, SO_mixing_grid_list, SO_mixing_Jvals)
    Q_delta = calc_phase_shifts(Q_T_grid_list, Q_LSJ_grid_list, Q_LSJ_names,
                                        Q_T_grid_list_mixing, Q_mixing_grid_list, Q_mixing_Jvals)
    
    delta = calc_phase_shifts(C_T_grid_list, sum_contribs(C_LSJ_grid_list, SS_LSJ_grid_list, T_LSJ_grid_list, SO_LSJ_grid_list, Q_LSJ_grid_list), C_LSJ_names,
                              C_T_grid_list_mixing, sum_contribs(C_mixing_grid_list, SS_mixing_grid_list, T_mixing_grid_list, SO_mixing_grid_list, Q_mixing_grid_list), C_mixing_Jvals)
    

    # Load Literature Results
    # Set your target directory
    directory = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/nnonline-phaseshifts"  # replace with your directory path

    # Find all .tsv files with 3-letter filenames
    tsv_files = glob.glob(os.path.join(directory, "[0-9][A-Z][0-9].tsv"))

    # Dictionary to hold data from each file
    data_by_file = {}

    for filepath in tsv_files:
        filename = os.path.basename(filepath)
        key = os.path.splitext(filename)[0]  # get the 3-letter code
        
        # Read the TSV file into a DataFrame
        df = pd.read_csv(filepath, sep=r"\s+", engine='python')
        
        # Store each column as a NumPy array in a sub-dictionary
        data_by_file[key] = {col: df[col].values for col in df.columns}

    # Example access:
    # Tlab values from file "abc.tsv" -> data_by_file['abc']['Tlab']
    # esc96 values from file "xyz.tsv" -> data_by_file['xyz']['esc96']

    for LSJ_name in C_LSJ_names:
        l, s, j = LSJ_from_spectr(LSJ_name)
        L = l

        if(LSJ_name not in data_by_file.keys()):
            continue

        plt.figure()
        plt.plot(C_T_grid_list[0], C_delta[l][L][s][j] * 180/np.pi, label="own")
        plt.plot(data_by_file[LSJ_name]['Tlab'] * 1E-3, data_by_file[LSJ_name]["nijm2"], label="nijmII")
        plt.title(LSJ_name)
        plt.legend()
        plt.show()









if __name__ == "__main__":
    main()