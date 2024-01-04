import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from visualization.plotM2 import plot_result, plot_form_factor


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


def plot_form_factors(data_path: str, plot_dir: str, z_range: float, X_range: float, save_plot=False):

    pd_tau_list = list()
    for tauIdx in range(5):
        pd_tau = pd.read_csv(data_path + f"/T_{tauIdx}.txt")
        pd_tau = pd_tau.map(lambda s: complex(s.replace('i', 'j')) if(isinstance(s, str)) else s)

        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_tau["Z"])) < z_range, np.array(pd_tau["X"]) > X_range))[0]
        pd_tau = pd_tau.iloc[idx_selector]

        pd_tau_list.append(pd_tau)

    for tauIdx, pd_tau in enumerate(pd_tau_list):
        plot_form_factor(pd_tau, tensorBasisNamesT[tauIdx], tauIdx, data_path + "/" + plot_dir + "/", save_plot)


def load_dirac_space_form_factors(data_path: str, tensorbase_type: str, Z_range: float, X_range: float):

    pd_dirac_form_factors = list()
    for baseElemIdx in range(5):
        pd_ff = pd.read_csv(data_path + f"/{tensorbase_type}_{baseElemIdx}.txt")
        pd_ff = pd_ff.map(lambda s: complex(s.replace('i', 'j')) if(isinstance(s, str)) else s)

        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_ff["Z"])) < Z_range, np.array(pd_ff["X"]) > X_range))[0]
        pd_ff = pd_ff.iloc[idx_selector]

        pd_dirac_form_factors.append(pd_ff)

    return pd_dirac_form_factors



def load_flavor_space_form_factors(flavor_data_path: str, amplitude_isospin, dq_1_type, dq_2_type):
    pd_quark_exchange_flavor = pd.read_csv(flavor_data_path + "/quark-exchange.csv", delimiter=";", decimal=".")
    pd_diquark_exchange_flavor = pd.read_csv(flavor_data_path + "/diquark-exchange.csv", delimiter=";", decimal=".")

    quark_exchange_flavor_factor = pd_quark_exchange_flavor[(pd_quark_exchange_flavor.amplitude_isospin == amplitude_isospin) &
                                                            (pd_quark_exchange_flavor.diquark_1_type == dq_1_type) &
                                                            (pd_quark_exchange_flavor.diquark_2_type == dq_2_type)]["flavor_factor"].values[0]
    
    diquark_exchange_flavor_factor = pd_diquark_exchange_flavor[(pd_quark_exchange_flavor.amplitude_isospin == amplitude_isospin) &
                                                                (pd_quark_exchange_flavor.diquark_1_type == dq_1_type) &
                                                                (pd_quark_exchange_flavor.diquark_2_type == dq_2_type)]["flavor_factor"].values[0]

    try:
        quark_exchange_flavor_factor = float(quark_exchange_flavor_factor)
    except:
        a , b = quark_exchange_flavor_factor.split("/")
        quark_exchange_flavor_factor = int(a) / int(b)

    try:
        diquark_exchange_flavor_factor = float(diquark_exchange_flavor_factor)
    except:
        a , b = diquark_exchange_flavor_factor.split("/")
        diquark_exchange_flavor_factor = int(a) / int(b)


    return quark_exchange_flavor_factor, diquark_exchange_flavor_factor



def full_amplitude_symmetric_basis(data_path, tensorbase_type, amplitude_isospin, dq_1_type, dq_2_type):
    # Check that both quark, and diquark data is accessible
    data_path_quark_exchange = data_path + "/quark_exchange/"
    if(not os.path.isdir(data_path_quark_exchange)):
        exit(-1)

    data_path_diquark_exchange = data_path + "/diquark_exchange/"
    if(not os.path.isdir(data_path_diquark_exchange)):
        exit(-1)


    # Find latest run
    data_path_quark_exchange_latest = data_path_quark_exchange + find_latest_run_dir(data_path_quark_exchange) + "/"
    data_path_diquark_exchange_latest = data_path_diquark_exchange + find_latest_run_dir(data_path_diquark_exchange) + "/"

    if(data_path_quark_exchange_latest is None or data_path_diquark_exchange_latest is None):
        exit(-1)


    # Check that we have the data in the T basis with expected meta-information
    file_quark = open(data_path_quark_exchange_latest + "/spec.json")
    quark_exchange_spec = json.load(file_quark)
    file_quark.close()

    file_diquark = open(data_path_diquark_exchange_latest + "/spec.json")
    diquark_exchange_spec = json.load(file_diquark)
    file_diquark.close()

    if(quark_exchange_spec["basis"] != "T" or diquark_exchange_spec["basis"] != "T"):
        exit(-1)

    if(quark_exchange_spec["amplitude_isospin"] != amplitude_isospin or diquark_exchange_spec["amplitude_isospin"] != amplitude_isospin):
        exit(-1)

    if(quark_exchange_spec["diquark_type_1"] != dq_1_type or diquark_exchange_spec["diquark_type_1"] != dq_1_type):
        exit(-1)

    if(quark_exchange_spec["diquark_type_2"] != dq_2_type or diquark_exchange_spec["diquark_type_2"] != dq_2_type):
        exit(-1)


    # Load flavor factors
    flavor_factor_quark_exchange, flavor_factor_diquark_exchange = load_flavor_space_form_factors("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/flavorspace",
                                                                                       amplitude_isospin, dq_1_type, dq_2_type)
        

    # Load Dirac space form factors
    pd_dirac_space_form_factors_quark_exchange = load_dirac_space_form_factors(data_path_quark_exchange_latest, tensorbase_type, 1, 0)
    pd_dirac_space_form_factors_diquark_exchange = load_dirac_space_form_factors(data_path_diquark_exchange_latest, tensorbase_type, 1, 0)


    # Calculate full form factors by combining spaces
    pd_form_factors_quark_exchange = list()
    pd_form_factors_diquark_exchange = list()
    for pd_dirac_space_ff_q_ex, pd_dirac_space_ff_diq_ex in zip(pd_dirac_space_form_factors_quark_exchange, pd_dirac_space_form_factors_diquark_exchange):
        
        pd_dirac_space_ff_q_ex["f"] = pd_dirac_space_ff_q_ex["f"].apply(lambda x: x * flavor_factor_quark_exchange)
        pd_dirac_space_ff_diq_ex["f"] = pd_dirac_space_ff_diq_ex["f"].apply(lambda x: x * flavor_factor_diquark_exchange)

        pd_form_factors_quark_exchange.append(pd_dirac_space_ff_q_ex)
        pd_form_factors_diquark_exchange.append(pd_dirac_space_ff_diq_ex)



    
    # Calculate full symmetric amplitude
    #   Add quark and diquark contributions together
    # TODO check that we add points of equal external parameters (Z, X) --> make sure external impulses are equal
    pd_form_factors = [pd_ff_qu_ex.copy() for pd_ff_qu_ex in pd_form_factors_quark_exchange]

    for i, pd_ff_diq_ex in enumerate(pd_form_factors_diquark_exchange):
        pd_form_factors[i]["f"] = pd_form_factors[i]["f"] - pd_ff_diq_ex["f"]
        pd_form_factors[i]["f"] *= 1/2

    # Plot full amplitude
    for base_idx, pd_ff in enumerate(pd_form_factors):
        plot_form_factor(pd_ff, tensorBasisNamesT[base_idx], base_idx, "", False)
    



    
def find_latest_run_dir(run_dirs_path: str):
    latest_run_dir = None

    run_dir_idx = -1
    for run_directory in os.listdir(run_dirs_path):
        if(not os.path.isdir(run_dirs_path + run_directory)):
            continue

        cur_run_dir_idx = int(run_directory[run_directory.find("_")+1 : ])
        if(cur_run_dir_idx >= run_dir_idx):
            latest_run_dir = run_directory
    
    return latest_run_dir






def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/"  #sys.argv[1]


    tensorbase_type = "T"   #sys.argv[2]

    amplitude_isospin = 0   #int(sys.argv[3])

    dq_1_type = "scalar"    #sys.argv[2]
    dq_2_type = "scalar"    #sys.argv[3]


    # Construct directory fitting specs
    data_path = data_base_path + f"/BASE-{tensorbase_type}_I-{amplitude_isospin}_DQ-{dq_1_type}-{dq_2_type}/"

    # Postprocess full symmetric amplitude
    full_amplitude_symmetric_basis(data_path, tensorbase_type, amplitude_isospin, dq_1_type, dq_2_type)






if __name__ == "__main__":
    main()
    exit()

"""
z_range = 1
X_range = 0


#cur_path = sys.argv[1]
cur_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/BASE-T_I-0_DQ-scalar-scalar/diquark_exchange/run_0/"

plot_form_factors(cur_path, "plots_z10", z_range, X_range, False)
#plot_form_factors("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteractionPython/data/run0", "", z_range, X_range, function_name, False)
"""