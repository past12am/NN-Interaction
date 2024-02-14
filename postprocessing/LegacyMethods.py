import os
import json
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from basis.BasisTauToSymAsym import BasisTauToSymAsym
from visualization.plotting import plot_form_factor, plot_full_amplitude



def plot_form_factors(data_path: str, tensorbase_type: str, z_range: float, X_range: float, process_type: str, tensorBasisNames: typing.List, save_plot=False, basis_transform: typing.Callable=None):

    pd_ff_list = list()
    for base_idx in range(5):
        pd_ff = pd.read_csv(data_path + f"/{tensorbase_type}_{base_idx}.txt")
        pd_ff = pd_ff.map(lambda s: complex(s.replace('i', 'j')) if(isinstance(s, str)) else s)

        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_ff["Z"])) < z_range, np.array(pd_ff["X"]) > X_range))[0]
        pd_ff = pd_ff.iloc[idx_selector]

        pd_ff_list.append(pd_ff)

    if(basis_transform is not None):
        pd_ff_list = basis_transform(pd_ff_list)

    for base_idx, pd_ff in enumerate(pd_ff_list):
        plot_form_factor(pd_ff, tensorBasisNames[base_idx] if tensorbase_type == "tau" else tensorBasisNames[base_idx], base_idx, data_path + f"/{process_type}_F{base_idx + 1}" , save_plot)


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
    pd_quark_exchange_flavor = pd.read_csv(flavor_data_path + "/quark_exchange.csv", delimiter=";", decimal=".")
    pd_diquark_exchange_flavor = pd.read_csv(flavor_data_path + "/diquark_exchange.csv", delimiter=";", decimal=".")

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



def full_amplitude_symmetric_basis(data_path, tensorbase_type, amplitude_isospin, dq_1_type, dq_2_type, Z_range, X_range_lower, tensorBasisNames: typing.List, savefig: bool=False):
    # Check that both quark, and diquark data is accessible
    data_path_quark_exchange = data_path + "/quark_exchange/"
    if(not os.path.isdir(data_path_quark_exchange)):
        exit(-1)

    data_path_diquark_exchange = data_path + "/diquark_exchange/"
    if(not os.path.isdir(data_path_diquark_exchange)):
        exit(-1)


    # Find latest run
    latest_run_dir_quark = find_latest_run_dir(data_path_quark_exchange)
    latest_run_dir_diquark = find_latest_run_dir(data_path_diquark_exchange)

    data_path_quark_exchange_latest = data_path_quark_exchange + latest_run_dir_quark + "/"
    data_path_diquark_exchange_latest = data_path_diquark_exchange + latest_run_dir_diquark + "/"

    if(data_path_quark_exchange_latest is None or data_path_diquark_exchange_latest is None):
        exit(-1)


    # Check which basis we are working in
    file_quark = open(data_path_quark_exchange_latest + "/spec.json")
    quark_exchange_spec = json.load(file_quark)
    file_quark.close()

    file_diquark = open(data_path_diquark_exchange_latest + "/spec.json")
    diquark_exchange_spec = json.load(file_diquark)
    file_diquark.close()



    # Load Dirac space form factors
    pd_dirac_space_form_factors_quark_exchange_proto = load_dirac_space_form_factors(data_path_quark_exchange_latest, tensorbase_type, 1, 0)
    pd_dirac_space_form_factors_diquark_exchange_proto = load_dirac_space_form_factors(data_path_diquark_exchange_latest, tensorbase_type, 1, 0)


    # data in the tau basis with expected meta-information
    if(quark_exchange_spec["basis"] == "tau"):
        # data in tau basis --> transform to T basis
        pd_dirac_space_form_factors_quark_exchange = BasisTauToSymAsym.build_alternate_basis_pds(pd_dirac_space_form_factors_quark_exchange_proto)
        
    elif(quark_exchange_spec["basis"] == "T"):
        # data in the T basis with expected meta-information
        pd_dirac_space_form_factors_quark_exchange = pd_dirac_space_form_factors_quark_exchange_proto
        pass

    else:
        exit(-1)

    if(diquark_exchange_spec["basis"] == "tau"):
        pd_dirac_space_form_factors_diquark_exchange = BasisTauToSymAsym.build_alternate_basis_pds(pd_dirac_space_form_factors_diquark_exchange_proto)

    elif(diquark_exchange_spec["basis"] == "T"):
        pd_dirac_space_form_factors_diquark_exchange = pd_dirac_space_form_factors_diquark_exchange_proto

    else:
        exit(-1)


    # Check that remaining parameters match the specification
    if(quark_exchange_spec["amplitude_isospin"] != amplitude_isospin or diquark_exchange_spec["amplitude_isospin"] != amplitude_isospin):
        exit(-1)

    if(quark_exchange_spec["diquark_type_1"] != dq_1_type or diquark_exchange_spec["diquark_type_1"] != dq_1_type):
        exit(-1)

    if(quark_exchange_spec["diquark_type_2"] != dq_2_type or diquark_exchange_spec["diquark_type_2"] != dq_2_type):
        exit(-1)


    # Load flavor factors
    flavor_factor_quark_exchange, flavor_factor_diquark_exchange = load_flavor_space_form_factors("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/flavorspace",
                                                                                       amplitude_isospin, dq_1_type, dq_2_type)


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
    pd_form_factors_restricted = list()
    for base_idx, pd_ff in enumerate(pd_form_factors):
        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_ff["Z"])) <= Z_range, X_range_lower <= np.array(pd_ff["X"])))[0]
        pd_ff = pd_ff.iloc[idx_selector]
        pd_form_factors_restricted.append(pd_ff)

    plot_full_amplitude(pd_form_factors_restricted, tensorBasisNames, f"{data_path}/plots/quark_{latest_run_dir_quark}_diquark_{latest_run_dir_diquark}.png", savefig)
    



    
def find_latest_run_dir(run_dirs_path: str):
    latest_run_dir = None

    run_dir_idx = -1
    for run_directory in os.listdir(run_dirs_path):
        if(not os.path.isdir(run_dirs_path + run_directory)):
            continue

        cur_run_dir_idx = int(run_directory[run_directory.find("_")+1 : ])
        if(cur_run_dir_idx >= run_dir_idx):
            latest_run_dir = run_directory
            run_dir_idx = cur_run_dir_idx
    
    return latest_run_dir




def plot_dirac_space_dressing_functions(data_path: str, tensorbase_type: str, process_type: str, run: str=None, savefig: bool=False):
    z_range = 1
    X_range = 0

    process_data_path = data_path + process_type + "/"
    process_data_path += find_latest_run_dir(process_data_path) + "/" if run is None else run + "/"

    plot_form_factors(process_data_path, tensorbase_type, z_range, X_range, process_type, savefig)


def plot_dirac_space_dressing_functions_tau_to_T(data_path: str, process_type: str, run: str=None, savefig: bool=False):
    z_range = 1
    X_range = 0

    process_data_path = data_path + process_type + "/"
    process_data_path += find_latest_run_dir(process_data_path) + "/" if run is None else run + "/"

    plot_form_factors(process_data_path, "tau", z_range, X_range, process_type, savefig, BasisTauToSymAsym.build_alternate_basis_pds)
