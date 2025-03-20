
from data.Dataloader import Dataloader
from visualization.plotting import Plotter
from NNPostprocess import tensorBasisNamesDict

import json

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt




def parabola(x, a, b, c):
    return a + b * x + c * x * x


def fit_and_extrapolate(eps, h, f, X, Z, plot_title: str=None, plot_fit: bool=False):
    popt_eps, pcov_eps = curve_fit(parabola, eps, h)
    popt_f, pcov_f = curve_fit(parabola, eps, f)
    
    eps_space = np.linspace(0, np.max(eps), 20)

    if(plot_fit):
        plt.figure()
        plt.plot(eps_space, parabola(eps_space, *popt_f), label="fit", c="orange")
        plt.scatter(eps, f, label=f"Data: X={X}, Z={Z}")

        if(plot_title is not None):
            plt.title(plot_title)
        
        plt.legend()
        plt.show()


    # Extrapolate via fit
    h_extrapolated = parabola(0, *popt_eps)
    f_extrapolated = parabola(0, *popt_f)

    return h_extrapolated, f_extrapolated


def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/data/"
    output_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/"

    tensorbase_type = "tau"

    dq_1_type = "scalar"
    dq_2_type = "scalar"

    process_type = "diquark_exchange"

    Z_range = 0.9
    X_range_lower = 0

    latest_run_dir = None #"run_53"



    # Load pandas and paths
    data_path = data_base_path + f"/BASE-{tensorbase_type}_I-0_DQ-{dq_1_type}-{dq_2_type}/"
    data_path_process = data_path + f"/{process_type}/"
    latest_run_dir = Dataloader.find_latest_run_dir(data_path_process) if latest_run_dir is None else latest_run_dir

    data_path_process_latest = data_path_process + latest_run_dir

    pd_data_list = Dataloader.load_dirac_space_form_factors(data_path_process + latest_run_dir + "/", tensorbase_type, Z_range, X_range_lower, True)

    # Load specs
    spec_file = open(data_path_process_latest + "/spec.json")
    process_spec = json.load(spec_file)
    spec_file.close()



    # Load plotter
    plotter = Plotter(data_base_path, tensorBasisNamesDict, process_type, latest_run_dir, process_spec, False, False)


    for basis_idx, pd_data in enumerate(pd_data_list):
        pd_data = pd_data.dropna()

        X_vals = pd_data["X"].unique()
        Z_vals = pd_data["Z"].unique()

        X_list = list()
        Z_list = list()
        h_list = list()
        f_list = list()
        
        for X in X_vals:
            for Z in Z_vals:
                
                pd_cur = pd_data[(pd_data["Z"] == Z) & (pd_data["X"] == X)]

                eps = pd_cur["eps"]
                h = pd_cur["h"]
                f = pd_cur["f"]

                h_extrapolated, f_extrapolated = fit_and_extrapolate(eps, np.real(h), np.real(f), X, Z, "Real")
                h_extrapolated_imag, f_extrapolated_imag = fit_and_extrapolate(eps, np.imag(h), np.imag(f), X, Z, "Imag")

                print(f"Extrapolation: Imag f = {f_extrapolated_imag}      Imag h = {h_extrapolated_imag}      Real f = {f_extrapolated}")
                
                X_list.append(X)
                Z_list.append(Z)
                h_list.append(h_extrapolated)
                f_list.append(f_extrapolated)

        plotter.plot_form_factor_np(np.array(X_list), np.array(Z_list), np.array(f_list), f"f_{basis_idx + 1}", "X", tensorBasisNamesDict["rho"][basis_idx], "rho", basis_idx, "dontsave", 999)

        pd_res = pd.DataFrame({
            "X": X_list,
            "Z": Z_list,
            "h": h_list,
            "f": f_list
        })

        pd_res.to_csv(data_path_process + latest_run_dir + f"/{tensorbase_type}_{basis_idx}.txt", index=False)
                




if __name__ == "__main__":
    main()