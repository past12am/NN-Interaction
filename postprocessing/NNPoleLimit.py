
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


def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/data/"
    output_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NN-Interaction-Data/postprocess-output/"

    tensorbase_type = "tau"

    dq_1_type = "scalar"
    dq_2_type = "scalar"

    process_type = "quark_exchange"

    Z_range = 0.9
    X_range_lower = 0



    # Load pandas and paths
    data_path = data_base_path + f"/BASE-{tensorbase_type}_I-0_DQ-{dq_1_type}-{dq_2_type}/"
    data_path_process = data_path + f"/{process_type}/"
    latest_run_dir = Dataloader.find_latest_run_dir(data_path_process)

    data_path_process_latest = data_path_process + latest_run_dir

    pd_data_list = Dataloader.load_dirac_space_form_factors(data_path_process + latest_run_dir + "/", tensorbase_type, Z_range, X_range_lower)



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
        
        for X in X_vals:
            for Z in Z_vals:
                
                pd_cur = pd_data[(pd_data["Z"] == Z) & (pd_data["X"] == X)]

                eps = pd_cur["eps"]
                h = pd_cur["h"]

                popt, pcov = curve_fit(parabola, eps, h)
                eps_space = np.linspace(0, np.max(eps), 20)

                #plt.figure()
                #plt.scatter(eps, h, label=f"Data: X={X}, Z={Z}")
                #plt.plot(eps_space, parabola(eps_space, *popt), label="fit", c="orange")
                #plt.legend()
                #plt.show()


                # Extrapolate via fit
                h_extrapolated = parabola(0, *popt)
                
                X_list.append(X)
                Z_list.append(Z)
                h_list.append(h_extrapolated)

        plotter.plot_form_factor_np(np.array(X_list), np.array(Z_list), np.array(h_list), f"h_{basis_idx + 1}", "X", tensorBasisNamesDict["rho"][basis_idx], "rho", basis_idx, "dontsave", 999)
                




if __name__ == "__main__":
    main()