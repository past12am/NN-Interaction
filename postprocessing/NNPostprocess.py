import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from extrapolation.SchlessingerPointMethod import SchlessingerPointMethod
from visualization.plotM2 import plot_result, plot_form_factor


tensorBasisNames = {
    0: "$1 \otimes 1$",
    1: "$1 \otimes \gamma_{\\nu} P^{\\nu}$",
    2: "$\gamma^{\mu} \otimes \gamma^{\mu}$",
    3: "$\gamma^{\mu} \otimes [\gamma^{\mu}, \gamma_{\\nu} P^{\\nu}]$",
    4: "$\gamma_{5} \otimes \gamma_{5}$",
    5: "$\gamma_{5} \otimes \gamma_{5} \gamma_{\\nu} P^{\\nu}$",
    6: "$\gamma_{5} \gamma^{\mu} \otimes \gamma_{5} \gamma^{\mu}$",
    7: "$\gamma_{5} \gamma^{\mu} \otimes \gamma_{5} [\gamma^{\mu}, \gamma_{\\nu} P^{\\nu}]$",
}


def plot_abs_squared():
    z_range = 0.7
    tau_range = 0.001
    pd_tau = pd.read_csv("/data_run8/tau_0.txt")
    idx_selector = np.where(np.logical_and(np.abs(np.array(pd_tau["z"])) < z_range, np.array(pd_tau["tau"]) > tau_range))[0]

    plot_result(pd_tau, idx_selector, "/data_run8/", True)


def plot_form_factors(data_path: str, plot_dir: str, z_range: float, X_range: float, save_plot=False):

    pd_tau_list = list()
    for tauIdx in range(8):
        pd_tau = pd.read_csv(data_path + f"/tau_{tauIdx}.txt")
        pd_tau = pd_tau.applymap(lambda s: complex(s.replace('i', 'j')) if(isinstance(s, str)) else s)

        a_closest_m1 = pd_tau["a"][np.argmin(np.abs(pd_tau["a"] - 1))]
        pd_tau = pd_tau.where(pd_tau["a"] == a_closest_m1).dropna().reindex()
        print(a_closest_m1)

        idx_selector = np.where(np.logical_and(np.abs(np.array(pd_tau["z"])) < z_range, np.array(pd_tau["X"]) > X_range))[0]
        pd_tau = pd_tau.iloc[idx_selector]

        pd_tau_list.append(pd_tau)

    for tauIdx, pd_tau in enumerate(pd_tau_list):
        plot_form_factor(pd_tau, tensorBasisNames[tauIdx], tauIdx, data_path + "/" + plot_dir + "/", save_plot)



def test_func(x_grid):
    return 1/(x_grid + 100)

def test_func(x_grid):
    return np.exp(x_grid)

def schlessinger_test():
    x_grid = np.linspace(0, 15, 500)
    source_vals = test_func(x_grid)

    selected_indices = range(80,100,5)
    sp = SchlessingerPointMethod(x_grid[selected_indices], source_vals[selected_indices])

    schlessinger_vals = list()
    for x in x_grid:
        schlessinger_vals.append(sp.calc_value_at(x))


    plt.figure()
    plt.ylim(-1000, 1000)
    plt.plot(x_grid, source_vals, label="Original")
    plt.plot(x_grid, schlessinger_vals, label="SPM")
    plt.show()



z_range = 1
X_range = 0

plot_form_factors(sys.argv[1], "plots_z10", z_range, X_range, False)
#plot_form_factors("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteractionPython/data/run0", "", z_range, X_range, function_name, False)
