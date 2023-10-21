import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extrapolation.SchlessingerPointMethod import SchlessingerPointMethod
from visualization.plotM2 import plot_result

def plot_abs_squared():
    z_range = 0.7
    tau_range = 0.001
    pd_tau = pd.read_csv("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data_run4_f2/tau_0.txt")
    idx_selector = np.where(np.logical_and(np.abs(np.array(pd_tau["z"])) < z_range, np.array(pd_tau["tau"]) > tau_range))[0]

    plot_result(pd_tau, idx_selector, "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data_run4_f2/", True)


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


plot_abs_squared()