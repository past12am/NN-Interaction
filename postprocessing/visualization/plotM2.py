import math

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt




def plot_form_factor(pd_tau: pd.DataFrame, tensor_basis_elem: str, tensor_basis_elem_idx: int, path=None, save_plot=False):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(10, 9))

    fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

    function_name = "h"

    # Subplot real h
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title(f"$\Re({function_name}(X, z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$z$")
    ax.set_zlabel(f"$\Re({function_name})$")


    # Subplot imag h
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title(f"$\Im({function_name}(X, z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$z$")
    ax.set_zlabel(f"$\Im({function_name})$")



    function_name = "f"

    # Subplot real f
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title(f"$\Re({function_name}(X, z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$z$")
    ax.set_zlabel(f"$\Re({function_name})$")

    # Subplot imag f
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_title(f"$\Im({function_name}(X, z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$z$")
    ax.set_zlabel(f"$\Im({function_name})$")


    if(path is not None and save_plot):
        plt.savefig(path + f"/f_{tensor_basis_elem_idx}.png", dpi=200)
    plt.show()


def plot_result(pd_tau, idx_selector, img_path=None, perf_norm=False):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(pd_tau["X"][idx_selector], pd_tau["z"][idx_selector], ((1.0/(2 * (2 * math.pi)**4))**2 if perf_norm else 1) * pd_tau["|scattering_amp|2"][idx_selector])
    ax.set_xlabel("X / 1")
    ax.set_ylabel("z / 1")
    ax.set_zlabel("$|M|^2$")
    if img_path is not None:
        plt.savefig(img_path + "/NN-Scattering.png", dpi=400)
    plt.show()
