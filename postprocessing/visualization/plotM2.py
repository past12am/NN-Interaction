import math

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt




def plot_form_factor(pd_tau: pd.DataFrame, tensor_basis_elem: str, tensor_basis_elem_idx: int, path=None):
    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

    # Subplot real
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("$\Re(f(\\tau, z))$")
    ax.plot_trisurf(pd_tau["tau"], pd_tau["z"], np.real(pd_tau["f"]), cmap=cm.coolwarm)
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$z$")
    ax.set_zlabel("$\Re(f)$")


    # Subplot imag
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("$\Im(f(\\tau, z))$")
    ax.plot_trisurf(pd_tau["tau"], pd_tau["z"], np.imag(pd_tau["f"]), cmap=cm.coolwarm)
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$z$")
    ax.set_zlabel("$\Im(f)$")

    if(path is not None):
        plt.savefig(path + f"/f_{tensor_basis_elem_idx}.png", dpi=200)
    plt.show()


def plot_result(pd_tau, idx_selector, img_path=None, perf_norm=False):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(pd_tau["tau"][idx_selector], pd_tau["z"][idx_selector], ((1.0/(2 * (2 * math.pi)**4))**2 if perf_norm else 1) * pd_tau["|scattering_amp|2"][idx_selector])
    ax.set_xlabel("tau / 1")
    ax.set_ylabel("z / 1")
    ax.set_zlabel("$|M|^2$")
    if img_path is not None:
        plt.savefig(img_path + "/NN-Scattering.png", dpi=400)
    plt.show()
