import math

import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt




def plot_form_factor(pd_tau: pd.DataFrame, tensor_basis_elem: str, tensor_basis_elem_idx: int, fig_path=None, save_plot=False):
    fig = plt.figure(figsize=(10, 9))

    fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

    function_name = "h"

    # Subplot real h
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title(f"$\Re({function_name}(X, Z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Re({function_name})$")


    # Subplot imag h
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title(f"$\Im({function_name}(X, Z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Im({function_name})$")



    function_name = "f"

    # Subplot real f
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_title(f"$\Re({function_name}(X, Z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.real(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Re({function_name})$")

    # Subplot imag f
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.set_title(f"$\Im({function_name}(X, Z))$")
    ax.plot_trisurf(pd_tau["X"], pd_tau["Z"], np.imag(pd_tau[function_name]), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Im({function_name})$")


    if(fig_path is not None and save_plot):
        plt.savefig(fig_path, dpi=200)
    plt.show()


def plot_full_amplitude(pd_ff_list, tensor_basis_names, fig_path: str, savefig: bool=False):
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)

    for basis_idx, pd_ff in enumerate(pd_ff_list):
        ax = fig.add_subplot(1, 5, basis_idx + 1, projection='3d')
        ax.set_title(tensor_basis_names[basis_idx])
        ax.plot_trisurf(pd_ff["X"], pd_ff["Z"], np.real(pd_ff["f"]), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$F_{basis_idx}(X, Z)$")


    wspace = 0.4   # the amount of width reserved for blank space between subplots

    fig.subplots_adjust(wspace=wspace, top=0.95, bottom=0.05, left=0.05, right=0.95)

    if(savefig):
        plt.savefig(fig_path, dpi=600)
    plt.show()


def plot_result(pd_tau, idx_selector, img_path=None, perf_norm=False):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(pd_tau["X"][idx_selector], pd_tau["Z"][idx_selector], ((1.0/(2 * (2 * math.pi)**4))**2 if perf_norm else 1) * pd_tau["|scattering_amp|2"][idx_selector])
    ax.set_xlabel("X / 1")
    ax.set_ylabel("Z / 1")
    ax.set_zlabel("$|M|^2$")
    if img_path is not None:
        plt.savefig(img_path + "/NN-Scattering.png", dpi=400)
    plt.show()





def plot_full_amplitude_np(X: np.ndarray, Z: np.ndarray, F: np.ndarray, tensor_basis_names, fig_path: str):
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)

    for basis_idx in range(F.shape[0]):
        ax = fig.add_subplot(1, 5, basis_idx + 1, projection='3d')
        ax.set_title(tensor_basis_names[basis_idx])
        ax.plot_trisurf(X, Z, F[basis_idx, :, :].flatten(), cmap=cm.coolwarm)
        ax.set_xlabel("$X$")
        ax.set_ylabel("$Z$")
        ax.set_zlabel(f"$F_{basis_idx}(X, Z)$")


    wspace = 0.4   # the amount of width reserved for blank space between subplots

    fig.subplots_adjust(wspace=wspace, top=0.95, bottom=0.05, left=0.05, right=0.95)

    if(fig_path is not None):
        plt.savefig(fig_path, dpi=600)
    plt.show()


def plot_form_factor_np(X: np.ndarray, Z: np.ndarray, dressing_f: np.ndarray, dressing_f_name: str, tensor_basis_elem: str, fig_path=None, save_plot=False):
    fig = plt.figure(figsize=(10, 5))

    fig.suptitle("Tensor Basis Element " + tensor_basis_elem)

    # Subplot real h
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title(f"$\Re({dressing_f_name}(X, Z))$")
    ax.plot_trisurf(X, Z, np.real(dressing_f.flatten()), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Re({dressing_f_name})$")


    # Subplot imag h
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title(f"$\Im({dressing_f_name}(X, Z))$")
    ax.plot_trisurf(X, Z, np.imag(dressing_f.flatten()), cmap=cm.coolwarm)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Z$")
    ax.set_zlabel(f"$\Im({dressing_f_name})$")


    if(fig_path is not None and save_plot):
        plt.savefig(fig_path, dpi=200)
    plt.show()