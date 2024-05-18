import os
import json
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from scipy.optimize import curve_fit
from pyhank import HankelTransform

from data.Dataloader import Dataloader

from basis.BasisTauToRho import BasisTauToRho

from data.AmplitudeHandler import AmplitudeHandler, AmplitudeHandlerFitfunc, AmplitudeHandlerSchlessinger
from pwave.PartialWaveExpansion import PartialWaveExpansion
from visualization.plotting import plot_full_amplitude_np, plot_form_factor_np, plot_form_factor_np_side_by_side

from utils.fitfunctions import *

from numerics.NumericQuadratureFT import NumericQuadratureFT

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

tensorBasisNamesRho = {
    0: "$1 \otimes 1$",
    1: "$\\vec{\sigma} \otimes \\vec{\sigma}$",
    2: "$\\frac{1}{4M^2} (\\vec{\sigma} \dot \\vec{q}) \otimes (\\vec{\sigma} \dot \\vec{q})$",
    3: "$\\frac{1}{4M^2} (\\vec{\sigma} \otimes 1 + 1 \otimes \\vec{\sigma} (\\vec{q} \\times \\vec{p})$",
    4: "$\\frac{1}{4M^2} \\vec{\sigma} \dot (\\vec{q} \\times \\vec{p}) \otimes \\vec{\sigma} \dot (\\vec{q} \\times \\vec{p})$",
}




def plotAmplitudes(dataloader, savefig: bool=False):
    for base_idx in range(5):
        plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.f[base_idx, :, :], "f", tensorBasisNamesTau[base_idx], fig_path=dataloader.data_path + f"/tau_{base_idx + 1}.png", save_plot=savefig)

    for base_idx in range(5):
        plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.F[base_idx, :, :], "F", tensorBasisNamesT[base_idx], fig_path=dataloader.data_path + f"/T_{base_idx + 1}", save_plot=savefig)


def plotAmplitudes_h(dataloader, savefig: bool=False):
    # TODO note both basis elements in contraction
    for base_idx in range(5):
        plot_form_factor_np(dataloader.X, dataloader.Z, dataloader.h[base_idx, :, :], "h", tensorBasisNamesTau[base_idx], fig_path=dataloader.data_path + f"/h_{base_idx + 1}", save_plot=savefig)


def plotFullSymAmplitude(dataloader_qx, dataloader_dqx):
    # Plot Full (Anti-) Symmetric Amplitude
    F_complete = 1/2 * (dataloader_qx.F_flavored - dataloader_dqx.F_flavored)
    plot_full_amplitude_np(dataloader_qx.X, dataloader_qx.Z, F_complete, tensorBasisNamesT, None)


def plotAmplitudesPartialWaveExpandedAndOriginal(X: np.ndarray, Z: np.ndarray, V_l: np.ndarray, V: np.ndarray, data_path: str=None):
    # Build original amplitudes from partial wave expanded ones
    num_Z_check = 21
    Z_check_linspace = np.linspace(np.min(Z), np.max(Z), num_Z_check)

    V_qx_check = np.zeros((V.shape[0], V.shape[1], num_Z_check))

    for basis_idx in range(V_l.shape[0]):
        for X_idx in range(V_l.shape[2]):
            #V_check = np.polynomial.Legendre(V_qx_l[basis_idx, :, X_idx], domain=[-1, 1])
        
            V_qx_check[basis_idx, X_idx, :] = np.polynomial.legendre.legval(Z_check_linspace, V_l[basis_idx, :, X_idx])

        X_qx_extended = np.repeat(X, len(Z))
        Z_qx_extended = np.tile(Z, len(X))

        X_check_extended = np.repeat(X, num_Z_check)
        Z_check_linspace_extended = np.tile(Z_check_linspace, len(X))

        plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V[basis_idx, :, :], "V", "X",
                                         X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "V_{check}", "X",
                                         tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        

def plot_pwave_amp_with_fits(V_qx_l, X_qx, ampHandler: AmplitudeHandler):
    for basis_idx in range(V_qx_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(V_qx_l.shape[1]):

            fit_vals = np.zeros_like(X_qx)
            for X_idx, X in enumerate(X_qx):
                fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)


            axs[0].plot(X_qx, fit_vals, label=f"{l}-wave-fit")
            axs[0].plot(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")

            axs[1].loglog(X_qx, fit_vals, label=f"{l}-wave-fit")
            axs[1].loglog(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()
        plt.show()
        plt.close()


def plot_pwave_amp(f_l, x, xlabel, max_wave: int=None):
    for basis_idx in range(f_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(f_l.shape[1] if max_wave is None else max_wave):
            axs[0].plot(x, f_l[basis_idx, l, :], label=f"{l}-wave")
            axs[1].loglog(x, f_l[basis_idx, l, :], label=f"{l}-wave")
            

        axs[0].set_xlabel(f"${xlabel}$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel(f"$\log {xlabel}$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()
        plt.show()
        plt.close()


def plot_pwave_amp_fits_seperated(X_qx_check, ampHandler: AmplitudeHandler, Ymax: float=None):
    for basis_idx in range(ampHandler.f_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l.shape[1]):
            
            fit_vals = np.zeros_like(X_qx_check)
            for X_idx, X in enumerate(X_qx_check):
                fit_vals[X_idx] = ampHandler.f_l_fit(basis_idx, l, X)

            axs[0].plot(X_qx_check, fit_vals, label=f"{l}-wave-fit")
            axs[1].loglog(X_qx_check, fit_vals, label=f"{l}-wave-fit")

            axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
            axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()

        #axs[0].set_ylim([-0.001, 0.02])

        if Ymax is not None:
            axs[1].set_ylim(Ymax)

        plt.show()
        plt.close()

def plot_pwave_q_amp_fits_seperated(q_check, ampHandler: AmplitudeHandler, Ymax: float=None):
    for basis_idx in range(ampHandler.f_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l.shape[1]):
            
            fit_vals = np.zeros_like(q_check)
            for q_idx, q in enumerate(q_check):
                fit_vals[q_idx] = ampHandler.f_l_q_fit_at(basis_idx, l, q)

            axs[0].plot(q_check, fit_vals, label=f"{l}-wave-fit")
            axs[1].loglog(q_check, fit_vals, label=f"{l}-wave-fit")

            axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
            axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"{l}-wave", linestyle="-.")
            

        axs[0].set_xlabel("$q$")
        axs[0].set_ylabel("$V_{l}(q)$")

        axs[1].set_xlabel("$\log q$")
        axs[1].set_ylabel("$\log V_{l}(q)$")

        axs[0].legend()
        axs[1].legend()

        #axs[0].set_ylim([-0.001, 0.02])

        if Ymax is not None:
            axs[1].set_ylim(Ymax)

        plt.show()
        plt.close()


def plot_pwave_amp_FT(r_grid, ampHandler: AmplitudeHandler):
    for basis_idx in range(ampHandler.f_l_q.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l_q.shape[1]):
            
            V_r = np.zeros_like(r_grid)
            for r_idx, r in enumerate(r_grid):
                V_r[r_idx] = ampHandler.f_l_r_at(basis_idx, l, r)

            axs[0].plot(r_grid, V_r, label=f"{l}-wave")
            axs[1].loglog(r_grid, V_r, label=f"{l}-wave")
            

        axs[0].set_xlabel("$r$")
        axs[0].set_ylabel("$V_{l}(r)$")

        axs[1].set_xlabel("$\log r$")
        axs[1].set_ylabel("$\log V_{l}(r)$")

        axs[0].legend()
        axs[1].legend()

        plt.show()
        plt.close()


def plot_pwave_amp_fits(X_qx_check, ampHandler: AmplitudeHandler, Ymax: float=None):
    for basis_idx in range(ampHandler.f_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l.shape[1]):
            
            fit_vals = np.zeros_like(X_qx_check)
            for X_idx, X in enumerate(X_qx_check):
                fit_vals[X_idx] = ampHandler.f_l_at(basis_idx, l, X)

            axs[0].plot(X_qx_check, fit_vals, label=f"{l}-wave-fit")
            axs[1].loglog(X_qx_check, fit_vals, label=f"{l}-wave-fit")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()

        if Ymax is not None:
            axs[1].set_ylim(Ymax)

        plt.show()
        plt.close()


def plot_pwave_amp_with_interpolation(ampHandler: AmplitudeHandler):
    X_check = np.linspace(ampHandler.X[0], ampHandler.X[-1], 1000)

    for basis_idx in range(ampHandler.f_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l.shape[1]):
            axs[0].plot(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"f_{l}(X)")
            axs[0].plot(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"f_{l}(X) interp")

            axs[1].loglog(ampHandler.X, ampHandler.f_l[basis_idx, l, :], label=f"f_{l}(X)")
            axs[1].loglog(X_check, ampHandler.f_l_interpolation(basis_idx, l, X_check), label=f"f_{l}(X) interp")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$f_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log f_{l}(X)$")

        axs[0].legend()
        axs[1].legend()

        plt.show()
        plt.close()


def plot_pwave_q_amp_with_interpolation(ampHandler: AmplitudeHandler):
    q_check = np.linspace(ampHandler.q[0], ampHandler.q[-1], 1000)

    for basis_idx in range(ampHandler.f_l_q.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(ampHandler.f_l_q.shape[1]):
            axs[0].plot(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")
            axs[0].plot(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"f_{l}(q) interp")

            axs[1].loglog(ampHandler.q, ampHandler.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")
            axs[1].loglog(q_check, ampHandler.f_l_q_interpolation(basis_idx, l, q_check), label=f"f_{l}(q) interp")
            

        axs[0].set_xlabel("$q$")
        axs[0].set_ylabel("$f_{l}(q)$")

        axs[1].set_xlabel("$\log q$")
        axs[1].set_ylabel("$\log f_{l}(q)$")

        axs[0].legend()
        axs[1].legend()

        plt.show()
        plt.close()






def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/"  #sys.argv[1]


    tensorbase_type = "tau"   #sys.argv[2]

    amplitude_isospin = 0   #int(sys.argv[3])

    dq_1_type = "scalar"    #sys.argv[2]
    dq_2_type = "scalar"    #sys.argv[3]

    Z_range = 0.99
    X_range_lower = 0



    # Construct directory string fitting specs
    data_path = data_base_path + f"/BASE-{tensorbase_type}_I-{amplitude_isospin}_DQ-{dq_1_type}-{dq_2_type}/"



    # Load data files
    # TODO error happens for S1 and A1 when doing tau --> T, seems correct for T --> tau
    dataloader_qx = Dataloader(data_base_path, tensorbase_type, amplitude_isospin, "quark_exchange", dq_1_type, dq_2_type, Z_range, X_range_lower)
    dataloader_dqx = Dataloader(data_base_path, tensorbase_type, amplitude_isospin, "diquark_exchange", dq_1_type, dq_2_type, Z_range, X_range_lower)




    # Basis Transform to rho basis
    #   V_qx has shape (basis, X, Z)
    X_qx, Z_qx, V_qx = BasisTauToRho.build_rho_base_dressing_functions_from_tau(dataloader_qx)




    # Plot amplitudes
    #plotAmplitudes_h(dataloader_qx, savefig=True)
    #plotAmplitudes(dataloader_qx, savefig=True)

    #plotAmplitudes_h(dataloader_dqx)
    #plotAmplitudes(dataloader_dqx)

    #plotFullSymAmplitude(dataloader_qx, dataloader_dqx)





    # The FT workaround
    fitonly = True
    fitfunc, p0, bounds = yukawa_2_exponentials_v3_fitparams()
    #fitfunc, p0, bounds = yukawa_2_exponentials_fitparams()
    #fitfunc, p0, bounds = yukawa_poly_exponentials_evenodd_fitparams()
    ampHandler = AmplitudeHandlerFitfunc(X_qx, Z_qx, V_qx, fitfunc, p0, bounds)

    ########################### (1) ##############################
    #Perform Partial Wave Expansion to get f_l(X) from f(X, Z)         (i.e. get V_qx_l[basis, l, X]   from    V_qx[basis, X, Z])

    degree_pwave_exp = 4
    ampHandler.partial_wave_expand(degree_pwave_exp)

    # Check result of partial wave expansion
    #plotAmplitudesPartialWaveExpandedAndOriginal(ampHandler.X, ampHandler.Z, ampHandler.f_l, V_qx, dataloader_qx.data_path)

    # Plot partial wave amplitudes
    #plot_pwave_amp(ampHandler.f_l, ampHandler.X, "X")



    
    ########################### (2) ##############################
    # Fit f_l(X) for X >>        (i.e. fit V_qx_l[basis, l, X])  for fixed basis, l
    # and interpolate for X <= X_max


    # V_qx_l_fitcoeff has shape (basis, l, coeffs)
    ampHandler.fit_large_X_behaviour(min_X=0)

    ampHandler.interpolate_in_X()
            

    # Plot partial wave amplitudes for X in (Log-Log) Plot
    #plot_pwave_amp_with_fits(ampHandler.f_l, ampHandler.X, ampHandler)


    # Check Fit behaviour for large X
    X_qx_check = np.linspace(0, 10, 1000)
    #plot_pwave_amp_fits_seperated(X_qx_check, ampHandler)



    ########################### (3) ##############################
    # Plug things together
    #   -> query interpolation for X <= X_max
    #   -> query fit for X > X_max
    
    # Check interpolation
    #plot_pwave_amp_with_interpolation(ampHandler)

    # Check reconstructed Amplitude
    X_qx_extended = np.repeat(ampHandler.X, len(ampHandler.Z))
    Z_qx_extended = np.tile(ampHandler.Z, len(ampHandler.X))

    X_qx_reconst = np.linspace(0, 1, 20)
    Z_qx_reconst = np.linspace(-1, 0.9, 15)
    V_qx_reconst = np.zeros((V_qx.shape[0], len(X_qx_reconst), len(Z_qx_reconst)))

    X_qx_extended_reconst = np.repeat(X_qx_reconst, len(Z_qx_reconst))
    Z_qx_extended_reconst = np.tile(Z_qx_reconst, len(X_qx_reconst))

    for basis_idx in range(V_qx.shape[0]):
        for X_idx in range(X_qx_reconst.shape[0]):
            for Z_idx in range(Z_qx_reconst.shape[0]):
                V_qx_reconst[basis_idx, X_idx, Z_idx] = ampHandler.f_at(basis_idx, X_qx_reconst[X_idx], Z_qx_reconst[Z_idx])

    for basis_idx in range(V_qx.shape[0]):
        #plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V_qx[basis_idx, :, :], "V_{orig}", "X",
        #                                 X_qx_extended_reconst, Z_qx_extended_reconst, V_qx_reconst[basis_idx, :, :], "V_{reconst}", "X",
        #                                 tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        pass
                



    ########################### (4) ##############################
    # Knowing f(X, Z) for all X, get access to f(q, Z) via q(X, Z)
                
    # Although the results seem weird, they make sense when thinking about the
    # transformation X -> q to kind of stretch the X plot along the diagonal to get the q plot
    
    # Already build q grid using transformer --> Doesn't work too well
    #n_points = 4096
    #max_radius = 1000
    #transformer = HankelTransform(order=0, max_radius=max_radius, n_points=n_points)

    # Check Amplitude in q
    #q_qx_reconst = np.logspace(-5, 1, 500)
    q_qx_reconst = np.linspace(0, 10, 300)
    #q_qx_reconst = transformer.kr

    q_qx_extended_reconst = np.repeat(q_qx_reconst, len(Z_qx_reconst))
    Z_qx_q_extended_reconst = np.tile(Z_qx_reconst, len(q_qx_reconst))

    V_qx_q_reconst = np.zeros((V_qx.shape[0], len(q_qx_reconst), len(Z_qx_reconst)))

    for basis_idx in range(V_qx.shape[0]):
        for q_idx in range(q_qx_reconst.shape[0]):
            for Z_idx in range(Z_qx_reconst.shape[0]):
                V_qx_q_reconst[basis_idx, q_idx, Z_idx] = ampHandler.f_q_at(basis_idx, q_qx_reconst[q_idx], Z_qx_reconst[Z_idx], fitonly=fitonly)


    for basis_idx in range(V_qx.shape[0]):
        #plot_form_factor_np_side_by_side(X_qx_extended_reconst, Z_qx_extended_reconst, V_qx_reconst[basis_idx, :, :], "V_{reconst}", "X",
        #                                 q_qx_extended_reconst, Z_qx_q_extended_reconst, V_qx_q_reconst[basis_idx, :, :], "V_{reconst}", "q",
        #                                 tensorBasisNamesRho[basis_idx], dressing_f2_params="(q, Z)", fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        pass
        

    ########################### (5) ##############################
    #Perform Partial Wave Expansion to get f_l(q) from f(q, Z)
    
    degree_pwave_exp_q = 4
    ampHandler.partial_wave_expand_q(degree_pwave_exp_q, q_qx_reconst, Z_qx_reconst, fitonly=fitonly)       # Note q_qx_reconst should be the one from the transformer
    ampHandler.interpolate_in_q()

    # Check result of partial wave expansion
    #plotAmplitudesPartialWaveExpandedAndOriginal(ampHandler.q, Z_qx_reconst, ampHandler.f_l_q, V_qx_q_reconst, dataloader_qx.data_path)    # TODO plot in restricted grid

    # Plot partial wave amplitudes for q in (Log-Log) Plot
    #plot_pwave_amp(ampHandler.f_l_q, ampHandler.q, "q")






    ########################### (6) ##############################
    # Fourier Transform f_l(q) -> f_l(r)
    r_grid = np.linspace(0, 10, 100)

    # Fourier Transform via Numeric Quadrature
    quad_ft = NumericQuadratureFT(100, 20)

    f_l_r = np.zeros((ampHandler.f_l_q.shape[0], ampHandler.f_l_q.shape[1], len(r_grid)))
    for basis_idx in range(ampHandler.f_l_q.shape[0]):
        for l in range(ampHandler.f_l_q.shape[1]):
            print(f"Fourier Transforming {}")
            f_l_r[basis_idx, l, :] = quad_ft.fourierTransform(lambda q : ampHandler.f_l_q_at(basis_idx, l, q), r_grid)

    plot_pwave_amp(f_l_r, r_grid, "r")


    exit()

    # (6.1) Fit known function to analytically do FT
    q_fitfunc, q_p0, q_bounds, FT_q_fitfunc = yukawa_poly_exponentials_v3_fitparams()
    ampHandler.fit_q_pwaves(q_fitfunc, q_p0, q_bounds, FT_q_fitfunc)


    # (6.2) Plot knonw functions and fit
    plot_pwave_q_amp_fits_seperated(X_qx_check, ampHandler)


    # (6.3) Use fit-coefficients to get fourier transformed set of functions
    r_grid = np.linspace(0.05, 2, 1000)
    plot_pwave_amp_FT(r_grid, ampHandler)


    exit()
    #transformer2 = HankelTransform(order=0, k_grid=ampHandler.q)
    

    for basis_idx in range(ampHandler.f_l_q.shape[0]):
        plt.figure()
        for l in range(ampHandler.f_l_q.shape[1]):

            #V_q = ampHandler.f_l_q[basis_idx, l, :]  # = Vtwidle_nu
            V_q = ampHandler.f_l_q_at(basis_idx, l, transformer.kr)
            V_r = -2 * transformer.iqdht(transformer.v * V_q)         # = IHT(nu * Vtwidle_nu)

            plt.plot(transformer.r, V_r, label=f"{l}-wave")
        
        plt.show()
    



    exit()

    # Partial Wave Expansion
    #       X_qx is already unique, Z_qx can be replaced by wave (s, p, d, ... <=> l = 0, 1, 2, ...)

    # 1.) Calculate q^2 from X values and nucleon mass for each combination of X and Z
    # TODO load nucleon mass via dataloader from spec.json
    M_nucleon = 0.94

    # q2 has shape (X, Z)
    q2 = np.zeros((len(X_qx), len(Z_qx)))
    q2[:, :] = np.square(M_nucleon) * (2.0 * X_qx[:, None]) * (1.0 - Z_qx[None, :])

    """
    X_qx_extended = np.repeat(X_qx, len(Z_qx))
    Z_qx_extended = np.tile(Z_qx, len(X_qx))
    plot_form_factor_np(X_qx_extended, Z_qx_extended, q2, "q2", "q2")
    """

    # 2.) Set parameters
    degree_fit = 3

    # V_qx_l has shape (basis, l, X)
    V_qx_l = np.zeros((V_qx.shape[0], degree_fit+1, V_qx.shape[1]))

    for basis_idx in range(V_qx.shape[0]):
        V_qx_l[basis_idx, :] = PartialWaveExpansion(V_qx[basis_idx, :], X_qx, Z_qx, degree_fit).get_f_x()

    # 4.) Check result of partial wave expansion
    plotAmplitudesPartialWaveExpandedAndOriginal(X_qx, Z_qx, V_qx_l, V_qx, dataloader_qx.data_path)




    # Fourier Transformation

    #   1.) Fit partial wave expanded amplitudes with functions of known fourier transform  (Yukawa Potential and additionally sum of exponentials for small q2)
            
    #       V_qx_l_fitcoeff has shape (basis, l, coeffs)     
    V_qx_l_fitcoeff = np.zeros((V_qx_l.shape[0], V_qx_l.shape[1], 7))

    for basis_idx in range(V_qx_l.shape[0]):
        for l in range(V_qx_l.shape[1]):
            popt, pcov = curve_fit(yukawa_potential_exp_sum, X_qx, V_qx_l[basis_idx, l, :], 
                                   p0=[1, 1, -1, -2, -1, -2, -0.1], 
                                   bounds=([-np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, 0, np.inf, 0, np.inf]), 
                                   maxfev=10000)
            
            V_qx_l_fitcoeff[basis_idx, l, :] = popt
            

    #   2.) Plot partial wave amplitudes for X in (Log-Log) Plot
    for basis_idx in range(V_qx_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(V_qx_l.shape[1]):
            axs[0].plot(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[0].plot(X_qx, yukawa_potential_exp_sum(X_qx, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

            axs[1].loglog(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[1].loglog(X_qx, yukawa_potential_exp_sum(X_qx, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

        #axs[1].loglog(X_qx, 1.0/np.square(X_qx), label="1/X^2")
        #axs[1].loglog(X_qx, np.exp(X_qx), label="exp(X)")

        axs[0].legend()
        axs[1].legend()
        plt.show()
        plt.close()



    # 3.) Plot Fourier Transformed Amplitude
    r_linspace = np.linspace(0.1, 1, 500)
    for basis_idx in range(V_qx_l.shape[0]):
        plt.figure()

        for l in range(V_qx_l.shape[1]):
            plt.plot(r_linspace, FT_yukawa_potential_exp_sum(r_linspace, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave")

        plt.legend()
        plt.show()
        plt.close()






    # Legacy Prostprocessing
    # Note: qx run_13, dqx run_9 perfectly work for (anti) symmetric result
    #plot_dirac_space_dressing_functions(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", "quark_exchange", savefig=True)
    #plot_dirac_space_dressing_functions(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", "diquark_exchange", savefig=True)

    #full_amplitude_symmetric_basis(data_base_path + f"/BASE-T_I-{0}_DQ-{dq_1_type}-{dq_2_type}/", "T", 0, dq_1_type, dq_2_type, Z_range, X_range_lower, savefig=True)

    # Plot tau basis result in symmetric basis
    #plot_dirac_space_dressing_functions_tau_to_T(data_path, "quark_exchange")


    #for base_idx, pd_ff in enumerate(dataloader_qx.pd_dirac_space_form_factors_list):
    #    plot_form_factor(pd_ff, tensorBasisNamesT[base_idx], base_idx, data_path + f"/{dataloader_qx.process_type}_F{base_idx + 1}", False)






if __name__ == "__main__":
    main()
    exit()