import os
import json
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from scipy.optimize import curve_fit

from data.Dataloader import Dataloader

from basis.BasisTauToRho import BasisTauToRho

from data.AmplitudeHandler import AmplitudeHandler
from pwave.PartialWaveExpansion import PartialWaveExpansion
from visualization.plotting import plot_full_amplitude_np, plot_form_factor_np, plot_form_factor_np_side_by_side

from utils.fitfunctions import *

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
    1: "$\pmb{\sigma} \otimes \pmb{\sigma}$",
    2: "$\frac{1}{4M^2} (\pmb{\sigma \dot q}) \otimes (\pmb{\sigma \dot q})$",
    3: "$\frac{1}{4M^2} (\pmb{\sigma} \otimes 1 + 1 \otimes \pmb{\sigma} (\pmb{q \times p})$",
    4: "$\frac{1}{4M^2} \pmb{\sigma} \dot (\pmb{q \times p}) \otimes \pmb{\sigma} \dot (\pmb{q \times p})$",
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

        plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V[basis_idx, :, :], "V",
                                         X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "V_{check}", 
                                         tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        

def plot_pwave_amp_with_fits(V_qx_l, X_qx, V_qx_l_fitcoeff):
    for basis_idx in range(V_qx_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(V_qx_l.shape[1]):
            axs[0].plot(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[0].plot(X_qx, yukawa_potential_exp_sum(X_qx, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

            axs[1].loglog(X_qx, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[1].loglog(X_qx, yukawa_potential_exp_sum(X_qx, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()
        plt.show()


def plot_pwave_amp_fits(X_qx_check, V_qx_l_fitcoeff, Ymax: float=None):
    for basis_idx in range(V_qx_l_fitcoeff.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(V_qx_l_fitcoeff.shape[1]):
            axs[0].plot(X_qx_check, yukawa_potential_exp_sum(X_qx_check, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

            axs[1].loglog(X_qx_check, yukawa_potential_exp_sum(X_qx_check, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")
            

        axs[0].set_xlabel("$X$")
        axs[0].set_ylabel("$V_{l}(X)$")

        axs[1].set_xlabel("$\log X$")
        axs[1].set_ylabel("$\log V_{l}(X)$")

        axs[0].legend()
        axs[1].legend()

        if Ymax is not None:
            axs[1].set_ylim(Ymax)

        plt.show()


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
    ampHandler = AmplitudeHandler(X_qx, Z_qx, V_qx)

    ########################### (1) ##############################
    #Perform Partial Wave Expansion to get f_l(X) from f(X, Z)         (i.e. get V_qx_l[basis, l, X]   from    V_qx[basis, X, Z])

    degree_pwave_exp = 6
    ampHandler.partial_wave_expand(degree_pwave_exp)

    # Check result of partial wave expansion
    #plotAmplitudesPartialWaveExpandedAndOriginal(ampHandler.X, ampHandler.Z, ampHandler.f_l, V_qx, dataloader_qx.data_path)



    
    ########################### (2) ##############################
    #Fit f_l(X) for X >>        (i.e. fit V_qx_l[basis, l, X])  for fixed basis, l

    #       V_qx_l_fitcoeff has shape (basis, l, coeffs)
    ampHandler.fit_large_X_behaviour()
            

    #       Plot partial wave amplitudes for X in (Log-Log) Plot
    #plot_pwave_amp_with_fits(ampHandler.f_l, ampHandler.X, ampHandler.f_l_fitcoeff)


    # Check Fit behaviour for large X           --> TODO better fit
    X_qx_check = np.linspace(0, 1E3, 10000)
    #plot_pwave_amp_fits(X_qx_check, ampHandler.f_l_fitcoeff)




    ########################### (3) ##############################
    # Plug things together
    #   -> interpolation for X <= X_max
    #   -> query fit for X > X_max
    ampHandler.interpolate_in_X()
    
    # Check interpolation
    #plot_pwave_amp_with_interpolation(ampHandler)

    # Check reconstructed Amplitude
    X_qx_extended = np.repeat(ampHandler.X, len(ampHandler.Z))
    Z_qx_extended = np.tile(ampHandler.Z, len(ampHandler.X))

    X_qx_reconst = np.linspace(0.05, 1.5, 20)
    Z_qx_reconst = np.linspace(-1, 1, 15)
    V_qx_reconst = np.zeros((V_qx.shape[0], len(X_qx_reconst), len(Z_qx_reconst)))

    X_qx_extended_reconst = np.repeat(X_qx_reconst, len(Z_qx_reconst))
    Z_qx_extended_reconst = np.tile(Z_qx_reconst, len(X_qx_reconst))

    for basis_idx in range(V_qx.shape[0]):
        for X_idx in range(X_qx_reconst.shape[0]):
            for Z_idx in range(Z_qx_reconst.shape[0]):
                V_qx_reconst[basis_idx, X_idx, Z_idx] = ampHandler.f_at(basis_idx, X_qx_reconst[X_idx], Z_qx_reconst[Z_idx])

    for basis_idx in range(V_qx.shape[0]):
        plot_form_factor_np_side_by_side(X_qx_extended, Z_qx_extended, V_qx[basis_idx, :, :], "V_orig",
                                         X_qx_extended_reconst, Z_qx_extended_reconst, V_qx_reconst[basis_idx, :, :], "V_reconst", 
                                         tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)

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



    # 3.) Plot Fourier Transformed Amplitude
    r_linspace = np.linspace(0.1, 1, 500)
    for basis_idx in range(V_qx_l.shape[0]):
        plt.figure()

        for l in range(V_qx_l.shape[1]):
            plt.plot(r_linspace, FT_yukawa_potential_exp_sum(r_linspace, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave")

        plt.legend()
        plt.show()






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