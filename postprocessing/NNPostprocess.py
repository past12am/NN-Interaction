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

from visualization.plotting import plot_full_amplitude_np, plot_form_factor_np


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

        plot_form_factor_np(X_qx_extended, Z_qx_extended, V[basis_idx, :, :], "V", tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        plot_form_factor_np(X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "V_{check}", tensorBasisNamesRho[basis_idx], fig_path=data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)


def yukawa_potential(q2, c, L2):
    return c/(q2 + L2)


def yukawa_potential_exp_sum(q2, c, L2, a0, b0, a1, b1, offset):
    return yukawa_potential(q2, c, L2) + a0 * np.exp(b0 * q2) + a1 * np.exp(b1 * q2) + offset



def FT_yukawa_potential_exp_sum(r, c, L2, a0, b0, a1, b1, offset):
    """ We used the FT as -1/(2 pi)^3 Integral_-inf^inf{dq f(q^2) e^{i q.r}}"""

    return - c/(4.0 * np.pi) * np.exp(-np.sqrt(L2) * np.abs(r)) / (np.abs(r)) \
           - 1/(8 * np.sqrt(np.power(np.pi, 3))) * (a0/np.sqrt(np.power(-b0, 3)) * np.exp(np.square(r)/(4.0 * b0)) + a1/np.sqrt(np.power(-b1, 3)) * np.exp(np.square(r)/(4.0 * b1)))


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







    # Partial Wave Expansion
    #       X_qx is already unique, Z_qx can be replaced by wave (s, p, d, ... <=> l = 0, 1, 2, ...)

    # 1.) Calculate q^2 from X values and nucleon mass for each of the l (~ Z_values)
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
        for X_idx in range(V_qx.shape[1]):

            # 3.) For each Basis, fit legendre polynomials on Z (= cos(Theta))
            legendreFit = np.polynomial.legendre.Legendre.fit(Z_qx, V_qx[basis_idx, X_idx, :], deg=degree_fit, domain=[-1, 1])
            V_qx_l[basis_idx, :, X_idx] = legendreFit.convert().coef

            """
            plt.figure()
            plt.plot(Z_qx, V_qx[basis_idx, X_idx, :], label="Amplitudes")

            x_poly, y_poly = legendreFit.linspace()
            plt.plot(x_poly, y_poly, label="Legendre Fit")

            plt.xlim(np.min(Z_qx), np.max(Z_qx))

            plt.legend()
            plt.show()
            """

    # 4.) Check result of partial wave expansion
    #plotAmplitudesPartialWaveExpandedAndOriginal(X_qx, Z_qx, V_qx_l, V_qx, dataloader_qx.data_path)




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