import os
import json
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

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







    # partial wave expansion
    #       X_qx is already unique, Z_qx can be replaced by wave (s, p, d, ... <=> l = 0, 1, 2, ...)

    degree_fit = 12

    # V_qx_l has shape (basis, l, X)
    V_qx_l = np.zeros((V_qx.shape[0], degree_fit+1, V_qx.shape[1]))

    for basis_idx in range(V_qx.shape[0]):
        for X_idx in range(V_qx.shape[1]):
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

    # Build original amplitudes from partial wave expanded ones
    num_Z_check = 21
    Z_check_linspace = np.linspace(np.min(Z_qx), np.max(Z_qx), num_Z_check)

    V_qx_check = np.zeros((V_qx.shape[0], V_qx.shape[1], num_Z_check))

    for basis_idx in range(V_qx_l.shape[0]):
        for X_idx in range(V_qx_l.shape[2]):
            #V_check = np.polynomial.Legendre(V_qx_l[basis_idx, :, X_idx], domain=[-1, 1])
        
            V_qx_check[basis_idx, X_idx, :] = np.polynomial.legendre.legval(Z_check_linspace, V_qx_l[basis_idx, :, X_idx])

        X_qx_extended = np.repeat(X_qx, len(Z_qx))
        Z_qx_extended = np.tile(Z_qx, len(X_qx))

        X_check_extended = np.repeat(X_qx, num_Z_check)
        Z_check_linspace_extended = np.tile(Z_check_linspace, len(X_qx))

        plot_form_factor_np(X_qx_extended, Z_qx_extended, V_qx[basis_idx, :, :], "V", tensorBasisNamesRho[basis_idx], fig_path=dataloader_qx.data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)
        plot_form_factor_np(X_check_extended, Z_check_linspace_extended, V_qx_check[basis_idx, :, :], "V_{check}", tensorBasisNamesRho[basis_idx], fig_path=dataloader_qx.data_path + f"/rho_{basis_idx + 1}.png", save_plot=False)




    exit()
    # TODO check that Z_qx_zero_idx needs to be always scalar (--> one Z-value for each iteration --> "replace Z by l")
    Z_qx_zero_idx = np.argwhere(Z_qx == 0)[0][0]

    # (wave_idx, X_idx, tensorbase_idx)
    X_partwaves = [X_qx]
    Z_partwaves = [0]
    Vi_partwaves = np.array([V_qx[:, Z_qx_zero_idx, :]])


    # TODO load nucleon mass via dataloader from spec.json
    M_nucleon = 0.94


    # Fourier Transform V
    # l ... partial wave
    Vi_r = np.zeros_like(Vi_partwaves, dtype="complex")

    phi_linspace = np.linspace(0, 2*np.pi, 6)
    z_linspace = np.linspace(-1, 1, 7)

    q_3_unitvec = np.zeros((Vi_partwaves.shape[1], 3))

    q_idx = 0
    for z in z_linspace:
        for phi in phi_linspace:
            unit_3vec = np.array([np.sqrt(1 - np.square(z)) * np.sin(phi), np.sqrt(1 - np.square(z)) * np.cos(phi), z])
            q_idx += 1

    for l, (X_partwave, Z_partwave, V_partwave) in enumerate(zip(X_partwaves, Z_partwaves, Vi_partwaves)):
        q = M_nucleon * np.sqrt(2.0 * X_partwave * (1.0 - Z_partwave))

        q_mirrored = np.zeros(2 * len(q))
        q_mirrored[:len(q)] = -1 * np.flip(q)
        q_mirrored[len(q):] = q


        for tensorbase_idx in range(V_partwave.shape[-1]):
            Vi_partwave = V_partwave[..., tensorbase_idx]

            Vi_partwave_mirrored = np.zeros(2 * len(Vi_partwave))
            Vi_partwave_mirrored[:len(Vi_partwave)] = np.flip(Vi_partwave)
            Vi_partwave_mirrored[len(Vi_partwave):] = Vi_partwave

            
            # build -i pi sign(q) function
            #       as we consider q > 0 for all values of V --> use -i pi
            
            r_inverse = np.fft.fft(-1j * np.sign(q_mirrored))   # TODO
            Vi_r[l, :, tensorbase_idx] = -1.0/np.power(2.0 * np.pi, 2) * 1.0/(1j) * np.fft.fft(q * Vi_partwave)

            Vi_r_mirrored = -1.0/np.power(2.0 * np.pi, 2) * 1.0/(1j) * r_inverse * np.fft.fft(q_mirrored * Vi_partwave_mirrored)


            # 3D FFT --> TODO check this is correct
            Vi_partwave_xyz = np.meshgrid(Vi_partwave, Vi_partwave, Vi_partwave)[0]
            Vi_r_3d = np.fft.fftn(Vi_partwave_xyz)


            # Plot fourier transformed potential
            plt.figure()
            #plt.plot(Vi_r[l, :, tensorbase_idx], label="1D FFT")
            #plt.plot(Vi_r_mirrored)
            plt.plot(np.real(Vi_r_3d[0, 0, :]), label="3D FFT")
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