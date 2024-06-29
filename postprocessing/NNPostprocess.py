import os
import json
import typing

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

from data.Dataloader import Dataloader

from basis.BasisTauToRho import BasisTauToRho

from data.AmplitudeHandler import AmplitudeHandler, AmplitudeHandlerFitfunc, AmplitudeHandlerSchlessinger
from pwave.PartialWaveExpansion import PartialWaveExpansion
from visualization.plotting import Plotter, PlotterFullAmplitude

from utils.fitfunctions import *

from numerics.NumericQuadratureFT import NumericQuadratureFT

tensorBasisNamesTau = {
    0: "$1 \otimes 1$",
    1: "$\gamma_{5} \otimes \gamma_{5}$",
    2: "$\gamma^{\mu} \otimes \gamma^{\mu}$",
    3: "$\gamma_{5} \gamma^{\mu} \otimes \gamma_{5} \gamma^{\mu}$",
    4: "$\\frac{1}{8} [\gamma^{\mu}, \gamma^{\\nu}] \otimes [\gamma^{\mu}, \gamma^{\\nu}]$",
}

tensorBasisNamesTauPrime = {
    0: "$\\tau'_0$",
    1: "$\\tau'_1$",
    2: "$\\tau'_2$",
    3: "$\\tau'_3$",
    4: "$\\tau'_4$",
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
    2: "$\\frac{1}{4M^2} (\\vec{\sigma} \cdot \\vec{q}) \otimes (\\vec{\sigma} \cdot \\vec{q})$",
    3: "$\\frac{1}{4M^2} (\\vec{\sigma} \otimes 1 + 1 \otimes \\vec{\sigma} (\\vec{q} \\times \\vec{p})$",
    4: "$\\frac{1}{4M^2} \\vec{\sigma} \cdot (\\vec{q} \\times \\vec{p}) \otimes \\vec{\sigma} \cdot (\\vec{q} \\times \\vec{p})$",
}






def main():

    data_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/"  #sys.argv[1]
    output_base_path = "/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/postprocess-output/"

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
    qx_process_type = "quark_exchange"
    dqx_process_type = "diquark_exchange"
    dataloader_qx = Dataloader(data_base_path, tensorbase_type, amplitude_isospin, qx_process_type, dq_1_type, dq_2_type, Z_range, X_range_lower)
    dataloader_dqx = Dataloader(data_base_path, tensorbase_type, amplitude_isospin, dqx_process_type, dq_1_type, dq_2_type, Z_range, X_range_lower)


    # Instantiate Plotter
    qx_run_path_name = dataloader_qx.latest_run_dir_process
    dqx_run_path_name = dataloader_dqx.latest_run_dir_process
    
    plotter_qx = Plotter(output_base_path, tensorBasisNamesT, tensorBasisNamesTau, tensorBasisNamesRho, qx_process_type, qx_run_path_name, dataloader_qx.process_spec, True)
    plotter_dqx = Plotter(output_base_path, tensorBasisNamesT, tensorBasisNamesTau, tensorBasisNamesRho, dqx_process_type, dqx_run_path_name, dataloader_dqx.process_spec, True)

    plotter_combined = PlotterFullAmplitude(output_base_path, dataloader_qx, dataloader_dqx, True)

    plotter_qx.show_plots = False
    plotter_dqx.show_plots = False






    # Plot amplitudes
    qx_Proj_Basis = None
    if (dataloader_qx.process_spec["projection_basis"] == "T"):
        qx_Proj_Basis = tensorBasisNamesT
    elif (dataloader_qx.process_spec["projection_basis"] == "tau_prime"):
        qx_Proj_Basis = tensorBasisNamesTauPrime
    elif (dataloader_qx.process_spec["projection_basis"] == "tau_prime"):
        qx_Proj_Basis = tensorBasisNamesTau

    plotter_qx.plotAmplitudes_h(dataloader_qx, "AmplitudeQuarkExchange_h", dataloader_qx.process_spec["projection_basis"], qx_Proj_Basis, 0)     # TODO fix S1, A1 bug for tau --> T
    plotter_qx.plotAmplitudes(dataloader_qx, "AmplitudeQuarkExchange_f", "AmplitudeQuarkExchange_F", 0)



    dqx_Proj_Basis = None
    if (dataloader_dqx.process_spec["projection_basis"] == "T"):
        dqx_Proj_Basis = tensorBasisNamesT
    elif (dataloader_dqx.process_spec["projection_basis"] == "tau_prime"):
        dqx_Proj_Basis = tensorBasisNamesTauPrime
    elif (dataloader_dqx.process_spec["projection_basis"] == "tau_prime"):
        dqx_Proj_Basis = tensorBasisNamesTau

    plotter_dqx.plotAmplitudes_h(dataloader_dqx, "AmplitudeDiquarkExchange_h", dataloader_dqx.process_spec["projection_basis"], dqx_Proj_Basis, 0)
    plotter_dqx.plotAmplitudes(dataloader_dqx, "AmplitudeDiquarkExchange_f", "AmplitudeDiquarkExchange_F", 0)

    plotter_combined.plotFullSymAmplitude(tensorBasisNamesT, "FullSymAmplitude")






    ##############################################################################################################################################################################################
    #                                                            The FT workaround --> only for Quark Exchange                                                                                   #
    ##############################################################################################################################################################################################

    # Select quark or diquark exchange
    dataloader = dataloader_dqx
    plotter = plotter_dqx

    # Basis Transform to rho basis
    #   V has shape (basis, X, Z)
    X_grid, Z_grid, V = BasisTauToRho.build_rho_base_dressing_functions_from_tau(dataloader)


    fitonly = True
    fitfunc, p0, bounds = yukawa_2_exponentials_v3_fitparams()
    #fitfunc, p0, bounds = yukawa_2_exponentials_fitparams()
    #fitfunc, p0, bounds = yukawa_poly_exponentials_evenodd_fitparams()
    ampHandler_rho = AmplitudeHandlerFitfunc(X_grid, Z_grid, V, fitfunc, p0, bounds)




    ########################### (1) ##############################
    #Perform Partial Wave Expansion to get V_l(X) from V(X, Z)         (i.e. get V_qx_l[basis, l, X]   from    V[basis, X, Z])

    degree_pwave_exp = 4
    ampHandler_rho.partial_wave_expand(degree_pwave_exp)

    # Plot partial wave amplitudes
    plotter.plot_pwave_amp(ampHandler_rho.f_l, ampHandler_rho.X, "X", "1", "PWaves_V_l(X)", "rho", 11)

    # Check result of partial wave expansion
    plotter.plotAmplitudesPartialWaveExpandedAndOriginal(ampHandler_rho.X, ampHandler_rho.Z, ampHandler_rho.f_l, V, "X", "Amplitude_CheckPwavesReconstruction_V(X, Z)", 12)




    
    ########################### (2) ##############################
    # Fit f_l(X) for X >>        (i.e. fit V_qx_l[basis, l, X])  for fixed basis, l
    # and interpolate for X <= X_max


    # V_qx_l_fitcoeff has shape (basis, l, coeffs)
    ampHandler_rho.fit_large_X_behaviour(min_X=0)
    ampHandler_rho.interpolate_in_X()
            

    # Plot partial wave amplitudes for X in (Log-Log) Plot
    plotter.plot_pwave_amp_with_fits(ampHandler_rho.f_l, ampHandler_rho.X, ampHandler_rho, "rho", "Fit_PWaves_V_l(X)__known_region", 21)


    # Check Fit behaviour for large X
    X_grid_check = np.linspace(0, 10, 1000)
    plotter.plot_pwave_amp_fits_seperated(X_grid_check, ampHandler_rho, "Fit_PWaves_V_l(X)__extended_region", "rho", 22)




    ########################### (3) ##############################
    # Plug things together
    #   -> query interpolation for X <= X_max
    #   -> query fit for X > X_max
    
    # Check interpolation
    plotter.plot_pwave_amp_with_interpolation(ampHandler_rho, "V", "rho", "PWaves_Interpolation_V_l(X)", 31)


    # Check reconstructed Amplitude
    X_grid_extended = np.repeat(ampHandler_rho.X, len(ampHandler_rho.Z))
    Z_grid_extended = np.tile(ampHandler_rho.Z, len(ampHandler_rho.X))

    X_grid_reconst = np.linspace(0, 1, 20)
    Z_grid_reconst = np.linspace(-1, 0.9, 15)
    V_qx_reconst = np.zeros((V.shape[0], len(X_grid_reconst), len(Z_grid_reconst)))

    X_grid_extended_reconst = np.repeat(X_grid_reconst, len(Z_grid_reconst))
    Z_grid_extended_reconst = np.tile(Z_grid_reconst, len(X_grid_reconst))

    for basis_idx in range(V.shape[0]):
        for X_idx in range(X_grid_reconst.shape[0]):
            for Z_idx in range(Z_grid_reconst.shape[0]):
                V_qx_reconst[basis_idx, X_idx, Z_idx] = ampHandler_rho.f_at(basis_idx, X_grid_reconst[X_idx], Z_grid_reconst[Z_idx])

    for basis_idx in range(V.shape[0]):
        plotter.plot_form_factor_np_side_by_side(X_grid_extended, Z_grid_extended, V[basis_idx, :, :], "V_{orig}", "X",
                                                 X_grid_extended_reconst, Z_grid_extended_reconst, V_qx_reconst[basis_idx, :, :], "V_{reconst}", "X",
                                                 tensorBasisNamesRho[basis_idx], basis_idx, "rho", "Amplitude_PwavesFitReconstruction_V(X, Z)", 32)
        pass
                



    ########################### (4) ##############################
    # Knowing f(X, Z) for all X, get access to f(q, Z) via q(X, Z)
                
    # Although the results seem weird, they make sense when thinking about the
    # transformation X -> q to kind of stretch the X plot along the diagonal to get the q plot

    # Check Amplitude in q  --> TODO as this is more important, move to amplitude handler
    q_qx_reconst = np.linspace(0, 10, 300)

    q_qx_extended_reconst = np.repeat(q_qx_reconst, len(Z_grid_reconst))
    Z_grid_q_extended_reconst = np.tile(Z_grid_reconst, len(q_qx_reconst))

    V_qx_q_reconst = np.zeros((V.shape[0], len(q_qx_reconst), len(Z_grid_reconst)))

    for basis_idx in range(V.shape[0]):
        for q_idx in range(q_qx_reconst.shape[0]):
            for Z_idx in range(Z_grid_reconst.shape[0]):
                V_qx_q_reconst[basis_idx, q_idx, Z_idx] = ampHandler_rho.f_q_at(basis_idx, q_qx_reconst[q_idx], Z_grid_reconst[Z_idx], fitonly=fitonly)


    for basis_idx in range(V.shape[0]):
        plotter.plot_form_factor_np_side_by_side(X_grid_extended_reconst, Z_grid_extended_reconst, V_qx_reconst[basis_idx, :, :], "V_{reconst}", "X",
                                         q_qx_extended_reconst, Z_grid_q_extended_reconst, V_qx_q_reconst[basis_idx, :, :], "V_{reconst}", "q",
                                         tensorBasisNamesRho[basis_idx], basis_idx, "rho", "Amplitudes_Comparison_V(X, Z)_vs_V(q, Z)", 41)
        pass
        



    ########################### (5) ##############################
    #Perform Partial Wave Expansion to get f_l(q) from f(q, Z)
    
    degree_pwave_exp_q = 4
    ampHandler_rho.partial_wave_expand_q(degree_pwave_exp_q, q_qx_reconst, Z_grid_reconst, fitonly=fitonly)       # Note q_qx_reconst should be the one from the transformer
    ampHandler_rho.interpolate_in_q()

    # Check result of partial wave expansion
    plotter.plotAmplitudesPartialWaveExpandedAndOriginal(ampHandler_rho.q, Z_grid_reconst, ampHandler_rho.f_l_q, V_qx_q_reconst, "q", "Amplitude_CheckPwavesReconstruction_V(q, Z)", 51)

    # Plot partial wave amplitudes for q in (Log-Log) Plot
    plotter.plot_pwave_amp(ampHandler_rho.f_l_q, ampHandler_rho.q, "q", "GeV", "PWaves_V_l(q)", "rho", 52)






    ########################### (6) ##############################
    # Fourier Transform f_l(q) -> f_l(r)
    r_grid = np.linspace(0, 5, 100)

    # Fourier Transform via Numeric Quadrature
    quad_ft = NumericQuadratureFT(100, 20)

    f_l_r = np.zeros((ampHandler_rho.f_l_q.shape[0], ampHandler_rho.f_l_q.shape[1], len(r_grid)))
    ylims = np.zeros((ampHandler_rho.f_l_q.shape[0], 2))
    for basis_idx in range(ampHandler_rho.f_l_q.shape[0]):
        for l in range(ampHandler_rho.f_l_q.shape[1]):
            print(f"Fourier Transforming basis/l {basis_idx}/{l}")
            f_l_r[basis_idx, l, :] = quad_ft.fourierTransform(lambda q : ampHandler_rho.f_l_q_at(basis_idx, l, q), r_grid)

        y_lim_upper = np.max(f_l_r[basis_idx, ~np.isnan(f_l_r[basis_idx, ...])])
        ylims[basis_idx, :] = np.array([-y_lim_upper, y_lim_upper]) * 0.1

    plotter.plot_pwave_amp(f_l_r, r_grid, "r", "1/GeV", "PWaves_V_l(r)", "rho", 61)
    plotter.plot_pwave_amp_scaled_side_by_side(f_l_r, r_grid, "r", "1/GeV", "PWaves_V_l(r)__scaled", "rho", 62, ylims)
    plotter.plot_pwave_amp_wave_sum(f_l_r, r_grid, "r", "1/GeV", "PWaves_V_l(r)__summed_l", "rho", 63)
    


    exit()

    # (6.1) Fit known function to analytically do FT
    q_fitfunc, q_p0, q_bounds, FT_q_fitfunc = yukawa_poly_exponentials_v3_fitparams()
    ampHandler.fit_q_pwaves(q_fitfunc, q_p0, q_bounds, FT_q_fitfunc)


    # (6.2) Plot knonw functions and fit
    plot_pwave_q_amp_fits_seperated(X_grid_check, ampHandler)


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
    #       X_grid is already unique, Z_grid can be replaced by wave (s, p, d, ... <=> l = 0, 1, 2, ...)

    # 1.) Calculate q^2 from X values and nucleon mass for each combination of X and Z
    # TODO load nucleon mass via dataloader from spec.json
    M_nucleon = 0.94

    # q2 has shape (X, Z)
    q2 = np.zeros((len(X_grid), len(Z_grid)))
    q2[:, :] = np.square(M_nucleon) * (2.0 * X_grid[:, None]) * (1.0 - Z_grid[None, :])

    """
    X_grid_extended = np.repeat(X_grid, len(Z_grid))
    Z_grid_extended = np.tile(Z_grid, len(X_grid))
    plot_form_factor_np(X_grid_extended, Z_grid_extended, q2, "q2", "q2")
    """

    # 2.) Set parameters
    degree_fit = 3

    # V_qx_l has shape (basis, l, X)
    V_qx_l = np.zeros((V.shape[0], degree_fit+1, V.shape[1]))

    for basis_idx in range(V.shape[0]):
        V_qx_l[basis_idx, :] = PartialWaveExpansion(V[basis_idx, :], X_grid, Z_grid, degree_fit).get_f_x()

    # 4.) Check result of partial wave expansion
    plotAmplitudesPartialWaveExpandedAndOriginal(X_grid, Z_grid, V_qx_l, V, "X", dataloader_qx.data_path)




    # Fourier Transformation

    #   1.) Fit partial wave expanded amplitudes with functions of known fourier transform  (Yukawa Potential and additionally sum of exponentials for small q2)
            
    #       V_qx_l_fitcoeff has shape (basis, l, coeffs)     
    V_qx_l_fitcoeff = np.zeros((V_qx_l.shape[0], V_qx_l.shape[1], 7))

    for basis_idx in range(V_qx_l.shape[0]):
        for l in range(V_qx_l.shape[1]):
            popt, pcov = curve_fit(yukawa_potential_exp_sum, X_grid, V_qx_l[basis_idx, l, :], 
                                   p0=[1, 1, -1, -2, -1, -2, -0.1], 
                                   bounds=([-np.inf, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, 0, np.inf, 0, np.inf]), 
                                   maxfev=10000)
            
            V_qx_l_fitcoeff[basis_idx, l, :] = popt
            

    #   2.) Plot partial wave amplitudes for X in (Log-Log) Plot
    for basis_idx in range(V_qx_l.shape[0]):
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        for l in range(V_qx_l.shape[1]):
            axs[0].plot(X_grid, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[0].plot(X_grid, yukawa_potential_exp_sum(X_grid, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

            axs[1].loglog(X_grid, V_qx_l[basis_idx, l, :], label=f"{l}-wave")
            axs[1].loglog(X_grid, yukawa_potential_exp_sum(X_grid, *V_qx_l_fitcoeff[basis_idx, l, :]), label=f"{l}-wave-fit")

        #axs[1].loglog(X_grid, 1.0/np.square(X_grid), label="1/X^2")
        #axs[1].loglog(X_grid, np.exp(X_grid), label="exp(X)")

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