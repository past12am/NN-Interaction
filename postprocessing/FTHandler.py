import typing

import numpy as np


from basis.BasisTauToRho import BasisTauToRho

from data.AmplitudeHandler import AmplitudeHandler, AmplitudeHandlerFitfunc, AmplitudeHandlerSchlessinger
from data.Dataloader import Dataloader

from visualization.plotting import Plotter

from numerics.NumericQuadratureFT import NumericQuadratureFT

from utils.fitfunctions import *




def perform_FT_of_amplitudes(dataloader: Dataloader, plotter: Plotter, tensorBasisNamesRho: typing.Dict, process_type):

    ##############################################################################################################################################################################################
    #                                                                           The FT workaround                                                                                                #
    ##############################################################################################################################################################################################

    # Basis Transform to rho basis
    #   V has shape (basis, X, Z)
    X_grid, Z_grid, V = BasisTauToRho.build_rho_base_dressing_functions_from_tau(dataloader)


    fitonly = True
    fitfunc, p0, bounds = yukawa_2_exponentials_v3_fitparams()
    #fitfunc, p0, bounds = yukawa_2_exponentials_fitparams()
    #fitfunc, p0, bounds = yukawa_poly_exponentials_evenodd_fitparams()
    ampHandler_rho = AmplitudeHandlerFitfunc(X_grid, Z_grid, V, fitfunc, p0, bounds)



    if (process_type == "quark_exchange"):
        process_shorthand = "q"
    elif (process_type == "diquark_exchange"):
        process_shorthand = "dq"




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
    #skip   plotter.plot_pwave_amp_with_interpolation(ampHandler_rho, "V", "rho", "PWaves_Interpolation_V_l(X)", 31)


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
        plotter.plot_form_factor_np_side_by_side(X_grid_extended, Z_grid_extended, V[basis_idx, :, :], "U", "X",
                                                 X_grid_extended_reconst, Z_grid_extended_reconst, V_qx_reconst[basis_idx, :, :], "U", "X",
                                                 tensorBasisNamesRho[basis_idx], basis_idx, "rho", "Amplitude_PwavesFitReconstruction_V(X, Z)", 32, left_pretitle="Numeric: ", right_pretitle="Fitted: ")
        pass
                



    ########################### (4) ##############################
    # Knowing f(X, Z) for all X, get access to f(q, Z) via q(X, Z)
                
    # Although the results seem weird, they make sense when thinking about the
    # transformation X -> q to kind of stretch the X plot along the diagonal to get the q plot

    # Check Amplitude in q  --> TODO as this is more important, move to amplitude handler
    q_qx_reconst = np.linspace(0, 8, 300)

    q_qx_extended_reconst = np.repeat(q_qx_reconst, len(Z_grid_reconst))
    Z_grid_q_extended_reconst = np.tile(Z_grid_reconst, len(q_qx_reconst))

    V_qx_q_reconst = np.zeros((V.shape[0], len(q_qx_reconst), len(Z_grid_reconst)))

    for basis_idx in range(V.shape[0]):
        for q_idx in range(q_qx_reconst.shape[0]):
            for Z_idx in range(Z_grid_reconst.shape[0]):
                V_qx_q_reconst[basis_idx, q_idx, Z_idx] = ampHandler_rho.f_q_at(basis_idx, q_qx_reconst[q_idx], Z_grid_reconst[Z_idx], fitonly=fitonly)


    for basis_idx in range(V.shape[0]):
        plotter.plot_form_factor_np_side_by_side(X_grid_extended_reconst, Z_grid_extended_reconst, V_qx_reconst[basis_idx, :, :], "U", "X",
                                         q_qx_extended_reconst, Z_grid_q_extended_reconst, V_qx_q_reconst[basis_idx, :, :], "U", "q",
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







    ########################### (7) ##############################
    # Some more specific plots
    plotter.plotAmplitudes_rhoBasis(X_grid_extended, Z_grid_extended, V, "AmplitudeRhoBasis", 10, "q")


    for basis_idx in range(V.shape[0]):
        plotter.plot_form_factor_np(q_qx_extended_reconst, Z_grid_q_extended_reconst, V_qx_q_reconst[basis_idx, :, :], f"V_{basis_idx + 1}^{{({process_shorthand})}}", tensorBasisNamesRho[basis_idx], "rho", basis_idx, "AmplitudeV(q, Z)", 40)

    plotter.plot_pwave_amp_scaled(f_l_r, r_grid, "r", "1/GeV", "PWaves_V_l(r)", "rho", 60, (0, 1))


    return ampHandler_rho, f_l_r, r_grid