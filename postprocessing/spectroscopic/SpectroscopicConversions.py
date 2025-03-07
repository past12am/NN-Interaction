import typing
import numpy as np

from scipy.interpolate import CubicSpline

from basis.BasisTauToRho import BasisTauToRho
from data.AmplitudeHandler import AmplitudeHandlerFitfunc
from data.Dataloader import Dataloader
from numerics.NumericQuadratureFT import NumericQuadratureFT
from pwave.PartialWaveExpansion import PartialWaveExpansion
from utils.fitfunctions import yukawa_2_exponentials_v3_fitparams
from visualization.plotting import Plotter, PlotterFullAmplitude


class SpectroscopicConversion:

    ########################### (5) ##############################
    # Calculate Spectroscopic Basis (things change in the FT)
    #   We need to work around this, because our choice of momentum causes the integrals to diverge for Z->1
    #       In explanation, we need to calculate the combinations for the LSJ projection first, and then perform the partial wave expansions of the contributions
    #       Afterwards, we can just cherrypick the needed parts and sum them up


    
    # isoscalar t channel   (mathbf{F_0})
    def V__tI0(self, basis_idx, q, Z):
        return self.amplitude_handler_dqx.f_q_at(basis_idx, q, Z, fitonly=True)

    # isovector t channel   (mathbf{F_1})
    def W__tI1(self, basis_idx, q, Z):
        return 2 * self.amplitude_handler_qx.f_q_at(basis_idx, q, Z, fitonly=True) - self.amplitude_handler_dqx.f_q_at(basis_idx, q, Z, fitonly=True)


    def U(self, basis_idx, I_tchannel, q, Z):
        if(I_tchannel == 0 or I_tchannel == 1):
            return self.V__tI0(basis_idx, q, Z) + (4 * I_tchannel - 3) * self.W__tI1(basis_idx, q, Z)
        
        raise Exception(f"There aint no Isospin I_tchannel = {I_tchannel}")
        
    
    def R(self, I, I0_tchannel_data, I1_tchannel_data):
        #Singlet
        if(I == 0): 
            return 0.25 * (I0_tchannel_data - I1_tchannel_data)
        elif(I == 1):
            return 0.25 * (3 * I0_tchannel_data + I1_tchannel_data)
        else:
            raise Exception(f"No Isospin I = {I} in the amplitude")
    
            

    def __init__(self, amplitude_handler_qx: AmplitudeHandlerFitfunc, amplitude_handler_dqx: AmplitudeHandlerFitfunc):
        self.amplitude_handler_qx = amplitude_handler_qx
        self.amplitude_handler_dqx = amplitude_handler_dqx


    def spectroscopic_basis_run(self, plotter: PlotterFullAmplitude):
        LSJ_singlet = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)]
        LSJ_triplet__L_eq_J_plus_1 = [(None, None, None), (1, 1, 0), (2, 1, 1), (3, 1, 2), (4, 1, 3), (5, 1, 4)]
        LSJ_triplet__L_eq_J = [(None, None, None), (1, 1, 1), (2, 1, 2), (3, 1, 3), (4, 1, 4), (5, 1, 5)]
        LSJ_triplet__L_eq_J_minus_1 = [(0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4), (4, 1, 5), (5, 1, 6)]


        ########################### (5) ##############################
        # Perform Partial Wave Expansion of integral kernels
        q_grid = np.linspace(0, 50, 200)
        Z_grid = np.linspace(-1, 0.9, 30)

        r_grid = np.linspace(0, 3, 100)

        degree_pwave_expansion = 4


        # Define which contributions to calculat (Note that array indices need to match)
        contribs = ["C", "SS", "SO"]

        singlet__callables = [self.singlet_C, self.singlet_SS, self.singlet_SO]

        triplet_l_is_j_Minus_1__callable_tuples = [(self.triplet_l_is_j_Minus_1__pwaveEqual__C, self.triplet_l_is_j_Minus_1__pwaveMinus__C), 
                                                   (self.triplet_l_is_j_Minus_1__pwaveEqual__SS, self.triplet_l_is_j_Minus_1__pwaveMinus__SS),
                                                   (self.triplet_l_is_j_Minus_1__pwaveEqual__SO, self.triplet_l_is_j_Minus_1__pwaveMinus__SO)]
        triplet_l_is_j__callable_tuples = [(self.triplet_l_is_j__pwaveEqual__C, self.triplet_l_is_j__pwavePlusMinus__C), 
                                                   (self.triplet_l_is_j__pwaveEqual__SS, self.triplet_l_is_j__pwavePlusMinus__SS),
                                                   (self.triplet_l_is_j__pwaveEqual__SO, self.triplet_l_is_j__pwavePlusMinus__SO)]
        triplet_l_is_j_Plus_1__callable_tuples = [(self.triplet_l_is_j_Plus_1__pwaveEqual__C, self.triplet_l_is_j_Plus_1__pwavePlus__C), 
                                                   (self.triplet_l_is_j_Plus_1__pwaveEqual__SS, self.triplet_l_is_j_Plus_1__pwavePlus__SS),
                                                   (self.triplet_l_is_j_Plus_1__pwaveEqual__SO, self.triplet_l_is_j_Plus_1__pwavePlus__SO)]

        I_tchannel_list = list()
        for I_tchannel in [0, 1]:
            # Calculate
            (singlet_contrib_grid_list, triplet_l_is_j_Minus_1__contrib_grid_list, triplet_l_is_j__contrib_grid_list, triplet_l_is_j_Plus_1__contrib_grid_list, singlet_contrib_r_list, triplet_l_is_j_Minus_1__contrib_r_list, triplet_l_is_j__contrib_r_list, triplet_l_is_j_Plus_1__contrib_r_list)\
                = self.calculate_LSJ_result_in_basis(I_tchannel, q_grid, Z_grid, degree_pwave_expansion, r_grid, singlet__callables, triplet_l_is_j_Minus_1__callable_tuples, triplet_l_is_j__callable_tuples, triplet_l_is_j_Plus_1__callable_tuples)
            
            I_tchannel_list.append((singlet_contrib_grid_list, 
                                    triplet_l_is_j_Minus_1__contrib_grid_list, 
                                    triplet_l_is_j__contrib_grid_list, 
                                    triplet_l_is_j_Plus_1__contrib_grid_list,
                                    singlet_contrib_r_list,
                                    triplet_l_is_j_Minus_1__contrib_r_list,
                                    triplet_l_is_j__contrib_r_list, 
                                    triplet_l_is_j_Plus_1__contrib_r_list))


        I0_tchannel_singlet_contrib_grid_list = I_tchannel_list[0][0]
        I0_tchannel_triplet_l_is_j_Minus_1__contrib_grid_list = I_tchannel_list[0][1]
        I0_tchannel_triplet_l_is_j__contrib_grid_list = I_tchannel_list[0][2]
        I0_tchannel_triplet_l_is_j_Plus_1__contrib_grid_list = I_tchannel_list[0][3]
        I0_tchannel_singlet_contrib_r_list = I_tchannel_list[0][4]
        I0_tchannel_triplet_l_is_j_Minus_1__contrib_r_list = I_tchannel_list[0][5]
        I0_tchannel_triplet_l_is_j__contrib_r_list = I_tchannel_list[0][6]
        I0_tchannel_triplet_l_is_j_Plus_1__contrib_r_list = I_tchannel_list[0][7]

        I1_tchannel_singlet_contrib_grid_list = I_tchannel_list[1][0]
        I1_tchannel_triplet_l_is_j_Minus_1__contrib_grid_list = I_tchannel_list[1][1]
        I1_tchannel_triplet_l_is_j__contrib_grid_list = I_tchannel_list[1][2]
        I1_tchannel_triplet_l_is_j_Plus_1__contrib_grid_list = I_tchannel_list[1][3]
        I1_tchannel_singlet_contrib_r_list = I_tchannel_list[1][4]
        I1_tchannel_triplet_l_is_j_Minus_1__contrib_r_list = I_tchannel_list[1][5]
        I1_tchannel_triplet_l_is_j__contrib_r_list = I_tchannel_list[1][6]
        I1_tchannel_triplet_l_is_j_Plus_1__contrib_r_list = I_tchannel_list[1][7]


        # Switch from t-channel isospin to NN, NP, PP reactions (--> back to the curly F)
        for I in [0, 1]:
            # Singlet I = 0: (PN - NP)
            process_singlet_contrib_grid_list = [self.R(I, I0_tchannel_singlet_contrib_grid, I1_tchannel_singlet_contrib_grid) for (I0_tchannel_singlet_contrib_grid, I1_tchannel_singlet_contrib_grid) in zip(I0_tchannel_singlet_contrib_grid_list, I1_tchannel_singlet_contrib_grid_list)]
            process_singlet_contrib_r_list = [self.R(I, I0_tchannel_singlet_contrib_r, I1_tchannel_singlet_contrib_r) for (I0_tchannel_singlet_contrib_r, I1_tchannel_singlet_contrib_r) in zip(I0_tchannel_singlet_contrib_r_list, I1_tchannel_singlet_contrib_r_list)]


            # Triplet I = 1: (PP = NN = PN + NP)
            process_triplet_l_is_j_Minus_1__contrib_grid_list = [self.R(I, I0_tchannel_triplet_l_is_j_Minus_1__contrib_grid, I1_tchannel_triplet_l_is_j_Minus_1__contrib_grid) for (I0_tchannel_triplet_l_is_j_Minus_1__contrib_grid, I1_tchannel_triplet_l_is_j_Minus_1__contrib_grid) in zip(I0_tchannel_triplet_l_is_j_Minus_1__contrib_grid_list, I1_tchannel_triplet_l_is_j_Minus_1__contrib_grid_list)]
            process_triplet_l_is_j__contrib_grid_list = [self.R(I, I0_tchannel_triplet_l_is_j__contrib_grid, I1_tchannel_triplet_l_is_j__contrib_grid) for (I0_tchannel_triplet_l_is_j__contrib_grid, I1_tchannel_triplet_l_is_j__contrib_grid) in zip(I0_tchannel_triplet_l_is_j__contrib_grid_list, I1_tchannel_triplet_l_is_j__contrib_grid_list)]
            process_triplet_l_is_j_Plus_1__contrib_grid_list = [self.R(I, I0_tchannel_triplet_l_is_j_Plus_1__contrib_grid, I1_tchannel_triplet_l_is_j_Plus_1__contrib_grid) for (I0_tchannel_triplet_l_is_j_Plus_1__contrib_grid, I1_tchannel_triplet_l_is_j_Plus_1__contrib_grid) in zip(I0_tchannel_triplet_l_is_j_Plus_1__contrib_grid_list, I1_tchannel_triplet_l_is_j_Plus_1__contrib_grid_list)]
            
            process_triplet_l_is_j_Minus_1__contrib_r_list = [self.R(I, I0_tchannel_triplet_l_is_j_Minus_1__contrib_r, I1_tchannel_triplet_l_is_j_Minus_1__contrib_r) for (I0_tchannel_triplet_l_is_j_Minus_1__contrib_r, I1_tchannel_triplet_l_is_j_Minus_1__contrib_r) in zip(I0_tchannel_triplet_l_is_j_Minus_1__contrib_r_list, I1_tchannel_triplet_l_is_j_Minus_1__contrib_r_list)]
            process_triplet_l_is_j__contrib_r_list = [self.R(I, I0_tchannel_triplet_l_is_j__contrib_r, I1_tchannel_triplet_l_is_j__contrib_r) for (I0_tchannel_triplet_l_is_j__contrib_r, I1_tchannel_triplet_l_is_j__contrib_r) in zip(I0_tchannel_triplet_l_is_j__contrib_r_list, I1_tchannel_triplet_l_is_j__contrib_r_list)]
            process_triplet_l_is_j_Plus_1__contrib_r_list = [self.R(I, I0_tchannel_triplet_l_is_j_Plus_1__contrib_r, I1_tchannel_triplet_l_is_j_Plus_1__contrib_r) for (I0_tchannel_triplet_l_is_j_Plus_1__contrib_r, I1_tchannel_triplet_l_is_j_Plus_1__contrib_r) in zip(I0_tchannel_triplet_l_is_j_Plus_1__contrib_r_list, I1_tchannel_triplet_l_is_j_Plus_1__contrib_r_list)]


            
            # Plot
            #  TODO different names for r and q plots
            #   Process Singlet
            for c_idx, singlet_contrib_grid in enumerate(process_singlet_contrib_grid_list):
                plotter.plot_pwave_LSJ(singlet_contrib_grid, LSJ_singlet, q_grid, "q", "GeV", contribs[c_idx], I, f"Singlet {contribs[c_idx]} for I = {I}", f"LSJ_Singlet_{contribs[c_idx]}_I={I}")

            for c_idx, singlet_contrib_r in enumerate(process_singlet_contrib_r_list):
                plotter.plot_pwave_LSJ(singlet_contrib_r, LSJ_singlet, r_grid, "r", "1/GeV", contribs[c_idx], I, f"Singlet {contribs[c_idx]} for I = {I}", f"LSJ_Singlet_{contribs[c_idx]}_I={I}")


            #   Process Triplet
            for c_idx, triplet_l_is_j_Minus_1__contrib_grid in enumerate(process_triplet_l_is_j_Minus_1__contrib_grid_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__contrib_grid, LSJ_triplet__L_eq_J_minus_1, q_grid, "q", "GeV", contribs[c_idx], I, f"Triplet L=J-1 {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J-1_{contribs[c_idx]}_I={I}")
            
            for c_idx, triplet_l_is_j_Minus_1__contrib_r in enumerate(process_triplet_l_is_j_Minus_1__contrib_r_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__contrib_r, LSJ_triplet__L_eq_J_minus_1, r_grid, "r", "1/GeV", contribs[c_idx], I, f"Triplet L=J-1 {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J-1_{contribs[c_idx]}_I={I}")


            for c_idx, triplet_l_is_j__contrib_grid in enumerate(process_triplet_l_is_j__contrib_grid_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j__contrib_grid, LSJ_triplet__L_eq_J, q_grid, "q", "GeV", contribs[c_idx], I, f"Triplet L=J {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J_{contribs[c_idx]}_I={I}")

            for c_idx, triplet_l_is_j__contrib_r in enumerate(process_triplet_l_is_j__contrib_r_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j__contrib_r, LSJ_triplet__L_eq_J, r_grid, "r", "1/GeV", contribs[c_idx], I, f"Triplet L=J {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J_{contribs[c_idx]}_I={I}")


            for c_idx, triplet_l_is_j_Plus_1__contrib_grid in enumerate(process_triplet_l_is_j_Plus_1__contrib_grid_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__contrib_grid, LSJ_triplet__L_eq_J_plus_1, q_grid, "q", "GeV", contribs[c_idx], I, f"Triplet L=J+1 {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J+1_{contribs[c_idx]}_I={I}")

            for c_idx, triplet_l_is_j_Plus_1__contrib_r in enumerate(process_triplet_l_is_j_Plus_1__contrib_r_list):
                plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__contrib_r, LSJ_triplet__L_eq_J_plus_1, r_grid, "r", "1/GeV", contribs[c_idx], I, f"Triplet L=J+1 {contribs[c_idx]} for I = {I}", f"LSJ_Triplet_L=J+1_{contribs[c_idx]}_I={I}")

            
            

    # TODO correction factor for J dependent elements
    def calculate_LSJ_result_in_basis(self, I_tchannel: int, q_grid: np.array, Z_grid: np.array, degree_pwave_expansion: int, r_grid: np.array,
                                      singlet__callables: typing.List[typing.Callable], 
                                      triplet_l_is_j_Minus_1__callable_tuples: typing.List[typing.Tuple[typing.Callable, typing.Callable]],
                                      triplet_l_is_j__callable_tuples: typing.List[typing.Tuple[typing.Callable, typing.Callable]],
                                      triplet_l_is_j_Plus_1__callable_tuples: typing.List[typing.Tuple[typing.Callable, typing.Callable]]):
        # Central
        #   Singlets
        singlet_amplitude_grid_list = list()
        for singlet__callable in singlet__callables:
            singlet_amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, singlet__callable, I_tchannel=I_tchannel)
            singlet_amplitude_grid_list.append(singlet_amplitude_grid)


        #   Triplets
        #       Raw
        #           L = J - 1
        triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid_list = list()
        triplet_l_is_j_Minus_1__pwaveMinus__amplitude_grid_list = list()

        for (triplet_l_is_j_Minus_1__pwaveEqual__callable, triplet_l_is_j_Minus_1__pwaveMinus__callable) in triplet_l_is_j_Minus_1__callable_tuples:
            triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j_Minus_1__pwaveEqual__callable, I_tchannel=I_tchannel)
            triplet_l_is_j_Minus_1__pwaveMinus__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j_Minus_1__pwaveMinus__callable, I_tchannel=I_tchannel)

            triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid_list.append(triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid)
            triplet_l_is_j_Minus_1__pwaveMinus__amplitude_grid_list.append(triplet_l_is_j_Minus_1__pwaveMinus__amplitude_grid)
        

        #           L = J
        triplet_l_is_j__pwaveEqual__amplitude_grid_list = list()
        triplet_l_is_j__pwavePlusMinus__amplitude_grid_list = list()

        for (triplet_l_is_j__pwaveEqual__callable, triplet_l_is_j__pwavePlusMinus__callable) in triplet_l_is_j__callable_tuples:
            triplet_l_is_j__pwaveEqual__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j__pwaveEqual__callable, I_tchannel=I_tchannel)
            triplet_l_is_j__pwavePlusMinus__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j__pwavePlusMinus__callable, I_tchannel=I_tchannel)

            triplet_l_is_j__pwaveEqual__amplitude_grid_list.append(triplet_l_is_j__pwaveEqual__amplitude_grid)
            triplet_l_is_j__pwavePlusMinus__amplitude_grid_list.append(triplet_l_is_j__pwavePlusMinus__amplitude_grid)


        #           L = J + 1
        triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid_list = list()
        triplet_l_is_j_Plus_1__pwavePlus__amplitude_grid_list = list()

        for (triplet_l_is_j_Plus_1__pwaveEqual__callable, triplet_l_is_j_Plus_1__pwavePlus__callable) in triplet_l_is_j_Plus_1__callable_tuples:
            triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j_Plus_1__pwaveEqual__callable, I_tchannel=I_tchannel)
            triplet_l_is_j_Plus_1__pwavePlus__amplitude_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, triplet_l_is_j_Plus_1__pwavePlus__callable, I_tchannel=I_tchannel)

            triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid_list.append(triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid)
            triplet_l_is_j_Plus_1__pwavePlus__amplitude_grid_list.append(triplet_l_is_j_Plus_1__pwavePlus__amplitude_grid)



        #       Construct Full combination
        triplet_l_is_j_Minus_1__amplitude_grid_list = [np.zeros_like(triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid) for i in range(len(triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid_list))]
        triplet_l_is_j__amplitude_grid_list = [np.zeros_like(triplet_l_is_j__pwaveEqual__amplitude_grid) for i in range(len(triplet_l_is_j__pwaveEqual__amplitude_grid_list))]
        triplet_l_is_j_Plus_1__amplitude_grid_list = [np.zeros_like(triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid) for i in range(len(triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid_list))]
        
        #       L = J - 1
        for idx in range(len(triplet_l_is_j_Minus_1__amplitude_grid_list)):
            for l in range(0, degree_pwave_expansion):
                triplet_l_is_j_Minus_1__amplitude_grid_list[idx][l, :] = triplet_l_is_j_Minus_1__pwaveEqual__amplitude_grid_list[idx][l + 1, :] + triplet_l_is_j_Minus_1__pwaveMinus__amplitude_grid_list[idx][l, :]

        #       L = J
        for idx in range(len(triplet_l_is_j__amplitude_grid_list)):
            for l in range(1, degree_pwave_expansion):
                triplet_l_is_j__amplitude_grid_list[idx][l, :] = triplet_l_is_j__pwaveEqual__amplitude_grid_list[idx][l, :] + (triplet_l_is_j__pwavePlusMinus__amplitude_grid_list[idx][l + 1, :] + triplet_l_is_j__pwavePlusMinus__amplitude_grid_list[idx][l - 1, :])

        #       L = J + 1
        for idx in range(len(triplet_l_is_j_Plus_1__amplitude_grid_list)):
            for l in range(1, degree_pwave_expansion + 1):
                triplet_l_is_j_Plus_1__amplitude_grid_list[idx][l, :] = triplet_l_is_j_Plus_1__pwaveEqual__amplitude_grid_list[idx][l-1, :] + triplet_l_is_j_Plus_1__pwavePlus__amplitude_grid_list[idx][l, :]   # Note that l = 0 is not a valid quantum number here




        singlet_amplitude_spline_list = [SpectroscopicConversion.interpolate_expanded_in_var(singlet_amplitude_grid, q_grid) for singlet_amplitude_grid in singlet_amplitude_grid_list]

        triplet_l_is_j_Minus_1__amplitude_spline_list = [SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Minus_1__amplitude_grid, q_grid) for triplet_l_is_j_Minus_1__amplitude_grid in triplet_l_is_j_Minus_1__amplitude_grid_list]
        triplet_l_is_j__amplitude_spline_list = [SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j__amplitude_grid, q_grid) for triplet_l_is_j__amplitude_grid in triplet_l_is_j__amplitude_grid_list]
        triplet_l_is_j_Plus_1__amplitude_spline_list = [SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Plus_1__amplitude_grid, q_grid) for triplet_l_is_j_Plus_1__amplitude_grid in triplet_l_is_j_Plus_1__amplitude_grid_list]




        ##############################################################
        # Fourier Transform LSJ(q) --> LSJ(r)

        # Fourier Transform via Numeric Quadrature
        quad_ft = NumericQuadratureFT(100, 20)

        singlet_amplitude_r_list = [np.zeros((len(singlet_amplitude_spline), len(r_grid))) for i, singlet_amplitude_spline in enumerate(singlet_amplitude_spline_list)]

        triplet_l_is_j_Minus_1__amplitude_r_list = [np.zeros((len(triplet_l_is_j_Minus_1__amplitude_spline), len(r_grid))) for i, triplet_l_is_j_Minus_1__amplitude_spline in enumerate(triplet_l_is_j_Minus_1__amplitude_spline_list)]
        triplet_l_is_j__amplitude_r_list = [np.zeros((len(triplet_l_is_j__amplitude_spline), len(r_grid))) for i, triplet_l_is_j__amplitude_spline in enumerate(triplet_l_is_j__amplitude_spline_list)]
        triplet_l_is_j_Plus_1__amplitude_r_list = [np.zeros((len(triplet_l_is_j_Plus_1__amplitude_spline), len(r_grid))) for i, triplet_l_is_j_Plus_1__amplitude_spline in enumerate(triplet_l_is_j_Plus_1__amplitude_spline_list)]

        # Singlet
        for idx, singlet_amplitude_spline in enumerate(singlet_amplitude_spline_list):
            for l in range(len(singlet_amplitude_spline)):
                print(f"Fourier Transforming Singlet L={l}")
                singlet_amplitude_r_list[idx][l, :] = quad_ft.fourierTransform(lambda q : singlet_amplitude_spline[l](q), r_grid)

        # Triplets
        for idx, triplet_l_is_j_Minus_1__amplitude_spline in enumerate(triplet_l_is_j_Minus_1__amplitude_spline_list):
            for l in range(len(triplet_l_is_j_Minus_1__amplitude_spline)):
                print(f"Fourier Transforming Triplet L = J-1 = {l}")
                triplet_l_is_j_Minus_1__amplitude_r_list[idx][l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Minus_1__amplitude_spline[l](q), r_grid)

        for idx, triplet_l_is_j__amplitude_spline in enumerate(triplet_l_is_j__amplitude_spline_list):
            for l in range(len(triplet_l_is_j__amplitude_spline)):
                print(f"Fourier Transforming Triplet L = J = {l}")
                triplet_l_is_j__amplitude_r_list[idx][l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j__amplitude_spline[l](q), r_grid)

        for idx, triplet_l_is_j_Plus_1__amplitude_spline in enumerate(triplet_l_is_j_Plus_1__amplitude_spline_list):
            for l in range(len(triplet_l_is_j_Plus_1__amplitude_spline)):
                print(f"Fourier Transforming Triplet L = J+1 = {l}")
                triplet_l_is_j_Plus_1__amplitude_r_list[idx][l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Plus_1__amplitude_spline[l](q), r_grid)



        return singlet_amplitude_grid_list,\
               triplet_l_is_j_Minus_1__amplitude_grid_list,\
               triplet_l_is_j__amplitude_grid_list,\
               triplet_l_is_j_Plus_1__amplitude_grid_list,\
               singlet_amplitude_r_list,\
               triplet_l_is_j_Minus_1__amplitude_r_list,\
               triplet_l_is_j__amplitude_r_list,\
               triplet_l_is_j_Plus_1__amplitude_r_list

        

    # TODO move to dedicated class
    #   expand_function must take params (var_keep, Z)
    @staticmethod
    def partial_wave_expand(degree, grid_keep, grid_pwave, expand_function: typing.Callable, **callable_args):
        input_grid = np.zeros((len(grid_keep), len(grid_pwave)))

        for keep_idx, keep_var in enumerate(grid_keep):
            for Z_idx, Z in enumerate(grid_pwave):
                input_grid[keep_idx, Z_idx] = expand_function(keep_var, Z, **callable_args)

        res_grid = PartialWaveExpansion(input_grid, grid_keep, grid_pwave, degree).get_f_x()        
        return res_grid
    

    @staticmethod
    def interpolate_expanded_in_var(to_interpolate, var_grid):
        return [CubicSpline(var_grid, to_interpolate[l]) for l in range(to_interpolate.shape[0])]




    # Triplet, L = J + 1
    #   For the triplet we need to split up the current and next partial wave parts and plug them together afterwards
    def triplet_l_is_j_Plus_1__pwaveEqual__C(self, q, Z, I_tchannel):
        return 0
    
    def triplet_l_is_j_Plus_1__pwaveEqual__SS(self, q, Z, I_tchannel):
        return 0
    
    def triplet_l_is_j_Plus_1__pwaveEqual__T_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(2, I_tchannel, q, Z)    # * 1/(2 * J + 1)
    
    def triplet_l_is_j_Plus_1__pwaveEqual__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(3, I_tchannel, q, Z)
    
    def triplet_l_is_j_Plus_1__pwaveEqual__Q_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -2 * np.power(r, 4) * Z * self.U(4, I_tchannel, q, Z)     # * 1/(2 * J + 1)



    def triplet_l_is_j_Plus_1__pwavePlus__C(self, q, Z, I_tchannel):
        return self.U(0, I_tchannel, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__SS(self, q, Z, I_tchannel):
        return self.U(1, I_tchannel, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__T_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * self.U(2, I_tchannel, q, Z)   # * 2/(2 * J + 1)
    
    def triplet_l_is_j_Plus_1__pwavePlus__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * 2 * Z * self.U(3, I_tchannel, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__Q_Jindep(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (1 - np.square(Z)) * self.U(4, I_tchannel, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__Q_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * self.U(4, I_tchannel, q, Z)      # * 2/(2 * J + 1)
    


    # Triplet, L = J
    def triplet_l_is_j__pwaveEqual__C(self, q, Z, I_tchannel):
        return self.U(0, I_tchannel, q, Z)
    
    def triplet_l_is_j__pwaveEqual__SS(self, q, Z, I_tchannel):
        return self.U(1, I_tchannel, q, Z)
    
    def triplet_l_is_j__pwaveEqual__T(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * (1 + Z) * self.U(2, I_tchannel, q, Z)
    
    def triplet_l_is_j__pwaveEqual__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -4 * np.square(r) * Z * self.U(3, I_tchannel, q, Z)
    
    def triplet_l_is_j__pwaveEqual__Q(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -np.power(r, 4) * (3 * np.square(Z) + 1) * self.U(4, I_tchannel, q, Z)



    def triplet_l_is_j__pwavePlusMinus__C(self, q, Z, I_tchannel):
        return 0

    def triplet_l_is_j__pwavePlusMinus__SS(self, q, Z, I_tchannel):
        return 0

    def triplet_l_is_j__pwavePlusMinus__T(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -2 * np.square(r) * self.U(2, I_tchannel, q, Z)

    def triplet_l_is_j__pwavePlusMinus__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(3, I_tchannel, q, Z)

    def triplet_l_is_j__pwavePlusMinus__Q(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.power(r, 4) * Z * self.U(4, I_tchannel, q, Z)
    


    # Triplet, L = J - 1
    def triplet_l_is_j_Minus_1__pwaveEqual__C(self, q, Z, I_tchannel):
        return 0
    
    def triplet_l_is_j_Minus_1__pwaveEqual__SS(self, q, Z, I_tchannel):
        return 0
    
    def triplet_l_is_j_Minus_1__pwaveEqual__T_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -2 * np.square(r) * self.U(2, I_tchannel, q, Z)    # * 1/(2 * J + 1)
    
    def triplet_l_is_j_Minus_1__pwaveEqual__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(3, I_tchannel, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveEqual__Q_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.power(r, 4) * Z * self.U(4, I_tchannel, q, Z)     # * 1/(2 * J + 1)



    def triplet_l_is_j_Minus_1__pwaveMinus__C(self, q, Z, I_tchannel):
        return self.U(0, I_tchannel, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveMinus__SS(self, q, Z, I_tchannel):
        return self.U(1, I_tchannel, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveMinus__T_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return np.square(r) * self.U(2, I_tchannel, q, Z)   # * 2/(2 * J + 1)
    
    def triplet_l_is_j_Minus_1__pwaveMinus__SO(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * 2 * Z * self.U(3, I_tchannel, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveMinus__Q_Jindep(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (1 - np.square(Z)) * self.U(4, I_tchannel, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveMinus__Q_Jdep_noncorr(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return -np.power(r, 4) * self.U(4, I_tchannel, q, Z)      # * 2/(2 * J + 1)
    


    # Singlet
    def singlet_kernel(self, q, Z, I_tchannel):
        return self.singlet_C(self, q, Z, I_tchannel) + self.singlet_SS(self, q, Z, I_tchannel) + self.singlet_T(self, q, Z, I_tchannel) + self.singlet_SO(self, q, Z, I_tchannel) + self.singlet_Q(self, q, Z, I_tchannel)

    def singlet_C(self, q, Z, I_tchannel):
        return self.U(0, I_tchannel, q, Z)

    def singlet_SS(self, q, Z, I_tchannel):
        return -3 * self.U(1, I_tchannel, q, Z)

    def singlet_T(self, q, Z, I_tchannel):
        return -np.square(q) * self.U(2, I_tchannel, q, Z)

    def singlet_SO(self, q, Z, I_tchannel):
        return 0

    def singlet_Q(self, q, Z, I_tchannel):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (np.square(Z) - 1) * self.U(4, I_tchannel, q, Z)
    

    @staticmethod    
    def r_of_q_Z(q, Z):
        return q / np.sqrt(2 * (1 - Z))
    




    def simple_test_case(self):
        q_grid = np.linspace(0, 50, 200)
        Z_grid = np.linspace(-1, 0.9, 30)

        r_grid = np.linspace(0, 3, 100)

        degree_pwave_expansion = 4
    
        ######################### Hard coded #########################
        # Central
        #   Singlets
        singlet_C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_C, I_tchannel=0)
        singlet_C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_C, I_tchannel=1)

        #   Triplets
        #       Raw
        #           L = J - 1
        triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveEqual__C, I_tchannel=0)
        triplet_l_is_j_Minus_1__pwaveMinus__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveMinus__C, I_tchannel=0)

        triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveEqual__C, I_tchannel=1)
        triplet_l_is_j_Minus_1__pwaveMinus__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveMinus__C, I_tchannel=1)

        
        #           L = J
        triplet_l_is_j__pwaveEqual__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j__pwaveEqual__C, I_tchannel=0)
        triplet_l_is_j__pwavePlusMinus__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j__pwavePlusMinus__C, I_tchannel=0)

        triplet_l_is_j__pwaveEqual__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j__pwaveEqual__C, I_tchannel=1)
        triplet_l_is_j__pwavePlusMinus__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j__pwavePlusMinus__C, I_tchannel=1)


        #           L = J + 1
        triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwaveEqual__C, I_tchannel=0)
        triplet_l_is_j_Plus_1__pwavePlus__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwavePlus__C, I_tchannel=0)

        triplet_l_is_j_Plus_1__pwaveEqual__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwaveEqual__C, I_tchannel=1)
        triplet_l_is_j_Plus_1__pwavePlus__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwavePlus__C, I_tchannel=1)


        #       Construct Full combination
        triplet_l_is_j_Plus_1__C_I0_grid = np.zeros_like(triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j_Plus_1__C_I1_grid = np.zeros_like(triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j__C_I0_grid = np.zeros_like(triplet_l_is_j__pwaveEqual__C_I0_grid)
        triplet_l_is_j__C_I1_grid = np.zeros_like(triplet_l_is_j__pwaveEqual__C_I1_grid)
        triplet_l_is_j_Minus_1__C_I0_grid = np.zeros_like(triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j_Minus_1__C_I1_grid = np.zeros_like(triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid)

        #       L = J - 1
        for l in range(0, degree_pwave_expansion):
            triplet_l_is_j_Minus_1__C_I0_grid[l, :] = triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid[l + 1, :] + triplet_l_is_j_Minus_1__pwaveMinus__C_I0_grid[l, :]
            triplet_l_is_j_Minus_1__C_I1_grid[l, :] = triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid[l + 1, :] + triplet_l_is_j_Minus_1__pwaveMinus__C_I1_grid[l, :]

        #       L = J
        for l in range(1, degree_pwave_expansion):
            triplet_l_is_j__C_I0_grid[l, :] = triplet_l_is_j__pwaveEqual__C_I0_grid[l, :] + (triplet_l_is_j__pwavePlusMinus__C_I0_grid[l + 1, :] + triplet_l_is_j__pwavePlusMinus__C_I0_grid[l - 1, :])
            triplet_l_is_j__C_I1_grid[l, :] = triplet_l_is_j__pwaveEqual__C_I1_grid[l, :] + (triplet_l_is_j__pwavePlusMinus__C_I1_grid[l + 1, :] + triplet_l_is_j__pwavePlusMinus__C_I1_grid[l - 1, :])

        #       L = J + 1
        for l in range(1, degree_pwave_expansion + 1):
            triplet_l_is_j_Plus_1__C_I0_grid[l, :] = triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid[l-1, :] + triplet_l_is_j_Plus_1__pwavePlus__C_I0_grid[l, :]   # Note that l = 0 is not a valid quantum number here
            triplet_l_is_j_Plus_1__C_I1_grid[l, :] = triplet_l_is_j_Plus_1__pwaveEqual__C_I1_grid[l-1, :] + triplet_l_is_j_Plus_1__pwavePlus__C_I1_grid[l, :]   # Note that l = 0 is not a valid quantum number here



        #singlet_C_I0_reconstructed = np.zeros((len(q_grid), len(Z_grid)))
        #for q_idx in range(q_grid.shape[0]):
        #    for Z_idx in range(Z_grid.shape[0]):
        #        singlet_C_I0_reconstructed[q_idx, Z_idx] = self.singlet_C(q_grid[q_idx], Z_grid[Z_idx], I_tchannel=0)


        # TODO specific plotting
        #q_grid_extended = np.repeat(q_grid, len(Z_grid))
        #Z_grid_extended = np.tile(Z_grid, len(q_grid))
        #plotter.plot_form_factor_np(q_grid_extended, Z_grid_extended, singlet_C_I0_reconstructed, "Singlet Central I_tchannel=0", "q", "1S0", "rho", 0, "tmp", 990)


        singlet_C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_C_I0_grid, q_grid)
        singlet_C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_C_I1_grid, q_grid)

        triplet_l_is_j_Minus_1__C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Minus_1__C_I0_grid, q_grid)
        triplet_l_is_j_Minus_1__C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Minus_1__C_I1_grid, q_grid)
        triplet_l_is_j__C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j__C_I0_grid, q_grid)
        triplet_l_is_j__C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j__C_I1_grid, q_grid)
        triplet_l_is_j_Plus_1__C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Plus_1__C_I0_grid, q_grid)
        triplet_l_is_j_Plus_1__C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Plus_1__C_I1_grid, q_grid)

        # The factors of 1/2 (2 l + 1) and 2/(2 l + 1) from the expansion
        # and orthogonality relation cancel and we can pick out what we need from
        # from the splines




        ########################### (6) ##############################
        # Fourier Transform LSJ(q) --> LSJ(r)
        r_grid = np.linspace(0, 3, 100)

        # Fourier Transform via Numeric Quadrature
        quad_ft = NumericQuadratureFT(100, 20)

        singlet_C_I0_r = np.zeros((len(singlet_C_I0_splines), len(r_grid)))
        singlet_C_I1_r = np.zeros((len(singlet_C_I1_splines), len(r_grid)))

        triplet_l_is_j_Minus_1__C_I0_r = np.zeros((len(triplet_l_is_j_Minus_1__C_I0_splines), len(r_grid)))
        triplet_l_is_j_Minus_1__C_I1_r = np.zeros((len(triplet_l_is_j_Minus_1__C_I1_splines), len(r_grid)))
        triplet_l_is_j__C_I0_r = np.zeros((len(triplet_l_is_j__C_I0_splines), len(r_grid)))
        triplet_l_is_j__C_I1_r = np.zeros((len(triplet_l_is_j__C_I1_splines), len(r_grid)))
        triplet_l_is_j_Plus_1__C_I0_r = np.zeros((len(triplet_l_is_j_Plus_1__C_I0_splines), len(r_grid)))
        triplet_l_is_j_Plus_1__C_I1_r = np.zeros((len(triplet_l_is_j_Plus_1__C_I1_splines), len(r_grid)))

        for l in range(len(singlet_C_I0_splines)):
            print(f"Fourier Transforming L={l}")
            singlet_C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : singlet_C_I0_splines[l](q), r_grid)
            singlet_C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : singlet_C_I1_splines[l](q), r_grid)

            triplet_l_is_j_Minus_1__C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Minus_1__C_I0_splines[l](q), r_grid)
            triplet_l_is_j_Minus_1__C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Minus_1__C_I1_splines[l](q), r_grid)
            triplet_l_is_j__C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j__C_I0_splines[l](q), r_grid)
            triplet_l_is_j__C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j__C_I1_splines[l](q), r_grid)
            triplet_l_is_j_Plus_1__C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Plus_1__C_I0_splines[l](q), r_grid)
            triplet_l_is_j_Plus_1__C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Plus_1__C_I1_splines[l](q), r_grid)
        

        ######################### Dynamic #########################
        singlet__callables = [self.singlet_C]

        triplet_l_is_j_Minus_1__callable_tuples = [(self.triplet_l_is_j_Minus_1__pwaveEqual__C, self.triplet_l_is_j_Minus_1__pwaveMinus__C)]
        triplet_l_is_j__callable_tuples = [(self.triplet_l_is_j__pwaveEqual__C, self.triplet_l_is_j__pwavePlusMinus__C)]
        triplet_l_is_j_Plus_1__callable_tuples = [(self.triplet_l_is_j_Plus_1__pwaveEqual__C, self.triplet_l_is_j_Plus_1__pwavePlus__C)]


        (check_singlet_C_I0_grid, check_triplet_l_is_j_Minus_1__C_I0_grid, check_triplet_l_is_j__C_I0_grid, check_triplet_l_is_j_Plus_1__C_I0_grid, check_singlet_C_I0_r, check_triplet_l_is_j_Minus_1__C_I0_r, check_triplet_l_is_j__C_I0_r, check_triplet_l_is_j_Plus_1__C_I0_r)\
            = self.calculate_LSJ_result_in_basis(0, q_grid, Z_grid, degree_pwave_expansion, r_grid, singlet__callables, triplet_l_is_j_Minus_1__callable_tuples, triplet_l_is_j__callable_tuples, triplet_l_is_j_Plus_1__callable_tuples)
        (check_singlet_C_I1_grid, check_triplet_l_is_j_Minus_1__C_I1_grid, check_triplet_l_is_j__C_I1_grid, check_triplet_l_is_j_Plus_1__C_I1_grid, check_singlet_C_I1_r, check_triplet_l_is_j_Minus_1__C_I1_r, check_triplet_l_is_j__C_I1_r, check_triplet_l_is_j_Plus_1__C_I1_r)\
            = self.calculate_LSJ_result_in_basis(1, q_grid, Z_grid, degree_pwave_expansion, r_grid, singlet__callables, triplet_l_is_j_Minus_1__callable_tuples, triplet_l_is_j__callable_tuples, triplet_l_is_j_Plus_1__callable_tuples)

        (check_singlet_C_I0_grid, check_triplet_l_is_j_Minus_1__C_I0_grid, check_triplet_l_is_j__C_I0_grid, check_triplet_l_is_j_Plus_1__C_I0_grid, check_singlet_C_I0_r, check_triplet_l_is_j_Minus_1__C_I0_r, check_triplet_l_is_j__C_I0_r, check_triplet_l_is_j_Plus_1__C_I0_r)\
            = (check_singlet_C_I0_grid[0], check_triplet_l_is_j_Minus_1__C_I0_grid[0], check_triplet_l_is_j__C_I0_grid[0], check_triplet_l_is_j_Plus_1__C_I0_grid[0], check_singlet_C_I0_r[0], check_triplet_l_is_j_Minus_1__C_I0_r[0], check_triplet_l_is_j__C_I0_r[0], check_triplet_l_is_j_Plus_1__C_I0_r[0])
        (check_singlet_C_I1_grid, check_triplet_l_is_j_Minus_1__C_I1_grid, check_triplet_l_is_j__C_I1_grid, check_triplet_l_is_j_Plus_1__C_I1_grid, check_singlet_C_I1_r, check_triplet_l_is_j_Minus_1__C_I1_r, check_triplet_l_is_j__C_I1_r, check_triplet_l_is_j_Plus_1__C_I1_r)\
            = (check_singlet_C_I1_grid[0], check_triplet_l_is_j_Minus_1__C_I1_grid[0], check_triplet_l_is_j__C_I1_grid[0], check_triplet_l_is_j_Plus_1__C_I1_grid[0], check_singlet_C_I1_r[0], check_triplet_l_is_j_Minus_1__C_I1_r[0], check_triplet_l_is_j__C_I1_r[0], check_triplet_l_is_j_Plus_1__C_I1_r[0])


        ######################### Check equal #########################
        assert np.allclose(singlet_C_I0_grid, check_singlet_C_I0_grid)
        assert np.allclose(singlet_C_I1_grid, check_singlet_C_I1_grid)
        assert np.allclose(triplet_l_is_j_Minus_1__C_I0_grid, check_triplet_l_is_j_Minus_1__C_I0_grid)
        assert np.allclose(triplet_l_is_j_Minus_1__C_I1_grid, check_triplet_l_is_j_Minus_1__C_I1_grid)
        assert np.allclose(triplet_l_is_j__C_I0_grid, check_triplet_l_is_j__C_I0_grid)
        assert np.allclose(triplet_l_is_j__C_I1_grid, check_triplet_l_is_j__C_I1_grid)
        assert np.allclose(triplet_l_is_j_Plus_1__C_I0_grid, check_triplet_l_is_j_Plus_1__C_I0_grid)
        assert np.allclose(triplet_l_is_j_Plus_1__C_I1_grid, check_triplet_l_is_j_Plus_1__C_I1_grid)

        assert np.allclose(singlet_C_I0_r, check_singlet_C_I0_r, equal_nan=True)
        assert np.allclose(singlet_C_I1_r, check_singlet_C_I1_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j_Minus_1__C_I0_r, check_triplet_l_is_j_Minus_1__C_I0_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j_Minus_1__C_I1_r, check_triplet_l_is_j_Minus_1__C_I1_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j__C_I0_r, check_triplet_l_is_j__C_I0_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j__C_I1_r, check_triplet_l_is_j__C_I1_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j_Plus_1__C_I0_r, check_triplet_l_is_j_Plus_1__C_I0_r, equal_nan=True)
        assert np.allclose(triplet_l_is_j_Plus_1__C_I1_r, check_triplet_l_is_j_Plus_1__C_I1_r, equal_nan=True)