import typing
import numpy as np

from scipy.interpolate import CubicSpline

from basis.BasisTauToRho import BasisTauToRho
from data.AmplitudeHandler import AmplitudeHandlerFitfunc
from data.Dataloader import Dataloader
from numerics.NumericQuadratureFT import NumericQuadratureFT
from pwave.PartialWaveExpansion import PartialWaveExpansion
from utils.fitfunctions import yukawa_2_exponentials_v3_fitparams
from visualization.plotting import Plotter


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


    def U(self, basis_idx, I, q, Z):
        if(I == 0 or I == 1):
            return self.V__tI0(basis_idx, q, Z) + (4 * I - 3) * self.W__tI1(basis_idx, q, Z)
        
        raise Exception(f"There aint no Isospin I = {I}")
        
        
            

    def __init__(self, amplitude_handler_qx: AmplitudeHandlerFitfunc, amplitude_handler_dqx: AmplitudeHandlerFitfunc):
        self.amplitude_handler_qx = amplitude_handler_qx
        self.amplitude_handler_dqx = amplitude_handler_dqx


    def spectroscopic_basis_run(self, plotter: Plotter):
        LSJ_singlet = [(0, 0, 0), (1, 0, 1), (2, 0, 2), (3, 0, 3), (4, 0, 4)]
        LSJ_triplet__L_eq_J_plus_1 = [(None, None, None), (1, 1, 0), (2, 1, 1), (3, 1, 2), (4, 1, 3), (5, 1, 4)]
        LSJ_triplet__L_eq_J_minus_1 = [(0, 1, 1), (1, 1, 2), (2, 1, 3), (3, 1, 4), (4, 1, 5), (5, 1, 6)]

        ########################### (5) ##############################
        # Perform Partial Wave Expansion of integral kernels
        q_grid = np.linspace(0, 50, 200)
        Z_grid = np.linspace(-1, 0.9, 30)

        degree_pwave_expansion = 4

        # Central
        #   Singlets
        singlet_C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_C, I=0)
        singlet_C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_C, I=1)

        #   Triplets
        #       Raw
        #           L = J - 1
        triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveEqual__C, I=0)
        triplet_l_is_j_Minus_1__pwaveMinus__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwavePlus__C, I=0)

        triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwaveEqual__C, I=1)
        triplet_l_is_j_Minus_1__pwaveMinus__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Minus_1__pwavePlus__C, I=1)

        
        #           L = J
        # TODO

        #           L = J + 1
        triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwaveEqual__C, I=0)
        triplet_l_is_j_Plus_1__pwavePlus__C_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwavePlus__C, I=0)

        triplet_l_is_j_Plus_1__pwaveEqual__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwaveEqual__C, I=1)
        triplet_l_is_j_Plus_1__pwavePlus__C_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.triplet_l_is_j_Plus_1__pwavePlus__C, I=1)


        #       Construct Full combination
        triplet_l_is_j_Plus_1__C_I0_grid = np.zeros_like(triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j_Plus_1__C_I1_grid = np.zeros_like(triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j_Minus_1__C_I0_grid = np.zeros_like(triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid)
        triplet_l_is_j_Minus_1__C_I1_grid = np.zeros_like(triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid)

        #       L = J - 1
        for l in range(0, degree_pwave_expansion):
            triplet_l_is_j_Minus_1__C_I0_grid[l, :] = triplet_l_is_j_Minus_1__pwaveEqual__C_I0_grid[l + 1, :] + triplet_l_is_j_Minus_1__pwaveMinus__C_I0_grid[l]
            triplet_l_is_j_Minus_1__C_I1_grid[l, :] = triplet_l_is_j_Minus_1__pwaveEqual__C_I1_grid[l + 1, :] + triplet_l_is_j_Minus_1__pwaveMinus__C_I1_grid[l]

        #       L = J + 1
        for l in range(1, degree_pwave_expansion + 1):
            triplet_l_is_j_Plus_1__C_I0_grid[l, :] = triplet_l_is_j_Plus_1__pwaveEqual__C_I0_grid[l-1, :] + triplet_l_is_j_Plus_1__pwavePlus__C_I0_grid[l, 0]   # Note that l = 0 is not a valid quantum number here
            triplet_l_is_j_Plus_1__C_I1_grid[l, :] = triplet_l_is_j_Plus_1__pwaveEqual__C_I1_grid[l-1, :] + triplet_l_is_j_Plus_1__pwavePlus__C_I1_grid[l, 0]   # Note that l = 0 is not a valid quantum number here



        #singlet_C_I0_reconstructed = np.zeros((len(q_grid), len(Z_grid)))
        #for q_idx in range(q_grid.shape[0]):
        #    for Z_idx in range(Z_grid.shape[0]):
        #        singlet_C_I0_reconstructed[q_idx, Z_idx] = self.singlet_C(q_grid[q_idx], Z_grid[Z_idx], I=0)


        # TODO specific plotting
        #q_grid_extended = np.repeat(q_grid, len(Z_grid))
        #Z_grid_extended = np.tile(Z_grid, len(q_grid))
        #plotter.plot_form_factor_np(q_grid_extended, Z_grid_extended, singlet_C_I0_reconstructed, "Singlet Central I=0", "q", "1S0", "rho", 0, "tmp", 990)


        singlet_C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_C_I0_grid, q_grid)
        singlet_C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_C_I1_grid, q_grid)

        triplet_l_is_j_Minus_1__C_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Minus_1__C_I0_grid, q_grid)
        triplet_l_is_j_Minus_1__C_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(triplet_l_is_j_Minus_1__C_I1_grid, q_grid)
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
        triplet_l_is_j_Plus_1__C_I0_r = np.zeros((len(triplet_l_is_j_Plus_1__C_I0_splines), len(r_grid)))
        triplet_l_is_j_Plus_1__C_I1_r = np.zeros((len(triplet_l_is_j_Plus_1__C_I1_splines), len(r_grid)))

        for l in range(len(singlet_C_I0_splines)):
            print(f"Fourier Transforming L={l}")
            singlet_C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : singlet_C_I0_splines[l](q), r_grid)
            singlet_C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : singlet_C_I1_splines[l](q), r_grid)

            triplet_l_is_j_Minus_1__C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Minus_1__C_I0_splines[l](q), r_grid)
            triplet_l_is_j_Minus_1__C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Minus_1__C_I1_splines[l](q), r_grid)
            triplet_l_is_j_Plus_1__C_I0_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Plus_1__C_I0_splines[l](q), r_grid)
            triplet_l_is_j_Plus_1__C_I1_r[l, :] = quad_ft.fourierTransform(lambda q : triplet_l_is_j_Plus_1__C_I1_splines[l](q), r_grid)



        # Singlet
        plotter.plot_pwave_LSJ(singlet_C_I0_grid, LSJ_singlet, q_grid, "q", "GeV", "rho", "C", 0, "Central LSJ Singlet for I = 0", "LSJ_Singlet_I=0", 100)
        plotter.plot_pwave_LSJ(singlet_C_I1_grid, LSJ_singlet, q_grid, "q", "GeV", "rho", "C", 1, "Central LSJ Singlet for I = 1", "LSJ_Singlet_I=1", 100)

        # Triplet L = J - 1
        plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__C_I0_grid, LSJ_triplet__L_eq_J_minus_1, q_grid, "q", "GeV", "rho", "C", 0, "Central LSJ Triplet for I = 0, L = J - 1", "LSJ_Triplet_I=0_L=J-1", 101)
        plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__C_I1_grid, LSJ_triplet__L_eq_J_minus_1, q_grid, "q", "GeV", "rho", "C", 1, "Central LSJ Triplet for I = 1, L = J - 1", "LSJ_Triplet_I=1_L=J-1", 101)

        # Triplet L = J + 1
        plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__C_I0_grid, LSJ_triplet__L_eq_J_plus_1, q_grid, "q", "GeV", "rho", "C", 0, "Central LSJ Triplet for I = 0, L = J + 1", "LSJ_Triplet_I=0_L=J+1", 101)
        plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__C_I1_grid, LSJ_triplet__L_eq_J_plus_1, q_grid, "q", "GeV", "rho", "C", 1, "Central LSJ Triplet for I = 1, L = J + 1", "LSJ_Triplet_I=1_L=J+1", 101)




        # Singlet
        plotter.plot_pwave_LSJ(singlet_C_I0_r, LSJ_singlet, r_grid, "r", "1/GeV", "rho", "C", 0, "Central LSJ Singlet for I = 0", "LSJ_Singlet_I=0", 110)
        plotter.plot_pwave_LSJ(singlet_C_I1_r, LSJ_singlet, r_grid, "r", "1/GeV", "rho", "C", 1, "Central LSJ Singlet for I = 1", "LSJ_Singlet_I=1", 110)

        # Triplet L = J - 1
        plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__C_I0_r, LSJ_triplet__L_eq_J_minus_1, r_grid, "r", "1/GeV", "rho", "C", 0, "Central LSJ Triplet for I = 0, L = J - 1", "LSJ_Triplet_I=0_L=J-1", 101)
        plotter.plot_pwave_LSJ(triplet_l_is_j_Minus_1__C_I1_r, LSJ_triplet__L_eq_J_minus_1, r_grid, "r", "1/GeV", "rho", "C", 1, "Central LSJ Triplet for I = 1, L = J - 1", "LSJ_Triplet_I=1_L=J-1", 101)

        # Triplet L = J + 1
        plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__C_I0_r, LSJ_triplet__L_eq_J_plus_1, r_grid, "r", "1/GeV", "rho", "C", 0, "Central LSJ Triplet for I = 0, L = J + 1", "LSJ_Triplet_I=0_L=J+1", 101)
        plotter.plot_pwave_LSJ(triplet_l_is_j_Plus_1__C_I1_r, LSJ_triplet__L_eq_J_plus_1, r_grid, "r", "1/GeV", "rho", "C", 1, "Central LSJ Triplet for I = 1, L = J + 1", "LSJ_Triplet_I=1_L=J+1", 101)



        

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


    def singlet_LSJ_of_q(self, singlet_spline_list, L, S, J, q):
        if(S != 0 or L != J):
            raise Exception("L,S,J = {L}, {S}, {J} is not a singlet")
        
        return singlet_spline_list[J](q)
    

    def triplet_LSJ_of_q(self, triplet_spline_list, L, S, J, q):
        pass


    # Triplet, L = J + 1
    #   For the triplet we need to split up the current and next partial wave parts and plug them together afterwards
    def triplet_l_is_j_Plus_1__pwaveEqual__C(self, q, Z, I):
        return 0
    
    def triplet_l_is_j_Plus_1__pwaveEqual__SS(self, q, Z, I):
        return 0
    
    def triplet_l_is_j_Plus_1__pwaveEqual__T_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(2, I, q, Z)    # * 1/(2 * J + 1)
    
    def triplet_l_is_j_Plus_1__pwaveEqual__SO(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(3, I, q, Z)
    
    def triplet_l_is_j_Plus_1__pwaveEqual__Q_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -2 * np.power(r, 4) * Z * self.U(4, I, q, Z)     # * 1/(2 * J + 1)



    def triplet_l_is_j_Plus_1__pwavePlus__C(self, q, Z, I):
        return self.U(0, I, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__SS(self, q, Z, I):
        return self.U(1, I, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__T_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * self.U(2, I, q, Z)   # * 2/(2 * J + 1)
    
    def triplet_l_is_j_Plus_1__pwavePlus__SO(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * 2 * Z * self.U(3, I, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__Q_Jindep(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (1 - np.square(Z)) * self.U(4, I, q, Z)
    
    def triplet_l_is_j_Plus_1__pwavePlus__Q_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * self.U(4, I, q, Z)      # * 2/(2 * J + 1)
    


    # Triplet, L = J
    def triplet_l_is_j__kernel(self, q, Z, I):
        pass


    # Triplet, L = J - 1
    def triplet_l_is_j_Minus_1__pwaveEqual__C(self, q, Z, I):
        return 0
    
    def triplet_l_is_j_Minus_1__pwaveEqual__SS(self, q, Z, I):
        return 0
    
    def triplet_l_is_j_Minus_1__pwaveEqual__T_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -2 * np.square(r) * self.U(2, I, q, Z)    # * 1/(2 * J + 1)
    
    def triplet_l_is_j_Minus_1__pwaveEqual__SO(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.square(r) * self.U(3, I, q, Z)
    
    def triplet_l_is_j_Minus_1__pwaveEqual__Q_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return 2 * np.power(r, 4) * Z * self.U(4, I, q, Z)     # * 1/(2 * J + 1)



    def triplet_l_is_j_Minus_1__pwavePlus__C(self, q, Z, I):
        return self.U(0, I, q, Z)
    
    def triplet_l_is_j_Minus_1__pwavePlus__SS(self, q, Z, I):
        return self.U(1, I, q, Z)
    
    def triplet_l_is_j_Minus_1__pwavePlus__T_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return np.square(r) * self.U(2, I, q, Z)   # * 2/(2 * J + 1)
    
    def triplet_l_is_j_Minus_1__pwavePlus__SO(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -np.square(r) * 2 * Z * self.U(3, I, q, Z)
    
    def triplet_l_is_j_Minus_1__pwavePlus__Q_Jindep(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (1 - np.square(Z)) * self.U(4, I, q, Z)
    
    def triplet_l_is_j_Minus_1__pwavePlus__Q_Jdep_noncorr(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return -np.power(r, 4) * self.U(4, I, q, Z)      # * 2/(2 * J + 1)
    


    # Singlet
    def singlet_kernel(self, q, Z, I):
        return self.singlet_C(self, q, Z, I) + self.singlet_SS(self, q, Z, I) + self.singlet_T(self, q, Z, I) + self.singlet_SO(self, q, Z, I) + self.singlet_Q(self, q, Z, I)

    def singlet_C(self, q, Z, I):
        return self.U(0, I, q, Z)

    def singlet_SS(self, q, Z, I):
        return -3 * self.U(1, I, q, Z)

    def singlet_T(self, q, Z, I):
        return -np.square(q) * self.U(2, I, q, Z)

    def singlet_SO(self, q, Z, I):
        return 0

    def singlet_Q(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return np.power(r, 4) * (np.square(Z) - 1) * self.U(4, I, q, Z)
    

    @staticmethod    
    def r_of_q_Z(q, Z):
        return q / np.sqrt(2 * (1 - Z))