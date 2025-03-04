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
        ########################### (5) ##############################
        # Perform Partial Wave Expansion of integral kernels
        q_grid = np.linspace(0, 50, 200)
        Z_grid = np.linspace(-1, 0.9, 30)

        degree_pwave_expansion = 4
        singlet_I0_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_kernel_I0)
        singlet_I1_grid = SpectroscopicConversion.partial_wave_expand(degree_pwave_expansion, q_grid, Z_grid, self.singlet_kernel_I1)


        U_I0_reconstructed = np.zeros((len(q_grid), len(Z_grid)))
        for q_idx in range(q_grid.shape[0]):
            for Z_idx in range(Z_grid.shape[0]):
                U_I0_reconstructed[q_idx, Z_idx] = self.singlet_kernel_I0(q_grid[q_idx], Z_grid[Z_idx])


        q_grid_extended = np.repeat(q_grid, len(Z_grid))
        Z_grid_extended = np.tile(Z_grid, len(q_grid))

        # TODO specific plotting
        plotter.plot_form_factor_np(q_grid_extended, Z_grid_extended, U_I0_reconstructed, "U_I0", "q", "1S0", "rho", 0, "tmp", 990)


        singlet_I0_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_I0_grid, q_grid)
        singlet_I1_splines = SpectroscopicConversion.interpolate_expanded_in_var(singlet_I1_grid, q_grid)

        # The factors of 1/2 (2 l + 1) and 2/(2 l + 1) from the expansion
        # and orthogonality relation cancel and we can pick out what we need from
        # from the splines

        # TODO specific plotting
        plotter.plot_pwave_amp(np.array([singlet_I0_grid]), q_grid, "q", "GeV", "1S0_I0", "rho", "Singlet", 101)



        ########################### (6) ##############################
        # Fourier Transform LSJ(q) --> LSJ(r)
        r_grid = np.linspace(0, 3, 100)

        # Fourier Transform via Numeric Quadrature
        quad_ft = NumericQuadratureFT(100, 20)

        singlet_I0_r = np.zeros((len(singlet_I0_splines), len(r_grid)))
        ylims = np.zeros(2)

        for l in range(len(singlet_I0_splines)):
            print(f"Fourier Transforming L={l}")
            singlet_I0_r[l, :] = quad_ft.fourierTransform(lambda q : singlet_I0_splines[l](q), r_grid)

        y_lim_upper = np.max(singlet_I0_r[~np.isnan(singlet_I0_r)])
        ylims[:] = np.array([-y_lim_upper, y_lim_upper]) * 0.1


        # TODO specific plotting functions for this case
        plotter.plot_pwave_amp(np.array([singlet_I0_r]), r_grid, "r", "1/GeV", "1S0_I0", "rho", "Singlet", 101)
        plotter.plot_pwave_amp_scaled_side_by_side(np.array([singlet_I0_r]), r_grid, "r", "1/GeV", "1S0_I0__scaled", "rho", 102, ylims)



        

    # TODO move to dedicated class
    #   expand_function must take params (var_keep, Z)
    @staticmethod
    def partial_wave_expand(degree, grid_keep, grid_pwave, expand_function: typing.Callable):
        input_grid = np.zeros((len(grid_keep), len(grid_pwave)))

        for keep_idx, keep_var in enumerate(grid_keep):
            for Z_idx, Z in enumerate(grid_pwave):
                input_grid[keep_idx, Z_idx] = expand_function(keep_var, Z)

        res_grid = PartialWaveExpansion(input_grid, grid_keep, grid_pwave, degree).get_f_x()        
        return res_grid
    

    @staticmethod
    def interpolate_expanded_in_var(to_interpolate, var_grid):
        return [CubicSpline(var_grid, to_interpolate[l]) for l in range(to_interpolate.shape[0])]


    def singlet_LSJ_of_q(self, singlet_spline_list, L, S, J, q):
        if(S != 0 or L != J):
            raise Exception("L,S,J = {L}, {S}, {J} is not a singlet")
        
        return singlet_spline_list[J](q)


    def singlet_kernel_I0(self, q, Z):
        return self.singlet_kernel(q, Z, 0)
    
    def singlet_kernel_I1(self, q, Z):
        return self.singlet_kernel(q, Z, 1)
    
    def triplet_lneqj_pj_kernel_I0(self, q, Z):
        pass

    def triplet_lneqj_pjpm1_kernel_I0(self, q, Z):
        pass

    def triplet_leqj_pjpm_kernel_I0(self, q, Z):
        pass

    def triplet_leqj_pj_kernel_I0(self, q, Z):
        pass

    def singlet_kernel(self, q, Z, I):
        r = self.r_of_q_Z(q, Z)
        return self.U(0, I, q, Z) - 3 * self.U(1, I, q, Z) - np.square(q) * self.U(2, I, q, Z) + np.power(r, 4) * (np.square(Z) - 1) * self.U(4, I, q, Z)

    def triplet_lneqj_pj_kernel(self, q, Z, I):
        pass

    def triplet_lneqj_pjpm1_kernel(self, q, Z, I):
        pass

    def triplet_leqj_pjpm_kernel(self, q, Z, I):
        pass

    def triplet_leqj_pj_kernel(self, q, Z, I):
        pass

    @staticmethod    
    def r_of_q_Z(q, Z):
        return q / np.sqrt(2 * (1 - Z))
    