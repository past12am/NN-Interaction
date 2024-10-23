from abc import abstractmethod
from inspect import signature
import typing

import numpy as np

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

from utils.fitfunctions import yukawa_potential_exp_sum
from pwave.PartialWaveExpansion import PartialWaveExpansion
from extrapolation.SchlessingerPointMethod import SchlessingerPointMethod

from utils.fitfunctions import *


class AmplitudeHandler:

    def __init__(self, X, Z, f) -> None:
        """
            f   has shape (basis, X, Z)
            X    has shape (X)
            Z    has shape (Z)

            fitfunc must be of the form callable(X, **params)
        """

        self.X = X
        self.Z = Z
        self.f = f

        self.X_max = X[-1]

        self.f_l = None             # shape (basis, l, X)

        self.f_l_cspline = None     # list of CubicSpline() of shape (basis, l)

        self.f_l_q = None           # shape (basis, l, q_gridpoints)

        self.q = None


    @abstractmethod
    def f_l_fit(self, basis_idx, l, X):
        pass

    @abstractmethod
    def fit_large_X_behaviour(self):
        pass

    
    def interpolate_in_X(self):
        self.f_l_cspline = [[CubicSpline(self.X, self.f_l[basis_idx, l]) for l in range(self.f_l.shape[1])] for basis_idx in range(self.f_l.shape[0])]
    

    def f_l_interpolation(self, basis_idx, l, X):
        return self.f_l_cspline[basis_idx][l](X)
    

    def interpolate_in_q(self):
        self.f_l_q_cspline = [[CubicSpline(self.q, self.f_l_q[basis_idx, l]) for l in range(self.f_l_q.shape[1])] for basis_idx in range(self.f_l_q.shape[0])]

    
    def f_l_q_interpolation(self, basis_idx, l, q):
        return self.f_l_q_cspline[basis_idx][l](q)


    def partial_wave_expand(self, degree_pwave_exp):
        # f_l has shape (basis, l, X)
        self.f_l = np.zeros((self.f.shape[0], degree_pwave_exp+1, self.f.shape[1]))

        for basis_idx in range(self.f.shape[0]):
            self.f_l[basis_idx, :, :] = PartialWaveExpansion(self.f[basis_idx, :], self.X, self.Z, degree_pwave_exp).get_f_x()


    def partial_wave_expand_q(self, degree_pwave_exp: int, q_grid: np.ndarray, Z_grid: np.ndarray, fitonly: bool=False):
        self.q = q_grid

        f_q_generated = np.zeros((self.f.shape[0], len(q_grid), len(Z_grid)))
        for basis_idx in range(f_q_generated.shape[0]):
            for q_idx, q in enumerate(q_grid):
                for Z_idx, Z in enumerate(Z_grid):
                    f_q_generated[basis_idx, q_idx, Z_idx] = self.f_q_at(basis_idx, q, Z, fitonly=fitonly)

        # f_l_q has shape (basis, l, num_q_gridpoints)
        self.f_l_q = np.zeros((self.f.shape[0], degree_pwave_exp+1, len(q_grid)))

        for basis_idx in range(f_q_generated.shape[0]):
            self.f_l_q[basis_idx, :, :] = PartialWaveExpansion(f_q_generated[basis_idx, :], q_grid, Z_grid, degree_pwave_exp).get_f_x()
        

    def f_at(self, basis_idx, X, Z):
        assert X >= 0
        assert Z <= 1 and Z >= -1

        res = 0.0

        for l in range(self.f_l.shape[1]):
            leg_coef_array = np.zeros(l + 1)
            leg_coef_array[l] = 1.0

            legpol_val = np.polynomial.legendre.legval(Z, leg_coef_array)

            res += legpol_val * (2*l + 1) * self.f_l_at(basis_idx, l, X)

        return res
    

    def f_fit_at(self, basis_idx, X, Z):
        assert X >= 0
        assert Z <= 1 and Z >= -1

        res = 0.0

        for l in range(self.f_l.shape[1]):
            leg_coef_array = np.zeros(l + 1)
            leg_coef_array[l] = 1.0

            legpol_val = np.polynomial.legendre.legval(Z, leg_coef_array)

            res += legpol_val * (2*l + 1) * self.f_l_fit(basis_idx, l, X)

        return res


    def f_l_at(self, basis_idx, l, X):
        #if X <= self.X_max:
        #    ## avoid using spline --> try to stick to known X for X < X_max
        #    #closest_idx = (np.abs(self.X - X)).argmin()
        #    #if(np.isclose(self.X[closest_idx], X)):
        #    #    return self.f_l[basis_idx, l, closest_idx]
        #    #print(f"Needed interpolation in known region: X_query={X}, X_closest={self.X[closest_idx]} --> dif={np.abs(self.X[closest_idx] - X)}")

        #    return self.f_l_interpolation(basis_idx, l, X)
        #

        return self.f_l_fit(basis_idx, l, X)
        
    
    def fit_q_pwaves(self, q_fitfunc: typing.Callable, p0: typing.List, bounds: typing.Tuple, FT_q_fitfunc: typing.Callable, max_l: int=None, min_q: float=0):
        self.q_fitfunc = q_fitfunc
        self.FT_q_fitfunc = FT_q_fitfunc

        self.f_l_q_fitcoeff = np.zeros((self.f_l_q.shape[0], self.f_l_q.shape[1], len(signature(self.q_fitfunc).parameters) - 1))

        min_q_idx = np.argwhere(self.q >= min_q)[0][0]

        for basis_idx in range(self.f_l_q.shape[0]):
            for l in range(self.f_l_q.shape[1] if max_l is None else max_l):
                try:
                    popt, pcov = curve_fit(q_fitfunc, self.q[min_q_idx:], self.f_l_q[basis_idx, l, min_q_idx:], 
                                        #p0=[1, 1, 0.5, 1, 1, 0, 1], 
                                        #bounds=([0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8]), 
                                        p0=p0,
                                        bounds=bounds,
                                        max_nfev=100000,
                                        ftol=1E-14,
                                        xtol=1E-14,
                                        gtol=1E-14)
                    
                    self.f_l_q_fitcoeff[basis_idx, l, :] = popt

                    print(f"n = {self.f_l_q_fitcoeff[basis_idx, l, -1]}")

                    import matplotlib.pyplot as plt
                    q_check = np.linspace(self.q[0], self.q[-1], 1000)
                    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

                    plt.title(f"basis_idx {basis_idx}")

                    axs[0].plot(self.q, self.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")
                    axs[0].plot(q_check, self.f_l_q_fit_at(basis_idx, l, q_check), label=f"f_{l}(q) fit")

                    axs[1].loglog(self.q, self.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")
                    axs[1].loglog(q_check, self.f_l_q_fit_at(basis_idx, l, q_check), label=f"f_{l}(q) fit")
                    
                    axs[0].set_xlabel("$q$")
                    axs[0].set_ylabel("$f_{l}(q)$")

                    axs[1].set_xlabel("$\\log q$")
                    axs[1].set_ylabel("$\\log f_{l}(q)$")

                    axs[0].legend()
                    axs[1].legend()

                    plt.show()
                    plt.close()
                
                except Exception as e:
                    print(e)

                    import matplotlib.pyplot as plt
                    q_check = np.linspace(self.q[0], self.q[-1], 1000)
                    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

                    plt.title(f"basis_idx {basis_idx}")

                    axs[0].plot(self.q, self.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")

                    axs[1].loglog(self.q, self.f_l_q[basis_idx, l, :], label=f"f_{l}(q)")
                    
                    axs[0].set_xlabel("$q$")
                    axs[0].set_ylabel("$f_{l}(q)$")

                    axs[1].set_xlabel("$\\log q$")
                    axs[1].set_ylabel("$\\log f_{l}(q)$")

                    axs[0].legend()
                    axs[1].legend()

                    plt.show()
                    plt.close()

        

    @staticmethod
    def X_at(q, Z):
        # from q = M sqrt(2X) sqrt(1 - Z), we know that q has to be 0 if Z = 1, and thus also X = 0 from the inverse relation
        if (Z == 1):
            return 0
        
        M_nucleon = 0.94 # TODO pass
        X = np.square(q) / (2.0 * np.square(M_nucleon) * (1.0 - Z))

        return X


    def f_q_at(self, basis_idx, q, Z, fitonly: bool=False):
        if (fitonly):
            return self.f_fit_at(basis_idx, self.X_at(q, Z), Z)
        
        else:
            return self.f_at(basis_idx, self.X_at(q, Z), Z)
        

    def f_l_q_at(self, basis_idx, l, q):

        if ((hasattr(q, "__len__") and (q[0] < np.min(self.q) or q[-1] > np.max(self.q))) or \
            q < np.min(self.q) or q > np.max(self.q)):

            # TODO better way than assuming 0 for out of bounds needed
            if(np.all(q > np.max(self.q))):
                return 0

            raise Exception(f"Out of range for interpolated q --> extrapolate --> q in [{np.min(self.q)}, {np.max(self.q)}] and tried q = {q}")
        
        else:
            return self.f_l_q_interpolation(basis_idx, l, q)
        

    def f_l_q_fit_at(self, basis_idx, l, q):
        return self.q_fitfunc(q, *self.f_l_q_fitcoeff[basis_idx, l, :])
    

    def f_l_r_at(self, basis_idx, l, r):
        return self.FT_q_fitfunc(r, *self.f_l_q_fitcoeff[basis_idx, l, :])

        

    
        
    

class AmplitudeHandlerFitfunc(AmplitudeHandler):

    def __init__(self, X, Z, f, fitfunc: typing.Callable[..., typing.Any] = yukawa_potential_exp_sum, p0: typing.List=None, bounds: typing.Tuple=None) -> None:
        super().__init__(X, Z, f)

        self.fitfunc = fitfunc
        self.p0 = p0
        self.bounds = bounds

        self.f_l_fitcoeff = None    # shape (basis, l, #coefs fitfunction)


    def fit_large_X_behaviour(self, max_l: int=None, min_X: float=0):
        self.f_l_fitcoeff = np.zeros((self.f_l.shape[0], self.f_l.shape[1], len(signature(self.fitfunc).parameters) - 1))

        min_X_idx = np.argwhere(self.X >= min_X)[0][0]

        for basis_idx in range(self.f_l.shape[0]):
            for l in range(self.f_l.shape[1] if max_l is None else max_l):
                popt, pcov = curve_fit(self.fitfunc, self.X[min_X_idx:], self.f_l[basis_idx, l, min_X_idx:], 
                                    #p0=[1, 1, 0.5, 1, 1, 0, 1], 
                                    #bounds=([0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8]), 
                                    p0=self.p0,
                                    bounds=self.bounds,
                                    max_nfev=100000,
                                    ftol=1E-15,
                                    xtol=1E-15,
                                    gtol=1E-15)
                
                self.f_l_fitcoeff[basis_idx, l, :] = popt

    
    def f_l_fit(self, basis_idx, l, X):
        return self.fitfunc(X, *self.f_l_fitcoeff[basis_idx, l, :])
    


class AmplitudeHandlerSchlessinger(AmplitudeHandler):

    def __init__(self, X, Z, f) -> None:
        super().__init__(X, Z, f)

        self.f_l_schless = None     # shape (basis, l)  of type SchlessingerPointMethod()



    def f_l_fit(self, basis_idx, l, X):
        return self.f_l_schless[basis_idx][l].calc_value_at(X)


    def fit_large_X_behaviour(self):
        self.f_l_schless = [[SchlessingerPointMethod(self.X, self.f_l[basis_idx, l, :]) for l in range(self.f_l.shape[1])] for basis_idx in range(self.f_l.shape[0])]
    