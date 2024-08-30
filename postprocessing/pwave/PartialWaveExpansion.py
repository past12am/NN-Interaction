import numpy as np


class PartialWaveExpansion:
    """
        We assume here, that f is an array of shape (x, z), thus f(x, z)
        where x in [0, inf), z in [-1, 1]

        Note that we will do the Partial wave expansion in z

        Note that the expansion coefficients are then f_l(x)    thus have shape (l, x)
    """

    def __init__(self, f, x, z, degree, exp_type: str="fit") -> None:
        self.f_x = np.zeros((degree + 1, x.shape[0]))

        if (exp_type == "fit"):
            self.__fit_based_pwave_expansion(f, z, degree)


    def __fit_based_pwave_expansion(self, f, z, degree):
        for X_idx in range(f.shape[0]):
            legendreFit = np.polynomial.legendre.Legendre.fit(z, f[X_idx, :], deg=degree, domain=[-1, 1])

            for l in range(len(legendreFit.convert().coef)):
                self.f_x[l, X_idx] = legendreFit.convert().coef[l] / (2 * l + 1)

    def get_f_x(self):
        return self.f_x
    


