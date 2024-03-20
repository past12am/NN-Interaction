import numpy as np


class SchlessingerPointMethod:

    def __init__(self, x_vals: np.ndarray, function_vals: np.ndarray):
        self.x_vals = x_vals
        self.function_vals = function_vals

        self.a = np.zeros(len(x_vals) - 1)
        self.__calc_a_coeffs()


    def __calc_a_coeffs(self):
        for i in range(len(self.x_vals) - 1):
            Z = np.zeros(i + 1)

            Z[0] = self.function_vals[0]/self.function_vals[i+1] - 1
            for k in range(i):
                Z[k+1] = (self.a[k] * (self.function_vals[i+1] - self.function_vals[k])) / Z[k]

            self.a[i] = Z[i]/(self.x_vals[i + 1] - self.x_vals[i])


    def __calc_Z0(self, x):
        Z = np.zeros(len(self.x_vals) + 1)

        # range from len(x)-1 to 0 exclusive
        Z[len(self.x_vals) - 1] = 0
        for k in range(len(self.x_vals) - 1, 0, -1):
            Z[k - 1] = (self.a[k-1] * (x - self.x_vals[k-1])) / (1 + Z[k])

        return Z[0]


    def calc_value_at(self, x):
        Z0 = self.__calc_Z0(x)
        r = self.function_vals[0]/(1 + Z0)

        return r
