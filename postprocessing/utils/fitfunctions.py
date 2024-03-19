import numpy as np

def yukawa_potential(q2, c, L2):
    return c/(q2 + L2)


def yukawa_potential_exp_sum(q2, c, L2, a0, b0, a1, b1, offset):
    return yukawa_potential(q2, c, L2) + a0 * np.exp(b0 * q2) + a1 * np.exp(b1 * q2) + offset



def FT_yukawa_potential_exp_sum(r, c, L2, a0, b0, a1, b1, offset):
    """ We used the FT as -1/(2 pi)^3 Integral_-inf^inf{dq f(q^2) e^{i q.r}}"""

    return - c/(4.0 * np.pi) * np.exp(-np.sqrt(L2) * np.abs(r)) / (np.abs(r)) \
           - 1/(8 * np.sqrt(np.power(np.pi, 3))) * (a0/np.sqrt(np.power(-b0, 3)) * np.exp(np.square(r)/(4.0 * b0)) + a1/np.sqrt(np.power(-b1, 3)) * np.exp(np.square(r)/(4.0 * b1)))
