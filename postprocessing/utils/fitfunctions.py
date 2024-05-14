import numpy as np

def yukawa_potential(q2, c, L2):
    return c/(q2 + L2)


def yukawa_potential_exp_sum(q2, c, L2, a0, b0, a1, b1, offset):
    return yukawa_potential(q2, c, L2) + a0 * np.exp(b0 * q2) + a1 * np.exp(b1 * q2) + offset


def yukawa_potential_exp_sum_polyn_rank2(q2, c, L2, a0, b0, offset, c1, c2):
    return yukawa_potential(q2, c, L2) + a0 * np.exp(b0 * q2) + offset + c1 * q2 + c2 * np.power(q2, 2)


def yukawa_2_exponentials_initial_n(X, c1, c2, c3, c4, c5, n):
    return (c1 + c2 * X) * np.exp(-c3 * X) + (c4 + c5 * X) / np.power(1 + X, n)



def yukawa_2_exponentials(X, c1, c2, c3, c4, c5, c6, n):
    return (c1 + c2 * X) * np.exp(-c5 - c3 * X) + c4 / np.power(c6 + X, n)

def yukawa_2_exponentials_fitparams():
    p0=[1, 1, 0.5, 1, 1, 0, 1]
    bounds=([0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
    return yukawa_2_exponentials, p0, bounds



def yukawa_2_exponentials_v2(X, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, n):
    return (c1 + c2 * X + c7 * np.square(X)) * np.exp(-c5 - c3 * X) + (c8 + c9 * np.power(X, 3)) * np.exp(-c10 - c11 * X) + c4 / np.power(c6 + X, n)

def yukawa_2_exponentials_v2_fitparams():
    p0=[1, 1, 0.5, 1, 1, 0, 0.1, 0, 1, 0, 0.2, 1]
    bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
    return yukawa_2_exponentials_v2, p0, bounds



#def yukawa_2_exponentials_v3(X, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, n):
#    return (c1 + c2 * X + c7 * np.square(X)) * np.exp(-c5 - c3 * X) + (c8 + c9 * np.power(X, 3)) * np.exp(-c10 - c11 * X) + c4 / np.power(c6 + X, n)
#
#def yukawa_2_exponentials_v3_fitparams():
#    p0=[1, 1, 0.5, 1, 1, 0, 0.1, 0, 1, 0, 0.2, 1]
#    bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
#    return yukawa_2_exponentials_v3, p0, bounds



def yukawa_2_exponentials_lennard_jones(X, c1, c2, c3, c4, c5, c6, n, eps, sigma):
    return yukawa_2_exponentials(X, c1, c2, c3, c4, c5, c6, n) + 4 * eps * (np.power(sigma/X, 12) - np.power(sigma/X, 6))

def yukawa_2_exponentials_lennard_jones_fitparams():
    p0=[1, 1, 0.5, 1, 1, 0, 1, 1, 2]
    bounds=([0, 0, 0, 0, 0, 0, 0.5, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8, np.inf, np.inf])
    return yukawa_2_exponentials_lennard_jones, p0, bounds




def FT_yukawa_potential_exp_sum(r, c, L2, a0, b0, a1, b1, offset):
    """ We used the FT as -1/(2 pi)^3 Integral_-inf^inf{dq f(q^2) e^{i q.r}}"""

    return - c/(4.0 * np.pi) * np.exp(-np.sqrt(L2) * np.abs(r)) / (np.abs(r)) \
           - 1/(8 * np.sqrt(np.power(np.pi, 3))) * (a0/np.sqrt(np.power(-b0, 3)) * np.exp(np.square(r)/(4.0 * b0)) + a1/np.sqrt(np.power(-b1, 3)) * np.exp(np.square(r)/(4.0 * b1)))
