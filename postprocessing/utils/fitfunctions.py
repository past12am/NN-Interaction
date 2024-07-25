import numpy as np
from scipy.special import gamma

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
    #p0=[1, 1, 0.1, 1, 5, 1, 1, 5, 1, 5, 0.1, 1]
    #bounds=([0, -np.inf, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])

    p0=[1, 1, 0.5, 1, 1, 0, 0.1, 0, 1, 0, 0.2, 1]
    bounds=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
    return yukawa_2_exponentials_v2, p0, bounds



def yukawa_2_exponentials_v3(X, c1, c2, a1, a2, b1, b2, d1, n):
    return (c1 + c2 * X) * np.exp(-a1 - b1 * X) + d1 / np.power(1 + X, n)

def yukawa_2_exponentials_v3_fitparams():
    p0=[1, 1, 0, 0, 1, 1, 1, 2]
    bounds=([-np.inf, -np.inf, 0, 0, 0.5, 0.5, -np.inf, 1.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
    return yukawa_2_exponentials_v3, p0, bounds


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



 

# Fitfunctions for q partial waves
def yukawa_poly_exponentials(q, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, d1, n1, n2):
    return c1 * np.exp(-a1 - b1 * q) + c2 * q * np.exp(-a2 - b2 * q) + c3 * np.square(q) * np.exp(-a3 - b3 * q) \
           + d1 / np.power(1 + q, n1) \
           + 1 / np.power(1 + q, n2)
           #+ c4 * np.power(q, 3) * np.exp(-a4 - b4 * q) \
           #+ c5 * np.power(q, 4) * np.exp(-a5 - b5 * q) \
           #+ c6 * np.power(q, 4) * np.exp(-a6 - b6 * q)

def yukawa_poly_exponentials_FT(r, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, d1, n1, n2):
    return poly_exp_0_FT(r, a1, b1, c1) + poly_exp_1_FT(r, a2, b2, c2) + poly_exp_2_FT(r, a3, b3, c3) \
           + yukawa_FT(r, d1, n1) + yukawa_FT(r, 1, n2)
    # + poly_exp_3_FT(r, a4, b4, c4) \
    #+ poly_exp_4_FT(r, a5, b5, c5) \

def yukawa_poly_exponentials_fitparams():
    #p0=[1, 1, 0.1, 1, 5, 1, 1, 5, 1, 5, 0.1, 1]
    #bounds=([0, -np.inf, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])

    #p0=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4]
    p0=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 2, 4]
    bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1, 1.5, 1.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8, 8])
    return yukawa_poly_exponentials, p0, bounds, yukawa_poly_exponentials_FT





# Fitfunctions for q partial waves
def yukawa_poly_exponentials_v2(q, a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, n1, n2):
    return c1 * np.exp(-a1 - b1 * q) + c2 * q * np.exp(-a2 - b2 * q) + c3 * np.square(q) * np.exp(-a3 - b3 * q) \
           + d1 / np.power(1 + q, n1) \
           + 1 / np.power(1 + q, n2)

def yukawa_poly_exponentials_v2_FT(r, a1, a2, a3, b1, b2, b3, c1, c2, c3, d1, n1, n2):
    return poly_exp_0_FT(r, a1, b1, c1) + poly_exp_1_FT(r, a2, b2, c2) + poly_exp_2_FT(r, a3, b3, c3) \
           + yukawa_FT(r, d1, n1) + yukawa_FT(r, 1, n2)

def yukawa_poly_exponentials_v2_fitparams():
    #p0=[1, 1, 0.1, 1, 5, 1, 1, 5, 1, 5, 0.1, 1]
    #bounds=([0, -np.inf, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])

    #p0=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4]
    p0=[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 2, 4]
    bounds=([-np.inf, -np.inf, -np.inf, 0, 0, 0, -np.inf, -np.inf, -np.inf, -1, 1.5, 1.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8, 8])
    return yukawa_poly_exponentials_v2, p0, bounds, yukawa_poly_exponentials_v2_FT





# Fitfunctions for q partial waves
def yukawa_poly_exponentials_v3(q, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, e1, e2, n1, n2):
    return c1 * np.exp(-a1 - b1 * q) + c2 * q * np.exp(-a2 - b2 * q) + c3 * np.square(q) * np.exp(-a3 - b3 * q) \
           + d1 / np.power(e1 + q, n1) \
           + d2 / np.power(e2 + q, n2) \
           + c4 * np.power(q, 3) * np.exp(-a4 - b4 * q) \
           #+ c5 * np.power(q, 4) * np.exp(-a5 - b5 * q) \
           #+ c6 * np.power(q, 4) * np.exp(-a6 - b6 * q)

def yukawa_poly_exponentials_v3_FT(r, a1, a2, a3, a4, b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, e1, e2, n1, n2):
    return poly_exp_0_FT(r, a1, b1, c1) + poly_exp_1_FT(r, a2, b2, c2) + poly_exp_2_FT(r, a3, b3, c3) \
           + yukawa_extended_FT(r, d1, e1, n1) + yukawa_extended_FT(r, d2, e2, n2) \
           + poly_exp_3_FT(r, a4, b4, c4)
    #+ poly_exp_4_FT(r, a5, b5, c5) \

def yukawa_poly_exponentials_v3_fitparams():
    #p0=[1, 1, 0.1, 1, 5, 1, 1, 5, 1, 5, 0.1, 1]
    #bounds=([0, -np.inf, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])

    #p0=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4]
    p0=[0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.25, 0, 1, 1, 0.5, 1, 2, 4]
    bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf, 1.5, 1.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8, 8])
    return yukawa_poly_exponentials_v3, p0, bounds, yukawa_poly_exponentials_v3_FT





def yukawa_poly_exponentials_evenodd(q, a1, a2, b1, b2, c1, c2, c3, c4, c5, d1, d2, n):
    return (c1 + c3 * np.square(q) + c5 * np.power(q, 4)) * np.exp(-a1 - b1 * q) + (c2 * q + c4 * np.power(q, 3)) * np.exp(-a2 - b2 * q) 
           #+ d1 / np.power(d2 + q, n)

def yukawa_poly_exponentials_evenodd_fitparams():
    #p0=[1, 1, 0.1, 1, 5, 1, 1, 5, 1, 5, 0.1, 1]
    #bounds=([0, -np.inf, 0, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, 0, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])

    p0=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    bounds=([-np.inf, -np.inf, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0, 0.5], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 8])
    return yukawa_poly_exponentials_evenodd, p0, bounds





# Fourier Transformed basis functions
#def yukawa_FT(r, d1, n):
#    return - 1/(2 * np.square(np.pi * r) * np.abs(r) * gamma(n)) * d1 * (2 * np.power(np.abs(r), 3) * gamma(-3 + n) * hyp1f2(2, 2-n/2, (5 - n)/2, -np.square(r)/4) \
#                                                                         - np.pi * np.power(np.abs(r), n) * csc(n * np.pi) * (np.abs(r) * np.cos(n * np.pi/2 + np.abs(r)) + (n - 1) * np.sin(n * np.pi/2 + np.abs(r))))
#
#def yukawa_extended_FT(r, d, e, n):
#    return - (d * np.power(e, -n) * csc(n * np.pi / 2) * sec(n * np.pi / 2) * (-np.power(e, 1 + n) * np.pi * np.power(np.abs(r), 1 + n) * np.cos(n * np.pi / 2 + e * np.abs(r)) \
#                                                                               + 2 * np.power(e, 3) * np.power(np.abs(r), 3) * gamma(n - 3) * hyp1f2(2, 2 - n/2, (5 - n)/2, -1/4 * np.square(e * r)) * np.sin(n * np.pi) \
#                                                                               - np.power(e, n) * (n - 1) * np.pi * np.power(np.abs(r), n) * np.sin(n * np.pi / 2 + e * np.abs(r)))) \
#            / (4 * np.square(np.pi * r) * np.abs(r) * gamma(n))
#
#def poly_exp_0_FT(r, a, b, c):
#    return - (b * c * np.exp(-a)) / (np.square(np.pi) * np.square(np.square(b) + np.square(r)))
#
#def poly_exp_1_FT(r, a, b, c):
#    return (c * np.exp(-a) * (-3 * np.square(b) + np.square(r))) / (np.square(np.pi) * np.power(np.square(b) + np.square(r), 3))
#
#def poly_exp_2_FT(r, a, b, c):
#    return - (12 * b * c * np.exp(-a) * (np.square(b) - np.square(r))) / (np.square(np.pi) * np.power(np.square(b) + np.square(r), 4))
#
#def poly_exp_3_FT(r, a, b, c):
#    return - (12 * c * np.exp(-a) * (5 * np.power(b, 4) - 10 * np.square(b * r) + np.power(r, 4))) / (np.square(np.pi) * np.power(np.square(b) + np.square(r), 5))
#
#def poly_exp_4_FT(r, a, b, c):
#    return - (120 * b * c * np.exp(-a) * (3 * np.power(b, 4) - 10 * np.square(b * r) + 3 * np.power(r, 4))) / (np.square(np.pi) * np.power(np.square(b) + np.square(r), 6))