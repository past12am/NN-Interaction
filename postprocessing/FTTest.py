import typing

import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftn, fftfreq, fftshift, ifht, fhtoffset
from pyhank import HankelTransform

from numerics.NumericQuadratureFT import NumericQuadratureFT


class GaussLegendre:

    def __init__(self, n_points) -> None:
        self.n_points = n_points
        self.x, self.w = np.polynomial.legendre.leggauss(n_points)

    
    def integrate(self, f: typing.Callable, a: float, b: float):
        jacob = (b - a)/2

        integ_sum = 0 + 0j
        for i in range(self.n_points):
            integ_sum += self.w[i] * f(jacob * self.x[i] + (a + b)/2)

        integral = jacob * integ_sum

        return integral





m = 3

def yukawa_mom(q):
    return 1/(np.square(q) + np.square(m))

def yukawa_pos(r):
    return -np.exp(-m * r)/(4 * np.pi * r)

def yukawa_mom_components(qx, qy, qz):
    return yukawa_mom(np.sqrt(np.square(qx) + np.square(qy) + np.square(qz)))


def f(x, mu):
    return x**(mu + 1)*np.exp(-x**2/2)



def gl_ft_wrapper(f: typing.Callable, r):
    return lambda q: q * f(q) * np.sin(q * r)



# https://pyhank.readthedocs.io/en/latest/hankel.html
# https://www.researchgate.net/publication/8918701_Computation_of_quasi-discrete_Hankel_transforms_of_integer_order_for_propagating_optical_wave_fields/link/0c96051b8e0c556ce8000000/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19












def phi_test(k):
    return k / (1 + np.square(k))



#N = 100         # Sum Integration
#n_max = 20      # Sum - Series Convergence

r_grid = np.linspace(0, 3, 100)
ft = NumericQuadratureFT(100, 20)
V_r = ft.fourierTransform(yukawa_mom, r_grid)

    

plt.figure()
plt.plot(r_grid, V_r)
plt.plot(r_grid, yukawa_pos(r_grid))
plt.show()






exit()

# Via Gaussian quadrature
n = 3000
gl = GaussLegendre(n)


r_grid = np.linspace(0, 5, 100)
V_r = np.zeros_like(r_grid, dtype=np.complex128)

for r_idx, r in enumerate(r_grid):
    V_r[r_idx] = gl.integrate(gl_ft_wrapper(yukawa_mom, r), 0, 5)


plt.figure()
plt.plot(r_grid, np.real(V_r))
plt.plot(r_grid, np.imag(V_r))
plt.show()





exit()

# Numeric
transformer = HankelTransform(order=0, max_radius=10, n_points=8192)
V_q = yukawa_mom(2 * np.pi * transformer.v)  # = Vtwidle_nu
V_r = -2 *  transformer.iqdht(transformer.v * V_q)         # = IHT(nu * Vtwidle_nu)


# Analytic
V_r_analyt = yukawa_pos(transformer.r)


plt.figure()
plt.plot(transformer.r, V_r, label="Hankel")
plt.plot(transformer.r, V_r_analyt, label="Analytic")
#plt.ylim([np.min(V_r_analyt), np.max(V_r_analyt)])
plt.show()







exit()

mu = 1                       # Order mu of Bessel function
start = -2
stop = 3

k = np.logspace(start, stop, 100000)  # Input evaluation points
dln = np.log(k[1]/k[0])      # Step size
offset = fhtoffset(dln, initial=(start + 1) * np.log(10), mu=mu)
r = np.exp(offset)/k[::-1]   # Output evaluation points


V_k = yukawa_mom(k)
V_r = 1/(4 * np.square(np.pi)) * ifht(np.exp(2 * offset) * V_k, dln, mu=mu, offset=offset)

V_r_exact = yukawa_pos(r)
#rel_err = abs((V_r - V_r_exact)/V_r)

print(V_r_exact)



fig, ax = plt.subplots()
#ax1.set_title(r'$r^{\mu+1}\ \exp(-r^2/2)$')
#ax1.loglog(r, V_r, 'k', lw=2)
#ax1.set_xlabel('r')
#ax2.set_title(r'$k^{\mu+1} \exp(-k^2/2)$')

ax.set_xscale('log')
#ax.set_yscale('log')
#ax.plot(r, V_r_exact, label='Analytical')
ax.plot(r, V_r, label='FFTLog')
ax.set_xlabel('r')
ax.legend(loc=3, framealpha=1)
#ax2.set_ylim([0, -1e13])
#ax.set_xlim([1E-6, None])

#ax2b = ax2.twinx()
#ax2b.loglog(k, rel_err, 'C0', label='Rel. Error (-)')
#ax2b.set_ylabel('Rel. Error (-)', color='C0')
#ax2b.tick_params(axis='y', labelcolor='C0')
#ax2b.legend(loc=4, framealpha=1)
#ax2b.set_ylim([1e-9, 1e-3])
plt.show()




exit()

# Via Hankel Transform
mu = 1
p_grid = np.logspace(-7, 1, num=128)
dln = np.log(p_grid[1]/p_grid[0])

offset = fhtoffset(dln, initial=-6 * np.log(10), mu=mu)
r_grid = np.exp(offset)/p_grid[::-1]

V_p = yukawa_mom(p_grid)
##V_r = -1/(np.square(2 * np.pi) * r_grid) * ifht(np.square(p_grid) * V_p, dln=dln, mu=mu, offset=offset)
#V_r = -1/(np.square(2 * np.pi)) * ifht(np.exp(2 * offset) * V_p, dln=dln, mu=mu, offset=offset)
f_r = fht(f(V_p, mu), dln, mu=mu, offset=offset)



plt.figure()
#plt.plot(np.log(r_grid))
#plt.loglog(r_grid, np.exp(yukawa_pos(r_grid)), label="V(r) analytic")
plt.loglog(r_grid, f_r, label="V(r) numeric")
plt.legend()
plt.show()

exit()


n = 100
q_grid = np.linspace(0, 100, n)

V_pos_pre = fft(q_grid * yukawa_mom(np.abs(q_grid)))
r_grid = fftshift(fftfreq(q_grid.shape[0]))

V_pos = - 1/(2 * np.square(np.pi) * r_grid) * 1/2j * V_pos_pre



plt.figure()
plt.plot(r_grid, V_pos.imag)
plt.plot(r_grid, yukawa_pos(r_grid))
plt.show()



exit()

n = 1000
V_mom = np.zeros((n, n, n))
q_grid = np.linspace(-100, 100, n)
for i, qi in enumerate(q_grid):
    for j, qj in enumerate(q_grid):
        for k, qk in enumerate(q_grid):
            V_mom[i, j, k] = yukawa_mom_components(qi, qj, qk)


V_pos = -1/(2 * np.power(np.pi, 3)) * fftn(V_mom)
x = fftshift(fftfreq(V_mom.shape[0]))
y = fftshift(fftfreq(V_mom.shape[1]))
z = fftshift(fftfreq(V_mom.shape[2]))

r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
V_pos_r = np.zeros_like(r)
for i in range(len(x)):
    V_pos_r[i] = V_pos[i, i, i]

plt.figure()
plt.plot(x, V_pos[:, 500, 500])
#plt.plot(x, yukawa_pos(x))
plt.show()


