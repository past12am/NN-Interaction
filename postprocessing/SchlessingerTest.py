import numpy as np
import matplotlib.pyplot as plt

from extrapolation.SchlessingerPointMethod import SchlessingerPointMethod



def test_func(x_grid):
    return 1/(x_grid + 100)

def test_func(x_grid):
    return np.exp(x_grid)

def schlessinger_test():
    x_grid = np.linspace(0, 15, 500)
    source_vals = test_func(x_grid)

    selected_indices = range(80,100,5)
    sp = SchlessingerPointMethod(x_grid[selected_indices], source_vals[selected_indices])

    schlessinger_vals = list()
    for x in x_grid:
        schlessinger_vals.append(sp.calc_value_at(x))


    plt.figure()
    plt.ylim(-1000, 1000)
    plt.plot(x_grid, source_vals, label="Original")
    plt.plot(x_grid, schlessinger_vals, label="SPM")
    plt.show()