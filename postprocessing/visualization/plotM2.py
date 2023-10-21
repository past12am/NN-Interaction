import math

import numpy as np
import matplotlib.pyplot as plt




def plot_result(pd_tau, idx_selector, img_path=None, perf_norm=False):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    ax.plot_trisurf(pd_tau["tau"][idx_selector], pd_tau["z"][idx_selector], ((1.0/(2 * (2 * math.pi)**4))**2 if perf_norm else 1) * pd_tau["|scattering_amp|2"][idx_selector])
    ax.set_xlabel("tau / 1")
    ax.set_ylabel("z / 1")
    ax.set_zlabel("$|M|^2$")
    if img_path is not None:
        plt.savefig(img_path + "/NN-Scattering.png", dpi=400)
    plt.show()
