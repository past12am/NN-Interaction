
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd_tau = pd.read_csv("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/tau_0.txt")


idx_selector = np.where(np.logical_and(np.abs(np.array(pd_tau["PK"])) < 0.6, np.array(pd_tau["QQ"]) < -0.01))[0]

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_trisurf(pd_tau["PK"][idx_selector], pd_tau["QQ"][idx_selector], pd_tau["|scattering_amp|2"][idx_selector])
ax.set_xlabel("PK / GeV$^2$")
ax.set_ylabel("QQ / GeV$^2$")
ax.set_zlabel("$|M|^2$")
#plt.savefig("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/NN-Scattering.png", dpi=400)
plt.show()
