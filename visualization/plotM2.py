
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd_tau = pd.read_csv("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/tau_0.txt")


fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.plot_trisurf(pd_tau["QQ"], pd_tau["PK"], pd_tau["|scattering_amp|2"])
ax.set_xlabel("QQ / GeV$^2$")
ax.set_ylabel("PK / GeV$^2$")
ax.set_zlabel("$|M|^2$")
#plt.show()
plt.savefig("/home/past12am/OuzoCloud/Studium/Physik/6_Semester/SE_Bachelorarbeit/NNInteraction/data/NN-Scattering_M2.png", dpi=400)