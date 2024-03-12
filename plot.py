import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

with open("result_plot/no_EM_trajectory.pkl","rb") as handle:
    trajectory = pickle.load(handle)

plot_range = slice(0, len(trajectory["steps"]))

t = np.array(trajectory["steps"][plot_range])* trajectory["h"]

fig, ax = plt.subplots(2,2,figsize = (12,6))
ax[0,0].plot(t, trajectory["dipole"][plot_range])
ax[0,1].plot(t, trajectory["H"][plot_range])
ax[1,0].plot(t, trajectory["T"][plot_range])
ax[1,1].plot(t, trajectory["V"][plot_range])

fig.savefig("result_plot/hamiltonians.jpeg", bbox_inches="tight",dpi=600)
