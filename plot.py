import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

with open("result_plot/trajectory_temp.pkl","rb") as handle:
    trajectory = pickle.load(handle)

kinetic_energy = np.array(trajectory["kinetic_energy"])
potential_energy = np.array(trajectory["potential_energy"])/2
hamiltonian = kinetic_energy + potential_energy

plot_range = slice(0, len(trajectory["steps"]))

t = np.array(trajectory["steps"][plot_range])

fig, ax = plt.subplots(2,2,figsize = (12,6))
#ax[0,0].plot(t, trajectory["dipole"][plot_range])
ax[0,1].plot(t, kinetic_energy[plot_range])
ax[1,0].plot(t, potential_energy[plot_range])
ax[1,1].plot(t, hamiltonian[plot_range])

fig.savefig("result_plot/hamiltonians.jpeg", bbox_inches="tight",dpi=600)
