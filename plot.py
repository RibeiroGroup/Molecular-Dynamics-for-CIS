from math import floor
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

with open("result_plot/trajectory_temp.pkl","rb") as handle:
    trajectory = pickle.load(handle)

kinetic_energy = np.array(trajectory["kinetic_energy"])
potential_energy = np.array(trajectory["potential_energy"])

rad_energy = np.array(trajectory["EM_energy"])
print(rad_energy.shape)
total_rad_energy = np.sum(rad_energy,axis = 1)

hamiltonian = kinetic_energy + potential_energy + total_rad_energy

t = np.array(trajectory["time"])
t = t[t < 0.006]

plot_range = slice(0, len(t))

fig, ax = plt.subplots(3,2,figsize = (12,12))

ax[0,0].plot(t, trajectory["total dipole"][plot_range])
ax[0,0].set_xlabel("Time")
ax[0,0].set_ylabel("Total dipole")

ax[0,1].plot(t, kinetic_energy[plot_range])
ax[0,1].set_xlabel("Time")
ax[0,1].set_ylabel("Kinetic energy")

ax[1,0].plot(t, total_rad_energy[plot_range])
ax[1,0].set_xlabel("Time")
ax[1,0].set_ylabel("EM_energy")

ax[1,1].plot(t, potential_energy[plot_range])
ax[1,1].set_xlabel("Time")
ax[1,1].set_ylabel("Potential energy")

ax[2,1].plot(t, hamiltonian[plot_range])
ax[2,1].set_xlabel("Time")
ax[2,1].set_ylabel("Hamiltonian")

fig.savefig("result_plot/hamiltonians.jpeg", bbox_inches="tight",dpi=600)


fig, ax = plt.subplots(5,3,figsize = (20,12))

for i in range(rad_energy.shape[1]):
    ax[i%5,floor(i/5)].plot(t, rad_energy[plot_range,i])

fig.savefig("result_plot/em.jpeg",bbox_inches ="tight", dpi = 600)
