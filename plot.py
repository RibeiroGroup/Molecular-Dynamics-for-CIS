from math import floor
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

from reduced_parameter import time_unit, dipole_unit, epsilon, sigma

with open("result_plot/trajectory_temp.pkl","rb") as handle:
    trajectory = pickle.load(handle)

print(trajectory.keys())

kinetic_energy = np.array(trajectory["kinetic_energy"])
potential_energy = np.array(trajectory["potential_energy"])

rad_energy = np.array(trajectory["EM_energy"])
print(rad_energy.shape)
total_rad_energy = np.sum(rad_energy,axis = 1)

hamiltonian = kinetic_energy + potential_energy + total_rad_energy

t = np.array(trajectory["time"]) * time_unit * 1e12 #(ps)
#t = t[t < 0.006]

plot_range = slice(0, len(t))
L = 20
print(L * sigma * 1e8)

##############
### DIPOLE ###
##############
###################################
### TOTAL RADIATION HAMILTONIAN ###
###################################

fig, ax = plt.subplots(2,figsize = (6,8))

dipole = np.array(trajectory["total dipole"][plot_range])
dipole *= dipole_unit
ax[0].plot(t, dipole)
ax[0].set_xlabel("Time (ps)")
ax[0].set_ylabel("Total dipole (statC.cm)")

Hem = np.array(total_rad_energy[plot_range])
Hem *= epsilon * 6.2415e11
ax[1].plot(t, Hem)
ax[1].set_xlabel("Time (ps)")
ax[1].set_ylabel("Radiation Hamiltonian (eV)")

fig.savefig("result_plot/radiation.jpeg", bbox_inches="tight",dpi=600)

#############################
### RADIATION HAMILTONIAN ###
#############################

fig, ax = plt.subplots(5,2,figsize = (8,12))

k_vector = np.vstack([
        np.array([[1,0,0],[0,1,0],[0,0,1]]),
        np.array([[1,1,0],[0,1,1],[1,0,1]]),
        np.array([[1,1,1]]),
        np.array([[1,0,0],[0,1,0],[0,0,1]]) * 2,
        np.array([[1,1,0],[0,1,1],[1,0,1]]) * 2,
        np.array([[1,1,1],[1,-1,1],[1,1,-1]]) * 2,
        np.array([[1,2,0],[0,1,2],[1,0,2]]),
        np.array([[2,1,0],[0,2,1],[2,0,1]]),
        np.array([[2,1,1],[1,2,1],[1,1,2]]),
        np.array([[2,2,1],[1,2,2],[2,1,2]]),
        ])

k_val = np.einsum("ki,ki->k",k_vector,k_vector)
unique_kval = list(set(list(k_val)))
print(unique_kval)

for i, k in enumerate(unique_kval):

    k_val_cm = np.sqrt(k) * 1e-2 / (L * sigma)
    print(k_val_cm)

    H_em_ = np.array(rad_energy)[:,k_val == k]
    H_em_ = np.sum(H_em_[plot_range],axis=1) * epsilon * 6.2415e11 * 1e6

    ax[i%5, int(i/5)].plot(t, H_em_)
    ax[i%5, int(i/5)].set_ylim(-1e-2,1)

    ax[i%5, int(i/5)].text(0.75, 0.9, r"$\tilde{\nu}$ =" + " {:.0f} ".format(k_val_cm)+r" $cm^{-1}$",
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[i%5, int(i/5)].transAxes)

    if int(i/5) == 0:
        ax[i%5, int(i/5)].set_ylabel(r"Mode energy $(10^{-6}eV)$")

    if i % 5 == 4:
        ax[i%5, int(i/5)].set_xlabel(r"Time (ps)")

Hem = np.array(total_rad_energy[plot_range])
Hem *= epsilon * 6.2415e11 * 1e6
ax[4,1].plot(t, Hem)
ax[4,1].text(0.75, 0.9, "total",
         horizontalalignment='center',
         verticalalignment='center',
         transform = ax[4,1].transAxes)

ax[4,1].set_xlabel(r"Time (ps)")


fig.savefig("result_plot/radiation_mode.jpeg",bbox_inches ="tight", dpi = 600)
"""
ax[0,1].plot(t, kinetic_energy[plot_range])
ax[0,1].set_xlabel("Time")
ax[0,1].set_ylabel("Kinetic energy")

ax[1,1].plot(t, potential_energy[plot_range])
ax[1,1].set_xlabel("Time")
ax[1,1].set_ylabel("Potential energy")

ax[2,1].plot(t, hamiltonian[plot_range])
ax[2,1].set_xlabel("Time")
ax[2,1].set_ylabel("Hamiltonian")

fig.savefig("result_plot/hamiltonians.jpeg", bbox_inches="tight",dpi=600)


fig, ax = plt.subplots(5,3,figsize = (20,12))

for i in range(rad_energy.shape[1]):
    try:
        ax[i%5,floor(i/5)].plot(t, rad_energy[plot_range,i])
    except:
        break

fig.savefig("result_plot/em.jpeg",bbox_inches ="tight", dpi = 600)
"""
