import warnings
warnings.filterwarnings('ignore')

import os, sys
from glob import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilities import reduced_parameter as red
from field.utils import profiling_rad

import utilities.reduced_parameter as red

PICKLE_PATH = "pickle_jar/*"
fig1, ax1 = plt.subplots(1,2,figsize = (12,4))
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots(1,2,figsize = (12,4))

ar_velocity_dist = []
xe_velocity_dist = []

file_list = []
for file in glob(PICKLE_PATH):

    if os.path.isdir(file):
        continue
    elif "note.txt" in file:
        continue

    file_list.append(file)

file_list =sorted(file_list)
final_time = 0
profilefile = None
for file in file_list:

    print(file)

    with open(file,"rb") as handle:
        result = pickle.load(handle)

    atoms = result["atoms"]
    Afield = result["field"]

    total_energy = np.array(atoms.observable["kinetic"]) + np.array(atoms.observable["potential"]) \
            + np.sum(Afield.history["energy"],axis = 1)

    print(total_energy[0])
    print(total_energy[1])

    time = np.array(atoms.observable["t"]) * red.time_unit * 1e12

    for i, e in enumerate(atoms.elements):

        v = atoms.trajectory["r_dot"][0][i]
        v = np.sqrt(np.sum(v * v))
        v = v * red.velocity_unit * 1e-2 * 1e-3

        if e == "Ar":
            ar_velocity_dist.append(v)
        elif e == "Xe":
            xe_velocity_dist.append(v)

    total_dipole = np.array(atoms.observable["total_dipole"])

    rad_energy = np.array(Afield.history["energy"]) * red.epsilon * 6.242e11 
    if Afield.history["t"][-1] > final_time:

        final_time = Afield.history["t"][-1]
        profilefile = file

        omega = Afield.k_val / red.sigma
        omega /= 2*np.pi
        omega_profile, rad_profile = profiling_rad(omega, rad_energy[-1])

    ax2.plot(time, total_dipole)
    ax2.set_ylabel("Total dipole")

    ax1[0].plot(time, np.sum(rad_energy,axis=1))
    ax1[0].set_ylabel("Radiation energy (eV)")

print(profilefile)
n,_,_ = ax3[1].hist(xe_velocity_dist, bins = np.arange(0,3,0.01))
ax3[0].hist(ar_velocity_dist, bins = np.arange(0,3,0.01))
ax3[0].set_ylim(0, np.max(n))
ax3[0].set_ylabel("Frequency")
ax3[0].set_xlabel("Argon velocity (km/s)")
ax3[1].set_xlabel("Xenon velocity (km/s)")

ax1[0].set_xlabel("Time (ps)")
ax1[1].scatter(omega_profile, rad_profile, s = 10)
ax1[1].set_xlabel("Wavenumber (cm^-1)")
ax1[1].set_ylabel("Final energy (eV)")

fig1.savefig("figure/full_simulation_radiation.jpeg",dpi = 600,bbox_inches="tight")
fig2.savefig("figure/full_simulation_dipole.jpeg",dpi = 600,bbox_inches="tight")
fig3.savefig("figure/full_simulation_velocity_profile.jpeg",dpi = 600,bbox_inches="tight")
