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

#PICKLE_PATH = "pickle_jar/result_Jul5th_2024_1024/*"
PICKLE_PATH = "pickle_jar/*"
KEYWORDS = "free"

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
    elif KEYWORDS not in file:
        continue

    file_list.append(file)

file_list =sorted(file_list)
final_time = 0
initial_times = 0
profilefile = None
for i,file in enumerate(file_list):

    print(file)

    if i >= 19: break

    with open(file,"rb") as handle:
        result = pickle.load(handle)

    atoms = result["atoms"]
    Afield = result["field"]

    total_energy = np.array(atoms.observable["kinetic"]) + np.array(atoms.observable["potential"]) \
            + np.sum(Afield.history["energy"],axis = 1)

    print(total_energy[0])
    print(total_energy[1])
    print(total_energy[1] * red.epsilon * 6.242e11 )

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

    omega = Afield.k_val / red.sigma
    omega /= 2*np.pi
    if np.isclose(Afield.history["t"][0], 0):
        initialfile = file
        omega_profile, initial_rad_profile = profiling_rad(omega, rad_energy[0])

    if Afield.history["t"][-1] > final_time:

        profilefile = file

        final_time = Afield.history["t"][-1]
        omega_profile, final_rad_profile = profiling_rad(omega, rad_energy[-1])

    ax2.plot(time, total_dipole)
    ax2.set_ylabel("Total dipole")

    ax1[0].plot(time, np.sum(rad_energy,axis=1))
    ax1[0].set_ylabel("Radiation energy (eV)")

rad_profile = np.array(final_rad_profile) - np.array(initial_rad_profile)
print(initialfile)
print(profilefile)

n,_,_ = ax3[1].hist(xe_velocity_dist, bins = np.arange(0,3,0.01))
ax3[0].hist(ar_velocity_dist, bins = np.arange(0,3,0.01))
ax3[0].set_ylim(0, np.max(n))
ax3[0].set_ylabel("Frequency")
ax3[0].set_xlabel("Argon velocity (km/s)")
ax3[1].set_xlabel("Xenon velocity (km/s)")

ax1[0].set_xlabel("Time (ps)")
ax1[1].scatter(omega_profile, rad_profile, s = 10)
print(len(omega_profile))
ax1[1].set_xlabel("Wavenumber (cm^-1)")
ax1[1].set_ylabel("Final energy (eV)")

fig1.savefig("figure/full_simulation_radiation"+KEYWORDS+".jpeg",dpi = 600,bbox_inches="tight")
fig2.savefig("figure/full_simulation_dipole"+KEYWORDS+".jpeg",dpi = 600,bbox_inches="tight")
fig3.savefig("figure/full_simulation_velocity_profile"+KEYWORDS+".jpeg",dpi = 600,bbox_inches="tight")
