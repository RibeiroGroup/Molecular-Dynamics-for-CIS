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
from utilities.etc import binning, moving_average

#PICKLE_PATH = os.path.expanduser("~/code/mm_polariton_pickle_ja/result_Jul11st_2024_1319/*")
#PICKLE_PATH = "pickle_jar/result_Jul12nd_2024_0939/*"
PICKLE_PATH = "pickle_jar/*"
#PICKLE_PATH = "pickle_jar/cluster/result_Jul15th_2024_1018/*"
#KEYWORDS = "cavity"

rad_profile_list = []

fig1, ax1 = plt.subplots(2,figsize = (6,8))
fig2, ax2 = plt.subplots(2,figsize = (6,8))

for j, KEYWORDS in enumerate(["cavity","free"]):
    fig3, ax3 = plt.subplots(1,2,figsize = (12,4))
    fig4, ax4 = plt.subplots(1,2,figsize = (12,4))

    ar_velocity_dist = []
    xe_velocity_dist = []

    file_dict = {}
    for file in glob(PICKLE_PATH):

        if os.path.isdir(file):
            continue
        elif KEYWORDS not in file or "result" not in file:
            continue

        number = file.split(".")[0]
        number = number.split("_")[-1]
        number = int(number)

        file_dict[number] = file

    final_time = 0
    initial_times = 0
    profilefile = None

    for i,file in file_dict.items():

        if i >= 4: continue

        print(file)

        with open(file,"rb") as handle:
            result = pickle.load(handle)

        atoms = result["atoms"]
        print("Total number of atoms: ",atoms.N_atoms)
        print("Temperature",result["temperature"])

        Afield = result["probe_field"]
        cave_field = result["cavity_field"]

        total_energy = np.array(atoms.observable["kinetic"]) + np.array(atoms.observable["potential"]) \
                + np.sum(Afield.history["energy"],axis = 1) 
        if KEYWORDS == "cavity":
            total_energy += np.sum(cave_field.history["energy"],axis = 1)
            print("cavity energy")
            print(np.sum(cave_field.history["energy"],axis = 1)[-1])
            print(np.sum(cave_field.history["energy"],axis = 1)[0])

        print("total energy")
        print(total_energy[0])
        print(total_energy[1])
        print("Probe field energy")
        print(np.sum(Afield.history["energy"],axis = 1)[0])
        print(np.sum(Afield.history["energy"],axis = 1)[-1])

        print("Kinetic energy")
        print(np.array(atoms.observable["kinetic"])[0])

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
        #rad_energy *= 8065.56 # convert from eV to cm^-1

        omega = Afield.k_val / red.sigma
        omega /= 2*np.pi
        if np.isclose(Afield.history["t"][0], 0):
            initialfile = file
            omega_profile, initial_rad_profile = profiling_rad(omega, rad_energy[0])

        if Afield.history["t"][-1] > final_time:

            profilefile = file

            final_time = Afield.history["t"][-1]
            omega_profile, final_rad_profile = profiling_rad(omega, rad_energy[-1])

        ax2[j].plot(time, total_dipole)
        ax2[j].set_ylabel("Total dipole")

        ax4[0].plot(time, np.sum(rad_energy,axis=1))
        ax4[0].set_ylabel("Radiation energy (eV)")

    print(initialfile)
    print(profilefile)
    rad_profile = np.array(final_rad_profile) - np.array(initial_rad_profile)

    """
    for i, o in enumerate(omega_profile):
        print(i+1,o)
    """

    n,_,_ = ax3[1].hist(xe_velocity_dist, bins = np.arange(0,10,0.1))
    ax3[0].hist(ar_velocity_dist, bins = np.arange(0,10,0.1))
    ax3[0].set_ylim(0, np.max(n))
    ax3[0].set_ylabel("Frequency")
    ax3[0].set_xlabel("Argon velocity (km/s)")
    ax3[1].set_xlabel("Xenon velocity (km/s)")

    ax4[0].set_xlabel("Time (ps)")

    o, r = moving_average(omega_profile, rad_profile,20)
    ax4[1].scatter(omega_profile, rad_profile, s = 5)
    ax4[1].plot(o, r)

    ax4[1].set_xlabel("Wavenumber (cm^-1)")
    ax4[1].set_ylabel("Final energy (cm^-1)")

    #ax1[0].scatter(omega_profile, rad_profile, s = 5, alpha = 0.5)
    ax1[0].plot(o, r, label = KEYWORDS)
    rad_profile_list.append(rad_profile)

    fig3.savefig("figure/full_simulation_velocity_profile_"+KEYWORDS+".jpeg",dpi = 600,bbox_inches="tight")
    fig4.savefig("figure/full_simulation_radiation_"+KEYWORDS+".jpeg",dpi = 600,bbox_inches="tight")

ax1[0].set_xlabel("Wavenumber (cm^-1)")
ax1[0].set_ylabel("Final energy (cm^-1)")
ax1[0].legend()

profile_diff = (rad_profile_list[0] - rad_profile_list[1])
#ax1[1].scatter(omega_profile, profile_diff, 
#        s = 5, alpha = 0.5)
o,r = moving_average(omega_profile, profile_diff, 20)
ax1[1].plot(o,r, label = "Spectra in cavity \n- 'free space")

lf, uf = ax1[1].get_xlim()
ax1[1].plot(np.linspace(lf,uf,10), [0]*10, linestyle="dashed", linewidth = 0.5,color="gray")
ax1[1].legend()

fig1.savefig("figure/full_simulation_radiation.jpeg",dpi = 600,bbox_inches="tight")

ax2[0].annotate('cavity',xy = (0.80,0.95), xycoords='axes fraction')
ax2[1].annotate('free',xy = (0.80,0.95), xycoords='axes fraction')
fig2.savefig("figure/full_simulation_dipole.jpeg",dpi = 600,bbox_inches="tight")
