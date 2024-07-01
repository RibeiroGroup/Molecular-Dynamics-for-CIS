import pickle
import numpy as np
import matplotlib.pyplot as plt

from utilities import reduced_parameter as red
from field.utils import profiling_rad

PICKLE_PATH = "pickle_jar/result.pkl"

with open(PICKLE_PATH,"rb") as handle:
    result = pickle.load(handle)

atoms = result["atoms"]
Afield = result["field"]

fig, ax = plt.subplots(2,2,figsize = (12,8))

time = np.array(atoms.observable["t"])

kinetic = np.array(atoms.observable["kinetic"])
potential = np.array(atoms.observable["total_dipole"])

rad_energy = np.array(Afield.history["energy"])
omega = Afield.omega
omega_profile, rad_profile = profiling_rad(omega, rad_energy[-1])

ax[0,0].plot(time, kinetic)
ax[0,0].set_ylabel("Kinetic energy")

ax[0,1].plot(time, potential)
ax[0,1].set_ylabel("Potential energy")

ax[1,0].plot(time, np.sum(rad_energy,axis=1))
ax[1,0].set_ylabel("Radiation energy")

ax[1,1].scatter(omega_profile, rad_profile)
#ax[1,1].set_ylabel("Radiation energy")

fig.savefig("figure/full_simulation.jpeg",dpi = 600)
