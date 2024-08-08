import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler
from calculator.calculator import Calculator

import utilities.reduced_parameter as red

atoms = AtomsInBox(
    Lxy = 1e3, Lz = 1e3, cell_width = (1e2,1e2), mass_dict = red.mass_dict
    )

sampler = AllInOneSampler(
        N_atom_pairs=100, Lxy=1e3, Lz=1e3,
        d_ar_xe = 4.0, d_impact = 1.5,
        red_temp_unit=red.temp, K_temp=50,
        ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
        )

sample = sampler()
r_ar, r_xe = sample["r"]
r_dot_ar, r_dot_xe = sample["r_dot"]

np.random.seed(1)
atoms.add(elements = ["Ar"]*len(r_ar),r = r_ar,r_dot = r_dot_ar)
atoms.add(elements = ["Xe"]*len(r_xe),r = r_xe,r_dot = r_dot_xe)
"""
atoms.add(
        elements = ["Ar"],
        r = np.array([[-1, 0, 0]]),
        r_dot = np.array([[0.9, 0, 0]]),
        )

atoms.add(
        elements = ["Xe"],
        r = np.array([[1, 0, 0]]),
        r_dot = np.array([[-0.5, 0, 0]]),
        )
#"""

idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")

epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

atoms.add_calculator(
    calculator_class = Calculator,
    calculator_kwargs = {
    "epsilon": epsilon_mat, "sigma" : sigma_mat, 
    "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
    "mu0" : red.mu0, "d" : red.d0, "a" : red.a, "d7" : red.d7
    })


t = 0
h = 1e-3

time_list = []
energy_list = []
potential_list = []
kinetic_list = []
dipole_list = []

atoms.update_distance()

dipole_vec_list = []

for i in tqdm(range(10000)):
    atoms.Verlet_update(h, t=t)

    potential = atoms.potential()
    potential_list.append(potential)

    kinetic = atoms.kinetic()
    kinetic_list.append(kinetic)

    total_energy = potential + kinetic

    t += h

    energy_list.append(total_energy)

    dipole_vec = atoms.dipole()
    dipole_vec = np.einsum("ni,ni->n",dipole_vec,dipole_vec)

    dipole_vec_list.append(dipole_vec)

    total_dipole = 0.5 * np.sum(dipole_vec)

    dipole_list.append(total_dipole)

    time_list.append(t)

dipole_vec_list = np.array(dipole_vec_list)

energy_list = np.array(energy_list)
dipole_list = np.array(dipole_list)
time_list = np.array(time_list)

fig,ax = plt.subplots(2,2,figsize = (12,8))

ax[0,0].plot(time_list,energy_list)
ax[0,0].set_ylabel("Total energy")

ax[0,1].plot(time_list,kinetic_list)
ax[0,1].set_ylabel("Kinetic energy")

ax[1,0].plot(time_list,potential_list)
ax[1,0].set_ylabel("Potential energy")

ax[1,1].plot(time_list,dipole_list)
ax[1,1].set_ylabel("Dipole")

fig.savefig("figure/no_field.jpeg", dpi = 600,bbox_inches = "tight")

