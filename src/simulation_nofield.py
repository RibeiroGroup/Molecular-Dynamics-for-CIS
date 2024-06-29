import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from matter.atoms import AtomsInBox
from calculator.calculator import Calculator

import utilities.reduced_parameter as red

atoms = AtomsInBox(
    box_length = 30, cell_width = 10, mass_dict = red.mass_dict
    )

np.random.seed(1)
atoms.random_initialize({"Ar":60, "Xe":60}, max_velocity = 100, min_velocity = 20)
"""
atoms.add(
        elements = ["Ar"],
        R = np.array([[-1,-1,-1]]),
        R_dot = np.array([[1,1,1]]),
        )

atoms.add(
        elements = ["Xe"],
        R = np.array([[1,1,1]]),
        R_dot = np.array([[-1,-1,-1]]),
        )
"""

idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")

epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

atoms.add_calculator(
    calculator_class = Calculator,
    calculator_kwargs = {
    "epsilon": epsilon_mat, "sigma" : sigma_mat, 
    "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
    "mu0" : red.mu0, "d" : red.d0, "a" : red.a
    })


t = 0
h = 1e-4

time_list = []
energy_list = []
potential_list = []
kinetic_list = []
dipole_list = []

for i in tqdm(range(10000)):
    atoms.Verlet_update(h, t=t)

    potential = atoms.potential()
    potential_list.append(potential)

    kinetic = atoms.kinetic()
    kinetic_list.append(kinetic)

    total_energy = potential + kinetic

    t += h

    energy_list.append(total_energy)

    dipole_vec = atoms.dipole(return_matrix = False)
    dipole_vec = np.einsum("ni,ni->n",dipole_vec,dipole_vec)
    total_dipole = 0.5 * np.sum(dipole_vec)

    dipole_list.append(total_dipole)

    time_list.append(t)

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

