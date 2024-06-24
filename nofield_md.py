from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from atoms import AtomsInBox
from calculator import Calculator

import reduced_parameter as red

atoms = AtomsInBox(box_length = 20, cell_width = 5, mass_dict = red.mass_dict)

np.random.seed(2024)
#atoms.random_initialize({"Ar":10,"Xe":10}, max_velocity = 100)
atoms.add(
        elements = ["Ar"],
        R = np.array([[-2,-2,-2]]),
        R_dot = np.array([[10,10,10]]),
        )

atoms.add(
        elements = ["Xe"],
        R = np.array([[2,2,2]]),
        R_dot = np.array([[-1,-1,-1]]),
        )

idxAr = atoms.element_idx(element = "Xe")
idxXe = atoms.element_idx(element = "Ar")

epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

atoms.add_calculator(calculator_kwargs = {
    "epsilon": epsilon_mat, "sigma" : sigma_mat, 
    "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
    "mu0" : red.mu0, "d" : red.d0, "a" : red.a
    })


t = 0
h = 1e-5

energy_list = []
potential_list = []
kinetic_list = []
time_list = []

for i in tqdm(range(50000)):
    atoms.Verlet_update(h)

    potential = atoms.potential()
    potential_list.append(potential)

    kinetic = atoms.kinetic_energy()
    kinetic_list.append(kinetic)

    total_energy = potential + kinetic

    t += h

    energy_list.append(total_energy)
    time_list.append(t)

energy_list = np.array(energy_list)
time_list = np.array(time_list)

fig, ax = plt.subplots()

ax.plot(time_list,energy_list)
ax.plot(time_list,potential_list)
ax.plot(time_list,kinetic_list)

fig.savefig("test_md.jpeg")
