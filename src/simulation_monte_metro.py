import time
import pickle
import os, sys
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from calculator.calculator import Calculator

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
from field.utils import EM_mode_generate

import utilities.reduced_parameter as red

            ########################
            ########################
            ### EMPTY PARAMETERS ###
            ########################
            ########################
L = 1e6
cell_width = 1e4

t = 0
h = 1e-3

np.random.seed(2)

            ##########################
            ##########################
            ### INITIATE THE FIELD ###
            ##########################
            ##########################

k_vector = np.array(
        EM_mode_generate(max_n = 15, min_n = 0, vector_per_kval = 3), 
        dtype=np.float64)
print(len(k_vector))

amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(k_vector))
    ]) * 0e-1 * np.sqrt(L**3)

##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################
k_vector *= (2 * np.pi / L)
Afield = FreeVectorPotential(
        k_vector = k_vector, amplitude = amplitude,
        V = L ** 3, constant_c = red.c,
        )
### FREE FIELD POTENTIAL END ###
"""
##############################
### CAVITY POTENTIAL BEGIN ###
##############################
kappa = k_vector[:,:2] * (2 * np.pi / L)

m = k_vector[:,-1].reshape(-1)

Afield = CavityVectorPotential(
    kappa = kappa, m = m, amplitude = amplitude,
    L = L, S = L ** 2, constant_c = red.c)

### CAVITY POTENTIAL END ###
"""

            ##########################
            ##########################
            ### INITIATE ATOMS BOX ###
            ##########################
            ##########################
atoms = AtomsInBox(
    box_length = L, cell_width = cell_width, 
    mass_dict = red.mass_dict)

N_atom_pairs = 512
# Generate a matrix of LJ potential parameter
# e.g. matrix P with Pij is LJ parameter for i- and j-th atoms
idxAr = [1]*N_atom_pairs + [0]*N_atom_pairs # atoms.element_idx(element = "Xe")
idxXe = [0]*N_atom_pairs + [1]*N_atom_pairs # atoms.element_idx(element = "Ar")
epsilon_mat, sigma_mat = red.generate_LJparam_matrix(idxAr = idxAr, idxXe = idxXe)

# calculator to the atoms object
atoms.add_calculator(
    calculator_class = Calculator, N_atoms = N_atom_pairs * 2,
    calculator_kwargs = {
        "epsilon": epsilon_mat, "sigma" : sigma_mat, 
        "positive_atom_idx" : idxXe, "negative_atom_idx" : idxAr,
        "mu0" : red.mu0 * 1e3, "d" : red.d0, "a" : red.a
    })

atoms.clear()

#sampler for atoms configurations
sampler = AllInOneSampler(
        N_atom_pairs=N_atom_pairs, angle_range=np.pi/4, L=L,
        red_temp_unit=red.temp, K_temp=2500,
        ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
        )

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

for i in range(1):
    sample = sampler()
    r_ar, r_xe = sample["r"]
    r_dot_ar, r_dot_xe = sample["r_dot"]

    atoms.add(elements = ["Ar"]*N_atom_pairs,r = r_ar,r_dot = r_dot_ar)
    atoms.add(elements = ["Xe"]*N_atom_pairs,r = r_xe,r_dot = r_dot_xe)

    atoms.update_distance()

    atoms.record(t)
    Afield.record(t)

    dipole_drop_flag = False

    while not dipole_drop_flag or abs(potential) > 1e-1:
        em_force_func = lambda t, atoms: Afield.force(t,atoms)

        atoms.Verlet_update(
                h = h, t = t,
                field_force = em_force_func
                )

        C_dot_tp1 = Afield.dot_amplitude(t+h,atoms)
        C_new = Afield.C + h * (C_dot_tp1)

        Afield.update_amplitude(C_new)
            
        t += h

        atoms.record(t)
        Afield.record(t)

        dipole = atoms.observable["total_dipole"][-1]
        potential = atoms.observable["potential"][-1]

        print(i,"\t",dipole, "\t", potential)

        if dipole < atoms.observable["total_dipole"][-2]:
            dipole_drop_flag = True

    atoms.clear()

result = {"atoms":atoms, "field":Afield}
with open("pickle_jar/result2.pkl","wb") as handle:
    pickle.dump(result, handle)
