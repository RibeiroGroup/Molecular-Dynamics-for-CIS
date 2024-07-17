import time
import pickle
import os, sys

import numpy as np

from calculator.calculator import Calculator

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
from field.utils import EM_mode_generate_,EM_mode_generate, EM_mode_generate3

import utilities.reduced_parameter as red

import config

            ########################
            ########################
            ### EMPTY PARAMETERS ###
            ########################
            ########################
t = 0
h = config.h

L = config.L

np.random.seed(config.seed1)

K_temp = config.K_temp

            ##########################
            ##########################
            ### INITIATE THE FIELD ###
            ##########################
            ##########################

#k_vector = np.array([[0,0,i] for i in range(1,401)], dtype = np.float64)
k_vector = config.probe_kvector

##################################
### FREE FIELD POTENTIAL BEGIN ###
##################################

Afield = config.probe_field

### FREE FIELD POTENTIAL END ###

            ##########################
            ##########################
            ### INITIATE ATOMS BOX ###
            ##########################
            ##########################
np.random.seed(config.seed2)

N_atom_pairs = config.N_atom_pairs

sampler = config.sampler
initiate_atoms_box = config.initiate_atoms_box

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

for i in range(config.num_cycles):
    sample = sampler()
    r_ar, r_xe = sample["r"]
    r_dot_ar, r_dot_xe = sample["r_dot"]

    atoms = initiate_atoms_box()
    atoms.add(elements = ["Ar"]*N_atom_pairs,r = r_ar,r_dot = r_dot_ar)
    atoms.add(elements = ["Xe"]*N_atom_pairs,r = r_xe,r_dot = r_dot_xe)

    atoms.update_distance()

    atoms.record(t)
    Afield.record(t)

    dipole_drop_flag = False
    potential_drop_flag = False
    steps = 0

    while (not dipole_drop_flag or abs(dipole) > 1e-5 or steps < 100) and steps < 10000:
        steps += 1

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
        elif dipole > atoms.observable["total_dipole"][-2]:
            dipole_drop_flag = False

        """
        if potential < atoms.observable["potential"][-2]:
            potential_drop_flag = True
        elif potential > atoms.observable["potential"][-2]:
            potential_drop_flag = False
        """

    result = {
            "atoms":atoms, "cavity_field":None, "probe_field":Afield,
            "temperature":K_temp, "mu0" : config.mu0
            }
    with open("pickle_jar/result_free_{}.pkl".format(i),"wb") as handle:
        pickle.dump(result, handle)

    del atoms
    new_Afield = FreeVectorPotential(
        k_vector = k_vector, V = L ** 3, 
        amplitude = Afield.C,
        constant_c = red.c,
        )

    del Afield
    Afield = new_Afield
