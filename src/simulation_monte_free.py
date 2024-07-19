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
from simulation.single import single_collision_simulation

import config

#####################################################
#####################################################
#####################################################

#path to the PICKLE JAR
pickle_jar_path = "pickle_jar/" + time.strftime("%d_%b_%Y_%H%M%S", time.localtime())

if os.path.isdir(pickle_jar_path):
    prompt = input("Directory for output jar is already exist. Do you want to overwrite?[y]/n")
    if prompt == "n":
        raise Exception

os.mkdir(pickle_jar_path)

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

            ###################################
            ###################################
            ### WRITING SIMULATION METADATA ###
            ###################################
            ###################################

info_dict = {
        "type":"free", "h":h, "num_cycles":config.num_cycles,
        "N_atoms_pairs":config.N_atom_pairs, "L_xy": config.L, "L_z": config.L,
        "temperature":K_temp, "mu0":config.mu0, "seed":[config.seed1, config.seed2],
        "cavity_mode_integer":None, "probe_mode_integer":config.probe_kvector_int
        }

with open(pickle_jar_path + '/' + "info.pkl","wb") as handle:
    pickle.dump(info_dict, handle)

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

    t, result = single_collision_simulation(
            cycle_number = i, atoms = atoms, t0 = t, h = h,
            probe_field = Afield, cavity_field = None, total_dipole_threshold = 1e-5, 
            )

    result.update({
            "temperature":K_temp, "mu0" : config.mu0, "seed":[config.seed1, config.seed2]
            })

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
