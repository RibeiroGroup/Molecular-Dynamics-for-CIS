import time
import pickle
import os, sys, glob
import argparse

import numpy as np

from calculator.calculator import Calculator

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
from field.utils import EM_mode_generate_,EM_mode_generate, EM_mode_generate3

import utilities.reduced_parameter as red
from utilities.etc import categorizing_pickle
from simulation.single import single_collision_simulation

import config

#####################################################

start_from_pickle_jar_path = "/home/ribeirogroup/code/mm_polariton/src/pickle_jar/19_Jul_2024_170346"

#####################################################
#####################################################
#####################################################
if start_from_pickle_jar_path:
    print("Start simulation from ", start_from_pickle_jar_path)

    pickle_jar_path = start_from_pickle_jar_path
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "cavity")
    print(file_dict)

    final_pickle_path = file_dict[len(file_dict) - 1]
    final_cycle_num = len(file_dict)

    with open(final_pickle_path,"rb") as handle:
        result = pickle.load(handle)

    cavity_field = result["cavity_field"]
    probe_field = result["probe_field"]

    with open(pickle_jar_path+"/info.pkl","rb") as handle:
        info = pickle.load(handle)

    seed_list = info["seed_list"]

    t = info["t_final"]
    h = info["h"]

    L = info["L_xy"]

    k_vector2 = info["cavity_mode_integer"]
    kappa = k_vector2[:,:2] * (2 * np.pi / L)
    m = k_vector2[:,-1].reshape(-1)

    K_temp = info["temperature"]

elif not start_from_pickle_jar_path:
    #path to the PICKLE JAR
    pickle_jar_path = "pickle_jar/" + time.strftime("%d_%b_%Y_%H%M%S", time.localtime())

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

    final_cycle_num = 0

    np.random.seed(config.seed2)
    seed_list = np.random.randint(low = 0, high = 1000, size = 1000)

                ##########################
                ##########################
                ### INITIATE THE FIELD ###
                ##########################
                ##########################

    min_cavmode = 60; max_cavmode = 80
    possible_cavity_k = [0] + list(range(min_cavmode,max_cavmode)) 
    k_vector2 = np.array(
            EM_mode_generate(possible_cavity_k, vector_per_kval = 3, max_kval = max_cavmode),
            dtype=np.float64)
    print(len(k_vector2))

    k_val2 = np.einsum("ki,ki->k",k_vector2, k_vector2)
    k_val2 = np.tile(k_val2[:,np.newaxis],(1,2))

    amplitude2 = np.vstack([
        np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
        for i in range(len(k_vector2))
        ]) * 1e4 * np.sqrt(L**3) / k_val2

            ##############################
            ##############################
            ### CAVITY POTENTIAL BEGIN ###
            ##############################
            ##############################
    kappa = k_vector2[:,:2] * (2 * np.pi / L)

    m = k_vector2[:,-1].reshape(-1)

    cavity_field = CavityVectorPotential(
        kappa = kappa, m = m, amplitude = amplitude2,
        L = L, S = L ** 2, constant_c = red.c, coupling_strength = 1e3
        )

    probe_field = config.probe_field
    ### CAVITY POTENTIAL END ###

            ##########################
            ##########################
            ### INITIATE ATOMS BOX ###
            ##########################
            ##########################
np.random.seed(seed_list[0])
N_atom_pairs = config.N_atom_pairs

sampler = config.sampler
initiate_atoms_box = config.initiate_atoms_box

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

for i in range(final_cycle_num, final_cycle_num + config.num_cycles):
    np.random.seed(seed_list[i])

    sample = sampler()
    r_ar, r_xe = sample["r"]
    r_dot_ar, r_dot_xe = sample["r_dot"]

    atoms = initiate_atoms_box()
    atoms.add(elements = ["Ar"]*N_atom_pairs,r = r_ar,r_dot = r_dot_ar)
    atoms.add(elements = ["Xe"]*N_atom_pairs,r = r_xe,r_dot = r_dot_xe)

    atoms.update_distance()

    t, result = single_collision_simulation(
            cycle_number = i, atoms = atoms, t0 = t, h = h,
            probe_field = probe_field, cavity_field = cavity_field, total_dipole_threshold = 1e-5, 
            )

    with open(pickle_jar_path + '/' + "result_cavity_{}.pkl".format(i),"wb") as handle:
        pickle.dump(result, handle)

    cavity_field = result["cavity_field"] 
    probe_field = result["probe_field"] 

    del atoms
    new_cavity_field = CavityVectorPotential(
        kappa = kappa, m = m, 
        L = cavity_field.L, S = cavity_field.L ** 2,
        amplitude = cavity_field.C,
        constant_c = red.c,
        )

    new_probe_field = FreeVectorPotential(
            k_vector = config.probe_kvector, 
            amplitude = probe_field.C,
            V = L ** 3, constant_c = red.c,
            )

    del probe_field
    probe_field = new_probe_field

    del cavity_field
    cavity_field = new_cavity_field

            ###################################
            ###################################
            ### WRITING SIMULATION METADATA ###
            ###################################
            ###################################

info_dict = {
        "type":"cavity","h":h, "num_cycles":config.num_cycles,
        "N_atoms_pairs":config.N_atom_pairs, "L_xy": config.L, "L_z": config.L,
        "temperature":K_temp, "mu0":config.mu0, 
        "cavity_mode_integer":k_vector2, "probe_mode_integer":config.probe_kvector_int,
        "seed":[config.seed1, config.seed2], "seed_list":seed_list, "t_final":t
        }

with open(pickle_jar_path + '/' + "info.pkl".format(i),"wb") as handle:
    pickle.dump(info_dict, handle)

