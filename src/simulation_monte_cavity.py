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

##############################
##############################
### PATH TO THE PICKLE JAR ###
##############################
##############################

parser = argparse.ArgumentParser()

parser.add_argument("seed", type = int, help = "random seed for Monte Carlo simulation")
parser.add_argument(
    "--date", "-d", help = "given date, start simulation from path /date_seed", 
    default = None)

args = parser.parse_args()

exist_jar_flag = False

if args.date == None:
    pickle_jar_path = "pickle_jar/" + time.strftime("%Y_%b_%d_", time.localtime()) \
        + str(args.seed)
    if os.path.isdir(pickle_jar_path):
        file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "cavity")
        if len(file_dict) == 0:
            exist_jar_flag = False
        else:
            exist_jar_flag = True
    else:
        os.mkdir(pickle_jar_path)

elif args.date:
    pickle_jar_path = "pickle_jar/" + args.date + "_" + str(args.seed)
    try:
        assert os.path.isdir(pickle_jar_path)
    except AssertionError:
        raise Exception(
            "There is no such folder {}! Please check the date or start a new run!".format(
                pickle_jar_path)
            )
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "cavity")
    if len(file_dict) == 0:
        exist_jar_flag = False
    else:
        exist_jar_flag = True

###########
### ETC ###
###########

initiate_atoms_box = config.initiate_atoms_box

#####################################################
#####################################################
#####################################################
if exist_jar_flag:
    # start the simulation from certain pickle_jar if the path is provided
    print("Start simulation from ",pickle_jar_path)

    # load other info/metadata of the simulation 
    with open(pickle_jar_path+"/metadata_cavity.pkl","rb") as handle:
        info = pickle.load(handle)

    seed_list = info["seed_list"]

    t = info["t_final"]
    h = info["h"]

    L = info["L_xy"]

    K_temp = info["temperature"]
    sampler = info["sampler"]
    N_atom_pairs = info["N_atom_pairs"]

    k_vector2 = info["cavity_mode_integer"]
    kappa = k_vector2[:,:2] * (2 * np.pi / L)
    m = k_vector2[:,-1].reshape(-1)

    # get dict of {"cycle numbers": path}
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "cavity")

    final_cycle_num = max(file_dict.keys())
    final_pickle_path = file_dict[final_cycle_num]

    with open(final_pickle_path,"rb") as handle:
        result = pickle.load(handle)

    # load the cavity field and the probe field
    old_cavity_field = result["cavity_field"]

    cavity_field = CavityVectorPotential(
        kappa = kappa, m = m, 
        L = old_cavity_field.L, S = old_cavity_field.L ** 2,
        amplitude = old_cavity_field.C, constant_c = red.c,
        coupling_strength = info["coupling_strength"]["cavity"]
        )

    del old_cavity_field

    old_probe_field = result["probe_field"]

    probe_field = FreeVectorPotential(
            k_vector = config.probe_kvector, amplitude = old_probe_field.C,
            V = L ** 3, constant_c = red.c,
            coupling_strength = info["coupling_strength"]["probe"]
            )

    del old_probe_field


elif not exist_jar_flag:
                ########################
                ########################
                ### EMPTY PARAMETERS ###
                ########################
                ########################
    t = 0
    h = config.h
    L = config.L

    np.random.seed(args.seed)

    K_temp = config.K_temp

    final_cycle_num = -1

    seed_list = np.random.randint(low = 0, high = 1000, size = 1000)

                ##########################
                ##########################
                ### INITIATE THE FIELD ###
                ##########################
                ##########################

    min_cavmode = 62; max_cavmode = 81
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
        L = L, S = L ** 2, constant_c = red.c, 
        coupling_strength = config.cavity_coupling_strength
        )

    probe_field = config.probe_field
    ### FIELD END ###

                ##########################
                ##########################
                ### INITIATE ATOMS BOX ###
                ##########################
                ##########################
    np.random.seed(seed_list[0])
    N_atom_pairs = config.N_atom_pairs

    sampler = config.sampler

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################

for i in range(final_cycle_num + 1, final_cycle_num + 1 + config.num_cycles):
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
            probe_field = probe_field, cavity_field = cavity_field, total_dipole_threshold = 1e-4, 
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
        coupling_strength = config.cavity_coupling_strength
        )

    new_probe_field = FreeVectorPotential(
            k_vector = config.probe_kvector, 
            amplitude = probe_field.C,
            V = L ** 3, constant_c = red.c,
            coupling_strength = config.probe_coupling_strength
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
        "N_atom_pairs":config.N_atom_pairs, "L_xy": config.L, "L_z": config.L,
        "temperature":K_temp, "mu0":config.mu0, 
        "cavity_mode_integer":k_vector2, "probe_mode_integer":config.probe_kvector_int,
        "seed":args.seed, "seed_list":seed_list, "t_final":t,
        "sampler":sampler, 
        "coupling_strength":{"cavity":config.cavity_coupling_strength, "probe":config.probe_coupling_strength}
        }

with open(pickle_jar_path + '/' + "metadata_cavity.pkl".format(i),"wb") as handle:
    pickle.dump(info_dict, handle)

print("Simulation finish, save to:",pickle_jar_path)
