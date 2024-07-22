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

parser.add_argument("--seed", "-s", type = int, help = "random seed for Monte Carlo simulation")
parser.add_argument(
    "--continue_from", "-c", help = "given the directory path, continue simulation from the last pickle file", 
    default = None)
parser.add_argument(
    "--min_cav_mode", "-m", type = int,  help = "minimum cavity mode integer"
        )
parser.add_argument(
    "--max_cav_mode", "-x", type = int,  help = "maximum cavity mode integer"
        )

args = parser.parse_args()

if args.continue_from == None:
    pickle_jar_path = "pickle_jar/" + str(config.K_temp) + "_" + str(config.N_atom_pairs) + "_" \
            + str(args.min_cav_mode) + "_" + str(args.max_cav_mode)+ "_" + str(args.seed)
    if os.path.isdir(pickle_jar_path):
        file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "free")
        if len(file_dict) == 0:
            exist_jar_flag = False
        else:
            exist_jar_flag = True
    else:
        os.mkdir(pickle_jar_path)
        exist_jar_flag = False

elif args.continue_from:
    pickle_jar_path = "pickle_jar/" + args.continue_from
    try:
        assert os.path.isdir(pickle_jar_path)
    except AssertionError:
        raise Exception(
            "There is no such folder {}! Please check the date or start a new run!".format(
                pickle_jar_path)
            )
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "free")
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
    with open(pickle_jar_path+"/metadata_free.pkl","rb") as handle:
        info = pickle.load(handle)

    seed_list = info["seed_list"]

    t = info["t_final"]
    h = info["h"]

    L = info["L_xy"]

    K_temp = info["temperature"]
    sampler = info["sampler"]
    N_atom_pairs = info["N_atom_pairs"]

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
            probe_field = probe_field, cavity_field = None, total_dipole_threshold = 1e-4, 
            )

    with open(pickle_jar_path + '/' + "result_free_{}.pkl".format(i),"wb") as handle:
        pickle.dump(result, handle)

    probe_field = result["probe_field"] 

    del atoms

    new_probe_field = FreeVectorPotential(
            k_vector = config.probe_kvector, 
            amplitude = probe_field.C,
            V = L ** 3, constant_c = red.c,
            coupling_strength = config.probe_coupling_strength
            )

    del probe_field
    probe_field = new_probe_field

            ###################################
            ###################################
            ### WRITING SIMULATION METADATA ###
            ###################################
            ###################################

info_dict = {
        "type":"free","h":h, "num_cycles":config.num_cycles,
        "N_atom_pairs":config.N_atom_pairs, "L_xy": config.L, "L_z": config.L,
        "temperature":K_temp, "mu0":config.mu0, 
        "cavity_mode_integer":None, "probe_mode_integer":config.probe_kvector_int,
        "seed":args.seed, "seed_list":seed_list, "t_final":t,
        "sampler":sampler, 
        "coupling_strength":{"cavity":None, "probe":config.probe_coupling_strength}
        }

with open(pickle_jar_path + '/' + "metadata_free.pkl".format(i),"wb") as handle:
    pickle.dump(info_dict, handle)

print("Simulation finish, save to:",pickle_jar_path)
