import pickle
import os, sys, glob
import argparse

import numpy as np

from calculator.calculator import Calculator

from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
from field.utils import EM_mode_generate_,EM_mode_exhaust

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
#parser.add_argument("--type","-t", type = str, help = "Simulation type, include only 'external', 'free','nofield'")
parser.add_argument("--external","-x", type = str, 
    help = "External field, include only 'external', 'free', None", default = None)
parser.add_argument("--probe","-p", type = int, 
    help = "Probe field, 0 for no probe field and 1 for free field")
parser.add_argument(
    "--continue_from", "-c", help = "given the directory path, continue simulation from the last pickle file", 
    default = None)
parser.add_argument(
    "--min_mode", "-m", type = int,  help = "minimum external laser mode integer"
        )
parser.add_argument(
    "--max_mode", "-n", type = int,  help = "maximum external laser mode integer"
        )

args = parser.parse_args()
assert isinstance(args.seed, int)
assert args.external == 'cavity' or args.external == 'free' or args.external == None

if args.external:
    assert isinstance(args.min_mode, int) and isinstance(args.max_mode, int)
    assert args.min_mode < args.max_mode

"""
Directory management
"""
sim_type = 'probe' if args.probe else 'nofield'

pickle_jar_path = "pickle_jar/" + sim_type + '-' + str(config.K_temp)\
    + "_" + str(config.N_atom_pairs) + "_" + str(args.seed)

if args.external:
    pickle_jar_path += "_"  + args.external + "_"  + str(args.min_mode) + "_" + str(args.max_mode)

if os.path.isdir(pickle_jar_path):
    # check if there are pickle file in the existing directory
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "")
    if len(file_dict) == 0:
        exist_jar_flag = False
    else:
        exist_jar_flag = True
else:
    # Create directory in case of starting brand new simulation
    os.mkdir(pickle_jar_path)
    exist_jar_flag = False

###########
### ETC ###
###########

initiate_atoms_box = config.initiate_atoms_box

#####################################################
if exist_jar_flag:
    """
    Load the last pickle in the existing jar. The simulation metadata and field amplitude are
    load and use as input for the simulation
    """
    first_run_flag = False
    ########################################################################
    # start the simulation from certain pickle_jar if the path is provided #
    ########################################################################
    print("Start simulation from ",pickle_jar_path)

    # load other info/metadata of the simulation 
    with open(pickle_jar_path+"/metadata.pkl","rb") as handle:
        info = pickle.load(handle)

    # seed list for sampling atoms position/velocities
    seed_list = info["seed_list"]

    # get last time of the loaded simulation and time step
    t = info["t_final"]
    h = info["h"]

    # loaded temperature and atomic sampler
    sampler = info["sampler"]
    Lxy = info["Lxy"]; Lz = info["Lz"]
    ExternalVectorPotential = info["ExternalVectorPotential"]

    # get dict of {"cycle numbers": path to pickle}
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "")

    # get pickle that is the final simulation
    final_cycle_num = max(file_dict.keys())
    final_pickle_path = file_dict[final_cycle_num]

    # read the last pickle file
    with open(final_pickle_path,"rb") as handle:
        result = pickle.load(handle)

    # load the external field
    if args.external:
        print('Load external vector potential field.')
        old_external_field = result["external_field"]
        external_field = CavityVectorPotential(
            k_vector_int = old_external_field.k_vector_int, 
            constant_c = red.c,
            Lxy = old_external_field.Lxy, Lz = old_external_field.Lz, 
            amplitude = old_external_field.C, 
            coupling_strength = old_external_field.coupling_strength
            )
        del old_external_field
    else: external_field = None

    # load the probe field
    if args.probe:
        print('Load probe (free) vector potential field.')
        old_probe_field = result["probe_field"]
        probe_field = FreeVectorPotential(
                k_vector_int = old_probe_field.k_vector_int, 
                amplitude = old_probe_field.C, constant_c = red.c,
                Lxy = old_probe_field.Lxy, Lz = old_probe_field.Lz, 
                coupling_strength = old_probe_field.coupling_strength
                )
        del old_probe_field

#####################################################
elif not exist_jar_flag:
                ########################
                ########################
                ### EMPTY PARAMETERS ###
                ########################
                ########################
    first_run_flag = True
    # start a brand new simulation
    # set time zero and time step h
    t = 0
    h = config.h

    # set random seed
    np.random.seed(args.seed)

    # set Kelvin temperature
    K_temp = config.K_temp 

    final_cycle_num = -1

    seed_list = np.random.randint(low = 0, high = 1000, size = 1000)

                ##########################
                ##########################
                ### INITIATE THE FIELD ###
                ##########################
                ##########################

    if args.external:
    ### START CAVITY MODE SPECS ###
        print("Initiate external vector field")
        if args.external == 'cavity':
            ExternalVectorPotential = CavityVectorPotential
            Lxy = config.Lxy; Lz = config.Lz
        elif args.external == 'free':
            ExternalVectorPotential = FreeVectorPotential
            Lxy = config.Lxy_free; Lz = config.Lz_free

        # get the minimum and maximum integer for generating the external modes
        possible_external_k = [0] + list(range(args.min_mode,args.max_mode)) 
        ext_mode_int = np.array(
            EM_mode_exhaust(possible_external_k, max_kval = args.max_mode), dtype=np.float64)
        print("There are {} external mode".format(len(ext_mode_int)))

        # calculate the magnitude of the wavevector for normalizing the energy
        k_val = np.sqrt(np.einsum("ki,ki->k",ext_mode_int, ext_mode_int))
        k_val = np.tile(k_val[:,np.newaxis],(1,2))

        amplitude2 = np.vstack([
            #np.ones( 2) + np.ones(2) * 1j
            np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
            for i in range(len(ext_mode_int))
            ]) * np.sqrt(Lxy * Lxy * Lz) / k_val \
            * config.external_amplitude_scaling 

        # in-plane wavevector
        # integer for generating kz (see field.electromagnetic.CavityVectorPotential module)

        external_field = ExternalVectorPotential(
            k_vector_int = ext_mode_int, amplitude = amplitude2,
            Lxy = Lxy, Lz = Lz, constant_c = red.c, 
            coupling_strength = config.external_coupling_strength
            )
    else: 
        ExternalVectorPotential = None
        external_field = None
        print("Warning, Lxy and Lz are not defined for no external field case, default value are used")
        Lxy = config.Lxy; Lz = config.Lz
    ### END CAVITY MODE SPECS ###

    if args.probe:
    ### START FREE MODE SPECS ###
        print("Initiate Probe (Free) vector field:")
        probe_field = FreeVectorPotential(
                k_vector_int = config.probe_kvector_int, 
                amplitude = np.zeros(
                    (len(config.probe_kvector_int), 2), dtype = np.complex128),
                Lxy = Lxy, Lz = Lz, 
                constant_c = red.c, 
                coupling_strength = config.probe_coupling_strength
                )
    else: probe_field = None
    ### END FREE MODE SPECS ###

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

    atoms = initiate_atoms_box(Lxy, Lz)
    atoms.add(elements = ["Ar"]*len(r_ar),r = r_ar,r_dot = r_dot_ar)
    atoms.add(elements = ["Xe"]*len(r_xe),r = r_xe,r_dot = r_dot_xe)

    atoms.update_distance()

    t, result = single_collision_simulation(
            cycle_number = i, atoms = atoms, t0 = t, h = h,
            probe_field = probe_field, external_field = external_field, total_dipole_threshold = 1e-4, 
            )

    with open(pickle_jar_path + '/' + "result_{}.pkl".format(i),"wb") as handle:
        pickle.dump(result, handle)

    del atoms

    if args.external:
        external_field = result["external_field"] 
        new_external_field = ExternalVectorPotential(
            k_vector_int = external_field.k_vector_int,
            Lxy = external_field.Lxy, Lz = external_field.Lz, 
            amplitude = external_field.C,constant_c = red.c,
            coupling_strength = config.external_coupling_strength
            )

        del external_field
        external_field = new_external_field

    if args.probe:
        probe_field = result["probe_field"] 
        new_probe_field = FreeVectorPotential(
                k_vector_int = config.probe_kvector_int, 
                amplitude = probe_field.C,
                Lxy = probe_field.Lxy, Lz = probe_field.Lz, 
                constant_c = red.c, coupling_strength = config.probe_coupling_strength
                )

        del probe_field
        probe_field = new_probe_field

            ###################################
            ###################################
            ### WRITING SIMULATION METADATA ###
            ###################################
            ###################################
if first_run_flag:
    info_dict = {
            "t_final":t, "h":h, "num_cycles":config.num_cycles,
            "temperature":K_temp, "mu0":config.mu0, 
            "seed":args.seed, "seed_list":seed_list, 
            "sampler":sampler, "coupling_strength":{},
            "ExternalVectorPotential": ExternalVectorPotential,
            "Lxy":Lxy, "Lz":Lz
            }

    if args.external:
        info_dict["coupling_strength"].update(
            {"external":config.external_coupling_strength}
            )
        info_dict.update({
            "external_mode_integer":ext_mode_int
            })
    if args.probe:
        info_dict["coupling_strength"].update(
            {"probe":config.probe_coupling_strength}
            )


    with open(pickle_jar_path + '/' + "metadata.pkl".format(i),"wb") as handle:
        pickle.dump(info_dict, handle)

    print("Simulation finish, save to:",pickle_jar_path)
