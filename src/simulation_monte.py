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
from config import generate_field_amplitude

##############################
##############################
### PATH TO THE PICKLE JAR ###
##############################
##############################

parser = argparse.ArgumentParser()

parser.add_argument(
        "--seed", "-s", 
        type = int, help = "random seed for Monte Carlo simulation")
parser.add_argument(
        "--K_temp", "-t", 
        type = float, help = "Temperature", required = True)
parser.add_argument(
        "--field","-f", 
        type = str, help = "Field, valid argument: 'cavity', 'free', None", default = None)
parser.add_argument(
    "--min_mode", "-m", type = int,  
    help = "minimum field mode integer, if None provide for both this and -n, use modes from config.py",
    default = 0
        )
parser.add_argument(
    "--max_mode", "-n", type = int,  
    help = "maximum field mode integer, if None provide for both this and -m, use modes from config.py",
    default = 0
        )
parser.add_argument(
    "--mode_amplitude", "-a", type = str, 
    help = "mode amplitude, accepted argument: 'zero', 'zpve', 'config', 'cont'",
    default = 'zero'
        )
parser.add_argument(
    "--coupling_strength", "-c", type = int,
    help = "whether to use coupling strength factor in config.py or not (coupling_strength = 1)", 
    default = 0
        )
parser.add_argument(
    "--reduce_zdim", "-r", type = int,
    help = 'Reduce the dimension along the z axis'
)
parser.add_argument(
    "--reset", type = int,
    help = "reset the field between simulation cycles",
    default = 1
        )
parser.add_argument(
    "--cont", type = bool,
    help = "whether to continue simulation"
        )

######################
# Checking arguments #
######################

args = parser.parse_args()
assert isinstance(args.seed, int)
assert args.field == 'cavity' or args.field == 'free' or args.field == None

###########################
# Generating modes vector #
###########################

if args.field and not args.cont:
    if (args.min_mode > 0 and args.max_mode == 0) \
            or (args.min_mode == 0 and args.max_mode > 0):
        raise Exception("Exception: argument m and n have to be both None or int at the same time")

    elif args.min_mode == 0 and args.max_mode == 0:
        print("Notice: load default k-vector integer array from config.py.")
        kvector_int = config.default_kvector_int

    else:
        #assert isinstance(args.max_mode, float) and isinstance(args.min_mode, float)

        print("Notice: exhaustively generating modes")
        # get the minimum and maximum integer for generating the external modes
        possible_external_k = [0] + list(range(args.min_mode,args.max_mode)) 
        kvector_int = np.array(
            EM_mode_exhaust(possible_external_k, max_kval = args.max_mode), dtype=np.float64)
        print("Notice: There are {} external mode".format(len(kvector_int)))

elif args.cont:
    print("Notice: mode generating is skip, existing mode will be used!")

"""
Directory management
Sample directory name:
    nofield-292_1024_157
    cavity-292_1024_157-zpve_50_70
"""
sim_type = args.field if args.field else 'nofield'

pickle_jar_path = "pickle_jar/" + sim_type + '-' + str(args.K_temp)\
    + "_" + str(config.N_atom_pairs) + "_" + str(args.seed)

if args.field:
    pickle_jar_path += "-" + args.mode_amplitude
    pickle_jar_path += "_"  + str(args.min_mode) + "_" + str(args.max_mode)
    pickle_jar_path += "-c" + str(args.coupling_strength)

    if args.coupling_strength:
        pickle_jar_path += '_' + config.ct_label

    if args.reduce_zdim:
        pickle_jar_path += '-' + config.zlabel

if os.path.isdir(pickle_jar_path):
    if not args.cont: 
        raise Exception("Exception: folder existed but args.cont is not provided.")
    exist_jar_flag = True
else:
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
        info_dict = pickle.load(handle)

    locals().update(info_dict)

    # get dict of {"cycle numbers": path to pickle}
    file_dict = categorizing_pickle(pickle_jar_path,KEYWORDS = "")

    # get pickle that is the final simulation
    final_cycle_num = max(file_dict.keys())
    final_pickle_path = file_dict[final_cycle_num]

    # read the last pickle file
    with open(final_pickle_path,"rb") as handle:
        result = pickle.load(handle)

    # load the external field
    if args.field:
        print('Load external vector potential field.')
        old_field = result["field"]
        field = VectorPotential(
            k_vector_int = old_field.k_vector_int, 
            constant_c = red.c,
            Lxy = Lxy, Lz = Lz, 
            amplitude = old_field.C, 
            coupling_strength = old_field.coupling_strength
            )
        del old_field
    else: field = None

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
    K_temp = args.K_temp 

    final_cycle_num = -1

    seed_list = np.random.randint(low = 0, high = 1000, size = 1000)

    if args.coupling_strength:
        coupling_strength = config.coupling_strength
        print("Notice: coupling strength is applied with code {}, see config.py".format(
            config.ct_label))
    else:
        coupling_strength = 1
        print("Notice: coupling strength is 1")

    Lxy = config.Lxy

    if args.reduce_zdim:
        print(
            "Notice: the z dimension of the simulated box is reduced, with code {} see config.py for info".format(
                config.zlabel)
            )
        Lz = config.Lz_red
    else:
        Lz = config.Lz

    sampler = AllInOneSampler(
            N_atom_pairs=config.N_atom_pairs, Lxy=Lxy - 6, Lz=Lz - 6,
            d_ar_xe = 5.0, d_impact = 1.8,
            red_temp_unit=red.temp, K_temp=K_temp,
            ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
            )

    mode_amplitude = args.mode_amplitude
    reset = args.reset
    num_cycles = config.num_cycles
                ##########################
                ##########################
                ### INITIATE THE FIELD ###
                ##########################
                ##########################

    if args.field:
    ### START FREE MODE SPECS ###
        print("Notice: Initiate vector potential field.")

        if args.field == 'cavity': VectorPotential = CavityVectorPotential
        elif args.field == 'free': VectorPotential = FreeVectorPotential

        try:
            assert mode_amplitude in ['zero', 'zpve', 'config']
        except AssertionError:
            raise Exception('Please revise the ampltiude arguments!')

        field = VectorPotential(
                k_vector_int = kvector_int, 
                amplitude = generate_field_amplitude(kvector_int, mode_amplitude),
                Lxy = Lxy, Lz = Lz, 
                constant_c = red.c, 
                coupling_strength = coupling_strength
                )
    else: 
        field = None
        VectorPotential = None
    ### END FREE MODE SPECS ###

                ##########################
                ##########################
                ### INITIATE ATOMS BOX ###
                ##########################
                ##########################

    np.random.seed(seed_list[0])
    N_atom_pairs = config.N_atom_pairs

            ###################################
            ###################################
            ### WRITING SIMULATION METADATA ###
            ###################################
            ###################################

if first_run_flag:
    os.makedirs(pickle_jar_path)
    info_dict = {
            "t_final":t, "h":h, "num_cycles":config.num_cycles,
            "temperature":K_temp, "mu0":config.mu0, 
            "seed":args.seed, "seed_list":seed_list, 
            "sampler":sampler, "coupling_strength": coupling_strength,
            "VectorPotential": VectorPotential,
            "Lxy":Lxy, "Lz":Lz, 'reset': args.reset,
            'mode_amplitude': args.mode_amplitude
            }

    if args.field:
        info_dict.update({
            "kvector_int":kvector_int
            })

    with open(pickle_jar_path + '/' + "metadata.pkl","wb") as handle:
        pickle.dump(info_dict, handle)

    print("Notice: metadata has been written to " + pickle_jar_path + '/' + "metadata.pkl")

            ############################
            ############################
            ### START THE SIMULATION ###
            ############################
            ############################
t = 0
for i in range(final_cycle_num + 1, final_cycle_num + 1 + num_cycles):
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
            field = field, total_dipole_threshold = 1e-4, 
            max_steps = 5000
            )

    with open(pickle_jar_path + '/' + "result_{}.pkl".format(i),"wb") as handle:
        pickle.dump(result, handle)

    del atoms

    if args.field:
        field = result["field"] 

        if reset: 
            new_amplitude = generate_field_amplitude(kvector_int, mode_amplitude)
        else:
            new_amplitude = field.C

        new_field = VectorPotential(
            k_vector_int = field.k_vector_int,
            Lxy = Lxy, Lz = Lz, 
            amplitude = new_amplitude,
            constant_c = red.c,
            coupling_strength = coupling_strength
            )

        del field
        field = new_field

    info_dict['t_final'] = t

print("Simulation finish, save to:",pickle_jar_path)
