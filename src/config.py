import numpy as np

from calculator.calculator import Calculator
from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
#from field.utils import EM_mode_generate_,EM_mode_generate, EM_mode_generate3

import utilities.reduced_parameter as red

####################
### DRIVER PARAM ###
####################

#seed1 = 2024 #1807 # 1998 #2024
#seed2 = 2020 #929 # 1507 #2020

num_cycles = 10
if num_cycles != 10: print("Warning, number of cycles is not 10!")

h = 1e-2

################
### BOX SIZE ### 
################

Lxy_free = 3e7
Lz_free = 3e7
Lxy = 3e7
Lz  = 5e2
cell_width = 100

##############
### MATTER ###
##############

N_atom_pairs = 16 #512
K_temp = 292

mu0 = red.mu0

sampler = AllInOneSampler(
        N_atom_pairs=N_atom_pairs, angle_range=np.pi/4, Lxy=Lxy, Lz=Lz,
        d_ar_xe=4,red_temp_unit=red.temp, K_temp=K_temp,
        ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
        )

def initiate_atoms_box(Lxy, Lz):
    atoms = AtomsInBox(
        Lxy = Lxy, Lz= Lz, cell_width = cell_width, 
        mass_dict = red.mass_dict)
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
            "mu0" : mu0, "d" : red.d0, "a" : red.a, "d7": red.d7
        })

    return atoms

#############
### FIELD ### 
#############

probe_kvector_int = np.array(
        [[i,0,0] for i in range(1,200)]\
       +[[0,i,0] for i in range(1,200)]
        )

probe_coupling_strength = 1e0

amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(probe_kvector_int))
    ]) * 0

external_coupling_strength = 1e0
external_amplitude_scaling = 1e4
