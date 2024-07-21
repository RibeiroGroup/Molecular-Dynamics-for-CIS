import numpy as np

from calculator.calculator import Calculator
from matter.atoms import AtomsInBox
from matter.utils import AllInOneSampler

from field.electromagnetic import FreeVectorPotential,CavityVectorPotential
from field.utils import EM_mode_generate_,EM_mode_generate, EM_mode_generate3

import utilities.reduced_parameter as red

#seed1 = 2024 #1807 # 1998 #2024
#seed2 = 2020 #929 # 1507 #2020

L = 3e7
cell_width = 1e4

h = 1e-2

num_cycles = 1

K_temp = 292

probe_kvector_int = np.array(
        EM_mode_generate3(min_n = 1, max_n = 300), dtype = np.float64)

N_atom_pairs = 64

mu0 = red.mu0

def initiate_atoms_box():
    atoms = AtomsInBox(
        box_length = L, cell_width = cell_width, 
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

#sampler for atoms configurations
sampler = AllInOneSampler(
        N_atom_pairs=N_atom_pairs, angle_range=np.pi/4, L=L,
        d_ar_xe=4,red_temp_unit=red.temp, K_temp=K_temp,
        ar_mass=red.mass_dict["Ar"], xe_mass=red.mass_dict["Xe"]
        )

VECTOR_POTENTIAL_CLASS = FreeVectorPotential
probe_kvector = probe_kvector_int * (2 * np.pi / L)
probe_coupling_strength = 1e3 
cavity_coupling_strength = 1e3

amplitude = np.vstack([
    np.random.uniform(size = 2) * 1 + np.random.uniform(size = 2) * 1j
    for i in range(len(probe_kvector))
    ]) * 0

probe_field = VECTOR_POTENTIAL_CLASS(
        k_vector = probe_kvector, amplitude = amplitude,
        V = L ** 3, constant_c = red.c, 
        coupling_strength = probe_coupling_strength
        )
