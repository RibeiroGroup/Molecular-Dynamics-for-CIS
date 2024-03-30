import pickle
import time
import numpy as np

from scipy.constants import physical_constants
from scipy.constants import epsilon_0, speed_of_light, proton_mass, neutron_mass, electron_mass

from utils import PBC_wrapping

from distance import DistanceCalculator, explicit_test
from neighborlist import neighbor_list_mask

from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction, DipoleFunctionExplicitTest

from forcefield_old import LennardJonesPotential as LennardJonesPotentialOld, \
    construct_param_matrix

np.random.seed(319)

########################
###### BOX LENGTH ######
########################

L = 150
cell_width = 15

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = int(L/2)
N_Xe = int(L/2)
N = N_Ar + N_Xe

# randomized initial coordinates
R_all = np.random.uniform(-L/2, L/2, (N, 3))
#R_all = np.array([[1.0,1.0,1.0],[-1.0,-1.0,-1.0]])

# randomized initial velocity
V_all = np.random.uniform(-1e1, 1e1, (N,3))
#V_all = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]])

# indices of atoms in the R_all and V_all
idxAr = np.hstack(
    [np.ones(N_Ar), np.zeros(N_Xe)]
)

idxXe = np.hstack(
    [np.zeros(N_Ar), np.ones(N_Xe)]
)

######################################
###### FORCE-RELATED PARAMETERS ######
######################################

epsilon_Ar_Ar = 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 * 1.59360e-3
epsilon_Xe_Xe = 1.904 * 1.59360e-3

sigma_Ar_Ar = 3.41 * (1e-10 / 5.29177e-11)
sigma_Ar_Xe = 3.735* (1e-10 / 5.29177e-11)
sigma_Xe_Xe = 4.06 * (1e-10 / 5.29177e-11)

epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
    + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
    + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
    + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
    + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

M_Ar = (18 * proton_mass + (40 - 18) * neutron_mass) / electron_mass
M_Xe = (54 * proton_mass + (131 - 54) * neutron_mass)/ electron_mass 

M_Xe = M_Xe / M_Ar
M_Ar = 1

mass = M_Ar * idxAr + M_Xe * idxXe
mass_x3 = np.tile(mass[:,np.newaxis], (1,3))

##########################################################################
###### INITIATE UTILITY CLASSES (PLEASE UPDATE THEM DURING LOOPING) ######  
##########################################################################

distance_calc = DistanceCalculator(
        n_points = len(R_all), 
        neighbor_mask = None,
        box_length = L
        )

force_field = LennardJonesPotential(
    sigma = sigma_mat, epsilon = epsilon_mat, distance_calc = distance_calc
)

dipole_function = SimpleDipoleFunction(
        distance_calc, 
        mu0=0.0124 , a=1.5121, d0=7.10,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

"""
dipole_function2 = DipoleFunctionExplicitTest(
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr,
        mu0=0.0124 , a=1.5121, d0=7.10, L=L
        )

dipole_tensor = dipole_function(R_all, return_tensor = True)

dipole_tensor_ = dipole_function2(R_all)
"""

###################################
###### SIMULATION START HERE ######
###################################

n_steps = 50000
h = 1e-4
r = R_all
v = V_all

trajectory = {
    "potential_energy" : [],
    "kinetic_energy" : [],
    "total dipole" : [],
    "sum of all dipole" : [],
    "time" : [], "step" : [],
}

sim_time = 0
i = 0

start = time.time()
while sim_time < 10:

    if i % 10 == 0: 
        mask = neighbor_list_mask(r, L, cell_width)
        distance_calc.update_global_mask(mask)

        force_field.update_distance_calc(distance_calc)
        dipole_function.update_distance_calc(distance_calc)

    k1v = force_field.force(r) / mass_x3
    k1r = v

    k2v = force_field.force(r + k1r * h/2) / mass_x3
    k2r = v + k1v * h/2

    k3v = force_field.force(r + k2r * h/2) / mass_x3
    k3r = v + k2v * h/2

    k4v = force_field.force(r + k3r * h) / mass_x3
    k4r = v + k3v * h

    v += (k1v + 2*k2v + 2*k3v + k4v) * h/6
    r += (k1r + 2*k2r + 2*k3r + k4r) * h/6
    r = PBC_wrapping(r,L)

    kinetic_energy = 0.5 * np.sum(np.einsum("ij,ij->i",v,v) * mass) 
    potential_energy = force_field.potential(r)
    potential_energy = np.sum(potential_energy)

    trajectory["potential_energy"].append(potential_energy)
    trajectory["kinetic_energy"].append(kinetic_energy)

    dipole_vec_tensor = dipole_function(r)

    total_dipole_vec = np.sum(dipole_vec_tensor, axis = 0)
    total_dipole = np.sqrt(total_dipole_vec @ total_dipole_vec)
    trajectory["total dipole"].append(total_dipole)

    sim_time += h
    i += 1
    trajectory["time"].append(sim_time)

    if potential_energy < 1:
        h = 1e-2
    elif potential_energy < 10:
        h = 1e-3
    elif potential_energy < 100:
        h = 1e-4
    else:
        h = 1e-5

    if i % 10 == 0:
        print("-- Iteration #", i,  " Simulated time: ",sim_time, "--")
        print("Total energy", kinetic_energy + potential_energy/2)

        print("\t + kinetic_energy",kinetic_energy)
        print("\t + potential_energy",potential_energy)
        print("\t + total dipole",total_dipole)

        print("Runtime: ", time.time() - start)

    if i % 1000 == 0:
        with open("result_plot/trajectory_temp.pkl","wb") as handle:
            pickle.dump(trajectory,handle)

        print("Autosave!")

print("############ JOB COMPLETE ############")
print("Total runtime: ", time.time() - start)

with open("result_plot/trajectory_temp.pkl","wb") as handle:
    pickle.dump(trajectory,handle)

