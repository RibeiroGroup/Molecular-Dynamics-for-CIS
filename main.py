import pickle
import time
import numpy as np

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

L = 300
cell_width = 15

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = int(L / 2)
N_Xe = int(L / 2)
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

epsilon_Ar_Ar = 1 # 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 / 0.996 # * 1.59360e-3
epsilon_Xe_Xe = 1.904 / 0.996 # * 1.59360e-3

sigma_Ar_Ar = 1 # 3.41 * (1e-10 / 5.29177e-11)
sigma_Ar_Xe = 3.735 / 3.41 #* (1e-10 / 5.29177e-11)
sigma_Xe_Xe = 4.06  / 3.41 #* (1e-10 / 5.29177e-11)

epsilon = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
    + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

sigma = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
    + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
    + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
    + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

mass = np.array(idxAr) * 1 + np.array(idxXe) * 131.293/39.948
mass_x3 = np.tile(mass[:,np.newaxis],(1,3))

##########################################################################
###### INITIATE UTILITY CLASSES (PLEASE UPDATE THEM DURING LOOPING) ######  
##########################################################################

distance_calc = DistanceCalculator(
        n_points = len(R_all), 
        mask = None,
        box_length = L
        )

force_field = LennardJonesPotential(
    sigma = sigma, epsilon = epsilon, distance_calc = distance_calc
)

"""
epsilon_pmat = construct_param_matrix(
    N, int(N/2), 
    [0.996 * 1.59360e-3, 1.904 * 1.59360e-3],
    1.377 * 1.59360e-3)
sigma_pmat = construct_param_matrix(
    N, int(N/2),
    [sigma_Ar_Ar, sigma_Xe_Xe],
    sigma_Ar_Xe
)

force_field2 = LennardJonesPotentialOld(
    epsilon_pmat, sigma_pmat, N, L)

dipole_function = SimpleDipoleFunction(
        distance_calc, mu0=0.0124 , a=1.5121, d0=7.10,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

potential = force_field.potential(R_all)
force = force_field.force(R_all)

print(force)

potential_, force_ = explicit_test_LJ(R_all, epsilon ,sigma, L)
force_ = np.sum(force_,axis = 1)

print(force_)

force2 = force_field2.get_force(R_all)
print(np.sum(force2 - force))
"""

###################################
###### SIMULATION START HERE ######
###################################

n_steps = 50000
h = 1e-4
r = R_all
v = V_all

trajectory = {
    "potential_energy":[],
    "kinetic_energy":[],
    "time":[]
}

time = 0
i = 0

while time < 100:

    if i % 10 == 0: 
        mask = neighbor_list_mask(r, L, cell_width)
        distance_calc.update_global_mask(mask)
        force_field.update_distance_calc(distance_calc)

    k1v = force_field.force(r) / mass_x3
    #_, k1v = explicit_test_LJ(r, epsilon ,sigma, L)
    #k1v = np.sum(k1v,axis = 1) / (1837 * 30)
    k1r = v

    k2v = force_field.force(r + k1r * h/2) / mass_x3
    #_, k2v = explicit_test_LJ(r + k1r*h/2, epsilon ,sigma, L)
    #k2v = np.sum(k2v,axis = 1) / (1837 * 30)
    k2r = v + k1v * h/2

    k3v = force_field.force(r + k2r * h/2) / mass_x3
    #_, k3v = explicit_test_LJ(r + k2r*h/2, epsilon ,sigma, L)
    #k3v = np.sum(k3v,axis = 1) / (1837 * 30)
    k3r = v + k2v * h/2

    k4v = force_field.force(r + k3r * h) / mass_x3
    #_, k4v = explicit_test_LJ(r + k3r*h, epsilon ,sigma, L)
    #k4v = np.sum(k4v,axis = 1) / (1837 * 30)
    k4r = v + k3v * h

    v += (k1v + 2*k2v + 2*k3v + k4v) * h/6
    r += (k1r + 2*k2r + 2*k3r + k4r) * h/6
    r = PBC_wrapping(r,L)

    kinetic_energy = 0.5 * np.sum(np.einsum("ij,ij->i",v,v) * mass) 
    potential_energy = force_field.potential(r)
    potential_energy = np.sum(potential_energy)

    dipole = 

    trajectory["potential_energy"].append(potential_energy)
    trajectory["kinetic_energy"].append(kinetic_energy)

    time += h
    i += 1
    trajectory["time"].append(time)

    if potential_energy < 1e-4:
        h = 1e-2
    elif potential_energy < 1e-3:
        h = 1e-3
    elif potential_energy < 1e-2:
        h = 1e-4
    else:
        h = 1e-5

    if i % 10 == 0:
        print(time)
        print("Total energy", kinetic_energy + potential_energy/2)

        print("\t + kinetic_energy",kinetic_energy)
        print("\t + potential_energy",potential_energy)

    if i % 1000 == 0:
        with open("result_plot/trajectory_temp.pkl","wb") as handle:
            pickle.dump(trajectory,handle)

with open("result_plot/trajectory_temp.pkl","wb") as handle:
    pickle.dump(trajectory,handle)
