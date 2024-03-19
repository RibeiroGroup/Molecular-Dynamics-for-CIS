import time
import numpy as np

from utils import PBC_wrapping

from distance import DistanceCalculator, explicit_test
from neighborlist import neighbor_list_mask

from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction, DipoleFunctionExplicitTest

np.random.seed(10)

########################
###### BOX LENGTH ######
########################

L = 40
cell_width = 10

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

####################################
###### FORCE FIELD PARAMETERS ######
####################################

epsilon_Ar_Ar = 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 * 1.59360e-3
epsilon_Xe_Xe = 1.904 * 1.59360e-3

epsilon = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
    + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

sigma = (np.outer(idxAr,idxAr) * 3.41 \
    + np.outer(idxAr, idxXe) * 3.735 \
    + np.outer(idxXe, idxAr) * 3.735 \
    + np.outer(idxXe, idxXe) * 4.06) * (1e-10 / 5.29177e-11)

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

dipole_function = SimpleDipoleFunction(
        distance_calc, mu0=0.0124 , a=1.5121, d0=7.10,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

potential = force_field.potential(R_all)
force = force_field.force(R_all)

"""
print(potential)

potential_, force_ = explicit_test_LJ(R_all, epsilon ,sigma, L)
force_ = np.sum(force_,axis = 1)

print(R_all)
print(force)
"""

###################################
###### SIMULATION START HERE ######
###################################

n_steps = 1000
h = 1e-4
r = R_all
v = V_all

for i in range(n_steps):

    #mask = neighbor_list_mask(r, L, cell_width)
    #distance_calc.update_global_mask(mask)
    #force_field.update_distance_calc(distance_calc)

    #k1v = force_field.force(r)
    _, k1v = explicit_test_LJ(r, epsilon ,sigma, L)
    k1v = np.sum(k1v,axis = 1)
    k1r = v

    #k2v = force_field.force(r + k1r * h/2)
    _, k2v = explicit_test_LJ(r + k1r*h/2, epsilon ,sigma, L)
    k2v = np.sum(k2v,axis = 1)
    k2r = v + k1v * h/2

    #k3v = force_field.force(r + k2r * h/2)
    _, k3v = explicit_test_LJ(r + k2r*h/2, epsilon ,sigma, L)
    k3v = np.sum(k3v,axis = 1)
    k3r = v + k2v * h/2

    #k4v = force_field.force(r + k3r * h)
    _, k4v = explicit_test_LJ(r + k3r*h, epsilon ,sigma, L)
    k4v = np.sum(k4v,axis = 1)
    k4r = v + k3v * h

    v += (k1v + 2*k2v + 2*k3v + k4v) * h/6
    r += (k1r + 2*k2r + 2*k3r + k4r) * h/6
    r = PBC_wrapping(r,L)

    kinetic_energy = 0.5 * np.sum(np.einsum("ij,ij->i",v,v))
    potential_energy, _ = explicit_test_LJ(r, epsilon ,sigma, L)
    potential_energy = np.sum(potential_energy)
    #potential_energy = force_field.potential(r)

    if i % 10 == 0 or i <100:
        print(i)
        print("Total energy", kinetic_energy + potential_energy/2)

        print("\t + kinetic_energy",kinetic_energy)
        print("\t + potential_energy",potential_energy)




