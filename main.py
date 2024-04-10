import pickle
import time
import numpy as np

from utils import PBC_wrapping, orthogonalize

from distance import DistanceCalculator, explicit_test
from neighborlist import neighbor_list_mask

from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction

from parameter import epsilon_Ar_Ar, epsilon_Xe_Xe, epsilon_Ar_Xe, sigma_Ar_Ar, sigma_Xe_Xe, \
    sigma_Ar_Xe, M_Ar, M_Xe, mu0_1, d0_1, a1

from electromagnetic import VectorPotential

import input_dat

np.random.seed(319)

########################
###### BOX LENGTH ######
########################

L = 60
#L = input_dat.L
cell_width = 20

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
#N_Ar = len(input_dat.r_ar) 
N_Ar = int(L/3)
#N_Xe = len(input_dat.r_xe)
N_Xe = int(L/3)
N = N_Ar + N_Xe

# randomized initial coordinates
#R_all = np.vstack([input_dat.r_xe, input_dat.r_ar]) 
R_all = np.random.uniform(-L/2, L/2, (N, 3))

# randomized initial velocity
#V_all = np.vstack([input_dat.v_xe, input_dat.v_ar]) #
V_all =np.random.uniform(-1e1, 1e1, (N,3))

# indices of atoms in the R_all and V_all
idxXe = np.hstack(
    [np.ones(N_Ar), np.zeros(N_Xe)]
)

idxAr = np.hstack(
    [np.zeros(N_Ar), np.ones(N_Xe)]
)

######################################
###### FORCE-RELATED PARAMETERS ######
######################################

epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
    + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
    + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
    + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
    + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

M_Xe /= M_Ar
M_Ar = 1

mass = M_Ar * idxAr + M_Xe * idxXe
mass_x3 = np.tile(mass[:,np.newaxis], (1,3))

#########################
###### FIELD INPUT ######
#########################

n_modes = 1
#k_vector = np.random.randint(low = -5, high = 5, size = (n_modes, 3))

k_vector = input_dat.k_vec #np.array([
    #orthogonalize(kvec) for kvec in k_vector
    #]) 

C = input_dat.C# (np.random.rand(len(k_vector),2) + np.random.rand(len(k_vector),2) * 1j) * 1e5

vector_potential = VectorPotential(k_vector, amplitude = C)

##########################################
###### INITIAL VARIABLES AND OTHERS ######
##########################################

n_steps = 50000
h = 1e-5
r = R_all
v = V_all

energy = {
    "potential_energy" : [],
    "kinetic_energy" : [],
    "total dipole" : [],
    "EM_energy" : [],
    "time" : [], "step" : [],
}

sim_time = 0
i = 0

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
        mu0=mu0_1, a=a1, d0=d0_1,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

###################################
###### SIMULATION START HERE ######
###################################

start = time.time()

#for i in range(1):
while sim_time < 10:

    if i % 10 == 0: 
        mask = neighbor_list_mask(r, L, cell_width)
        distance_calc.update_global_mask(mask)

        force_field.update_distance_calc(distance_calc)
        dipole_function.update(distance_calc)

    gradD = dipole_function.gradient(r)
    k1c = vector_potential.dot_C(r,v,gradD=gradD,C=C)

    ff = force_field.force(r) 
    emf = vector_potential.transv_force(r,v,gradD=gradD,C=C)

    k1v =  (ff + emf) / mass_x3
    k1r = v

    """
    print(k1r)
    print(k1v)
    print(k1c)
    """

    gradD = dipole_function.gradient(r + k1r*h/2)
    k2c = vector_potential.dot_C(r + k1r*h/2, v + k1v*h/2,gradD=gradD, C=C+k1c*h/2)

    ff = force_field.force(r + k1r*h/2)
    emf = vector_potential.transv_force(r+k1r*h/2, v+k1v*h/2, gradD=gradD,C=C+k1c*h/2)

    k2v = (ff + emf) / mass_x3
    k2r = v + k1v*h/2

    """
    print(k2r)
    print(k2v)
    print(k2c)
    #"""

    gradD = dipole_function.gradient(r + k2r*h/2)
    k3c = vector_potential.dot_C(r+k2r*h/2, v+k2v*h/2,gradD=gradD,C=C+k2c*h/2)

    ff = force_field.force(r + k2r*h/2) 
    emf = vector_potential.transv_force(r + k2r*h/2, v + k2v*h/2,gradD=gradD,C=C+k2c*h/2)
    k3v = (ff + emf) / mass_x3
    k3r = v + k2v*h/2

    """
    """

    gradD = dipole_function.gradient(r + k3r*h)
    k4c = vector_potential.dot_C(r+k3r*h,v+k3v*h,gradD=gradD,C=C+k3c*h)

    ff =  force_field.force(r + k3r * h)
    emf = vector_potential.transv_force(r + k3r*h, v + k3v*h, gradD=gradD,C=C+k3c*h)

    k4v = (ff + emf) / mass_x3
    k4r = v + k3v * h
    
    ##############
    ### UPDATE ###
    ##############

    C += (1*k1c + 2*k2c + 2*k3c + 1*k4c) * h/6
    v += (1*k1v + 2*k2v + 2*k3v + 1*k4v) * h/6
    r += (1*k1r + 2*k2r + 2*k3r + 1*k4r) * h/6
    r = PBC_wrapping(r,L)

    #print(r)
    #print(v)
    #print(C)

    vector_potential.update_amplitude(amplitude = C)

    ########################
    ### CALCULATE ENERGY ###
    ########################

    kinetic_energy = 0.5 * np.sum(np.einsum("ij,ij->i",v,v) * mass) 

    #print(kinetic_energy)

    potential_energy = force_field.potential(r)
    potential_energy = np.sum(potential_energy)

    #print(potential_energy)

    H_em = vector_potential.hamiltonian()
    H_em_total = np.sum(H_em)

    #print(H_em_total)

    ####################
    ### SAVING STUFF ###
    ####################

    trajectory["potential_energy"].append(potential_energy)
    trajectory["kinetic_energy"].append(kinetic_energy)
    trajectory["EM_energy"].append(H_em_total)

    dipole_vec_tensor = dipole_function(r)

    total_dipole_vec = np.sum(dipole_vec_tensor, axis = 0)
    total_dipole = np.sqrt(total_dipole_vec @ total_dipole_vec)
    trajectory["total dipole"].append(total_dipole)

    sim_time += h
    i += 1
    trajectory["time"].append(sim_time)

    if potential_energy < 10:
        h = 1e-4
    elif potential_energy < 100:
        h = 1e-5
    elif potential_energy < 1000:
        h = 1e-6
    else:
        h = 1e-7

    if i % 100 == 0:
        print("-- Iteration #", i,  " Simulated time: ",sim_time, "--")
        print("Total energy", kinetic_energy + potential_energy + H_em_total)

        print("\t + kinetic_energy",kinetic_energy)
        print("\t + potential_energy",potential_energy)
        print("\t + field Hamiltonian",H_em_total)
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

