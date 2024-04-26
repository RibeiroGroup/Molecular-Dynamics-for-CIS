import os
import pickle
import time
import numpy as np

from utils import PBC_wrapping, orthogonalize

from distance import DistanceCalculator, explicit_test
from neighborlist import neighbor_list_mask

from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction

from reduced_parameter import sigma_ as len_unit, epsilon_ as energy_unit, time_unit, \
    M, c as v_light

from reduced_parameter import epsilon_Ar_Ar, epsilon_Xe_Xe, epsilon_Ar_Xe, sigma_Ar_Ar, sigma_Xe_Xe, \
    sigma_Ar_Xe, M_Ar, M_Xe, mu0, d0, a

import constants

from electromagnetic import VectorPotential

import input_dat
import constants

########################
########################
########################

free_em_field = 1
np.random.seed(39)

########################
###### BOX LENGTH ######
########################

L = 20
V = L ** 3
cell_width = 4

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = int(L * 2)
N_Xe = int(L * 2)
N = N_Ar + N_Xe

# randomized initial coordinates
R_all = np.random.uniform(-L/2, L/2, (N, 3))
#R_all = np.array([[1.0,1.0,1.0],[-1.0,-1.0,-1.0]]) * 0.3

# randomized initial velocity
V_all =np.random.uniform(-1e3, 1e3, (N,3))
#V_all = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]]) * 10

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

mass = M_Ar * idxAr + M_Xe * idxXe
mass_x3 = np.tile(mass[:,np.newaxis], (1,3))

#########################
###### FIELD INPUT ######
#########################

n_modes = 5

#k_vector = np.random.randint(low = -3, high = 3, size = (n_modes,3))
#k_vector += np.tile(np.array([[1,0,0]]),(n_modes,1))
k_vector = np.vstack([
        np.array([[1,0,0],[0,1,0],[0,0,1]]),
        np.array([[1,1,0],[0,1,1],[1,0,1]]),
        np.array([[1,1,1]]),
        np.array([[1,0,0],[0,1,0],[0,0,1]]) * 2,
        np.array([[1,1,0],[0,1,1],[1,0,1]]) * 2,
        np.array([[1,1,1],[1,-1,1],[1,1,-1]]) * 2,
        np.array([[1,2,0],[0,1,2],[1,0,2]]),
        np.array([[2,1,0],[0,2,1],[2,0,1]]),
        np.array([[2,1,1],[1,2,1],[1,1,2]]),
        np.array([[2,2,1],[1,2,2],[2,1,2]]),
        ])

k_vector = np.array(k_vector, dtype= np.float64) 

k_vector *= (2 * np.pi / L)

k_vector = np.array([
    orthogonalize(kvec) for kvec in k_vector
    ]) 

C = (np.random.rand(len(k_vector),2) + np.random.rand(len(k_vector),2) * 1j)* 0e0 \
        * V**-0.5
        
##########################################
###### INITIAL VARIABLES AND OTHERS ######
##########################################

h = 1e-10
r = R_all
v = V_all

def generate_empty_record():
    energy_data = {
        "potential_energy" : [],
        "kinetic_energy" : [],
        "total dipole" : [],
        "EM_energy" : [],
        "time" : [], "step" : [],
    }

    trajectory_data = {
        "position" : [],
        "velocity" :[],
        "field amplitude":[]
            }
    return energy_data, trajectory_data

energy_data, _ = generate_empty_record()

sim_time = 0

data_save_point = 0
data_save_interval = 1e-5

check_point = 0.0
chkp_interval = 1e-3

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
        mu0=mu0,
        a=a, d0=d0,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

vector_potential = VectorPotential(
        k_vector, amplitude = C,
        V = V, 
        #epsilon_0 = 1 * M_Ar, 
        speed_of_light = v_light,
        )

###################################
###### SIMULATION START HERE ######
###################################

start = time.time()

prev_energy = 1

max_h = 1e-8

foo = int(cell_width / (max_h * 1e5))
print(foo)

while sim_time < 1000:
   
    if i % foo == 0: 

        mask = neighbor_list_mask(r, L, cell_width)
        distance_calc.update_global_mask(mask)

        force_field.update_distance_calc(distance_calc)
        dipole_function.update(distance_calc)

    ############
    ### RK 1 ###
    ############
    if free_em_field:
        gradD = dipole_function.gradient(r)
        k1c = vector_potential.dot_C(r,v,gradD=gradD,C=C)
        emf = vector_potential.transv_force(r,v,gradD=gradD,C=C)
    else: emf = 0

    ff = force_field.force(r) 
    k1v =  (ff + emf) / mass_x3
    k1r = v

    ############
    ### RK 2 ###
    ############
    if free_em_field:
        gradD = dipole_function.gradient(r + k1r*h/2)
        k2c = vector_potential.dot_C(r + k1r*h/2, v + k1v*h/2,gradD=gradD, C=C+k1c*h/2)
        emf = vector_potential.transv_force(r+k1r*h/2, v+k1v*h/2, gradD=gradD,C=C+k1c*h/2)

    ff = force_field.force(r + k1r*h/2)
    k2v = (ff + emf) / mass_x3
    k2r = v + k1v*h/2

    ############
    ### RK 3 ###
    ############
    if free_em_field:
        gradD = dipole_function.gradient(r + k2r*h/2)
        k3c = vector_potential.dot_C(r+k2r*h/2, v+k2v*h/2,gradD=gradD,C=C+k2c*h/2)
        emf = vector_potential.transv_force(r + k2r*h/2, v + k2v*h/2,gradD=gradD,C=C+k2c*h/2)

    ff = force_field.force(r + k2r*h/2) 
    k3v = (ff + emf) / mass_x3
    k3r = v + k2v*h/2

    ############
    ### RK 4 ###
    ############
    if free_em_field:
        gradD = dipole_function.gradient(r + k3r*h)
        k4c = vector_potential.dot_C(r+k3r*h,v+k3v*h,gradD=gradD,C=C+k3c*h)
        emf = vector_potential.transv_force(r + k3r*h, v + k3v*h, gradD=gradD,C=C+k3c*h)

    ff =  force_field.force(r + k3r * h)
    k4v = (ff + emf) / mass_x3
    k4r = v + k3v * h
    
    ##############
    ### UPDATE ###
    ##############

    v += (1*k1v + 2*k2v + 2*k3v + 1*k4v) * h/6
    r += (1*k1r + 2*k2r + 2*k3r + 1*k4r) * h/6
    r = PBC_wrapping(r,L)

    if free_em_field:
        deltaC = (1*k1c + 2*k2c + 2*k3c + 1*k4c) * h/6
        vector_potential.update_amplitude(deltaC = deltaC)

    sim_time += (h)
    i += 1

    potential_energy = force_field.potential(r)
    potential_energy = np.sum(potential_energy)

    #########################################
    ### TOTAL DIPOLE and POTENTIAL ENERGY ###
    #########################################

    dipole_vec_tensor = dipole_function(r)

    total_dipole_vec = np.sum(dipole_vec_tensor, axis = 0)

    total_dipole = np.sqrt(total_dipole_vec @ total_dipole_vec)

    h = max_h
    """
    if potential_energy < 1000:# and total_dipole < 10000:
        h = 1e-8
    elif potential_energy < 10000:# and total_dipole < 100000:
        h = 1e-9
    else:
        h = 1e-10
    """

    ###################
    ### SAVING DATA ###
    ###################

    if sim_time > data_save_point:
        data_save_point += data_save_interval 

        energy_data["time"].append(sim_time)

        energy_data["total dipole"].append(total_dipole)

        ##########################################
        ### CALCULATING AND SAVING OBSERVABLES ###
        ##########################################

        kinetic_energy = 0.5 * np.sum(np.einsum("ij,ij->i",v,v) * mass)
        energy_data["kinetic_energy"].append(kinetic_energy)

        energy_data["potential_energy"].append(potential_energy)

        if free_em_field:
            H_em = vector_potential.hamiltonian()
            H_em_total = np.sum(H_em)
            energy_data["EM_energy"].append(H_em)
        else: 
            H_em_total = 0

        total_energy = kinetic_energy + potential_energy + H_em_total

        ######################
        ### PRINTING STUFF ###
        ######################

        print("-- Data saving... -- Iteration #", i,  " Simulated time: ",sim_time, "--")
        print("Total energy (reduced unit)", total_energy)
        #print("Total energy (kj/mol)", total_energy * energy_unit)

        print("\t + kinetic_energy",kinetic_energy)
        print("\t + potential_energy",potential_energy)

        if free_em_field:
            #print("\t + field Hamiltonian")
            #[print("\t + \t", i) for i in H_em]
            print("\t + total field Hamiltonian",H_em_total)

        print("\t + total dipole",total_dipole)

        print("Runtime: ", time.time() - start)

    if sim_time > check_point:
        check_point += chkp_interval
        with open("result_plot/trajectory_temp.pkl","wb") as handle:
            pickle.dump(energy_data,handle)

        #print("Autosave!")

print("############ JOB COMPLETE ############")
print("Total runtime: ", time.time() - start)

with open("result_plot/trajectory_temp.pkl","wb") as handle:
    pickle.dump(energy_data,handle)

