import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import DistanceCalculator, get_dist_matrix, PBC_wrapping, timeit
from scipy.constants import m_e, m_n, m_p

from forcefield import MorsePotential, LennardJonesPotential, construct_param_matrix

########### BOX DIMENSION ##################

L = 6

########### PARTICLES ##################

n_points = 2

np.random.seed(100)
all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))
all_r = np.array([[-5,-5,-5],[5,5,5]])
print(all_r.shape)

all_v = np.random.uniform(-1e2, 1e2, size=(n_points,3))
all_v = np.array([[1,1,1],[-1,-1,-1]]) * 1e1
print(all_v.shape)

###########################################################
############# MATERIAL SPECIFICATION   ####################
###########################################################

half_n_points = int(n_points/2)

Z_Ar = 18
m_Ar = Z_Ar * (m_p / m_e) + Z_Ar + (40 - Z_Ar) * (m_n / m_e) 
Z_Xe = 54
m_Xe = Z_Xe * (m_p / m_e) + Z_Xe + (131 - Z_Xe) * (m_n / m_e)

weight_tensor = np.hstack([
    [m_Ar] * half_n_points,
    [m_Xe] * half_n_points
    ])

###########################################################
############# POTENTIAL SPECIFICATION   ###################
###########################################################

pure_epsilon = np.array([0.996, 1.904]) * 1.59360e-3
mixed_epsilon = 1.377 * 1.59360e-3

pure_sigma = np.array([3.41, 4.06]) * (1e-10 / 5.29177e-11)
mixed_sigma = 3.735 * (1e-10 / 5.29177e-11)

epsilon = construct_param_matrix(n_points,half_n_points,pure_epsilon,mixed_epsilon)
sigma = construct_param_matrix(n_points,half_n_points,pure_sigma,mixed_sigma)

morse = MorsePotential(
    n_points = n_points,
    De =  1495 / 4.35975e-18 / 6.023e23,
    Re = 3.5e-10 / 5.29177e-11,
    a = 1/ ( (1/3 * 1e-10) / 5.29177e-11),
    L = L
)

lennardj = LennardJonesPotential(
    n_points = n_points,
    epsilon = epsilon,
    sigma = sigma,
    L = L)

#######################################################################
##################### SIMULATION START ################################
#######################################################################

@timeit
def run_md_sim(n_points, weight_tensor, r, v, potential, h, n_steps, L, n_records):

    trajectory = {"steps": [0], "T":[], "V":[], "H":[], "r":[], "L": L, "h": h}

    T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T) * weight_tensor)
    trajectory["T"].append(T)
    V = potential.get_potential(r)
    trajectory["V"].append(V)
    H = T + V
    trajectory["H"].append(H)
    H0 = H
    trajectory["r"].append(r)

    n_records = int(n_steps/n_records)

    for i in range(1, n_steps + 1):
        weight_tensor_x3 = np.tile(weight_tensor[:,np.newaxis], (1,3))

        k1v = potential.get_force(r) / weight_tensor_x3
        k1r = v

        k2v = potential.get_force(r + k1r*h/2) / weight_tensor_x3
        k2r = v + k1v*h/2

        k3v = potential.get_force(r + k2r*h/2) / weight_tensor_x3
        k3r = v + k2v*h/2

        k4v = potential.get_force(r + k3r*h) / weight_tensor_x3
        k4r = v + k3v*h

        r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
        r = PBC_wrapping(r, L)
        v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)

        T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T) * weight_tensor)
        V = potential.get_potential(r)
        H = T + V

        if i % n_records == 0:
            trajectory["steps"].append(i)
            trajectory["T"].append(T)
            trajectory["V"].append(V)
            trajectory["H"].append(H)
            trajectory["r"].append(r)

        if i % 1000 == 0:
            print("H = ",H, " V = ", V, " T = ", T)

    print("Total Hamiltonian variation: ", 
            max(trajectory["H"]) - min(trajectory["H"]))

    print("Total Hamiltonian deviation: ", 
            np.std(trajectory["H"]) )

    return trajectory

h = 1e-4
n_steps = 10000

trajectory = run_md_sim(
    n_points = n_points, weight_tensor = weight_tensor, r = all_r , v = all_v,
    potential = lennardj, h = h, n_steps = n_steps, L = L, n_records = 500
        )

with open("result_plot/trajectory.pkl","wb") as handle:
    pickle.dump(trajectory, handle)



