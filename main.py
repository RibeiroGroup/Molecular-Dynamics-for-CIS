import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import DistanceCalculator, get_dist_matrix, PBC_wrapping, timeit
from scipy.constants import m_e, m_n, m_p

from forcefield import MorsePotential, LennardJonesPotential, construct_param_matrix
from dipole import DipoleFunction

with open("result_plot//trajectory_2.pkl","rb") as handle:
    trajectory = pickle.load(handle)

########### BOX DIMENSION ##################

L = 40 # trajectory["L"]

########### PARTICLES ##################


np.random.seed(100)
#all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))
all_r = np.vstack([
    trajectory["initial r_Ar"],
    trajectory["initial r_Xe"]
    ])
print(all_r.shape)

#all_v = np.random.uniform(-1e2, 1e2, size=(n_points,3))
all_v = np.vstack([
    trajectory["initial v_Ar"],
    trajectory["initial v_Xe"],
])
print(all_v.shape)

n_points = len(all_r)
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

lennardj = LennardJonesPotential(
    n_points = n_points,
    epsilon = epsilon,
    sigma = sigma,
    L = L)

gri_dipole_func = DipoleFunction(
        parameters = {
            "mu0":0.0124, "R0": 7.10, "a":1.5121, "D7":0},
        engine = "grigoriev"
)

#######################################################################
##################### SIMULATION START ################################
#######################################################################

class DipoleCalculator:
    def __init__(self, n_points, distance_calc, dipole_function):

        self.distance_calc = distance_calc
        self.dipole_function = dipole_function

    def total_dipole_vector(self, r):
        distance_vector = self.distance_calc(r)
        distance_mat = get_dist_matrix(distance_vector)

        # take right upper part of the distance matrix
        rvec_ar_xe = distance_vector[
            : int(n_points/2) , int(n_points/2) : , :]

        r_ar_xe = distance_mat[
            : int(n_points/2) , int(n_points/2) :].ravel()

        # only need the upper triangle matrix part of the distance vector
        # to avoid double counting
        rvec_ar_xe = rvec_ar_xe.reshape(-1,3)

        dipole = self.dipole_function(r_ar_xe)
        dipole /= r_ar_xe
        dipole = np.tile(dipole[:,np.newaxis],(1,3))

        dipole = dipole * rvec_ar_xe 

        total_dipole_vec = np.sum(dipole,axis = 0)

        return total_dipole_vec

@timeit
def run_md_sim(
    n_points, weight_tensor, 
    r, v, 
    potential, 
    h, n_steps, L, n_records, 
    dipole_function = None):

    if dipole_function is not None:
        dipole_calc = DipoleCalculator(
            n_points = n_points, 
            distance_calc = potential.distance_calc,
            dipole_function = dipole_function)

    # for recording the trajectory
    trajectory = {"steps": [0], "T":[], "V":[], "H":[], "r":[], "L": L, "h": h, "dipole": []}

    T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T) * weight_tensor)
    trajectory["T"].append(T)
    V = potential.get_potential(r)
    trajectory["V"].append(V)
    H = T + V
    trajectory["H"].append(H)
    H0 = H
    trajectory["r"].append(r)
    total_dipole_vec = dipole_calc.total_dipole_vector(r)
    total_dipole = np.sqrt(total_dipole_vec @ total_dipole_vec.T)
    trajectory["dipole"].append(total_dipole)

    n_records = int(n_steps/n_records)

    for i in range(1, n_steps + 1):

        # the mentioned calculation of distance vector and matrix 
        distance_vector = potential.distance_calc(r)
        distance_matrix = get_dist_matrix(distance_vector)
        ### DIPOLE CALCULATION START ###
        if dipole_function is not None:
            total_dipole_vec = dipole_calc.total_dipole_vector(r)
            total_dipole = np.sqrt(total_dipole_vec @ total_dipole_vec.T)

        ### DIPOLE CALCULATION END ###

        weight_tensor_x3 = np.tile(weight_tensor[:,np.newaxis], (1,3))

        # RUNGE - KUTTA 4TH ORDER CALCULATION  
        # Note, since the distance vector and matrix is calculated later (with updated r)
        # for calculating the dipole, they is very well be used for the first calculation of 
        # the update based on force.
        k1v = potential.get_force(r) / weight_tensor_x3
        k1r = v

        k2v = potential.get_force(r + k1r*h/2) / weight_tensor_x3
        k2r = v + k1v*h/2

        k3v = potential.get_force(r + k2r*h/2) / weight_tensor_x3
        k3r = v + k2v*h/2

        k4v = potential.get_force(r + k3r*h) / weight_tensor_x3
        k4r = v + k3v*h
        # RUNGE - KUTTA 4TH ORDER UPDATE 

        r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
        r = PBC_wrapping(r, L)
        v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
        # RUNGE - KUTTA 4TH ORDER FINISH 

        T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T) * weight_tensor)
        V = potential.get_potential(r)
        H = T + V

        if i % n_records == 0:
            trajectory["steps"].append(i)
            trajectory["T"].append(T)
            trajectory["V"].append(V)
            trajectory["H"].append(H)
            trajectory["r"].append(r)
            trajectory["dipole"].append(total_dipole)

        if i % 1000 == 0:
            print("H = ",H, " V = ", V, " T = ", T)
            print("dipole = ", total_dipole)

    print("Total Hamiltonian variation: ", 
            max(trajectory["H"]) - min(trajectory["H"]))

    print("Total Hamiltonian deviation: ", 
            np.std(trajectory["H"]) )

    return trajectory

h = 1e-4
n_steps = 10000

trajectory = run_md_sim(
    n_points = n_points, weight_tensor = weight_tensor, r = all_r , v = all_v,
    potential = lennardj, h = h, n_steps = n_steps, L = L, n_records = 10000,
    dipole_function = gri_dipole_func
        )

with open("result_plot/no_EM_trajectory.pkl","wb") as handle:
    pickle.dump(trajectory, handle)



