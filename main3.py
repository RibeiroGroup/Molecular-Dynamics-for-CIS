import pickle
import numpy as np
import matplotlib.pyplot as plt

from utils import DistanceCalculator, get_dist_matrix, PBC_wrapping
from scipy.constants import m_e, m_n, m_p

########### BOX DIMENSION ##################

n_points = 16
L = 10

########### PARTICLES ##################

np.random.seed(100)
all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))
print(all_r.shape)
all_v = np.random.uniform(-1e1, 1e1, size=(n_points,3))
print(all_v.shape)


class BasePotential:
    """
    Base potential class
    Args:
    + n_points (int): number of particles in the simulation
    + L (float): box length for the wrapping effect of the periodic boundary condition
    """
    def __init__(self, n_points, L = None):

        self.distance_calc = DistanceCalculator(n_points,L)
        self.n_points = n_points

    def get_potential(self,R):
        """
        Calculate the potential.
        Args:
        + R (np.array): have shape N x 3 for cartesian coordinates of N particles
        Returns:
        + float: potential energy summing from all atom-atoms interactions
        """

        distance_vector = self.distance_calc(R)
        distance_matrix = get_dist_matrix(distance_vector)

        distance = distance_matrix[
            np.triu(
                np.ones(distance_matrix.shape, dtype=bool),
                k=1)
            ]

        return np.sum(self.potential(distance))

    def get_force(self, R):
        """
        Calculate the force.
        Args:
        + R (np.array): have shape N x 3 for cartesian coordinates of N particles
        Returns:
        + np.array: 
        """

        distance_vector = self.distance_calc(R)
        distance_matrix = get_dist_matrix(distance_vector)

        f = self.force(
                distance_matrix = distance_matrix, 
                distance_vector = distance_vector) 

        f = np.sum(f, axis = 1)

        return f

class LennardJonesPotential(BasePotential):
    def __init__(self, epsilon, sigma, n_points, L = None):

        super().__init__(n_points, L)
        self.epsilon = epsilon
        self.sigma = sigma

    def potential(self, distance_matrix):
        epsilon = self.epsilon[
                np.triu(
                    np.ones((self.n_points, self.n_points), dtype = bool),
                    k = 1
                    )]

        sigma = self.sigma[
                np.triu(
                    np.ones((self.n_points, self.n_points), dtype = bool),
                    k = 1
                    )]

        V = 4 * epsilon * ( (sigma/distance_matrix)**12 - (sigma/distance_matrix)**6 )

        return V

    def force(self,distance_matrix,distance_vector):
        distance_matrix += np.eye(self.n_points)

        f = 4 * self.epsilon * (
                12 * (self.sigma**12 / distance_matrix**14) - 6 * (self.sigma**6 / distance_matrix**8)
                )

        f = np.tile(f[:,:,np.newaxis],(1,1,3)) * (distance_vector)

        return f

class MorsePotential(BasePotential):
    def __init__(self, De, Re, a, n_points, L = None):

        super().__init__(n_points, L)

        self.De = De
        self.Re = Re
        self.a = a

    def potential(self, distance):

        return self.De*(1 - np.exp(-self.a*(-self.Re + distance)))**2

    def force(self,distance_matrix, distance_vector):

        f= -2*self.De*self.a*(1 - np.exp(-self.a*(-self.Re + distance_matrix)))\
            *np.exp(-self.a*(-self.Re + distance_matrix)) \
            /( distance_matrix + np.eye(len(distance_matrix)) )

        f = np.tile(f[:,:,np.newaxis], (1,1,3))

        f *= distance_vector# (-1.0*Rx + 1.0*x) 

        return f

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

def construct_param_matrix(n_points, half_n_points, pure_param, mixed_param):

    param_matrix = np.zeros((n_points, n_points))

    param_matrix[0:half_n_points, 0:half_n_points] = pure_param[0]
    param_matrix[half_n_points:n_points, half_n_points:n_points] = pure_param[1]

    param_matrix[0:half_n_points, half_n_points:n_points] = mixed_param
    param_matrix[half_n_points:n_points, 0:half_n_points] = mixed_param

    param_matrix[np.eye(n_points, dtype=bool)] = 0

    return param_matrix

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

potential = lennardj

#######################################################################
##################### SIMULATION START ################################
#######################################################################

r = all_r
v = all_v
h = 1e-4
n_steps = 50000

trajectory = {"steps": [0], "T":[], "V":[], "H":[], "r":[], "L": L}

T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T) * weight_tensor)
trajectory["T"].append(T)
V = potential.get_potential(r)
trajectory["V"].append(V)
H = T + V
trajectory["H"].append(H)
H0 = H
trajectory["r"].append(r)

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

    if i % 100 == 0:
        trajectory["steps"].append(i)
        trajectory["T"].append(T)
        trajectory["V"].append(V)
        trajectory["H"].append(H)
        trajectory["r"].append(r)

    if i % 100 == 0:
        print("H = ",H, " V = ", V, " T = ", T)

print(H0 - H)

with open("result_plot/trajectory.pkl","wb") as handle:
    pickle.dump(trajectory, handle)
