import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import PBC_wrapping, timeit
from distance import DistanceCalculator, explicit_test

run_test = True

class BasePotential:
    """
    Base potential class
    Args:
    + n_points (int): number of particles in the simulation
    + L (float): box length for the wrapping effect of the periodic boundary condition
    """
    def __init__(self, distance_calc):

        assert isinstance(distance_calc, DistanceCalculator)

        self.distance_calc = distance_calc

        self.n_points = self.distance_calc.n_points

    def update_distance_calc(self,distance_calc):

        self.distance_calc = distance_calc

    def potential(self, R, return_matrix = False):
        """
        Calculate the potential.
        Args:
        + R (np.array): have shape N x 3 for cartesian coordinates of N particles
        Returns:
        + float: potential energy summing from all atom-atoms interactions
        """

        potential_array = self.distance_calc.apply_function(
            R, func=self.get_potential)

        if return_matrix: 
            potential_matrix = self.distance_calc.construct_matrix(
                potential_array, output_shape = 1, symmetry = 1)
            return potential_matrix

        return np.sum(potential_array)

    def force(self, R, return_matrix = False):
        """
        Calculate the force.
        Args:
        + R (np.array): have shape N x 3 for cartesian coordinates of N particles
        Returns:
        + np.array: 
        """

        force_array = self.distance_calc.apply_function(
            R, func=self.get_force)

        force_matrix = self.distance_calc.construct_matrix(
            force_array, output_shape = 3, symmetry = -1)

        if return_matrix: return force_matrix

        f = np.sum(force_matrix, axis = 1)

        return f

class LennardJonesPotential(BasePotential):
    def __init__(self, epsilon, sigma, distance_calc):

        """
        Args:
        + epsilon (np.ndarray):
        + sigma (np.ndarray): Note, both epsilon and sigma have to be a matrix
            where the ij element is the parameter describe the force between 
            i-th and j-th atoms (e.g. Ar-Ar, Xe- Xe, or Ar-Xe)
        """

        super().__init__(distance_calc)
        assert epsilon.shape == (self.n_points, self.n_points)
        self.epsilon = epsilon

        assert sigma.shape == (self.n_points, self.n_points)
        self.sigma = sigma

    def get_potential(self, distance, distance_vec):
        epsilon = self.epsilon[self.distance_calc.utriang_mask]
        sigma = self.sigma[self.distance_calc.utriang_mask]

        V = 4 * epsilon * ( (sigma/distance)**12 - (sigma/distance)**6 )

        return V

    def get_force(self,distance,distance_vec):
        epsilon = self.epsilon[self.distance_calc.utriang_mask]
        sigma = self.sigma[self.distance_calc.utriang_mask]

        f = 4 * epsilon * (
                12 * (sigma**12 / distance**14) - 6 * (sigma**6 / distance**8)
                )

        f = np.tile(f[:,np.newaxis],(1,3)) * (distance_vec)

        return f

class MorsePotential(BasePotential):
    def __init__(self, De, Re, a, distance_calc): 

        super().__init__(distance_calc)

        self.De = De
        self.Re = Re
        self.a = a

    def potential(self, distance):

        return self.De*(1 - np.exp(-self.a*(-self.Re + distance)))**2

    def force(self,distance, distance_vector):

        f= -2*self.De*self.a*(1 - np.exp(-self.a*(-self.Re + distance_matrix)))\
            *np.exp(-self.a*(-self.Re + distance_matrix)) \
            /( distance_matrix + np.eye(len(distance_matrix)) )

        f = np.tile(f[:,:,np.newaxis], (1,1,3))

        f *= distance_vector# (-1.0*Rx + 1.0*x) 

        return f

def explicit_test_LJ(R, epsilon ,sigma, L):

    N = len(R)
    distance_, distance_vec_ = explicit_test(R, L)

    epsilon_ = epsilon[~np.eye(N,dtype=bool)].reshape(N,N-1)
    sigma_ = sigma[~np.eye(N,dtype=bool)].reshape(N,N-1)

    potential_ = 4 * epsilon_ * ( (sigma_/distance_)**12 - (sigma_/distance_)**6 )

    epsilon_ = epsilon[~np.eye(N,dtype=bool)].reshape(N,N-1)
    sigma_ = sigma[~np.eye(N,dtype=bool)].reshape(N,N-1)

    force_ =  4 * epsilon_ * (
            12 * (sigma_**12 / distance_**14) - 6 * (sigma_**6 / distance_**8)
        )

    force_ = np.tile(force_[:,:,np.newaxis],(1,1,3)) * (distance_vec_)

    return potential_, force_

if run_test:
    try:
        from neighborlist import neighbor_list_mask
        neighbor_list_module_availability = True
    except:
        print("Neighborlist module cannot be found. Testing without neighborlist.")

    ########################
    ######### TEST #########
    ########################
    ###### BOX LENGTH ######
    ########################

    L = 10
    cell_width = 20

    ##########################
    ###### ATOMIC INPUT ######
    ##########################

    # number of atoms
    N_Ar = int(L/2)
    N_Xe = int(L/2)
    N = N_Ar + N_Xe

    # randomized initial coordinates

    R_all = np.random.uniform(-L/2, L/2, (N, 3))

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

    ############################################
    ##### Test without neighbor cell list. #####
    ############################################

    distance_calc = DistanceCalculator(n_points = N, box_length = L)
    forcefield = LennardJonesPotential(epsilon_mat, sigma_mat, distance_calc)

    potential = forcefield.potential(R_all, return_matrix = True)
    force = forcefield.force(R_all, return_matrix = True)

    potential_,force_ = explicit_test_LJ(R_all, epsilon_mat, sigma_mat, L)

    print(np.sum(abs(potential - potential_)))
    print(np.sum(abs(force - force)))


