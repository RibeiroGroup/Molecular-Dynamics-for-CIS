import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import PBC_wrapping, timeit
from distance import DistanceCalculator, explicit_test

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

        potential_matrix = self.distance_calc.apply_function(
            R, func=self.get_potential, output_shape = 1)

        if return_matrix: return potential_matrix

        return np.sum(potential_matrix)

    def force(self, R, return_matrix = False):
        """
        Calculate the force.
        Args:
        + R (np.array): have shape N x 3 for cartesian coordinates of N particles
        Returns:
        + np.array: 
        """

        force_matrix = self.distance_calc.apply_function(
            R, func=self.get_force, output_shape = 3)

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

def construct_param_matrix(n_points, half_n_points, pure_param, mixed_param):

    """
    Constructing the parameter matrix for potential of mixture of atoms:
    Given pure parameter r_aa and r_bb, and "mixed parameter" r_ab, the result look like:
    [r_aa ... r_aa r_ab ... r_ab
    ...
    r_ab ... r_ab r_bb ... r_bb
    ...]
    where the block of first n rows and first n columns are r_aa,
    block of first n_rows and last m_columns are r_bb
    block of last m_rows and first n_columns are r_ab
    block of last m_rows and last m_columns are r_bb
    """

    param_matrix = np.zeros((n_points, n_points))

    param_matrix[0:half_n_points, 0:half_n_points] = pure_param[0]
    param_matrix[half_n_points:n_points, half_n_points:n_points] = pure_param[1]

    param_matrix[0:half_n_points, half_n_points:n_points] = mixed_param
    param_matrix[half_n_points:n_points, 0:half_n_points] = mixed_param

    param_matrix[np.eye(n_points, dtype=bool)] = 0

    return param_matrix

