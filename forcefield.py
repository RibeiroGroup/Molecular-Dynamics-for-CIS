import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import DistanceCalculator, get_dist_matrix, PBC_wrapping, timeit

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
        
        f[np.eye(self.n_points, dtype = bool)] = 0

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

