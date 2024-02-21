import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
from sympy.abc import a as sm_a
from sympy.abc import d as sm_D
from sympy.abc import r as sm_Re
from sympy.abc import x as sm_x, y as sm_y, z as sm_z

import constants

sm_Rx = sm.symbols("Rx")
sm_Ry = sm.symbols("Ry")
sm_Rz = sm.symbols("Rz")
sm_D = sm.symbols("D")
sm_a = sm.symbols("a")
sm_Re = sm.symbols("Re")

class MorsePotential:
    """
    Class for calculating Morse potential
    """
    def __init__(self, De, Re, a):
        self.De = De
        self.Re = Re
        self.a = a

        Rij = ( (sm_x - sm_Rx)**2 \
                + (sm_y - sm_Ry)**2 \
                + (sm_z - sm_Rz)**2 )**0.5

        self.morse_exp = De*(1 - sm.exp(-a*(Rij - Re)))**2
        self.morse = sm.lambdify(
                [sm_Rx, sm_Ry, sm_Rz, sm_x, sm_y, sm_z],
                self.morse_exp)

        self.force_exp = [
            -sm.diff(self.morse_exp, sm_sym)
            for sm_sym in [sm_x, sm_y, sm_z]
                ]

        self.force = [
            sm.lambdify([sm_Rx,sm_Ry,sm_Rz,sm_x,sm_y,sm_z], 
                force_component)
            for force_component in self.force_exp
                ]

    def __call__(self,center,R):
        assert len(R) == 3
        Rx, Ry, Rz = center
        x , y , z  = R
        return self.morse(Rx,Ry,Rz,x,y,z)

    def compute_force(self,center,R):
        Rx, Ry, Rz = center
        x , y , z  = R
        ma = np.array([force(Rx,Ry,Rz,x,y,z)
            for force in self.force])
        return ma

def compute_Morse_force(r, potential):
    force_list = []
    for i, ri in enumerate(r):
        force = np.zeros(3)
        for j, rj in enumerate(r):
            if i == j: continue
            force += potential.compute_force(
                    center = rj, R = ri)

        force_list.append(force)
    return np.array(force_list)

def compute_Hmorse(r,potential):
    V = 0
    for i, ri in enumerate(r):
        for j, rj in enumerate(r):
            if i == j: continue
            # compute force on ri
            #=> rj is the center of Morse potential
            V += potential(center = rj, R = ri)
    return V/2

print("Morse Potential")

Rij = ((sm_x - sm_Rx)**2 + (sm_y - sm_Ry)**2 + (sm_z - sm_Rz)**2) ** 0.5
V = sm_D * (1 - sm.exp(-sm_a * (Rij - sm_Re))) ** 2

print(V)
print(sm.diff(-V, sm_x))

print("Lennard-Jones Potential")

sm_e = sm.symbols("e")
sm_s = sm.symbols("s")

Rij = ((sm_x - sm_Rx)**2 + (sm_y - sm_Ry)**2 + (sm_z - sm_Rz)**2) ** 0.5
V = 4 * sm_e * ( (sm_s/Rij)**12 - (sm_s/Rij)**6 )

print(V)
print(sm.diff(-V, sm_x))
"""

class BasePotential:
    def __init__(self, n_points, L = None):

        self.distance_calc = DistanceCalculator(n_points,L)
        self.n_points = n_points

    def get_potential(self,R):

        distance_vector = self.distance_calc(R)
        distance_matrix = get_dist_matrix(distance_vector)

        distance = distance_matrix[
            np.triu(
                np.ones(distance_matrix.shape, dtype=bool),
                k=1)
            ]

        return np.sum(self.potential(distance))

    def get_force(self, R):

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
"""
