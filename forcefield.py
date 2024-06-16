import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

import constants
from utils import PBC_wrapping, timeit
from distance import DistanceCalculator, explicit_test

run_test = 0

def LJ_potential(sigma, epsilon, distance):

    V = 4 * epsilon * ( (sigma/distance)**12 - (sigma/distance)**6 )

    return V

def LJ_force(sigma, epsilon, distance, distance_vec):

    f = 4 * epsilon * (
            12 * (sigma**12 / distance**14) - 6 * (sigma**6 / distance**8)
            )

    f = np.tile(f[:,np.newaxis],(1,3)) * (distance_vec)

    return f

@timeit
def explicit_test_LJ(R, epsilon ,sigma, L):

    N = len(R)

    potential = np.zeros((N,N))
    force = np.zeros((N,N,3))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j: continue
            
            ep = epsilon[i,j]
            sig = sigma[i,j]
            
            dvec = ri - rj
            dvec = PBC_wrapping(dvec,L)

            d = np.sqrt(dvec @ dvec)

            potential[i,j] = 4 * ep * ( (sig/d)**12 - (sig/d)**6 )
            f = 4 * ep * (
                12 * (sig**12 / d**14) - 6 * (sig**6 / d**8)
            )

            force[i,j,:] = f * (dvec)

    return potential, force

