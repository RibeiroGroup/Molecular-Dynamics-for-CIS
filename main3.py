import numpy as np
from utils import DistanceCalculator, \
    test_for_distance_vector, test_for_distance_matrix

from forcefield import MorsePotential
from simpleForceField import MorsePotential as smMorsePotential

n_points = 100
all_r = np.random.rand( n_points, 3) * 5

########### BOX DIMENSION ##################

L = 15

rij_vec_tensor = np.zeros((n_points, n_points,3))
rij_matrix = np.zeros((n_points, n_points))

for i in range(len(all_r)):
    for j in range(i, len(all_r)):

        if i == j: continue

        ri = all_r[i]; rj = all_r[j];

        rij_vec = ri - rj
        rij = np.sqrt(np.sum((ri - rj)**2))

        rij_vec_tensor[i,j,:] = rij_vec
        rij_matrix[i,j] = rij

