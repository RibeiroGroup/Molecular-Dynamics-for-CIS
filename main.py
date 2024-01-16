import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
import constants

from utils import compute_dist_mat1, verify_distant_matrix

np.random.seed(2023)

# defining parameters for the EM field

k_vec = np.array([
    [1,0,0],[0,1,0]
    ]) * np.random.rand(2,3)
epsilon = np.array([
    [[0, 1, 0], [0, 0, 1]],
    [[1, 0, 0], [0, 0, 1]]
    ])
C0 = (np.random.rand(2) + 1j *  np.random.rand(2))

k = np.sqrt(k_vec @ k_vec.T)
omega = constants.c * k

A = MultiModeField(C=C0, k=k_vec, epsilon=epsilon)

# particle with random position and velocity
n_points = 5
ra = np.random.rand(n_points,3)
va = np.random.rand(n_points,3)
qa = np.array([1] * n_points)

# choosing time step

h = 0.001

time_step = 500

# 


n_points = 10000
dist_mat_calc = DistantMatrix(n_points)

np_time = 0
py_time = 0

ra = np.random.rand(n_points,3)

start = time.time()
dist_mat = dist_mat_calc(ra)
np_time += time.time() - start

start = time.time()
dist_mat_ = verify_distant_matrix(ra)
py_time += time.time() - start

if np.any(dist_mat - dist_mat_ > 1e-5):
    raise Exception

print(np_time, py_time)

