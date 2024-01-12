from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
import constants

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
ra0 = np.random.rand(3) # np.random.rand() * np.array([1,0,0]) #
va0 = np.random.rand(3) # np.random.rand() * np.array([1,1,1]) #
qa = np.array([1])

# choosing time step

h = 0.001

time_step = 500

###################
n_points = 1

ra_list = [ra0]
va_list = [va0]
C_list = [A.C]
energy_list = []

C = C_list[-1]
ra = ra_list[-1]
va = va_list[-1]

print(A.C)
