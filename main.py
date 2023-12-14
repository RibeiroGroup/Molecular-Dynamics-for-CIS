import numpy as np

from EM_field import vector_potential
from num_pde import RK4
from test_cases import A_test_case

np.random.seed(2022)

k_vec = np.array([1,0,0])
epsilon = np.array([
    [0, 1, 0], [0, 0, 1]
    ])
C0 = np.random.rand(size=2) +1j *  np.random.rand(size=2)

c = 1.37036e2
k = np.sqrt(k_vec @ k_vec)
omega = c * k

x = np.array([0.1,0.2,0.3])

ra = np.random.rand(3)
va = np.random.rand(3)

