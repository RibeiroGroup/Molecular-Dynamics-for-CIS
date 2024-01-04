import numpy as np
import matplotlib.pyplot as plt

from EM_field_simple import SingleModeField

def A_test_case(single = False, seed = None):
    if seed: np.random.seed(seed)
    C = np.array([
        [0.5 + 1.j, 1 + 0.5j], #C_{k = 1, epsilon = 1,2)
        [1 + 2.j, 2 + 1.j],
        [3 + 4.j, 4 + 3.j],
        [5 + 6.j, 6 + 5.j],
        ])

    k = np.array([
        [1,0,0], # k vector along x axis
        [0,1,0],
        [0,0,1],
        [0,1,0],
        ]) 

    epsilon = np.array([
        [[0,1,0] ,[0,0,1]], # polarization vectors can chosen
        [[1,0,0] ,[0,0,1]], # to be orthogonal to k
        [[1,0,0] ,[0,1,0]],
        [[1,0,0] ,[0,0,1]],
        ])
    
    if single:
        idx = np.random.randint(low = 0, high = len(C), size = 3)

        return C[idx[0],:].reshape(1,-1), \
            k[idx[1]].reshape(1,-1), epsilon[idx[1]]

    return C, k, epsilon

np.random.seed(123)

C = np.random.rand(2)
k_vec = np.array([1,1,1])
epsilon = np.array([
    [1, -0.5, -0.5], [-0.5,-0.5,1]
    ])

A = SingleModeField(C,k_vec,epsilon)
k_vec = A.k

ra = np.random.rand(3)
va = np.random.rand(3)
qa = 1

print(A.d_dx(ra))

Ax = A(ra)

dA_dx = np.tile(A.k[np.newaxis,:], (3,1)) \
    * np.tile(Ax.reshape(-1,1), (1,3))

print(dA_dx)

