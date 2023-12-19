import numpy as np
import matplotlib.pyplot as plt

from EM_field import vector_potential

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


C, k_vec , epsilon = A_test_case(True,2020)

A = vector_potential(C,k_vec,epsilon)
k_vec = A.k
n_mode = A.n_mode

np.random.seed(1010)
n_charge = 5
ra = np.random.rand(n_charge,3)
va = np.random.rand(n_charge,3)
qa = np.array([1,-1,1,-1,-1]).reshape(-1,1)

# compute J_k for each qa and sum over 

exp_ikx = np.tile(
    np.exp(-1j * k_vec @ ra.T)[:,:,np.newaxis],
    (1,1,3))

ra_ = np.tile(ra,(n_mode,1,1))

qa_ = np.tile(
    qa[np.newaxis,:,:],
    (n_mode, 1,3)
    )

jk = (1/(2*np.pi**1.5)) * exp_ikx * ra_ * qa_

class electric_field:
    def __init__(self, ra, va, qa):
        pass

print(ra[0])
print(A(ra[0]))
