import numpy as np
import matplotlib.pyplot as plt

def polarization_vectors(k):
    pass

class vector_potential:
    def __init__(self, C, k, epsilon, V = 1):
        """
        class for scallar potential A
        A = sum_k C_k(t) exp(-i k.x) + c.c.
          = sum_k real(C_k) cos(k.x) - imag(C_k) sin(k.x)
        inputs:
        + C (numpy.ndarray): array of complex values for C_k
        + k (numpy.ndarray): array of real vector k in R^3
        + epsilon (numpy.ndarray): set of polarization vector
        Note that len(C) == len(k)
        """
        assert C.shape[0] == k.shape[0]

        self.C = C
        self.k = k
        self.V = V

    def __call__(self,x):
        kx = k @ x
        kx = np.tile(kx.reshape(-1,1),(1,2))

        """
        Ckx = 2 * (np.real(C) * np.cos(kx) - np.imag(C) * np.sin(kx))
        """
        Ckx = C * np.exp(kx * np.array(1.j))
        Ckx += np.conjugate(C) * np.exp(kx * np.array(-1.j))

        Ckx = Ckx.reshape(*epsilon.shape[:-1], -1)
        Ckx = epsilon * Ckx 

        result = np.sum(Ckx, axis = 0)

        return result / self.V**(0.5)

    def curl(self,x):
        raise Exception("To be implemented")

    def div(self,x):
        raise Exception("To be implemented")

    def dot(self,x):
        """
        Derivative with respect with time
        """
        raise Exception("To be implemented")

class EMfield:
    def __init__(self, k_vectors):
        self.k_list = k_vector

def A_test_case(single = False, seed = None):
    if seed: np.random.seed(seed)
    C = np.array([
        [0.5 + 1.j, 1 + 0.5j],
        [1 + 2.j, 2 + 1.j],
        [3 + 4.j, 4 + 3.j],
        [5 + 6.j, 6 + 5.j],
        ])

    k = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [0,1,0],
        ]) # N x 3

    epsilon = np.array([
        [[0,1,0] ,[0,0,1]],
        [[1,0,0] ,[0,0,1]],
        [[1,0,0] ,[0,1,0]],
        [[1,0,0] ,[0,0,1]],
        ])
    
    if single:
        idx = np.random.randint(low = 0, high = len(C), size = 3)

        return C[idx[0],:].reshape(1,-1), \
            k[idx[1]].reshape(1,-1), epsilon[idx[2]]

    return C, k, epsilon

x = np.array([0.1,0.2,0.3]) # 3
C,k,epsilon = A_test_case(True, 2022)

A = vector_potential(C=C, k=k, epsilon=epsilon)
print(A(x))

kx = k @ x
kx = np.tile(kx.reshape(-1,1),(1,2))

"""
Ckx = 2 * (np.real(C) * np.cos(kx) - np.imag(C) * np.sin(kx))
"""
Ckx = C * np.exp(kx * np.array(1.j))
Ckx += - np.conjugate(C) * np.exp(kx * np.array(-1.j))
Ckx *= np.array(1.j)

Ckx = Ckx.reshape(*epsilon.shape[:-1], -1)
Ckx = epsilon * Ckx 

Ckx = np.sum(Ckx, axis = 1)
print(Ckx)

dAdr = []
for i in range(len(Ckx)):
    dAdr.append(np.outer(k[i,:], Ckx[i,:] ))
dAdr = np.array(dAdr)

print(dAdr)


