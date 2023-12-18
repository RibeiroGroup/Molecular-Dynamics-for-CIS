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
        self.n_mode = 1 if len(k.shape) == 1 else len(k)

        self.C = C.reshape(1,2) if self.n_mode == 1 else C # n_modes x 2
        self.k = k.reshape(1,3) if self.n_mode == 1 else k # n_modes x 3
        self.epsilon = epsilon.reshape(1,2,3) if self.n_mode == 1 \
            else epsilon # n_modes x 2 x 3

        self.V = V

       # assert C.shape[0] == k.shape[0]

    def update(self, C):
        self.C = C.reshape(1,2) if self.n_mode == 1 else C # n_modes x 2

    def __call__(self,x):
        kx = self.k @ x # matmul
        #copy along one axis so that kx shape match C's
        kx = np.tile(kx.reshape(-1,1),(1,2))

        #following the formula
        Ckx = self.C * np.exp(kx * np.array(1.j))
        Ckx += np.conjugate(self.C) * np.exp(kx * np.array(-1.j))

        #reshape to match with epsilon shape => element-wise multiplication
        Ckx = Ckx.reshape(*self.epsilon.shape[:-1], -1)
        Ckx = self.epsilon * Ckx 

        result = np.sum(Ckx, axis = 0)
        #if self.n_mode > 1:
        result = np.sum(result, axis = 0)

        return result / self.V**(0.5)

    def curl(self,x):
        raise Exception("To be implemented")

    def div(self,x):
        raise Exception("To be implemented")

    def diff_ra(self,x):
        """
        Derivative with respect to position
        The Jacobian matrix will have the form:

        dAx/drx & dAx/dry & dAx/drz
        dAy/drx & dAy/dry & dAy/drz  
        dAz/drx & dAz/dry & dAz/drz

        Tips: dH/dr = dH/dA . dA/dr
        where dA/dr = k*sth is this matrix
        """

        kx = self.k @ x
        kx = np.tile(kx.reshape(-1,1),(1,2))

        Ckx =  self.C * np.exp(kx * np.array(1.j))
        Ckx -=  np.conjugate(self.C) * np.exp(kx * np.array(-1.j))
        Ckx *= np.array(1.j)

        Ckx = Ckx.reshape(*self.epsilon.shape[:-1], -1)
        Ckx = self.epsilon * Ckx 

        """
        if self.n_mode == 1:
            Ckx = np.sum(Ckx, axis = 0)
            dAdr = np.outer(Ckx, self.k )
        else:
        """
        Ckx = np.sum(Ckx, axis = 1)
        dAdr = []
        for i in range(len(Ckx)):
            dAdr.append(np.outer(Ckx[i,:], self.k[i,:]))
        dAdr = np.array(dAdr)
        dAdr = np.sum(dAdr, axis = 0)

        return dAdr

    def transverse_project(self, vector):
        try:
            result = np.array([mat @ vector for mat in self.projection_mat])
            return result
        except:
            projection_mat = []        

            for k in self.k:
                k_norm = k @ k
                projection_mat.append(np.eye(3) -  np.outer(k,k)/k_norm)

            self.projection_mat = np.array(projection_mat)

            return self.transverse_project(vector)



