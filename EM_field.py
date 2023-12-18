import numpy as np
import matplotlib.pyplot as plt

from utils import outer_along_0axis

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

        result = np.sum(Ckx, axis = 1) # sum C_l [e^{ikx} e_l + c.c.] for each mode
        result = np.sum(result, axis = 0) # sum over all mode

        return result / self.V**(0.5)

    def curl(self,x):
        raise Exception("To be implemented")

    def partial_partial_t(self,x, j_ktrans):
        """
        Differentiate w.r.t. to time t
        """
        pass

    def partial_partial_x(self,x):
        """
        Derivative with respect to position
        The Jacobian matrix will have the form:

        dAx/drx & dAx/dry & dAx/drz
        dAy/drx & dAy/dry & dAy/drz  
        dAz/drx & dAz/dry & dAz/drz

        other formula
        dA/dx = sum(i C_l e^(ikx) e_l + c.c.) . k^T 
        """

        kx = self.k @ x # matmul
        #copy along one axis so that kx shape match C's
        kx = np.tile(kx.reshape(-1,1),(1,2))

        #following the formula
        Ckx = 1j * self.C * np.exp(kx * np.array(1.j))
        Ckx -= 1j * np.conjugate(self.C) * np.exp(kx * np.array(-1.j))

        #reshape to match with epsilon shape => element-wise multiplication
        Ckx = Ckx.reshape(*self.epsilon.shape[:-1], -1)
        Ckx = self.epsilon * Ckx 

        Ckx = np.sum(Ckx, axis = 1) # sum C_l [e^{ikx} e_l + c.c.] for each mode
        print(Ckx.shape)

        return outer_along_0axis(Ckx, self.k)

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



