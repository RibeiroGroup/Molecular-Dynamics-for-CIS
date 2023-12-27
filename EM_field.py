import numpy as np
import matplotlib.pyplot as plt

from utils import outer_along_0axis

"""
! TODO: add update C function to vector_potential
"""

def polarization_vectors(k):
    pass

class MultiModeField:
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
        assert k.shape[-1] == 3
        self.n_mode = 1 if len(k.shape) == 1 else k.shape[0]
        self.k = k.reshape(1,3) if self.n_mode == 1 else k # n_modes x 3

        self.update(C)

        self.epsilon = epsilon.reshape(1,2,3) if self.n_mode == 1 \
            else epsilon # n_modes x 2 x 3

        self.V = V

    def update(self, C):
        assert C.shape[-1] == 2
        self.C = C.reshape(1,2) if self.n_mode == 1 else C # n_modes x 2

    def calc_kx(self,x):

        n_points = 1 if len(x.shape) == 1 else x.shape[0]
        # assume shape k = n_mode x 3, shape ra = n_charge x 3
        kx = (self.k @ x.T).reshape(self.n_mode, n_points)
        #print("kx",kx.shape)# shape = n_mode x n_charge

        return n_points,  kx

    def __call__(self,x):
        n_points = 1 if len(x.shape) == 1 else x.shape[0]
        # assume shape k = n_mode x 3, shape ra = n_points x 3

        kx = (self.k @ x.T).reshape(self.n_mode, n_points)
        #print("kx",kx.shape)# shape = n_mode x n_points

        kx = np.tile(kx[:,np.newaxis,:],(1,2,1)) #using np.tile to repeat along one axis
        #print("kx 2",kx.shape) #shape = n_mode x 2 x n_points

        # C shape is only n_mode x 2
        C_ = np.tile(self.C[:,:,np.newaxis], (1,1,n_points))
        #print("C_",C_.shape) # shape = n_mode x 2 x n_points

        ckx = C_ * np.exp(1j * kx) + np.conjugate(C_) * np.exp(-1j * kx)
        #print("ckx",ckx.shape) # shape = n_mode x 2 x n_points

        epsilon_ = np.tile(self.epsilon[:,:,np.newaxis,:], (1,1,n_points,1))
        #print("epsilon_",epsilon_.shape) #shape = n_mode x 2 x n_points x 3

        ckx_ = np.tile(ckx[:,:,:,np.newaxis], (1,1,1,3))
        #print("ckx_", ckx_.shape) #shape = n_mode x 2 x n_points x 3

        Ax = np.sum(epsilon_ * ckx_, axis = 1) # shape = n_mode x n_points x 3
        Ax = np.sum(Ax, axis = 0) 
        #print(Ax.shape) # shape = n_points x 3

        return Ax / self.V**(-1.5)

    def curl(self,x):
        raise Exception("To be implemented")

    def partial_partial_t(self, x, C_dot):
        """
        Partial differentiate of field A w.r.t. to time t
        """
        assert C_dot.shape == self.C.shape
        kx, n_points = self.calc_kx(x) 

        kx = np.tile(kx[:,np.newaxis,:],(1,2,1)) #using np.tile to repeat along one axis
        #print("kx 2",kx.shape) #shape = n_mode x 2 x n_charge

        # C shape is only n_mode x 2
        C_ = np.tile(C_dot[:,:,np.newaxis], (1,1,n_points))
        #print("C_",C_.shape) # shape = n_mode x 2 x n_charge

        ckx = C_ * np.exp(1j * kx) + np.conjugate(C_) * np.exp(-1j * kx)
        #print("ckx",ckx.shape) # shape = n_mode x 2 x n_charge

        epsilon_ = np.tile(self.epsilon[:,:,np.newaxis,:], (1,1,n_points,1))
        #print("epsilon_",epsilon_.shape) #shape = n_mode x 2 x n_charge x 3

        ckx_ = np.tile(ckx[:,:,:,np.newaxis], (1,1,1,3))
        #print("ckx_", ckx_.shape) #shape = n_mode x 2 x n_charge x 3

        result = np.sum(epsilon_ * ckx_, axis = 1) # shape = n_mode x n_charge x 3
        result = np.sum(result, axis = 0) 
        #print(Ax.shape) # shape = n_charge x 3

        return result

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

        n_points, kx = self.calc_kx(x)

        kx = np.tile(kx[:,np.newaxis,:],(1,2,1)) #using np.tile to repeat along one axis

        C_ = np.tile(self.C[:,:,np.newaxis], (1,1,n_points))

        ckx = 1j * C_ * np.exp(1j * kx) - 1j * np.conjugate(C_) * np.exp(-1j * kx)

        epsilon_ = np.tile(self.epsilon[:,:,np.newaxis,:], (1,1,n_points,1))

        ckx_ = np.tile(ckx[:,:,:,np.newaxis], (1,1,1,3))

        Ax = np.sum(epsilon_ * ckx_, axis = 1) # shape = n_mode x n_charge x 3
        outer = np.einsum("ijk,il->ijkl",Ax,self.k)# shape = n_mode x n_charge x 3 x 3

        outer = np.sum(outer,axis = 0) #shape = n_charge x 3 x 3

        return outer

    def transverse_project(self, vector):
        try:
            n_mode = vector.shape[0]
            n_points = vector.shape[1]

            assert n_mode == self.n_mode
            
            projection_mat = self.projection_mat[:,np.newaxis,:,:]
            projection_mat = np.tile(projection_mat, (1,n_points,1,1))

            result = np.einsum("ijkl,ijl->ijk",projection_mat,vector)

            return result

        except AttributeError:
            projection_mat = []        

            #can be optimized with outer_along_0axis
            for k in self.k:
                k_norm = k @ k
                projection_mat.append(np.eye(3) -  np.outer(k,k)/k_norm)

            self.projection_mat = np.array(projection_mat)

            return self.transverse_project(vector)

    def get_jk(self.ra, va, qa):
        k_vec = self.k
        n_mode = self..n_mode
        n_charge = 1 if len(ra.shape) == 1 else ra.shape[0]

        ra = ra.reshape(n_charge, 3)
        va = va.reshape(n_charge, 3)
        qa = qa.reshape(n_charge, 1)

        # compute J_k for each qa and sum over 
        kx = (k_vec @ ra.T).reshape(n_mode, n_charge)
        exp_ikx = np.tile(np.exp(-1j * kx)[:,:,np.newaxis],(1,1,3))
        #print("exp_ikx",exp_ikx.shape) #shape: n_mode x n_charge x 3

        #va should have shape: n_charge x 3
        va_ = np.tile(va[np.newaxis,:,:],(n_mode,1,1))
        #print("va_",va_.shape) #shape: n_mode x n_charge x 3

        qa_ = np.tile(
            qa[np.newaxis,:,:],
            (n_mode, 1,3)
            )
        #print("qa_",qa_.shape)

        jk = (1/(2*np.pi**1.5)) * exp_ikx * va_ * qa_

        return jk
