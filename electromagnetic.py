import numpy as np
import numpy.linalg as la

from utils import PBC_wrapping, orthogonalize

run_test = True

class VectorPotential:
    def __init__(self, k_vector, amplitudes):
        """
        Args:
        + k_vector (np.array): should have shape N_modes x 3 x 3
            where the 1st dim -> mode, 2nd dim -> k vector, 1st & 2nd 
            polarization vector and the last dim is vector length (coordinate)
        """

        self.N_modes = len(k_vector)

        if kvec.shape == (self.N_modes, 3, 3):
            self.k_vector = k_vector
        else:
            assert k_vector.shape == (N_modes, 3)
            self.k_vector = np.array([
                orthogonalize(kvec) for kvec in k_vector
                ]) # each k_vector should have shape N_mode x 3 x 3

        assert amplitudes.shape == (N_modes, 2)

        self.C = amplitudes

    def __call__(self, R):
        k_vec = k_vector[:,0,:]
        pol_vec = k_vec[:,1:,:]
        
        f_R = np.einsum("kj,ij->ik",k_vec,R)

    def gradient(self, R):
        pass

    def dot(self, R):
        pass

if run_test == True:
    k_vector = np.array([
            [1,0,0],
            [1,1,0]
            ], dtype= np.float64)

    k_vector = np.array([
        orthogonalize(kvec) for kvec in k_vector
        ]) 

    print(k_vector[:,0,:].shape)
