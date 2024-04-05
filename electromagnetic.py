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

        if k_vector.shape == (self.N_modes, 3, 3):
            self.k_vector = k_vector
        else:
            assert k_vector.shape == (self.N_modes, 3)
            self.k_vector = np.array([
                orthogonalize(kvec) for kvec in k_vector
                ]) # each k_vector should have shape N_mode x 3 x 3

        assert amplitudes.shape == (self.N_modes, 2)

        self.C = amplitudes

    def __call__(self, R):
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]
        
        # free field mode function and c.c., a.k.a. exp(ikr)
        f_R = np.exp(
            1j * np.einsum("kj,nj->kn",k_vec,R))

        fs_R = np.exp(
            -1j * np.einsum("kj,nj->kn",k_vec,R))

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki",self.C,pol_vec)

        #
        A_R = np.einsum("ki,kn->ni",C_epsilon_k, f_R) \
            + np.einsum("ki,kn->ni",np.conjugate(C_epsilon_k), fs_R)

        return A_R

    def gradient(self, R):
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]
        
        # free field mode function and c.c., a.k.a. exp(ikr)
        f_R = np.exp(
            1j * np.einsum("kj,nj->kn",k_vec,R))

        fs_R = np.exp(
            -1j * np.einsum("kj,nj->kn",k_vec,R))

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki",self.C,pol_vec)

        k_o_C_epsilon_k = np.einsum("kj,ki->kij",k_vec,C_epsilon_k)

        #
        gradA_R = np.einsum("kij,kn->nij",k_o_C_epsilon_k, f_R) \
            + np.einsum("kij,kn->nij",np.conjugate(k_o_C_epsilon_k), fs_R)

        return gradA_R

    def dot(self, R):
        pass

class ExplicitTestVectorPotential:
    def __init__(self, k_vector, amplitudes):
        """
        Args:
        + k_vector (np.array): should have shape N_modes x 3 x 3
            where the 1st dim -> mode, 2nd dim -> k vector, 1st & 2nd 
            polarization vector and the last dim is vector length (coordinate)
        """

        self.N_modes = len(k_vector)

        if k_vector.shape == (self.N_modes, 3, 3):
            self.k_vector = k_vector
        else:
            assert k_vector.shape == (N_modes, 3)
            self.k_vector = np.array([
                orthogonalize(kvec) for kvec in k_vector
                ]) # each k_vector should have shape N_mode x 3 x 3

        assert amplitudes.shape == (self.N_modes, 2)

        self.C = amplitudes

    def __call__(self,R):
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]

        A_list = []

        for i, r in enumerate(R):
            A= np.zeros(3) + 1j * np.zeros(3)
            for j, k in enumerate(k_vec):
                f = np.exp(1j * k @ r)
                fs = np.exp(-1j * k @ r)

                A += pol_vec[j,0,:] * (self.C[j,0] * f + np.conjugate(self.C[j,0]) * fs)
                A += pol_vec[j,1,:] * (self.C[j,1] * f + np.conjugate(self.C[j,1]) * fs)

            A_list.append(np.array(A))

        A = np.array(A_list)
        return A

if run_test == True:
    np.random.seed(100)

    k_vector = np.array([
            [1,0,0],
            [1,1,0],
            [1,1,1],
            [0,1,1],
            ], dtype= np.float64)

    k_vector = np.array([
        orthogonalize(kvec) for kvec in k_vector
        ]) 

    amplitudes = np.random.rand(len(k_vector),2) + np.random.rand(len(k_vector),2) * 1j

    N_atoms = 10
    R = np.random.rand(N_atoms, 3)

    AField = VectorPotential(k_vector, amplitudes)
    print(AField.gradient(R).shape)

    #AFieldTest = ExplicitTestVectorPotential(k_vector, amplitudes)
    #print(AField(R) - AFieldTest(R))






