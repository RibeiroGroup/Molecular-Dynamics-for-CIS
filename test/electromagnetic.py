import numpy as np
import numpy.linalg as la

from utils import PBC_wrapping, orthogonalize
import constants

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

        k_vec = self.k_vector[:,0,:]

        self.k_val = np.sqrt(np.einsum("kj,kj->k",k_vec,k_vec))
        self.omega = self.k_val * constants.c

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

    def gradient(self, R):
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]

        gradA_list = []

        for i, r in enumerate(R):
            gradA = np.zeros((3,3)) + 1j * np.zeros((3,3))
            for j, k in enumerate(k_vec):
                f = 1j * np.exp(1j * k @ r)
                fs = -1j * np.exp(-1j * k @ r)

                A = np.zeros(3) + 1j * np.zeros(3)
                A += pol_vec[j,0,:] * (self.C[j,0] * f + np.conjugate(self.C[j,0]) * fs)
                A += pol_vec[j,1,:] * (self.C[j,1] * f + np.conjugate(self.C[j,1]) * fs)

                gradA += np.outer(A, k)
                # |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
                # |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|
                # |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|

            gradA_list.append(np.array(gradA))

        gradA = np.array(gradA_list)
        return gradA

    def dot_C(self, r, r_dot, gradD):
        all_k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]

        C_dot = []

        for j, k_vec in enumerate(all_k_vec):

            jk = 0

            for i, ri in enumerate(r):
                for l,rl in enumerate(r):
                    jk += np.exp(-1j * k_vec @ ri) * gradD[i,l].T @ r_dot[i]

                    if np.sum(gradD[i,l]) > 1:
                        print(i,l)
                        print(gradD[i,l])
            print(jk)
            #jk = (np.eye(3) - np.outer(k_vec, k_vec) / (self.k_val[j]**2)) @ jk

            proj_jk_transv = np.array([
                jk @ e for e in pol_vec[j]
                ])

            C_dot.append( -1j * self.omega[j] * self.C[j] + \
                (2 * np.pi * 1j / self.k_val[j]) * proj_jk_transv)

        C_dot = np.array(C_dot)
        return C_dot

    def transv_force(self,r, v, dipole_func):
        """
        args:
        + r
        + v	
        + dipole_func (class ExplicitTestDipoleFunction)
		"""
        all_k_vec = self.k_vector[:,0,:]
        epsilon = self.k_vector[:,1:,:]

        C = self.C

        ma_list = []
        force1_list = []
        force2_list = []
        force3_list = []

        gradD = []
        gradD_ = dipole_func.gradient(r,change_of_basis=None,return_all=True)

        for l, k_vec in enumerate(all_k_vec):
            epsilon_k = k_vec / self.k_val[l]
            change_of_basis_mat = np.vstack([epsilon_k, epsilon[l]])
            gD = dipole_func.gradient(r,change_of_basis_mat)
            gradD.append(gD)

        for j, rj in enumerate(r):
            _ma_ = np.array([0+0j,0+0j,0+0j])
            # sum over all wavevector k
            force1 = 0
            force2 = 0
            force3 = 0
            for l, k_vec in enumerate(all_k_vec):
                epsilon_k = k_vec / self.k_val[l]

                # k part
                vk  = epsilon_k     @ v[j].T # projection of v on k_vec
                vk1 = epsilon[l][0] @ v[j].T # projection of v on epsilon_k1
                vk2 = epsilon[l][1] @ v[j].T # projection of v on epsilon_k2
                vkj = [vk1, vk2]

                # C[0] = 0;  C[1] = C_{k1}; C[2] = C_{k2}
                k = self.k_val[l]

                mu_grad = gradD[l][j]
                C_dot = self.dot_C(r, v, gradD_)

                for m in [1,2]:
                    for n in [1,2]:

                        force1 +=  vkj[n-1] * (1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj) \
                                + np.conjugate(1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj)) ) \
                                * mu_grad[n][m] * epsilon_k

                    force2 +=       (C_dot[l][m-1] * np.exp(1j * k_vec @ rj) +  \
                        np.conjugate(C_dot[l][m-1] * np.exp(1j * k_vec @ rj)) ) \
                        * mu_grad[0][m] * epsilon_k

                # epsilon part
                for i in [1,2]:
                    for m in [1,2]:

                        force2 +=       (C_dot[l][m-1] * np.exp(1j * k_vec @ rj) + \
                            np.conjugate(C_dot[l][m-1] * np.exp(1j * k_vec @ rj)) )\
                            * mu_grad[i][m] * epsilon[l][i-1]

                        force3 +=    vk * (1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj) \
                            + np.conjugate(1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj)) )\
                            * mu_grad[i][m] * epsilon[l][i-1]

            force = force1 - force2 - force3
            force /= constants.c

            ma_list.append(np.real(force))
            force1_list.append(force1/constants.c)
            force2_list.append(force2/constants.c)
            force3_list.append(force3/constants.c)

        return np.array(ma_list)
        #return np.array(force1_list), np.array(force2_list), np.array(force3_list)
