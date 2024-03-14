import pickle
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from EM_field import MultiModeField
from simpleForceField import MorsePotential, compute_Morse_force, compute_Hmorse

import constants

#####################################################
### COMPUTE FORCE ###################################
#####################################################


def compute_long_force(q, r, v):
    ma_list = []
    for j, rj in enumerate(r):
        _ma_ = np.zeros(3) + 1.j * np.zeros(3)
        for l, rl in enumerate(r):
            if l == j: continue
            d = np.sqrt(np.sum( (r[l] - r[j])**2 ))
            _ma_ += 0.5 * q[j] * q[l] * (r[j] - r[l] ) / d**3

        ma_list.append(_ma_)

    return np.array(ma_list)

def compute_oscillator_force(r, k_const):
    ma = []
    for ri in r:
        ma.append( - k_const * ri)
    return np.array(ma)

#####################################################
### COMPUTE HAMILTONIAN #############################
#####################################################

def compute_Hmat_transv(q,r,v):
    K = 0
    for i, vi in enumerate(v):
        K += 0.5 * vi @ vi.T
    return K

def compute_Hmat_long(q,r,v):
    K = 0
    for i, vi in enumerate(v):
        if i == len(v) - 1: continue
        for j in range(len(v)):
            if i == j: continue
            d = np.sqrt( np.sum( (r[i] - r[j])**2 ) )
            K += 0.5 * q[i] * q[j] / d
    return K

def compute_Hem(k_vec,C):
    H_em = []

    for i, Ci in enumerate(C):

        H_em.append((2 * np.pi)**-1 * (k_vec[i] @ k_vec[i].T) \
            * (Ci @ np.conjugate(Ci).T) )

    return np.array(H_em)

def compute_H_oscillator(r,k_const):
    K = 0
    for ri in r:
        K += 0.5 * k_const * (ri @ ri.T).item()
    return K

#####################################################
### WRAPPER #########################################
#####################################################

class SimpleDynamicModel:
    def __init__(
            self, q, k_vec, epsilon, k_const=None, 
            potential=None, coulomb_interaction = False, exclude_EM = False):

        """
        Class of simple model for molecular dyanamic simulation
        for charged particle in EM field
        Args: 
        + q (list/ np.array of int): charge value of particles
        + k_vec (np.array): list of array of k vectors for the EM field
            have shape of n_mode x 3 with n_mode is int >= 1
        + epsilon (np.array): list or array of pairs of polarization vector modes
            for the EM field. Have the shape of n_mode x 2 x 3
        + k_const (float): constant for the Harmonic Oscillator
        + potential (None, MorsePotential, will add more class later):
            'engine' for potential or force computing
        """

        #assert len(q) == len(v) and len(v) == len(r)
        #assert C.shape[0] == k_vec.shape[0]

        self.q = q

        if len(k_vec.shape) < 2:
            k_vec = k_vec.reshape(1,3)
        else: 
            assert k_vec.shape[1] == 3
        self.k_vec = k_vec

        if len(epsilon.shape) < 3:
            epsilon = epsilon.reshape(1,2,3)
        else:
            assert epsilon.shape[1] == 2 and epsilon.shape[2] == 3
        self.epsilon = epsilon

        self.k_const = k_const
        self.potential = potential
        self.coulomb_interaction = coulomb_interaction
        self.exclude_EM = exclude_EM

        self.k = []; self.omega = []
        for vec in k_vec:
            k = np.sqrt(vec @ vec.T)
            self.k.append(k)
            self.omega.append( constants.c * k)

    def dot_C(self, r, v, C):
        """
        Computing partial derivative of C w.r.t. time, a.k.a. C_dot
        Args:
        + r (np.array): list/array of postition of charged particles. Shape: n_particles x 3
        + v (np.array): list/array of velocities of charged particles. Shape: n_particles x 3
        + C (np.array): list/array of pair of modes
        """

        if self.exclude_EM:
            return -1j * self.omega * C

        C_dot = []

        for j, k_vec in enumerate(self.k_vec):

            jk = 0
            for i,qi in enumerate(self.q):
                jk += np.exp(-1j * k_vec @ r[i]) * qi * v[i] # * (2 * np.pi)**(-1.5)

            jk_transv = (np.eye(3) - np.outer(k_vec, k_vec) / (self.k[j]**2)) @ jk

            proj_jk_transv = np.array([
                jk_transv @ e for e in self.epsilon[j] 
                ])

            C_dot.append( -1j * self.omega[j] * C[j] + \
                (2 * np.pi * 1j / self.k[j]) * proj_jk_transv)

        C_dot = np.array(C_dot)
        return C_dot

    def compute_force(self, r, v, C):
        if self.exclude_EM == False:
            F = self.compute_transv_force(r=r, v=v, C=C)
        else: F = 0

        if isinstance(self.potential, MorsePotential):
            F += compute_Morse_force(r,self.potential)
        elif self.potential == "coulomb":
            F += compute_long_force(q, r, v)

        if self.k_const:
            F += compute_oscillator_force(r,self.k_const)

        return F

    def compute_H(self, r, v, C):
        H_mat = compute_Hmat_transv(q=self.q, r=r, v=v)
        H_em = compute_Hem(self.k_vec, C)

        if isinstance(self.potential, MorsePotential):
            H_mat += compute_Hmorse(r,self.potential)
        elif self.potential == "coulomb":
            H_mat += compute_Hmat_long(q=self.q, r=r, v=v)

        if self.k_const != None:
            H_osci = compute_H_oscillator(r,self.k_const)
        else: H_osci = 0

        return H_em, H_mat, H_osci

    def pde_step(self, r, v, C):
        mv = self.compute_force(r,v,C)
        C = self.dot_C(r,v,C)
        r = v
        return r, mv, C

    def rk4_step(self, r, v, C, h):
        k1r, k1v, k1c = self.pde_step(r=r ,v=v, C=C)
        k2r, k2v, k2c = self.pde_step(r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2)
        k3r, k3v, k3c = self.pde_step(r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2)
        k4r, k4v, k4c = self.pde_step(r=r+k3r*h, v=v+k3v*h, C=C+k3c*h)

        r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
        v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
        C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)

        return r, v, C

    def compute_transv_force(self, r, v, C):
        C_dot = self.dot_C(r,v,C)

        # list of unit vector along the wavevector k, hence epsilon_k
        epsilon_k = [self.k_vec[i] / self.k[i] for i in range(len(self.k))]

        ma_list = []

        for j, rj in enumerate(r):
            _ma_ = np.array([0+0j,0+0j,0+0j])
            # sum over all wavevector k
            for l, k_vec in enumerate(self.k_vec):
                # k part
                vk  =       epsilon_k[l] @ v[j].T # projection of v on k_vec
                vk1 = self.epsilon[l][0] @ v[j].T # projection of v on epsilon_k1
                vk2 = self.epsilon[l][1] @ v[j].T # projection of v on epsilon_k2

                # C[0] = 0;  C[1] = C_{k1}; C[2] = C_{k2}
                k = self.k[l]

                _ma_k =     vk1 * (1j * k * C[l][0] * np.exp(1j * k_vec @ rj) \
                    + np.conjugate(1j * k * C[l][0] * np.exp(1j * k_vec @ rj)) )

                _ma_k +=   vk2 * (1j * k * C[l][1] * np.exp(1j * k_vec @ rj) \
                   + np.conjugate(1j * k * C[l][1] * np.exp(1j * k_vec @ rj)) )

                _ma_ += _ma_k * epsilon_k[l]

                # epsilon part
                for i in [1,2]:
                    _ma_ki =         -C_dot[l][i-1] * np.exp(1j * k_vec @ rj) + \
                        np.conjugate(-C_dot[l][i-1] * np.exp(1j * k_vec @ rj))

                    _ma_ki += -vk * (1j * k * C[l][i-1] * np.exp(1j * k_vec @ rj) \
                      + np.conjugate(1j * k * C[l][i-1] * np.exp(1j * k_vec @ rj)) )

                    _ma_ += _ma_ki * self.epsilon[l][i-1]

            _ma_ *= self.q[j] / constants.c

            ma_list.append(np.real(_ma_))

        return np.array(ma_list)


#################################################################################################################################################
#################################################################################################################################################
######## SIMPLE DYNAMIC MODEL FOR A PAIR OF ARGON AND XENON ATOMS IN EM FIELD ###################################################################
#################################################################################################################################################
#################################################################################################################################################


class SimpleDynamicModelEM:
    def __init__(
            self, k_vec, epsilon, M, dipole_function, 
            potential=None, L=None):

        """
        Class of simple model for molecular dyanamic simulation
        for charged particle in EM field
        Args: 
        + q (list/ np.array of int): charge value of particles
        + k_vec (np.array): list of array of k vectors for the EM field
            have shape of n_mode x 3 with n_mode is int >= 1
        + epsilon (np.array): list or array of pairs of polarization vector modes
            for the EM field. Have the shape of n_mode x 2 x 3
        + k_const (float): constant for the Harmonic Oscillator
        + potential (None, MorsePotential, will add more class later):
            'engine' for potential or force computing
        """

        #assert len(q) == len(v) and len(v) == len(r)
        #assert C.shape[0] == k_vec.shape[0]

        if len(k_vec.shape) < 2:
            k_vec = k_vec.reshape(1,3)
        else: 
            assert k_vec.shape[1] == 3
        self.k_vec = k_vec

        if len(epsilon.shape) < 3:
            epsilon = epsilon.reshape(1,2,3)
        else:
            assert epsilon.shape[1] == 2 and epsilon.shape[2] == 3
        self.epsilon = epsilon

        self.potential = potential
      
        self.k = []; self.omega = []
        for vec in k_vec:
            k = np.sqrt(vec @ vec.T)
            self.k.append(k)
            self.omega.append( constants.c * k)
            
        self.M = np.tile(M[:,np.newaxis], (1,3) )
        self.dipole_func = dipole_function

        # list of unit vector along the wavevector k, hence epsilon_k
        self.epsilon_k = [self.k_vec[i] / self.k[i] for i in range(len(self.k))]

        self.change_of_basis_mat = [
            np.vstack([self.epsilon_k[i], self.epsilon[i]]) for i in range(len(self.k_vec))
        ]

        self.L = L
        self.potential.L = L
        self.dipole_func.L = L

    def dot_C(self, r, v, C):
        """
        Computing partial derivative of C w.r.t. time, a.k.a. C_dot
        Args:
        + r (np.array): list/array of postition of charged particles. Shape: n_particles x 3
        + v (np.array): list/array of velocities of charged particles. Shape: n_particles x 3
        + C (np.array): list/array of pair of modes
        """

        C_dot = []

        for j, k_vec in enumerate(self.k_vec):

            rk = r #np.einsum("ji,ki->kj",md.change_of_basis_mat[0], r)
            
            mu_grad = self.dipole_func.gradient(rk[0],rk[1])
            
            jk = 0
            
            for i, mu_grad_i in enumerate(mu_grad):
                jk += np.exp(-1j * k_vec @ r[i]) * mu_grad_i.T @ v[i] # * (2 * np.pi)**(-1.5)

            jk_transv = (np.eye(3) - np.outer(k_vec, k_vec) / (self.k[j]**2)) @ jk

            proj_jk_transv = np.array([
                jk_transv @ e for e in self.epsilon[j] 
                ])
            
            C_dot.append( -1j * self.omega[j] * C[j] + \
                (2 * np.pi * 1j / self.k[j]) * proj_jk_transv)

        C_dot = np.array(C_dot)
        return C_dot

    def compute_force(self, r, v, C):
        
        F = self.compute_transv_force(r=r, v=v, C=C)

        if self.potential is not None:
            F += self.potential.get_force(r)

        return F
    
    def compute_Hmat_transv(self,r,v):
        K = 0
        for i, vi in enumerate(v):
            K += 0.5 * (self.M[i] * vi) @ vi.T
        return K

    def compute_H(self, r, v, C):
        H_mat = self.compute_Hmat_transv(r=r, v=v)
        H_em = compute_Hem(self.k_vec, C)

        H_mat += self.potential.get_potential(r)

        return H_em, H_mat

    def pde_step(self, r, v, C):
        a = self.compute_force(r,v,C) / self.M
        C = self.dot_C(r,v,C)
        r = v
        return r, a, C

    def rk4_step(self, r, v, C, h):
        k1r, k1v, k1c = self.pde_step(
            r=r ,v=v, C=C)
        k2r, k2v, k2c = self.pde_step(
            r=r+k1r*h/2, v=v+k1v*h/2, C=C+k1c*h/2)
        k3r, k3v, k3c = self.pde_step(
            r=r+k2r*h/2, v=v+k2v*h/2, C=C+k2c*h/2)
        k4r, k4v, k4c = self.pde_step(
            r=r+k3r*h, v=v+k3v*h, C=C+k3c*h)

        r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
        r = PBC_wrapping(r, self.L)
        v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
        C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)

        return r, v, C

    def compute_transv_force(self, r, v, C):
        C_dot = self.dot_C(r,v,C)
        
        ma_list = []

        for j, rj in enumerate(r):
            _ma_ = np.array([0+0j,0+0j,0+0j])
            # sum over all wavevector k
            for l, k_vec in enumerate(self.k_vec):

                mu_grad = self.dipole_func.gradient(r[0],r[1],self.change_of_basis_mat[l])
                
                # k part
                vk  =  self.epsilon_k[l] @ v[j].T # projection of v on k_vec
                vk1 = self.epsilon[l][0] @ v[j].T # projection of v on epsilon_k1
                vk2 = self.epsilon[l][1] @ v[j].T # projection of v on epsilon_k2
                vkj = [vk1, vk2]

                # C[0] = 0;  C[1] = C_{k1}; C[2] = C_{k2}
                k = self.k[l]
                
                _ma_k = 0
                for m in [1,2]:
                    for n in [1,2]:
                    
                        _ma_k +=   vkj[n-1] * (1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj) \
                                + np.conjugate(1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj)) ) \
                                * mu_grad[j][n][m]
                        
                    foo =           (-C_dot[l][m-1] * np.exp(1j * k_vec @ rj) +  \
                        np.conjugate(-C_dot[l][m-1] * np.exp(1j * k_vec @ rj)) ) \
                        * mu_grad[j][0][m]
                    
                    _ma_k += foo
                    
                _ma_ += _ma_k * self.epsilon_k[l]

                # epsilon part
                for i in [1,2]:
                    _ma_ki = 0
                    for m in [1,2]:
                        _ma_ki +=       (-C_dot[l][m-1] * np.exp(1j * k_vec @ rj) + \
                            np.conjugate(-C_dot[l][m-1] * np.exp(1j * k_vec @ rj)) )\
                            * mu_grad[j][i][m]

                        _ma_ki +=   -vk * (1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj) \
                            + np.conjugate(1j * k * C[l][m-1] * np.exp(1j * k_vec @ rj)) )\
                            * mu_grad[j][i][m]
                       
                    _ma_ += _ma_ki * self.epsilon[l][i-1]

            _ma_ /= constants.c

            ma_list.append(np.real(_ma_))

        return np.array(ma_list)

