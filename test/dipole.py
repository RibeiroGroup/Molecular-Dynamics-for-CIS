import argparse
import numpy as np
import sympy as sm

from distance import DistanceCalculator
from utils import PBC_wrapping

class ExplicitTestDipoleFunction:
    """
    Note: This class is for testing and verifying the above function only
    """
    def __init__(self, positive_atom_idx, negative_atom_idx, mu0, a, d0, L):
 
        self.mu0 = mu0
        self.a = a
        self.d0 = d0
        
        self.positive_atom_idx = positive_atom_idx
        self.negative_atom_idx = negative_atom_idx

        self.L = L

    def __call__(self,R, change_of_basis=None):

        mat = change_of_basis if change_of_basis is not None else np.eye(3)

        dipole_vec = np.zeros((len(R),len(R),3))
        for i, ri in enumerate(R):
            for j, rj in enumerate(R):
                if i == j: continue
                if self.positive_atom_idx[i] and self.negative_atom_idx[j]:
                    sign = 1
                elif self.positive_atom_idx[j] and self.negative_atom_idx[i]:
                    sign = -1
                else:
                    continue
                #if j > i: j -= 1
                rij = PBC_wrapping(ri - rj, self.L)

                rij = mat @ rij

                d = np.sqrt(rij @ rij)

                dipole = self.mu0 * np.exp(-self.a * (d - self.d0)) 

                dipole *= rij
                dipole *= sign
                dipole_vec[i,j,:] = dipole

        return dipole_vec

    def gradient(self, R, change_of_basis=None, return_all = False):
        mat = change_of_basis if change_of_basis is not None else np.eye(3)

        H = np.zeros((len(R), len(R) , 3 , 3) )

        for i, ri in enumerate(R):
            for j, rj in enumerate(R):
                if i == j: continue
                if self.positive_atom_idx[i] and self.negative_atom_idx[j]:
                    sign = +1
                elif self.positive_atom_idx[j] and self.negative_atom_idx[i]:
                    sign = -1
                else:
                    continue
                #if j > i: j -= 1
                rij = PBC_wrapping(ri - rj, self.L)

                rij = mat @ rij

                d2 = np.sum(rij**2)
                d = np.sqrt(d2)

                exp_ad = self.mu0 * np.exp(-self.a * (d - self.d0)) 

                for k in range(3):
                    for l in range(3):
                        H[i,j,k,l] = - self.a * rij[k] * rij[l] * exp_ad / d2
                        H[i,j,k,l] -= rij[k] * rij[l] * exp_ad / (d2 * d)

                        if k == l:
                            H[i,j,k,l] += exp_ad / d

                        H[i,j,k,l] *= sign
        if return_all:
            return H

        H = np.sum(H,axis=1)
        return H
