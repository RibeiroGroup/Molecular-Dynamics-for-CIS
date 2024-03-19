import numpy as np
import sympy as sm

from distance import DistanceCalculator
from utils import PBC_wrapping

class BaseDipoleFunction:
    def __init__(
            self, distance_calc, 
            positive_atom_idx, negative_atom_idx
            ):

        assert len(positive_atom_idx) == distance_calc.n_points
        assert len(negative_atom_idx) == distance_calc.n_points

        self.distance_calc = distance_calc
        #boolean mask for the distance matrix to extract distance from positive to negative atoms
        r_pn = np.outer(positive_atom_idx, negative_atom_idx)

        #boolean mask for the distance matrix to extract distance from negative to positive atoms
        r_np = np.outer(negative_atom_idx,positive_atom_idx)

        self.dipole_mask = np.array(r_pn + r_np, dtype = bool)

        self.sign_correction = np.tile(
                (r_pn - r_np)[:,:,np.newaxis],(1,1,3))

        self.sign_correction = self.sign_correction[~distance_calc.identity_mat_x3].reshape(
            distance_calc.n_points, distance_calc.n_points-1,3)

    def __call__(self, R_all):
        
        dipole_vec_tensor = self.distance_calc.apply_function(
                R_all, self.dipole_func, output_shape = 3, mask = self.dipole_mask
                )

        dipole_vec_tensor *= self.sign_correction
        return dipole_vec_tensor

    def gradient(self, R_all):

        return self.distance_calc.apply_function(
                R_all, self.gradient_func, output_shape = 9)

class SimpleDipoleFunction(BaseDipoleFunction):
    def __init__(self, distance_calc, positive_atom_idx, negative_atom_idx, mu0, a, d0):

        super().__init__(distance_calc, positive_atom_idx, negative_atom_idx)
        self.mu0 = mu0
        self.a = a
        self.d0 = d0

    def dipole_func(self, distance, distance_vec):

       dipole = self.mu0 * np.exp(-self.a * (distance - self.d0)) 
       dipole = np.tile(
               dipole[:,np.newaxis], (1,3))

       dipole *= distance_vec

       return dipole

class DipoleFunctionExplicitTest:
    def __init__(self, positive_atom_idx, negative_atom_idx, mu0, a, d0, L):
 
        self.mu0 = mu0
        self.a = a
        self.d0 = d0
        
        self.positive_atom_idx = positive_atom_idx
        self.negative_atom_idx = negative_atom_idx

        self.L = L

    def __call__(self,R):
        dipole_vec = np.zeros((len(R),len(R)-1,3))
        for i, ri in enumerate(R):
            for j, rj in enumerate(R):
                if i == j: continue
                if self.positive_atom_idx[i] and self.negative_atom_idx[j]:
                    sign = 1
                elif self.positive_atom_idx[j] and self.negative_atom_idx[i]:
                    sign = -1
                else:
                    continue
                if j > i: j -= 1

                rij = PBC_wrapping(ri - rj, self.L)
                d = np.sqrt(rij @ rij)

                dipole = self.mu0 * np.exp(-self.a * (d - self.d0)) 

                dipole *= rij
                dipole *= sign
                dipole_vec[i,j,:] = dipole

        return dipole_vec




