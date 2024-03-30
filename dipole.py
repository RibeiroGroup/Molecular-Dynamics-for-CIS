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
        r_pn = np.array(np.outer(positive_atom_idx, negative_atom_idx), dtype = bool)

        self.r_pn_mask = r_pn[~distance_calc.identity_mat].reshape(distance_calc.n_points,-1)

        self.r_pn_mask = np.array(
                np.tile(self.r_pn_mask[:,:,np.newaxis], (1,1,3)),
                dtype = bool)

        self.r_pn_mask_x3 = np.array(
                np.tile(self.r_pn_mask[:,:,:,np.newaxis], (1,1,1,3)),
                dtype = bool)

        #boolean mask for the distance matrix to extract distance from negative to positive atoms
        r_np = np.outer(negative_atom_idx,positive_atom_idx)
        r_np = np.array(np.outer(negative_atom_idx, positive_atom_idx), dtype = bool)

        self.dipole_mask = self.distance_calc.generate_custom_mask(r_pn + r_np)

    def __call__(self, R_all, return_tensor = False):
        """
        Return dipole vectors in two ways (1) array of all unique dipole vector from positive atom
        (r_+) to negative atom (r_-) (2) N x N matrix where i,j element is dipole vector between 
        i and j if exist dipole vector between them or zero (note that the actual return matrix is 
        of size N x (N-1) since the diagonal is removed; and the vector still point from r_+ to 
        r_-)
        """
        
        dipole_vec_tensor = self.distance_calc.apply_function(
                R_all, self.dipole_func, output_shape = 3,
                custom_mask = self.dipole_mask
                )

        if return_tensor:
            dipole_vec_tensor *= -1
            dipole_vec_tensor[self.r_pn_mask] *= -1

            return dipole_vec_tensor

        else:
            dipole_vec = dipole_vec_tensor[self.r_pn_mask].reshape(-1,3)

            return dipole_vec

    def gradient(self, R_all):
        """
        Return gradient of the dipole function w.r.t. position r = r_+.
        The matrix will have the shape N x (N - 1) x 3 x 3 where the 
        3 x 3 tensor at i, j -th element will be
            |~ dmu_x/dr_x dmu_y/dr_x dmu_z/dr_x ~|
            |  dmu_x/dr_y dmu_y/dr_y dmu_z/dr_y  |
            |_ dmu_x/dr_z dmu_y/dr_z dmu_z/dr_z _|
        with mu = mu(r_i, r_j), e.g. dipole vector between i-th and j-th atoms
        """
        gradient = self.distance_calc.apply_function(
                R_all, self.gradient_func, output_shape = (3,3),
                custom_mask = self.dipole_mask)

        gradient *= -1
        gradient[self.r_pn_mask_x3] *= -1

        return gradient

    def update_distance_calc(self,distance_calc):

        self.distance_calc = distance_calc

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

    def gradient_func(self,distance, distance_vec):
        distance = np.tile(distance[:,np.newaxis,np.newaxis], (1,3,3))

        exp_ad = np.exp(-self.a * (distance - self.d0))

        distance_outer = np.einsum("ij,ik->ijk", distance_vec, distance_vec)

        gradient = - self.a * self.mu0 * distance_outer * exp_ad / distance**2
        gradient -= self.mu0 * distance_outer * exp_ad / distance ** 3
        gradient += (self.mu0 * exp_ad / distance) * np.eye(3)

        return gradient

class DipoleFunctionExplicitTest:
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

    def __call__(self,R):
        dipole_vec = np.zeros((len(R),len(R)-1,3))
        for i, ri in enumerate(R):
            for j, rj in enumerate(R):
                if i == j: continue
                if self.positive_atom_idx[i] and self.negative_atom_idx[j]:
                    sign = 1
                elif self.positive_atom_idx[j] and self.negative_atom_idx[i]:
                    sign = -1
                    #continue
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

    def gradient(self, R):
        H = np.zeros((len(R), len(R)-1 , 3 , 3) )

        for i, ri in enumerate(R):
            for j, rj in enumerate(R):
                if i == j: continue
                if self.positive_atom_idx[i] and self.negative_atom_idx[j]:
                    sign = +1
                elif self.positive_atom_idx[j] and self.negative_atom_idx[i]:
                    sign = -1
                else:
                    continue
                if j > i: j -= 1

                rij = PBC_wrapping(ri - rj, self.L)
                print(rij)
                d = np.sqrt(rij @ rij)

                exp_ad = np.exp(-self.a * (d - self.d0)) 

                for k in range(3):
                    for l in range(3):
                        H[i,j,k,l] = - self.a * self.mu0 * rij[k] * rij[l] * exp_ad / d**2
                        H[i,j,k,l] -= self.mu0 * rij[k] * rij[l] * exp_ad / d**3

                        if k == l:
                            H[i,j,k,l] += self.mu0 * exp_ad / d

                        H[i,j,k,l] *= sign
        return H

########################
######### TEST #########
########################
###### BOX LENGTH ######
########################

L = 8

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = 1# int(L/4)
N_Xe = 1# int(L/4)
N = N_Ar + N_Xe

# randomized initial coordinates

#R_all = np.random.uniform(-L/2, L/2, (N, 3))
r_xe = np.array([0,0,0])
r_ar = np.array([2,1,3])
R_all = np.vstack([r_ar, r_xe])

# indices of atoms in the R_all and V_all
idxAr = np.hstack(
    [np.ones(N_Ar), np.zeros(N_Xe)]
)

idxXe = np.hstack(
    [np.zeros(N_Ar), np.ones(N_Xe)]
)

##########################################################################
###### INITIATE UTILITY CLASSES (PLEASE UPDATE THEM DURING LOOPING) ######  
##########################################################################

distance_calc = DistanceCalculator(
        n_points = len(R_all), 
        neighbor_mask = None,
        box_length = L
        )

dipole_function = SimpleDipoleFunction(
        distance_calc, 
        mu0=0.0284 , a=1.22522, d0=7.10,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr
        )

print(dipole_function.r_pn_mask_x3)

dipole_function_ = DipoleFunctionExplicitTest(
        mu0=0.0284 , a=1.22522, d0=7.10,
        positive_atom_idx = idxXe,
        negative_atom_idx = idxAr, L = L
        )

H = dipole_function.gradient(R_all)
H_ = dipole_function_.gradient(R_all)

print(H)
print("#####")
print(H_)
