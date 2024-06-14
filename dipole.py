import argparse
import numpy as np
import sympy as sm

from distance import DistanceCalculator
from utils import PBC_wrapping

# Set run_test to 1 to run test
run_test = 0

class BaseDipoleFunction:
    """
    Base dipole function class for constructing other dipole function
    """
    def __init__(
            self,positive_atom_idx, negative_atom_idx
            ):

        assert len(positive_atom_idx) == distance_calc.n_points
        assert len(negative_atom_idx) == distance_calc.n_points

        #boolean matrix which has element is 1 if the location is r_p - r_n (p:positive, n:negative) and 
        # 0 otherwise
        self.r_pn = np.array(np.outer(positive_atom_idx, negative_atom_idx), dtype = bool)

        self.update(distance_calc)

    def __call__(self, R_all, distance_calc, return_tensor = False):
        """
        Return dipole vectors in two ways (1) array of all unique dipole vector from positive atom
        (r_+) to negative atom (r_-) (2) N x N matrix where i,j element is dipole vector between 
        i and j if exist dipole vector between them or zero (note that the actual return matrix is 
        of size N x (N-1) since the diagonal is removed; and the vector still point from r_+ to 
        r_-)
        """
        
        dipole_vec_array = distance_calc.apply_function(
                R_all, self.dipole_func, custom_mask = self.r_pn_mask_x3
                )

        if return_tensor:
            dipole_tensor = self.distance_calc.construct_matrix(
                    dipole_vec_array, output_shape = 3, symmetry = 0,
                    custom_mask = self.r_pn_mask_x3, remove_diagonal = False
                    )

            dipole_tensor += np.transpose(dipole_tensor, (1,0,2))

            return dipole_tensor

        else:
            return dipole_vec_array

    def gradient(self, R_all, return_all = False):
        """
        Return gradient of the dipole function w.r.t. position r = r_+.
        The matrix will have the shape N x N x 3 x 3 where the 
        3 x 3 tensor at i, j -th element will be
            |~ dmu_x/dr_x dmu_y/dr_x dmu_z/dr_x ~|
            |  dmu_x/dr_y dmu_y/dr_y dmu_z/dr_y  |
            |_ dmu_x/dr_z dmu_y/dr_z dmu_z/dr_z _|; 
            OR (D_r mu)_(ij) = dmu_j / dr_i 
        with mu = mu(r_i, r_j), e.g. dipole vector between i-th and j-th atoms
        """
        gradient_array = self.distance_calc.apply_function(
                R_all, self.gradient_func, custom_mask = self.r_pn_mask_x3)

        gradient_tensor = self.distance_calc.construct_matrix(
                gradient_array, output_shape = (3,3),
                custom_mask = self.r_pn_mask_x3x3, symmetry = 0, 
                remove_diagonal = False)

        gradient_tensor -= np.transpose(gradient_tensor, (1,0,2,3))

        if return_all:
            return gradient_tensor

        gradient_tensor = np.sum(gradient_tensor, axis = 1)
        return gradient_tensor

    def update(self,distance_calc):

        self.distance_calc = distance_calc

        r_pn_mask = self.distance_calc.generate_custom_mask(
                self.r_pn, only_neighbor_mask = True)

        self.r_pn_mask_x3 = np.array(
                np.tile(r_pn_mask[:,:,np.newaxis], (1,1,3)),
                dtype = bool)

        self.r_pn_mask_x3x3 = np.array(
                np.tile(self.r_pn_mask_x3[:,:,:,np.newaxis], (1,1,1,3)),
                dtype = bool)

class SimpleDipoleFunction(BaseDipoleFunction):
    """
    Class for Dipole Function evaluation based on work by Grigoriev et al.
    """
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


if run_test:
    from test.dipole import ExplicitTestDipoleFunction
    try:
        from neighborlist import neighbor_list_mask
        neighbor_list_module_availability = True
    except:
        print("Neighborlist module cannot be found. Testing without neighborlist.")

    import input_dat
    ########################
    ######### TEST #########
    ########################
    ###### BOX LENGTH ######
    ########################

    L = 40
    cell_width = 20

    ##########################
    ###### ATOMIC INPUT ######
    ##########################

    R_all = np.vstack([input_dat.r_ar, input_dat.r_xe])
    V = np.vstack([input_dat.v_ar, input_dat.v_xe])

    N_atoms = R_all.shape[0]

    # indices of atoms in the R_all and V_all
    idxAr = np.hstack([np.ones(int(N_atoms/2)),np.zeros(int(N_atoms/2))])
    idxXe = np.hstack([np.zeros(int(N_atoms/2)),np.ones(int(N_atoms/2))])

    ############################################
    ##### Test without neighbor cell list. #####
    ############################################

    print("##### Test without neighbor cell list. #####")

    distance_calc = DistanceCalculator(
            n_points = len(R_all), 
            neighbor_mask = None,
            box_length = L
            )

    dipole_function = SimpleDipoleFunction(
            distance_calc, 
            mu0=0.0124 , a=1.5121, d0=7.10,
            positive_atom_idx = idxXe,
            negative_atom_idx = idxAr
            )

    dipole_function_test = ExplicitTestDipoleFunction(
            mu0=0.0124 , a=1.5121, d0=7.10,
            positive_atom_idx = idxXe,
            negative_atom_idx = idxAr, L = L
            )

    dipole_vec_mat = dipole_function(R_all, return_tensor = True)
    dipole_vec_mat_ = dipole_function_test(R_all)

    print("+ Difference w/ explicit test for dipole vector: ", 
            np.sum(abs(dipole_vec_mat - dipole_vec_mat_)))

    total_dipole_vec = dipole_function(R_all, return_tensor = False)
    total_dipole_vec = np.sum(total_dipole_vec,axis = 0)
    total_dipole_vec_ = np.sum(np.sum(dipole_vec_mat_,axis = 0),axis = 0) / 2
 
    print("+ Difference w/ explicit test for total dipole vector: ", 
            np.sum(abs(total_dipole_vec - total_dipole_vec_)))

    H = dipole_function.gradient(R_all,return_all=False)
    H_ = dipole_function_test.gradient(R_all,return_all=False)

    print("+ Difference w/ explicit test for dipole gradient: ", 
            np.sum(abs(H - H_)))

    #########################################
    ##### Test with neighbor cell list. #####
    #########################################
    print("##### Test with neighbor cell list. #####")

    distance_calc = DistanceCalculator(
            n_points = len(R_all), 
            neighbor_mask = neighbor_list_mask(R_all, L, cell_width),
            box_length = L
            )

    dipole_function = SimpleDipoleFunction(
            distance_calc, 
            mu0=0.0124 , a=1.5121, d0=7.10,
            positive_atom_idx = idxXe,
            negative_atom_idx = idxAr
            )

    dipole_vec_mat = dipole_function(R_all, return_tensor = True)
    dipole_vec_mat_ = dipole_function_test(R_all)

    print("+ Difference w/ explicit test for dipole vector: ", 
            np.sum(abs(dipole_vec_mat - dipole_vec_mat_)))

    total_dipole_vec = dipole_function(R_all, return_tensor = False)
    total_dipole_vec = np.sum(total_dipole_vec,axis = 0)
    total_dipole_vec_ = np.sum(np.sum(dipole_vec_mat_,axis = 0),axis = 0) / 2
 
    print("+ Difference w/ explicit test for total dipole vector: ", 
            np.sum(abs(total_dipole_vec - total_dipole_vec_)))

    H = dipole_function.gradient(R_all,return_all = True)
    H_ = dipole_function_test.gradient(R_all,return_all=True)

    print("+ Difference w/ explicit test for dipole gradient: ", 
            np.sum(abs(H - H_)))

    #########################################################################
    #########################################################################
    #########################################################################

    R_all = np.array([[0,0,0],[2,1,3]])

    # indices of atoms in the R_all and V_all
    idxAr = np.array([0,1])
    idxXe = np.array([1,0])
    
    distance_calc = DistanceCalculator(
            n_points = 2, neighbor_mask = None,
            box_length = None
            )

    dipole_function = SimpleDipoleFunction(
            distance_calc, 
            mu0=0.0284 , a=1.22522, d0=7.10,
            positive_atom_idx = idxXe,
            negative_atom_idx = idxAr
            )

    print(dipole_function.gradient(R_all,return_all=True))



