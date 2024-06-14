import argparse
import numpy as np

from utils import PBC_wrapping

run_test = True

"""
The follow distance calculator class employ mask of array. Mask is the boolean matrix that when apply
on array A with the same size, e.g. A[mask] would return element of A that correspond to the True 
element of the mask.
"""

class DistanceCalculator:
    """
    Utility class for calculating atomic pair-wise distance
    Methods:
        + update_global_mask (return None): set attribute utriang_mask
        + get_local_mask
    Attribute:
        + utriang_mask (np.ndarray, shape (N,N), dtype = bool): This mask-matrix serve two purpose, to
            retrieve the element of the triangular matrix (above the diagonal, not include the diagonal), 
            and to set elements of the said matrix to particular values. This limit the calculation
            to only the upper triangular matrix, and, by assuming that the matrix is symmetric, the lower 
            triangular part of the matrix can be deduced.
    """
    def __init__(self,N , box_length  = None):
        """
        Args:
        + N (int): number of particles
        + neighbor_mask (np.ndarray of shape (N,N) ): include the neighbor-list mask calculating by using 
			neighbor_list_mask function from neighborlist.py module
        + box_lenth (float): for incorporating the Periodic Boundary condition 
        """
        self.n_points = N

        self.L = box_length

		# Pre-generate boolean matrix for masking position-matrix
        # -> accelerating distance calculation
        self.identity_mat = np.eye(self.n_points, dtype = bool)

        mask = np.ones((self.n_points,self.n_points),dtype=bool)
        # upper triangular 2d boolean matrix
        self.utriang_mask = np.triu(mask ,k = 1)

    def repeat_x3(self, matrix):
        """
        Convert a Nd matrix to a (N+1)d matrix by repeating this Nd matrix along the last
        axis of the (N+1)d matrix (N = 2,3)
        Args:
        + matrix (np.array): 2d matrix
        """
        if len(matrix.shape) == 2:
            new_matrix = np.tile(matrix[:,:,np.newaxis],(1,1,3))

        elif len(matrix.shape) == 3:
            new_matrix = np.tile(matrix[:,:,:,np.newaxis],(1,1,1,3))

        return new_matrix

    def get_all_distance_vector_array(
            self, R, mask , neighborlist=None
            ):
        """
        Return array of distances with format:
            [r1 - r2, r1 - r3, r1 - r4 ... r1 - rN
            r2 - r3, r2 - r4, ... r2 - rN
            r3 - r4, r3 - r5, ... r3 - rN
            ...
            r[N-1] - rN ]
        Args:
        + R (np.array): particle postion whose pair-wise distances are evaluated
            SIZE: N x 3 with N is the number of particle
        + mask (np.array): custom mask. 
            SIZE: N x N with N is the number of particles
        + neighborlist (np.array): neighborlist matrix, if provide, will be
            multiplied with the mask matrix
        """

        assert mask.shape == (self.n_points, self.n_points)
        utriang_mask_x3 = mask

        if neighborlist:
            assert neighborlist.shape == (self.n_points, self.n_points)
            utriang_mask_x3 *= neighborlist

        utriang_mask_x3 = self.repeat_x3(utriang_mask_x3)

        R_mat1 = np.tile(
                R[np.newaxis,:,:], (self.n_points,1,1))
        """
        shape of R_mat1 should be:
        [r1 r2 r3 r4 ... rN
         r1 r2 r3 r4 ... rN
         r1 r2 r3 r4 ... rN
         ...............
         r1 r2 r3 r4 ... rN] (N rows and N-1 columns)
        """
        R_mat2 = np.transpose(R_mat1, (1,0,2))
        """
        shape of R_mat2 should be:
        [r1 r1 r1 r1 ... r1
         r2 r2 r2 r2 ... r2
         r3 r3 r3 r3 ... r3
         ...............
         rN rN rN rN ... rN] (N rows and N-1 columns)
        """

        R_mat1 = R_mat1[utriang_mask_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r2 r3 r4 ... rN r3 r4 ... rN r4 rN ... ...r(N-1) rN rN
        """
        R_mat2 = R_mat2[utriang_mask_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r1 r1 ...(N-1 times)... r1 r2 ...(N-2 times) ... r2 r3 ...... r_(N-1) 
        """

        d_vec = R_mat2 - R_mat1 # r1 - r2, r1 - r3, ... , r2 - r3, ...

        if self.L is not None:
            d_vec = PBC_wrapping(d_vec, self.L)

        return d_vec

    def apply_function(
            self, R, func, custom_mask=None, neighborlist=None,
            matrix_reconstruction=True
            ):
        """
        Compute a square matrix that has element ij = func( |rij| , rij)
        e.g. output of function that takes distance btw atom i and atom j (|rij|)
        and the distance vector between the two (rij = ri - rj)
        Args:
        + R (np.array): position. SIZE: N x 3
        + func (python function): a function that take two array of size N, N x 3
            with the first arg is the distance and the second arg is the distance vector
        + custom_mask (np.array): custom mask. If None, will use a boolean triangle matrix
            SIZE: N x N
        + neighborlist (np.array): 
        """

        if custom_mask is None:
            mask = self.utriang_mask
        else:
            assert custom_mask.shape == (self.n_points, self.n_points)
            mask = custom_mask

        distance_vec_array = self.get_all_distance_vector_array(R, mask)

        distance_array = np.sqrt(
                np.einsum("ij,ij->i", distance_vec_array, distance_vec_array)
                )

        some_array = func(distance_array, distance_vec_array)

        if not matrix_reconstruction:
            return some_array

        return_matrix = np.zeros((self.n_points,self.n_points) + some_array[1:])

    def construct_matrix(
            self, array, output_shape, custom_mask = None, 
            symmetry = -1, remove_diagonal = True
        ):
        """
        Args:
        + symmetry: -1 for antisymmetry (M[i,j] = -M[j,i]), 0 for no symmetry
            and 1 for symmetry matrix
        """

        if output_shape == 3:

            if custom_mask is None:
                mask_x3 = self.repeat_x3(self.utriang_mask)
            else:
                mask_x3 = self.repeat_x3(custom_mask)

            some_tensor = np.zeros((self.n_points,self.n_points,3))

            some_tensor[mask_x3] = array.ravel()

            identity_mat_x3 = self.repeat_x3(self.identity_mat)

            if symmetry:
                some_tensor += np.transpose(some_tensor, (1,0,2)) * symmetry

            if remove_diagonal:
                some_tensor = some_tensor[~identity_mat_x3].reshape(
                    self.n_points, self.n_points-1,3)

            return some_tensor

        elif output_shape == 1:

            if custom_mask is None:
                mask = self.utriang_mask
            else:
                mask = custom_mask

            some_tensor = np.zeros((self.n_points, self.n_points))

            some_tensor[mask] = array.ravel()

            if symmetry:
                some_tensor += some_tensor.T * symmetry

            if remove_diagonal:
                some_tensor = some_tensor[~self.identity_mat].reshape(
                        self.n_points, self.n_points-1)

            return some_tensor

        elif output_shape == (3,3):

            if custom_mask is None: raise Exception("Mask is required!")
            else: mask_x3x3 = custom_mask

            some_bigger_tensor = np.zeros((self.n_points, self.n_points, 3, 3))

            some_bigger_tensor[mask_x3x3] = array.ravel()

            if symmetry:
                some_bigger_tensor -= np.transpose(some_bigger_tensor, (1,0,2,3))

            if remove_diagonal:
                some_bigger_tensor = some_bigger_tensor[~self.identity_mat_x3x3].reshape(
                    self.n_points, self.n_points-1,3,3)

            return some_bigger_tensor

        else:
            raise Exception("DistanceCalculator.apply_function only accept output shape 1,3, and (3,3)")

    def get_all_distance_vector_tensor(self,R):
        """
        Sample for self.apply_function function
        Return the distance vector in the tensor with the format 
        0,       r1 - r2, r1 - r3, ... , r1 - rN
        r2 - r1, 0      , r2 - r3, ... , r2 - rN
        r3 - r1, r3 - r2, 0      , ... , r3 - rN
        ........................................
        rN - r1, rN - r2, rN - r3, ... , 0
        """

        def get_distance_vector(distance, distance_vec):
            return distance_vec

        return self.apply_function(R, get_distance_vector, output_shape = 3)

    def get_all_distance_matrix(self,R):
        """
        Sample for self.apply_function function
        Return the distance vector in the tensor with the format 
        0,       r1 - r2, r1 - r3, ... , r1 - rN
        r2 - r1, 0      , r2 - r3, ... , r2 - rN
        r3 - r1, r3 - r2, 0      , ... , r3 - rN
        ........................................
        rN - r1, rN - r2, rN - r3, ... , 0
        """

        def get_distance(distance, distance_vec):
            return distance

        return self.apply_function(R, get_distance, output_shape = 1)


def explicit_test(R , L = None):
    n = len(R)

    all_distance_vec_tensor = np.zeros((n,n-1,3))
    all_distance_matrix = np.zeros((n,n-1))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j: continue
            dvec = ri - rj
            dvec = PBC_wrapping(dvec,L)
            if j < i:
                all_distance_vec_tensor[i,j] = dvec
                all_distance_matrix[i,j] = np.sqrt(dvec @ dvec)
            elif j > i:
                j -= 1  
                all_distance_vec_tensor[i,j] = dvec
                all_distance_matrix[i,j] = np.sqrt(dvec @ dvec)

    return all_distance_matrix, all_distance_vec_tensor
            
if run_test:

    try:
        from neighborlist import neighbor_list_mask
        neighbor_list_module_availability = True
    except:
        print("Neighborlist module cannot be found. Testing without neighborlist.")

    ########################
    ###### BOX LENGTH ######
    ########################

    L = 200
    cell_width = 20

    ##########################
    ###### ATOMIC INPUT ######
    ##########################

    # number of atoms
    N_Ar = int(L/5)
    N_Xe = int(L/5)
    N = N_Ar + N_Xe

    # randomized initial coordinates
    R_all = np.random.uniform(-L/2, L/2, (N, 3))

    # Calculation from explicit test
    true_distance_mat, true_distance_vec = explicit_test(R_all, L) 
    De = 10
    true_potential = np.exp(De - true_distance_mat)

    ############################################
    ##### Test without neighbor cell list. #####
    ############################################
    print("##### Test without neighbor cell list. #####")

    distance_calc = DistanceCalculator(N = N, box_length = L)

    distance_mat = distance_calc.apply_function(R_all, func = lambda d, ar: d)
    print(distance_mat)

    """
    distance_mat = distance_calc.construct_matrix(distance_mat, output_shape = 1, symmetry = 1)

    print("+++ Difference between DistanceCalculator class and ExpliciTest for distance matrix +++")
    print(np.sum(distance_mat - true_distance_mat))
        
    distance_vec = distance_calc.apply_function(R_all, func = lambda d, ar: ar)
    distance_vec = distance_calc.construct_matrix(distance_vec, output_shape = 3)

    print("+++ Difference between DistanceCalculator class and ExpliciTest for distance vector array +++")
    print(np.sum(abs(distance_vec - true_distance_vec)))

    #########################################
    ##### Test with neighbor cell list. #####
    #########################################
    print("##### Test with neighbor cell list. #####")

    distance_calc = DistanceCalculator(
        n_points = N, neighbor_mask = neighbor_list_mask(R_all, L, cell_width) , 
        box_length = L)

    distance_mat = distance_calc.apply_function(R_all, func = lambda d, ar: d)
    distance_mat = distance_calc.construct_matrix(distance_mat, output_shape = 1)

    print("+++ Difference between DistanceCalculator class and ExpliciTest for distance matrix +++")
    print(np.sum(distance_mat - true_distance_mat))

    potential = distance_calc.apply_function(R_all, func = lambda d, ar: np.exp(De - d))
    potential = distance_calc.construct_matrix(potential, output_shape = 1)

    print("+++ Difference between DistanceCalculator class and ExpliciTest for potential +++")
    print(np.sum(potential - true_potential))
        
    distance_vec = distance_calc.apply_function(R_all, func = lambda d, ar: ar)
    distance_vec = distance_calc.construct_matrix(distance_vec, output_shape = 3)

    print("+++ Difference between DistanceCalculator class and ExpliciTest for distance vector array +++")
    print(np.sum(abs(distance_vec - true_distance_vec)))


    """
