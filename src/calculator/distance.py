import numpy as np

from .utils import PBC_wrapping

"""
Note:
The follow distance calculator class employ mask of array. Mask is the boolean matrix that when apply
on array A with the same size, e.g. A[mask] would return element of A that correspond to the True 
element of the mask.

Distance matrix refer to a matrix whose ij-element is the distance between the i-element and j-element
of the R position vector 

Distance vector matrix refer to a matrix whose ij-element is the vector distance between the i-element 
and j-element of the R position vector 
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
        + mask (np.array):  masking for extracting relevant elements
            SIZE: N x N x 3 with N is the number of particles
        + neighborlist (np.array): neighborlist matrix, if provide, will be
            multiplied with the mask matrix
        """

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

        R_mat1 = R_mat1[mask].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r2 r3 r4 ... rN r3 r4 ... rN r4 rN ... ...r(N-1) rN rN
        """
        R_mat2 = R_mat2[mask].reshape(-1,3)
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
            self, R, func,
            custom_mask=None, neighborlist=None,
            matrix_reconstruction=True, symmetric_padding = 1
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
        + matrix_reconstruction (bool): whether return the array 
        + symmetric_padding (int):
        """

        if custom_mask is None:
            #using the upper triangular bool matrix as default mask
            mask = self.utriang_mask
        else:
            #ensure provided custom mask matrix should have shape N x N
            assert custom_mask.shape == (self.n_points, self.n_points)
            mask = custom_mask

        if neighborlist is not None:
            assert neighborlist.shape == (self.n_points, self.n_points)
            #the final mask is the element-wise product of the mask and neighborlist 
            mask *= neighborlist

        #extend the mask matrix to N x N x 3 by tiling the last dim
        utriang_mask_x3 = self.repeat_x3(mask)

        #calculating the distance vector matrix
        distance_vec_array = self.get_all_distance_vector_array(R, utriang_mask_x3)

        #calculating the distance matrix 
        distance_array = np.sqrt(
                np.einsum("ij,ij->i", distance_vec_array, distance_vec_array)
                )

        some_array = func(distance_array, distance_vec_array)

        if not matrix_reconstruction:
            return some_array

        return self.matrix_reconstruct(some_array, mask, utriang_mask_x3, symmetric_padding)
    
    def matrix_reconstruct(
        self, some_array, mask = None, 
        utriang_mask_x3 = None, symmetric_padding = None
        ):

        return_matrix = np.zeros((self.n_points,self.n_points) + some_array.shape[1:])

        if return_matrix.shape == (self.n_points, self.n_points):
            if mask is None: raise Exception("Mask needed to be provide")
            out_mask = mask

        elif return_matrix.shape == (self.n_points, self.n_points, 3):
            if utriang_mask_x3 is None: raise Exception("Mask needed to be provide")
            out_mask = utriang_mask_x3
        
        elif return_matrix.shape == (self.n_points, self.n_points, 3, 3):
            if utriang_mask_x3 is None: raise Exception("Mask needed to be provide")
            out_mask = self.repeat_x3(utriang_mask_x3)

        else: 
            raise Exception("UNexpected output")

        return_matrix[out_mask] = some_array.ravel()

        if symmetric_padding:
            return_matrix += symmetric_padding * np.swapaxes(return_matrix,0,1)
            
        return return_matrix

    def calculate_distance_matrix(self,R, neighborlist=None):
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

        return self.apply_function(
            R, get_distance, neighborlist = neighborlist, symmetric_padding = 1)

    def calculate_distance_vector_tensor(self,R,neighborlist=None):
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

        return self.apply_function(
            R, get_distance_vector, neighborlist = neighborlist, 
            symmetric_padding = -1)

def explicit_test(R , L = None):
    """
    Function for explicitly calculating the pair-wisee distance for 
    testing the above distance class
    Args:
    + R (np.array): array of position. SIZE: N x 3
    + L (float): length of the box, for applying periodic boundary condition (PBC)
        default is None, for no PBC
    """
    n = len(R)

    all_distance_vec_tensor = np.zeros((n,n,3))
    all_distance_matrix = np.zeros((n,n))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j: continue

            dvec = ri - rj
            dvec = PBC_wrapping(dvec,L)

            all_distance_vec_tensor[i,j] = dvec
            all_distance_matrix[i,j] = np.sqrt(dvec @ dvec)

    return all_distance_matrix, all_distance_vec_tensor
            
