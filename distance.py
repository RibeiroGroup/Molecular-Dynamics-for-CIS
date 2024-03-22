import numpy as np

from utils import PBC_wrapping

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
    def __init__(self,n_points, neighbor_mask = None, box_length  = None):
        """
        Args:
        + n_points (int)
        + mask (np.ndarray of shape (N,N) ): recommend for incorporating the neighbor list by using neighbor_list_mask
            function from neighborlist.py module
        + box_lenth (float): for incorporating the Periodic Boundary condition 
        """
        self.n_points = n_points

        self.identity_mat = np.eye(self.n_points, dtype = bool)

        self.identity_mat_x3 = np.tile(
                self.identity_mat[:,:,np.newaxis],(1,1,3)
        )

        self.update_global_mask(neighbor_mask)

        self.L = box_length

    def update_global_mask(self, neighbor_mask = None):

        if neighbor_mask is None:
            self.neighbor_mask = None
            mask = np.ones((self.n_points,self.n_points),dtype=bool)

        else:
            assert isinstance(neighbor_mask, np.ndarray)
            assert neighbor_mask.shape == (self.n_points, self.n_points)
            self.neighbor_mask = neighbor_mask
            mask = neighbor_mask

        self.utriang_mask = np.triu(mask ,k = 1)

        self.utriang_mask_x3 = np.tile(
                self.utriang_mask[:,:,np.newaxis],(1,1,3))

    def get_all_distance_vector_array(self, R):
        """
        Return array of distances with format:
        [r1 - r2, r1 - r3, r1 - r4 ... r1 - rN
        r2 - r3, r2 - r4, ... r2 - rN
        r3 - r4, r3 - r5, ... r3 - rN
        ...
        r[N-1] - rN
        """

        utriang_mask_x3 = self.utriang_mask_x3

        R_mat1 = np.tile(
                R[np.newaxis,:,:], (self.n_points,1,1))
        """
        shape of ra_mat1 should be:
        [r1 r2 r3 r4 ... rN
         r1 r2 r3 r4 ... rN
         r1 r2 r3 r4 ... rN
         ...............
         r1 r2 r3 r4 ... rN] (N rows and N-1 columns)
        """
        R_mat2 = np.transpose(R_mat1, (1,0,2))
        """
        shape of ra_mat1 should be:
        [r1 r1 r1 r1 ... r1
         r2 r2 r2 r2 ... r2
         r3 r3 r3 r3 ... r3
         ...............
         rN rN rN rN ... rN] (N rows and N-1 columns)
        """

        R_mat1 = R_mat1[self.utriang_mask_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r2 r3 r4 ... rN r3 r4 ... rN r4 rN ... ...r(N-1) rN rN
        """
        R_mat2 = R_mat2[self.utriang_mask_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r1 r1 ...(N-1 times)... r1 r2 ...(N-2 times) ... r2 r3 ...... r_(N-1) 
        """

        d_vec = R_mat2 - R_mat1 # r1 - r2, r1 - r3, ... , r2 - r3, ...

        if self.L is not None:
            d_vec = PBC_wrapping(d_vec, self.L)

        return d_vec

    def apply_function(self, R, func, output_shape):
        """
        Compute a square function that has element rij = func( |rij| , rij)
        e.g. output of function that takes distance btw atom i and atom j (|rij|)
        and the distance vector between the two (rij)
        """

        distance_vec_array = self.get_all_distance_vector_array(R)

        distance_array = np.sqrt(
                np.einsum("ij,ij->i", distance_vec_array, distance_vec_array)
                )

        if output_shape == 3:

            some_array = func(distance_array, distance_vec_array)

            some_tensor = np.zeros((self.n_points,self.n_points,3))

            some_tensor[self.utriang_mask_x3] = some_array.ravel()

            some_tensor -= np.transpose(some_tensor, (1,0,2))

            some_tensor = some_tensor[~self.identity_mat_x3].reshape(
                self.n_points, self.n_points-1,3)

            return some_tensor

        elif output_shape == 1:

            some_scalar = func(distance_array, distance_vec_array)

            some_matrix = np.zeros((self.n_points, self.n_points))

            some_matrix[self.utriang_mask] = some_scalar.ravel()

            some_matrix += some_matrix.T

            some_matrix = some_matrix[~self.identity_mat].reshape(self.n_points, self.n_points-1)

            return some_matrix

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
            







