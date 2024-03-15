import numpy as np

from utils import PBC_wrapping

class DistanceCalculator:
    def __init__(self,n_points, mask = None ,box_length  = None):
        self.n_points = n_points
        self.update_mask(mask)
        self.L = box_length

    def update_mask(self, mask = None):
        if mask is None:
            mask = np.ones((self.n_points,self.n_points),dtype=bool)
        else:
            assert isinstance(mask, np.ndarray)
            assert mask.shape == (self.n_points, self.n_points)

        self.utriang_bool_mat = np.triu(mask ,k = 1 )

        self.utriang_bool_mat_x3 = np.tile(
                self.utriang_bool_mat[:,:,np.newaxis],(1,1,3))

    def get_all_distance_vector_array(self,R):
        """
        First, then return vector of distances with format:
        [r2 - r1, r3 - r1 , ... , rN - r1
        r3 - r2, r4 - r2, ... , rN - r2
        r4 - r3, r5 - r3, ... , rN - r3
        ... ... rN - r{N-1} ]

        Then arange the distance vector in the tensor with the format 
        0,       r1 - r2, r1 - r3, ... , r1 - rN
        r2 - r1, 0      , r2 - r3, ... , r2 - rN
        r3 - r1, r3 - r2, 0      , ... , r3 - rN
        ........................................
        rN - r1, rN - r2, rN - r3, ... , 0
        """

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

        R_mat1 = R_mat1[self.utriang_bool_mat_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r2 r3 r4 ... rN r3 r4 ... rN r4 rN ... ...r(N-1) rN rN
        """
        R_mat2 = R_mat2[self.utriang_bool_mat_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r1 r1 ...(N-1 times)... r1 r2 ...(N-2 times) ... r2 r3 ...... r_(N-1) 
        """

        d_vec = R_mat1 - R_mat2 # r2 - r1, r3 - r1, ... , r3 - r2, ...

        if self.L is not None:
            d_vec = PBC_wrapping(d_vec, self.L)

        return d_vec

    def apply_dvector_function(self, R, func):

    def get_all_distance_vector_tensor(self,R):

        d_vec = self.get_all_distance_vector_array(R)

        vec_tensor = np.zeros((self.n_points,self.n_points,3))

        vec_tensor[self.utriang_bool_mat_x3] = - d_vec.ravel()

        vec_tensor -= np.transpose(vec_tensor, (1,0,2))

        return vec_tensor

    def get_all_distance_matrix(self,R):

        all_distance_vector_array = self.get_all_distance_vector_array(R)

        all_distance = np.sqrt(
                np.einsum("ij,ij->i",
                    all_distance_vector_array, all_distance_vector_array)
                )

        distance_mat = np.zeros((self.n_points, self.n_points))

        distance_mat[self.utriang_bool_mat] = all_distance.ravel()

        distance_mat += distance_mat.T

        return distance_mat

