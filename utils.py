import time
import numpy as np

def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(time.time() - start)
        return result
    return inner

def PBC_wrapping(r, L):
    r = np.where(r >= L/2, r - L, r)
    r = np.where(r < -L/2, r + L, r)
    return r

class DistanceCalculator:
    def __init__(self,n_points,box_length  = None):
        self.n_points = n_points
        self.utriang_bool_mat = np.triu(
                np.ones((n_points,n_points),dtype=bool),
                k = 1
                )
        self.utriang_bool_mat_x3 = np.tile(
                self.utriang_bool_mat[:,:,np.newaxis],(1,1,3))

        self.L = box_length

    def get_distance(self,R):
        """
        Return vector of distances with format:
        [r2 - r1, r3 - r1 , ... , rN - r1
        r3 - r2, r4 - r2, ... , rN - r2
        r4 - r3, r5 - r3, ... , rN - r3
        ... ... rN - r{N-1} ]
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

        R_mat1 = R_mat1[self.utriang_bool_mat_x3].reshape(-1,3)
        """
        Get the flattened upper triangular matrix of R_mat1 one line above the
        diagonal:
        r2 r3 r4 ... rN r3 r4 ... rN r4 rN ... ...r(N-1) rN rN
        """
        R_mat2 = R_mat2[self.utriang_bool_mat_x3].reshape(-1,3)

        return R_mat1 - R_mat2

    #@timeit
    def __call__(self,R):
        """
        Arrange the distance vector in the tensor with the format 
        0,       r1 - r2, r1 - r3, ... , r1 - rN
        r2 - r1, 0      , r2 - r3, ... , r2 - rN
        r3 - r1, r3 - r2, 0      , ... , r3 - rN
        ........................................
        rN - r1, rN - r2, rN - r3, ... , 0
        """
        d_vec = self.get_distance(R)

        vec_tensor = np.zeros((self.n_points,self.n_points,3))

        vec_tensor[self.utriang_bool_mat_x3] = - d_vec.ravel()

        vec_tensor -= np.transpose(vec_tensor, (1,0,2))

        if self.L is not None:
            vec_tensor = PBC_wrapping(vec_tensor, self.L)

        return vec_tensor

    def get_distance_vector(self,R):
        d_vec = self.get_distance(R)

        dist = np.sqrt(np.sum((d_vec)**2,axis=-1))

        dist_mat = np.zeros((self.n_points,self.n_points))
        dist_mat[self.utriang_bool_mat] = dist
        dist_mat += dist_mat.T

        """
        dist_mat = dist_mat[
                ~np.eye(self.n_points,dtype=bool)].reshape(
                        self.n_points,self.n_points-1)
        """

        return dist_mat

def get_dist_matrix(distance_vector):

    distance_matrix = np.sqrt(
        np.sum(distance_vector**2, axis = -1))
    
    return distance_matrix

"""
L = 100
n_points = 1000
all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))

dc = DistanceCalculator(n_points, box_length= L)
dc(all_r)
"""

def test_for_distance_matrix(ra):
    n_points = ra.shape[0]
    dist_mat_ = np.zeros((n_points,n_points))
    for i, ri in enumerate(ra):
        for j,rj in enumerate(ra):
            if i == j: continue
            #j_ = j if j < i else j-1
            dist_mat_[i][j] = np.sqrt(np.sum((ri - rj)**2))

    return dist_mat_

def test_for_distance_vector(x):
    n_points = x.shape[0]

    d_ = np.zeros((n_points, n_points, 3))
    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            if i == j: continue
            d_[i,j,:] = x1 - x2

    return d_

"""
@PBC_decorator(L = L)
def get_dist_vector(R):
    rij_vec_tensor = np.zeros((n_points, n_points,3))
    for i in range(len(R)):
        for j in range(i, len(R)):
            
            ri = R[i]; rj = R[j];

            rij_vec = ri - rj
            #rij = np.sqrt(np.sum((ri - rj)**2))

            rij_vec_tensor[i,j,:] = rij_vec
            rij_vec_tensor[j,i,:] = - rij_vec

    return rij_vec_tensor

def PBC_wrapping(r, L):
    r = np.where(r >= L/2, r - L, r)
    r = np.where(r < -L/2, r + L, r)
    return r

def PBC_decorator(L):
    def real_PBC_wrap(func):
        def PBC_wrapping_wrapper(*args,**kwargs):
            dist = func(*args, **kwargs)
            dist = PBC_wrapping(dist, L)

            return dist

        return PBC_wrapping_wrapper
    return real_PBC_wrap


"""
