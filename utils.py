import numpy as np

class DistanceCalculator:
    def __init__(self,n_points):
        self.n_points = n_points
        self.utriang_bool_mat = np.triu(
                np.ones((n_points,n_points),dtype=bool),
                k = 1
                )
        self.utriang_bool_mat_x3 = np.tile(
                self.utriang_bool_mat[:,:,np.newaxis],(1,1,3))

    def get_distance_vector(self,R):

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

    def __call__(self,R):
        d_vec = self.get_distance_vector(R)

        dist = np.sqrt(np.sum((d_vec)**2,axis=-1))

        dist_mat = np.zeros((self.n_points,self.n_points))
        dist_mat[self.utriang_bool_mat] = dist
        dist_mat += dist_mat.T

        dist_mat = dist_mat[
                ~np.eye(self.n_points,dtype=bool)].reshape(
                        self.n_points,self.n_points-1)

        return dist_mat

def compute_dist_mat1(ra):
    """
    Compute distance matrix with the form:
    [ d{12} d{13} d{14} .. d{1N}
      d{21} d{23} d{24} .. d{2N}
      d{31} d{32} d{34} .. d{3N}
      ..... ..... ..... .. .....
      d{N1} d{N2} d{N3} .. d{N,N-1}
    Args:
    + ra (np.array): array of coordinates with shape n_points x 3
    """
    n_points = ra.shape[0]
    ra_mat1 = np.tile(
            ra[np.newaxis,:,:],(n_points,1,1))[
                    ~np.eye(n_points,dtype=bool)].reshape(
                            n_points,n_points-1,3)
    # give the array a "new axis" -> repeat the array along the new axis
    # -> remove element along the diagonal

    #print(ra_mat1.shape)

    ra_mat2 = np.tile(
            ra[:,np.newaxis,:],(1,n_points-1,1)) 
    #print(ra_mat2.shape)
    """
    shape of ra_mat2 should be:
    [r1 r1 r1 .. r1
     r2 r2 r2 .. r2
     r3 r3 r3 .. r3
     .. .. .. .. ..
     rN rN rN .. rN] (N rows and N-1 columns)
    """
    ra_mat2 = ra_mat2[upper_triang_bool_by3_mat]

    dist_mat = np.sqrt(np.sum((ra_mat1 - ra_mat2)**2,axis=-1))
    #print(dist_mat.shape)
    return dist_mat

def verify_distant_matrix(ra):
    n_points = ra.shape[0]
    dist_mat_ = np.zeros((n_points,n_points-1))
    for i, ri in enumerate(ra):
        for j,rj in enumerate(ra):
            if i == j: continue
            j_ = j if j < i else j-1
            dist_mat_[i][j_] = np.sqrt(np.sum((ri - rj)**2))

    return dist_mat_




