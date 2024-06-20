import time
import numpy as np
import numpy.linalg as la
from itertools import combinations, combinations_with_replacement, permutations

def repeat_x3(array):
    array = np.tile(
        array[:,np.newaxis], (1,3))
    return array

def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("Runtime by timeit: ",time.time() - start)
        return result
    return inner

def PBC_wrapping(r, L):
    """
    Function for wrapping position or distance vector with periodic boundary condition
    Args:
    + r (np.array): array of position or vector
    + L (float): box length
    """
    if L is None:
        return r
    else:
        assert isinstance(L,float) or isinstance(L,int)
        r = np.where(r >= L/2, r - L, r)
        r = np.where(r < -L/2, r + L, r)
    return r

def orthogonalize(vec, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    """
    
    U = np.vstack([vec, np.random.rand(2,3)])
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]

		# each entry is np.dot(V[j], V[i]) for all j < i
        coeff_vec = np.dot(prev_basis, V[i].T)  

        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if la.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= la.norm(V[i])

    V[0] = vec
    return V

def EM_mode_generate(
        max_n, min_n = 0,
        vector_per_kval=None,
        align_vector = None
        ):

    """
    Exhautively generate all combination for mode vector
    """

    all_combs = combinations_with_replacement(
            list(range(min_n,max_n)), 3
            )
    #generate all combinations of sorted integers, e.g. (0,1,2) or (1,1,2)

    modes_list = []

    for comb in all_combs:
        """
        Permuting element in each combination
        """
        comb = list(comb)
        comb_modes_list = []

        if np.sum(comb) < 1: 
            continue

        if comb[0] == comb[1] and comb[1] == comb[2]:
            # [1,1,1] -> [-1,1,1]
            comb[0] = - comb[0]

        perm = set(permutations(comb))

        for mode in perm:
            comb_modes_list.append(np.array(mode))

        comb_modes_list = np.array(comb_modes_list)

        if align_vector is not None:
            alignment = list(map(
                    lambda x: x @ align_vector, comb_modes_list))

            sort_idx = np.argsort(alignment)[::-1]

            comb_modes_list = comb_modes_list[sort_idx,:]

        if vector_per_kval:
            comb_modes_list = comb_modes_list[:vector_per_kval,:]

        modes_list.append(comb_modes_list)

    return np.vstack(modes_list)

def neighborlist_mask(R_all, L, cell_width):
    """
    Class for generating neighborlist mask for accelerating calculation of distance
    Args:
    + R_all (np.array): position of all particles/atoms
    + L (float) dimension of the simulating cubic box
    + cell_width (float): dimension of cell
    """

    assert cell_width < L

    #binning the cubic box width to cells width
    L_bin = np.arange(-L/2,L/2+1,cell_width)

    #calculate the center of the cell
    cell_center_list = np.array(
            [(L + L_bin[i+1])/2 for i,L in enumerate(L_bin[:-1])]
            )

    # Repeating R_all to get an array w dim: (N atoms, 3, len(cell_center_list))
    tiled_R_all = np.tile(R_all[:,:,np.newaxis],(1,1,len(cell_center_list)) )

    # Repeating cell_center_list to get an array w dim: (N atoms, 3, num cell center)
    # Assuming the cell centers coordinates are the same in x,y,z dim
    tiled_cell_center = np.tile(cell_center_list[np.newaxis,np.newaxis,:],(R_all.shape[0],3,1))

    # Calculating the distance (in either x, y, z dim) to corresponding cell center
    # The smallest absolute distance => cell center index/bin
    cell_bin = np.argmin(abs(tiled_cell_center - tiled_R_all), axis = -1)

    # Calculating the differences of cell center indices/bin for all atoms in all
    # 3 dim, cell center difference by one in either x, y, z => nearby cell
    R_bin_diff = abs(
            np.tile(cell_bin[:,np.newaxis,:],(1,len(cell_bin),1)) \
            - np.tile(cell_bin[np.newaxis,:,:],(len(cell_bin),1,1))
            )

    # Considering the Periodic Boundary condition
    R_bin_diff = np.where(R_bin_diff == len(cell_center_list) - 1, 1, R_bin_diff)

    mask = np.sum(R_bin_diff,axis = -1)
    mask = np.where(mask <= 3, True, False) 

    return mask


"""
k_vector = EM_mode_generate(3, vector_per_kval=3, align_vector = None)# np.array([1,0,0]))
print(k_vector)

k_vector = np.array(k_vector, dtype= np.float64) 

k_vector *= (2 * np.pi / 100)

k_vector = np.array([
    orthogonalize(kvec) for kvec in k_vector
    ]) 

print(k_vector[:,0,:])
"""
