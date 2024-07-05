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

def EM_mode_generate_(max_n, n_vec_per_kz = 1, min_n = 1):
    modes_list = []
    for i in range(min_n, max_n + 1):
        sample_range = np.arange(1,i + 1)
        if n_vec_per_kz < len(sample_range):
            ky = np.random.choice(
                    sample_range, size = n_vec_per_kz - 1, 
                    replace = False)
            ky = np.hstack([[0], ky])
        else: ky = sample_range

        kz = np.array([i] * len(ky)).reshape(-1,1)
        ky = ky.reshape(-1,1)
        kx = np.zeros(ky.shape)

        mode_vector = np.hstack([kx,ky,kz])
        modes_list.append(mode_vector)
    return np.vstack(modes_list)

def EM_mode_generate3(max_n,min_n = 1, max_n111 = None):
    mode_list = []
    for i in range(min_n, max_n+1):
        mode_list.append([i,0,0])
        mode_list.append([0,i,0])
        mode_list.append([0,0,i])

        """
        if max_n111 and i < max_n111:
            for j in np.arange(max(0,i-2),min(max_n,i+2)):
                mode_list.append([j,j,i])
                mode_list.append([j,i,j])
                mode_list.append([i,j,j])
        """

    return np.vstack(mode_list)

def profiling_rad(omega_list,Hrad):

    unique_omega = list(set(np.round(omega_list,decimals = 6)))
    unique_omega = np.sort(unique_omega)
    rad_profile = []

    for i, omega in enumerate(unique_omega):
        rad_profile.append(
                np.sum(Hrad[np.isclose(omega_list, omega)])
                )

    return unique_omega, rad_profile


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
