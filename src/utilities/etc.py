import os, sys, glob
import time
import numpy as np
import numpy.linalg as la
from itertools import combinations, combinations_with_replacement, permutations

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

def binning(x,y,width):
    bins = np.arange(x[0],x[-1],width)
    new_x = []; new_y = []
    for i, x1 in enumerate(bins[:-1]):
        x2 = bins[i+1]
        new_x.append((x1 + x2)/2)
        new_y.append(
                np.mean(
                    np.where((x > x1) * (x < x2),y,0)
                    )
                )

    return np.array(new_x), np.array(new_y)

def moving_average(x, y, w):
    #x = np.convolve(x, np.ones(w), 'valid') / w
    y_new = np.convolve(y, np.ones(w), 'valid') / w
    halfw = int(w/2)
    y_new = np.hstack(
        [y[halfw], y_new, y[len(x) + 1 - halfw]]
        )
    return x,y_new

def categorizing_pickle(pickle_jar_path, KEYWORDS = ""):
    """
    Get and categorize all pickle files corresponding to either 'pickled' 
    monte_carlo simulation in the cavity or the free field.
    Args:
    + pickle_jar_path (str): path to directory of all pickle files. Just the path
        to the directory. No need *
    + KEYWORDS (str): either 'free' or 'cavity'
    Returns:
    + file_dict (Python dictionary): keys is the number of the cycle that is pickled 
        and value is the path to the pickle file
    """

    file_dict = {}
    pickle_jar_path += "/*"
    for file in glob.glob(pickle_jar_path):
        if os.path.isdir(file):
            continue
        elif KEYWORDS not in file or "result" not in file:
            continue

        # the format of the pickle file to be e.g result_cavity_5.pkl
        number = file.split(".")[0]
        number = number.split("_")[-1]
        number = int(number)

        file_dict[number] = file

    return file_dict





