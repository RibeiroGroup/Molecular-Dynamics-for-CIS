import time
import numpy as np
import numpy.linalg as la

from scipy.stats import chi as Chi

def repeat_x3(array):
    array = np.tile(
        array[:,np.newaxis], (1,3))
    return array

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

def init_amplitude(k_value, mode = 'zero', T = None):
    """
    Initialize the amplitude for the field given array of k_vector
    Args:
    + k_vector (np.array, shape N x 3): k vector for iniitalizing the amplitude
    + k_value (np.array, shape N): norm of all k_vector given in k_vector
    + mode ('str'): mode of amplitude initialization. Available modes:
        - 'zero': init all amplitude to be zeros
        - 'boltzmann': init all amplitude according to Boltzmann distribution
    """
    if mode == 'zero':
        amplitude = np.zeros(
            (len(k_value), 2), dtype = np.complex128)

    elif mode == 'boltzmann':
        print('Please make sure your temperature {} is in reduced unit.'.format(T))
        assert T is not None
        amplitude = []

        for i, kvec in enumerate(k_value):
            kval = k_value[i]

            chi2_dist = Chi(2, scale = np.sqrt(T * np.pi) / kval)
            ck0 = chi2_dist.rvs(size = 2)

            theta = np.random.uniform(0, np.pi * 2)
            C = np.array([
                ck0[0] * (np.cos(theta) + 1j * np.sin(theta)),
                ck0[1] * (np.cos(theta) + 1j * np.sin(theta))
            ])
            
            amplitude.append(C)

        amplitude = np.array(amplitude)
        
    return amplitude

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
