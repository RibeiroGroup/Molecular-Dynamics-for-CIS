import numpy as np
from distance import DistanceCalculator, explicit_test
from utils import PBC_wrapping, timeit

test = True

def LJ_potential(sigma, epsilon, distance):

    V = 4 * epsilon * ( (sigma/distance)**12 - (sigma/distance)**6 )

    return V

def LJ_force(sigma, epsilon, distance, distance_vec):

    f = 4 * epsilon * (
            12 * (sigma**12 / distance**14) - 6 * (sigma**6 / distance**8)
            )

    f = np.tile(f[:,np.newaxis],(1,3)) * (distance_vec)

    return f

@timeit
def explicit_test_LJ(R, epsilon ,sigma, L):

    N = len(R)

    potential = np.zeros((N,N))
    force = np.zeros((N,N,3))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j: continue
            
            ep = epsilon[i,j]
            sig = sigma[i,j]
            
            dvec = ri - rj
            dvec = PBC_wrapping(dvec,L)

            d = np.sqrt(dvec @ dvec)

            potential[i,j] = 4 * ep * ( (sig/d)**12 - (sig/d)**6 )
            f = 4 * ep * (
                12 * (sig**12 / d**14) - 6 * (sig**6 / d**8)
            )

            force[i,j,:] = f * (dvec)

    return potential, force

class Calculator(DistanceCalculator):

    def __init__(self, N, box_length, epsilon, sigma):

        super().__init__(N, box_length)

        self.epsilon = epsilon
        self.sigma = sigma

    def calculate_distance(self, R, neighborlist = None):

        self.neighborlist = neighborlist

        self.distance_matrix = self.calculate_distance_matrix(
            R, neighborlist)

        self.distance_vec_tensor = self.calculate_distance_vector_tensor(
            R, neighborlist)

    @timeit
    def LennardJones_potential(self, return_force_matrix = False):

        mask = self.utriang_mask
        mask_x3 = self.repeat_x3(mask)

        if self.neighborlist is not None:
             mask *= self.neighborlist

        epsilon = self.epsilon[mask]
        sigma = self.sigma[mask]

        darray = self.distance_matrix[mask]
        dvec_array = self.distance_vec_tensor[mask]

        potential = LJ_potential(sigma, epsilon, darray)
        potential = self.matrix_reconstruct(
            potential, symmetric_padding = 1, mask = mask)

        force = LJ_force(sigma, epsilon, darray, dvec_array)
        force = self.matrix_reconstruct(
            force, symmetric_padding = -1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_force_matrix: 
            force = np.sum(force, axis = 1)

        return potential, force

if test == True:
    from utils import neighborlist_mask

    ########################
    ###### BOX LENGTH ######
    ########################

    L = 1000
    cell_width = 20

    ##########################
    ###### ATOMIC INPUT ######
    ##########################

    # number of atoms
    N_Ar = int(L/4)
    N_Xe = int(L/4)
    N = N_Ar + N_Xe

    # randomized initial coordinates
    R_all = np.random.uniform(-L/2, L/2, (N, 3))

    N = R_all.shape[0]

    idxXe = np.hstack([np.ones(int(N/2)),np.zeros(int(N/2))])
    idxAr = np.hstack([np.zeros(int(N/2)),np.ones(int(N/2))])

    ######################################
    ###### FORCE-RELATED PARAMETERS ######
    ######################################

    epsilon_Ar_Ar = 0.996 * 1.59360e-3
    epsilon_Ar_Xe = 1.377 * 1.59360e-3
    epsilon_Xe_Xe = 1.904 * 1.59360e-3

    sigma_Ar_Ar = 3.41 * (1e-10 / 5.29177e-11)
    sigma_Ar_Xe = 3.735* (1e-10 / 5.29177e-11)
    sigma_Xe_Xe = 4.06 * (1e-10 / 5.29177e-11)

    epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
        + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
        + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

    sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
        + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
        + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
        + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

    ############
    ### TEST ###
    ############
    neighborlist = neighborlist_mask(R_all, L, cell_width)

    calculator = Calculator(
        N, box_length = L, 
        epsilon = epsilon_mat, sigma=sigma_mat)

    calculator.calculate_distance(R_all, neighborlist)

    potential,force = calculator.LennardJones_potential(return_force_matrix = True)

    potential_, force_ = explicit_test_LJ(R_all, epsilon_mat, sigma_mat, L)

    print(np.sum(abs(potential - potential_)))
    print(np.sum(abs(force - force_)))





