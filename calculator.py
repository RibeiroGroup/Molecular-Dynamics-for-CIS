import numpy as np
from distance import DistanceCalculator, explicit_test
from utils import PBC_wrapping, timeit

from forcefield import LJ_force, LJ_potential
from dipole import Grigoriev_dipole, Grigoriev_dipole_grad, generate_dipole_mask

test = False

class Calculator(DistanceCalculator):
    """
    Class for calculating various properties that based on interactomic distance.
    Args:
    + N (int): number of atoms
    + box_length (float): the length of the simulated box
    Follow is various properties parameters. Need to add/modify parameters if existing function 
    to be modified or new function to be added. Current function: Lennard-Jones potential
    and Griegoriev dipole function.
    - Lennard-Jones potential parameters:
        + epsilon (np.ndarray): epsilon of Lennard-Jones potential
            SIZE: N x N
        + sigma (np.ndarray): sigma of Lennard-Jones potential
            SIZE: N x N
    - Argon-Xenon dipole function:
        + positive_atom_idx (np.array): array with SIZE: N (N is number of atoms)
            and element is either 1/True or 0/False
        + negative_atom_idx (np.array): array with SIZE: N (N is number of atoms)
            and element is either 1/True or 0/False
        + mu0 (float): mu_0 parameter
        + a (float): a parameter
        + d(float): d parameter
    """
    def __init__(
        self, N, box_length, epsilon, sigma, 
        positive_atom_idx, negative_atom_idx, mu0, a, d
        ):

        super().__init__(N, box_length)

        assert epsilon.shape == (self.n_points, self.n_points)
        self.epsilon = epsilon

        assert sigma.shape == (self.n_points, self.n_points)
        self.sigma = sigma

        self.dipole_mask = generate_dipole_mask(
                positive_atom_idx, negative_atom_idx)

        self.mu0 = mu0
        self.a = a
        self.d = d

    def calculate_distance(self, R, neighborlist = None):
        """
        Calculating all relevant distances and distances vector.
        Note: ALways running this method to update all distances matrix 
        before calculating dipole or force/ potential.
        Args:
        + R (np.array): list of all atoms' positions
            SIZE: N x N
        + neighborlist (np.array): neighborlist
            SIZE: N x N
        """

        # using methods from DistanceCalculator parent class to calculate 
        # distance matrix
        self.distance_matrix = self.calculate_distance_matrix(
            R, neighborlist)

        # and distance vector matrix
        self.distance_vec_tensor = self.calculate_distance_vector_tensor(
            R, neighborlist)

        # generating mask for subsequent calculation
        # usual upper triangle boolean matrix
        self.mask = self.utriang_mask

        if neighborlist is not None:
             self.mask *= neighborlist

        self.mask_x3 = self.repeat_x3(self.mask)

    def potential(self,return_matrix = False):
        """
        Method for calculating Lennard-Jones potential
        """

        # extracting epsilon and sigma in the parameter matrix
        # that correspond to the mask position
        epsilon = self.epsilon[self.mask]
        sigma = self.sigma[self.mask]

        # extracting distance in the distance matrix
        # that correspond to the mask position
        darray = self.distance_matrix[self.mask]

        # apply LJ_potential function
        potential = LJ_potential(sigma, epsilon, darray)
        
        if return_matrix:
            # rearrange the result array into matrix form
            return self.matrix_reconstruct(
                potential, symmetric_padding = 1, mask = self.mask)

        return np.sum(potential)

    def force(self, return_matrix = False):
        """
        Method for calculating Lennard-Jones force
        """
        # retrieving parameters that corresponding to relevant pair of atoms
        epsilon = self.epsilon[self.mask]
        sigma = self.sigma[self.mask]

        # retrieving distance value and vector similarly
        darray = self.distance_matrix[self.mask]
        dvec_array = self.distance_vec_tensor[self.mask_x3].reshape(-1,3)

        # calculating force
        force = LJ_force(sigma, epsilon, darray, dvec_array)
        force = self.matrix_reconstruct(
            force, symmetric_padding = -1, 
            mask = self.mask, utriang_mask_x3 = self.mask_x3)

        if not return_matrix: 
            force = np.sum(force, axis = 1)

        return force 

    def dipole(self, return_matrix=True):

        #generate the mask for retrieving relevant distance and parameters
        mask = self.dipole_mask
        mask_x3 = self.repeat_x3(mask)

        # generating distance array rather than matrix (similar for distance vec)
        distance = self.distance_matrix[mask]
        distance_vec = self.distance_vec_tensor[mask_x3].reshape(-1,3)

        dipole = Grigoriev_dipole(
                distance = distance, distance_vec = distance_vec, 
                mu0 = self.mu0, a = self.a, d = self.d)

        # rearrange the result array into N x N x 3 matrix form 
        dipole = self.matrix_reconstruct(
            dipole, symmetric_padding = 1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_matrix: 
            dipole = np.sum(dipole, axis = 1)

        return dipole

    def dipole_grad(self, return_matrix = False):
        """
        Return gradient of the dipole function w.r.t. position r = r_+.
        The matrix will have the shape N x N x 3 x 3 where the 
        3 x 3 tensor at i, j -th element will be
            |~ dmu_x/dr_x dmu_y/dr_x dmu_z/dr_x ~|
            |  dmu_x/dr_y dmu_y/dr_y dmu_z/dr_y  |
            |_ dmu_x/dr_z dmu_y/dr_z dmu_z/dr_z _|; 
            OR (D_r mu)_(ij) = dmu_j / dr_i 
        with mu = mu(r_i, r_j), e.g. dipole vector between i-th and j-th atoms
        """

        mask = self.dipole_mask
        mask_x3 = self.repeat_x3(mask)

        distance = self.distance_matrix[mask]
        distance_vec = self.distance_vec_tensor[mask_x3].reshape(-1,3)

        gradient = Grigoriev_dipole_grad(
            distance = distance, distance_vec = distance_vec,
            mu0 = self.mu0, a = self.a, d = self.d)

        # rearrange the result array into N x N x 3 matrix form 
        gradient = self.matrix_reconstruct(
            gradient, symmetric_padding = -1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_matrix:
            gradient = np.sum(gradient, axis = 1)

        return gradient

if test == True:
    from utils import neighborlist_mask
    from forcefield import explicit_test_LJ
    from dipole import explicit_test_dipole

    ########################
    ###### BOX LENGTH ######
    ########################

    L = 20
    cell_width = 10

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

    dipole_mask = generate_dipole_mask(idxXe, idxAr)

    calculator = Calculator(
        N, box_length = L, 
        epsilon = epsilon_mat, sigma=sigma_mat,
        positive_atom_idx = idxXe, negative_atom_idx = idxAr,
        mu0=0.0124 , a=1.5121, d=7.10,
        )

    calculator.calculate_distance(R_all, neighborlist)

    potential_, force_ = explicit_test_LJ(R_all, epsilon_mat, sigma_mat, L)

    print("### Potential test ###")
    potential = calculator.potential()
    print(np.sum(abs(potential - potential_)))

    print("### Force test ###")
    force = calculator.force(return_matrix = True)
    print(np.sum(abs(force - force_)))

    print("### Dipole test ###")
    dipole = calculator.dipole()
    dipole_, gradD_ = explicit_test_dipole(R_all, dipole_mask,
        mu0=0.0124 , a=1.5121, d=7.10, L = L)

    print(np.sum(abs(dipole - dipole_)))

    print("### Dipole gradient test ###")
    gradD = calculator.dipole_grad()
    
    print(np.sum(abs(gradD - gradD_)))


