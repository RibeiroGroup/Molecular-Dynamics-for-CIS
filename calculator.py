import numpy as np
from distance import DistanceCalculator, explicit_test
from utils import PBC_wrapping, timeit
from forcefield import LJ_force, LJ_potential

test = True

def generate_dipole_mask(positive_atom_idx, negative_atom_idx):
    """
    Generating dipole mask (potentially for Calculator object's dipole method)
    Output will be a N x N mask with True at ij element if there is dipole between
    them, e.g. positive and negative atoms, and False otherwise.
    Args:
    + positive_atom_idx (np.array): array with SIZE: N (N is number of atoms)
        and element is either 1/True or 0/False
    + negative_atom_idx (np.array): array with SIZE: N (N is number of atoms)
        and element is either 1/True or 0/False
    """

    dipole_mask = np.array(
        np.outer(positive_atom_idx, negative_atom_idx), dtype = bool)

    dipole_mask += dipole_mask.transpose()

    return dipole_mask


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
        + Dipole mask (np.ndarray): a N x N boolean matrix mask with ij-element is True if
            dipole exist between i-th and j-th atom and zero otherwise
        + mu0 (float): \mu_0 parameter
        + a (float): a parameter
        + d(float): d parameter
    """
    def __init__(
        self, N, box_length, epsilon, sigma, 
        dipole_mask, mu0, a, d
        ):

        super().__init__(N, box_length)

        assert epsilon.shape == (self.n_points, self.n_points)
        self.epsilon = epsilon

        assert sigma.shape == (self.n_points, self.n_points)
        self.sigma = sigma

        assert dipole_mask.shape == (self.n_points, self.n_points)
        self.dipole_mask = dipole_mask

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

        self.neighborlist = neighborlist

        # using methods from DistanceCalculator parent class to calculate 
        # distance matrix
        self.distance_matrix = self.calculate_distance_matrix(
            R, neighborlist)

        # and distance vector matrix
        self.distance_vec_tensor = self.calculate_distance_vector_tensor(
            R, neighborlist)

    def potential(self):
        """
        Method for calculating Lennard-Jones potential
        """

        # usual upper triangle boolean matrix
        mask = self.utriang_mask

        if self.neighborlist is not None:
             # add in neighborlist mask
             mask *= self.neighborlist

        # extracting epsilon and sigma in the parameter matrix
        # that correspond to the mask position
        epsilon = self.epsilon[mask]
        sigma = self.sigma[mask]

        # extracting distance in the distance matrix
        # that correspond to the mask position
        darray = self.distance_matrix[mask]

        # apply LJ_potential function
        potential = LJ_potential(sigma, epsilon, darray)
        # rearrange the result array into matrix form
        potential = self.matrix_reconstruct(
            potential, symmetric_padding = 1, mask = mask)

        return potential

    def force(self, return_matrix = False):

        # similar to the potential method
        mask = self.utriang_mask
        mask_x3 = self.repeat_x3(mask)

        if self.neighborlist is not None:
             mask *= self.neighborlist

        epsilon = self.epsilon[mask]
        sigma = self.sigma[mask]

        darray = self.distance_matrix[mask]
        dvec_array = self.distance_vec_tensor[mask].reshape(-1,3)

        force = LJ_force(sigma, epsilon, darray, dvec_array)
        force = self.matrix_reconstruct(
            force, symmetric_padding = -1, 
            mask = mask, utriang_mask_x3 = mask_x3)

        if not return_matrix: 
            force = np.sum(force, axis = 1)

        return force 

    def dipole(self):

        # similar to the potential method
        mask = self.utriang_mask * self. dipole_mask
        mask_x3 = self.repeat_x3(mask)

        if self.neighborlist is not None:
             mask *= self.neighborlist

        distance = self.distance_matrix[mask]
        distance_vec = self.distance_vec_tensor[mask_x3].reshape(-1,3)

        dipole = self.mu0 * np.exp(-self.a * (distance - self.d)) 

        dipole = np.tile(
               dipole[:,np.newaxis], (1,3))


        dipole *= distance_vec

        return dipole

    def dipole_grad(self):
        """!TODO"""
        pass

if test == True:
    from utils import neighborlist_mask
    from forcefield import explicit_test_LJ

    ########################
    ###### BOX LENGTH ######
    ########################

    L = 200
    cell_width = 40

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
        dipole_mask = dipole_mask,
        mu0=0.0124 , a=1.5121, d=7.10,
        )

    calculator.calculate_distance(R_all, neighborlist)

    potential = calculator.potential()
    force = calculator.force(return_matrix = True)

    potential_, force_ = explicit_test_LJ(R_all, epsilon_mat, sigma_mat, L)

    print(np.sum(abs(potential - potential_)))
    print(np.sum(abs(force - force_)))


    dipole = calculator.dipole()



