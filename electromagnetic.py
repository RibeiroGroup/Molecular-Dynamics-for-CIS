import numpy as np
from utils import orthogonalize, timeit

test = True

class FreeFieldVectorPotential:
    """
    Class for vector potential of the electromagnetic field in free (e.g. not confined) field
    Args:
    + k_vector (np.array): wavevector, SIZE: N x 3 with N is arbitrary number of modes
    + amplitude (np.array): amplitude of the vector potential for each mode, thus, it should
        have SIZE: N x 2 with N is the number of modes and should be consistent with the number
        of wavevector
    + V (float): volume
    + epsilon (np.array, optional): polarization vector, SIZE N x 2 x 3 with N should be consistent
        with above arguments
    """
    def __init__(self, k_vector, amplitude, V, epsilon = None):
        self.number_modes = k_vector.shape[0]

        assert k_vector.shape[1] == 3
        assert amplitude.shape == (self.number_modes, 2)
        assert isinstance(V, float)

        self.k_vector = k_vector
        self.C = amplitude
        self.V = V

        if epsilon == None:
            self.epsilon = []

            for k_vec in self.k_vector:
                self.epsilon.append(orthogonalize(k_vec)[1:,:])

            self.epsilon = np.array(self.epsilon)

        else:
            self.epsilon = epsilon

        assert self.epsilon.shape == (self.number_modes, 2, 3)

        self.omega = np.einsum("ni,ni->n",self.k_vector,self.k_vector)

    def __call__(self, t, R, amplitude = None):
        """
        Evaluate the vector potential at time t and multiple positions specified in R
        Args:
        + t (float): time
        + R (np.array): position. SIZE: M x 3 with M is the number of positions
        Returns:
        + np.array: SIZE M x 3 with M specified in R argument
        """

        C = self.C if amplitude = None else amplitude
        k_vec = self.k_vector
        pol_vec = self.epsilon

        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # free field mode function and c.c., a.k.a. exp(ikr) exp(-i \omega t) + c.c
        f_R = np.exp(
            1j * np.einsum("kj,mj->km",k_vec,R) - 1j * omega * t) # j = 3

        fs_R = np.exp(
            -1j * np.einsum("kj,mj->km",k_vec,R) + 1j * omega * t) # j = 3

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki" ,C ,pol_vec)

        #
        A_R = np.einsum("ki,km->mi",C_epsilon_k, f_R) \
            + np.einsum("ki,km->mi",np.conjugate(C_epsilon_k), fs_R)

        return A_R / np.sqrt(V)

    def dot_amplitude(self, t, R, R_dot):
        """
        Calculate the time derivative of the amplitude of the vector potential
            in the presence of moving charge particle / dipole
        Args:
        + t (float): time
        + R (np.ndarray): position of charged particle 
            SIZE: M x 3 w/ M is number of particles
        + R_dot (np.ndarray): velocity of charge particle
            SIZE: M x 3
        + gradD (np.array): SIZE M x 3 x 3 where the i-th 3x3 matrix in the tensor is 
            the total gradient of dipole of all M atoms with the i-th atom
        """
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]
        omega = np.tile(self.omega[:,np.newaxis],(1,2))
        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        C = C if C is not None else self.C

        grad_mu_r_dot = np.einsum("nij,ni->nj", gradD, R_dot) # i = j = 3

        exp_ikr = np.exp(np.einsum("ki,ni->kn",-1j * k_vec,R)) # i = 3

        grad_mu_eikr = np.einsum("nj,kn->kj",grad_mu_r_dot,exp_ikr) # j = 3

        C_dot = np.einsum("kij,kj->ki",pol_vec,grad_mu_eikr) #i = 2, j = 3

        C_dot *= (2 * np.pi * 1j / k_val) * np.exp(1j * self.omega * t)

        return C_dot

    def time_diff(self, t, R, Rp, Rp_dot):
        """
        Calculate the time derivative of the 
        """
        C_dot = self.dot_amplitude(t, Rp, Rp_dot)
        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        dA1 = self.__call__(t, R, C_dot)
        dA2 = self.__call__(t, R, -1j * omega * self.C)

        return dA1 + dA2

    def gradient(self, R):
        pass

    def get_electric_field(self, R):
        pass

    def get_magnetic_field(self, R):
        pass

class CavityFieldPotentialVector:
    """
    Class for vector potential of the field in the cavity
    Args:
    + kappa (np.array): wavevector in the x and y direction
        SIZE: N x 2 with N is the number of modes
    + m (np.array): quantum number of wavevector in z direction
        SIZE: N with N consistent with above definition
    + amplitude (np.array): ampltitude of both the TE and TM mode
        SIZE: N x 2 with N consistent with above definition
    + S (float): xy cross-section of the cavity
    + L (float): length in z direction of the cavity
    """
    def __init__(self, kappa, m, amplitude, S, L):
        
        self.n_modes = kappa.shape[0]
        assert kappa.shape == (self.n_modes, 2)
        assert m.shape == (self.n_modes)
        assert amplitude.shape == (self.n_modes, 2)
        assert isinstance(S, float)
        assert isinstance(L, float)

        self.kappa_vec = kappa_vec
        self.kappa = np.sqrt(
                np.einsum("ki,ki->k",self.kappa_vec,self.kappa_vec)
                )

        self.kz = 2 * np.pi * m / L
        self.C = amplitude
        self.S = S
        self.L = L

        # constructing the TE mode polarization vector
        self.epsilon_TE = np.zeros(self.n_modes, 3)
        self.epsilon_TE[:,0] = self.kappa_vec[:,1] / self.kappa
        self.epsilon_TE[:,1] = -self.kappa_vec[:,0] / self.kappa

    def __call__(self, t, R):
        """
        Evaluate the vector potential at time t and multiple positions specified in R
        Args:
        + t (float): time
        + R (np.array): position. SIZE: M x 3 with M is the number of positions
        Returns:
        + np.array: SIZE M x 3 with M specified in R argument
        """

        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # TE mode
        f_R = np.exp(
                1j * np.einsum("kj,mj->km",self.kapa_vec,R[:,:2]) - 1j * omega * t)

        fs_R = np.exp(
            -1j * np.einsum("kj,mj->km",k_vec,R) + 1j * omega * t)
        
        C_TE_sinkz = self.C[:,0] * np.sin(self.kz * R[:,-1])
        C_TE_sinkz_ekmu = np.tile(C_TE_sinkz[:,np.newaxis],(1,3)) * self.epsilon_TE #k x 3

        A_TE = np.einsum("ki,km->mi",C_TE_sinkz_ekmu, f_R) \
            + np.einsum("ki,km->mi",np.conjugate(C_TE_sinkz_ekmu), fs_R)

        # TM mode

        return A_R / np.sqrt(V)


if test:
    from utils import EM_mode_generate

    k_vec = EM_mode_generate(20)
    print(k_vec.shape)

    """
    EXAMPLE CALCULATION FOR VECTOR POTENTIAL OF NON-CONFINED FIELD
    """

    n_mode = k_vec.shape[0]

    amplitude = np.random.rand(n_mode,2) + 1j * np.random.rand(n_mode,2)

    A = FreeFieldVectorPotential(k_vector = k_vec, amplitude = amplitude, V = 100.0)

    kappa = np.array([[1,1]])
    m = np.array([1])

    """
    EXAMPLE CALCULATION FOR VECTOR POTENTIAL OF CONFINED FIELD
    """

    n_mode = kappa.shape[0]

    amplitude = np.random.rand(n_mode,2) + 1j * np.random.rand(n_mode,2)

    A_cav = CavityFieldPotentialVector(
            kappa = k_vec, m = m, amplitude = amplitude, S = 100.0, L = 10.0)

    #R = np.random.rand(10,3)

    #A(0, R)






