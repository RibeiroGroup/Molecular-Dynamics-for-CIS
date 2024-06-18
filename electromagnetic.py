import numpy as np
from utils import orthogonalize, timeit

test = False

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
    def __init__(self, k_vector, amplitude, V, constant_c, pol_vec = None):
        self.number_modes = k_vector.shape[0]

        assert k_vector.shape[1] == 3
        assert amplitude.shape == (self.number_modes, 2)
        assert isinstance(V, float)

        self.k_vector = k_vector
        self.C = amplitude
        self.V = V
        self.constant_c = constant_c

        if pol_vec == None:
            self.pol_vec = []

            for k_vec in self.k_vector:
                self.pol_vec.append(orthogonalize(k_vec)[1:,:])

            self.pol_vec = np.array(self.pol_vec)

        else:
            self.pol_vec = pol_vec

        assert self.pol_vec.shape == (self.number_modes, 2, 3)

        self.k_val = np.sqrt(
                np.einsum("ni,ni->n",self.k_vector,self.k_vector)) 

        self.omega = self.k_val * constant_c

    def update_amplitude(self, amplitude):
        assert amplitude.shape == (self.number_modes, 2)
        self.C = amplitude

    def __call__(self, t, R, amplitude = None):
        """
        Evaluate the vector potential at time t and multiple positions specified in R
        Args:
        + t (float): time
        + R (np.array): position. SIZE: M x 3 with M is the number of positions
        Returns:
        + np.array: SIZE M x 3 with M specified in R argument
        """

        C = self.C if amplitude is None else amplitude
        k_vec = self.k_vector
        pol_vec = self.pol_vec

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

        return A_R / np.sqrt(self.V)

    def dot_amplitude(self, t, R, current):
        """
        Calculate the time derivative of the amplitude of the vector potential
            in the presence of moving charge particle / dipole
        Args:
        + t (float): time
        + R (np.ndarray): position of charged particle 
            SIZE: M x 3 w/ M is number of particles
        + current (np.ndarray): notation J, is the current
            SIZE: M x 3
        """
        k_vec = self.k_vector
        pol_vec = self.pol_vec

        omega = self.omega

        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        C = self.C

        #grad_mu_r_dot = np.einsum("nij,ni->nj", gradD, R_dot) # i = j = 3

        exp_ikr = np.exp(np.einsum("ki,ni->kn",-1j * k_vec,R)) # i = 3

        Jk = np.einsum("nj,kn->kj",current,exp_ikr) # j = 3

        C_dot = np.einsum("kij,kj->ki",pol_vec, Jk) #i = 2, j = 3

        C_dot *= (2 * np.pi * 1j / k_val) * np.exp(1j * self.omega * t)

        return C_dot

    def time_diff(self, t, R, Rp, current):
        """
        Calculate the time derivative of the 
        """
        C_dot = self.dot_amplitude(t, Rp, current)
        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        dA1 = self.__call__(t, R, C_dot)
        dA2 = self.__call__(t, R, -1j * omega * self.C)

        return dA1 + dA2

    def gradient(self, t, R):
        """
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; (gradA)_(ij) = dA_i/dr_j
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        k_vec = self.k_vector
        pol_vec = self.pol_vec

        C = self.C
        
        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # free field mode function and c.c., a.k.a. exp(ikr) exp(-i \omega t) + c.c
        f_R = 1j * np.exp(
            1j * np.einsum("kj,mj->km",k_vec,R) - 1j * omega * t) # j = 3

        fs_R = - 1j * np.exp(
            -1j * np.einsum("kj,mj->km",k_vec,R) + 1j * omega * t) # j = 3

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki",C,pol_vec)

        k_o_C_epsilon_k = np.einsum("kj,ki->kij",k_vec,C_epsilon_k)

        #
        gradA_R = np.einsum("kij,kn->nij",k_o_C_epsilon_k, f_R) \
            + np.einsum("kij,kn->nij",np.conjugate(k_o_C_epsilon_k), fs_R)

        return gradA_R

    def hamiltonian(self, return_sum_only = False):
        k_sum = np.einsum("ki,ki->k",self.k_vector,self.k_vector)
        c_sum = np.einsum("ki,ki->k",self.C,np.conjugate(self.C))

        H = (2*np.pi)**-1 * k_sum * c_sum

        if return_sum_only:
            return np.sum(H)

        return H

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






