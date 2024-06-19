import numpy as np
from utils import orthogonalize, timeit

test = True

class BaseVectorPotential:
    def __init__(self):
        self.C = amplitude

    def update_amplitude(self, amplitude):
        assert amplitude.shape == (self.number_modes, 2)
        self.C = amplitude

    def mode_function(self, t, R):
        pass

    def __call__(self, t, R, amplitude = None):
        """
        Evaluate the vector potential at time t and multiple positions specified in R
        Args:
        + t (float): time
        + R (np.array): position. SIZE: M x 3 with M is the number of positions
        + amplitude (np.array): optional amplitude, if None provided,
            self.C is used for evaluation
        Returns:
        + np.array: SIZE M x 3 with M specified in R argument
        """

        C = self.C if amplitude is None else amplitude

        f_R = self.mode_function(t, R)

        fs_R = np.conjugate(f_R)

        A_R = np.einsum("ki,kimj->mj",C, f_R) \
            + np.einsum("ki,kimj->mj",np.conjugate(C), fs_R)

        return A_R / np.sqrt(self.V)

    def dot_amplitude(self, t, charge_assemble):
        """
        Calculate the time derivative of the amplitude of the vector potential
            in the presence of moving charge particle / dipole
        Args:
        + t (float): time
        + charge_assemble (python object): can be any object, but it must have 
            'current' method that: 1/ take in arg 'k' and 'mode_function', 
            2/ mode_function must take in None, 'TE' or 'TM' corresponding to
            current in k-space for free field (exponential Fourier), current 
            projected by TE or TM mode function
        """
        k_vec = self.k_vector
        pol_vec = self.pol_vec

        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        C = self.C

        Jk = charge_assemble.current(k_vec, mode_function = None)

        C_dot = np.einsum("kij,kj->ki",pol_vec, Jk) #i = 2, j = 3

        C_dot *= (2 * np.pi * 1j / k_val) * np.exp(1j * omega * t)

        return C_dot

    def time_diff(self, t, charge_assembly):
        """
        Calculate the time derivative of the vector potential
        Args: 
        + t (float): time
        + charge_assemble (python object): can be any object, but it must have 
            'current' method that: 1/ take in arg 'k' and 'mode_function', 
            2/ mode_function must take in None, 'TE' or 'TM' corresponding to
            current in k-space for free field (exponential Fourier), current 
            projected by TE or TM mode function
        """
        C_dot = self.dot_amplitude(t, charge_assembly)
        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        dA = self.__call__(
                t, R, amplitude = -1j * omega * self.C + C_dot)

        return dA

    def hamiltonian(self, return_sum_only = False):
        """
        Evaluate the Hamiltonian/total energy of the field
        Args:
        + return_sum_only (bool, default = False): if True, return 
            only the total value, if False, will return the Hamiltonian
            of the field for each mode 
        """
        k_sum = np.einsum("ki,ki->k",self.k_vector,self.k_vector)
        c_sum = np.einsum("ki,ki->k",self.C,np.conjugate(self.C))

        H = (2*np.pi)**-1 * k_sum * c_sum / np.sqrt(self.V)

        if return_sum_only:
            return np.sum(H)

        return H

class FreeFieldVectorPotential(BaseVectorPotential):
    """
    Class for vector potential of the electromagnetic field in free (e.g. not confined) field
    Args:
    + k_vector (np.array): wavevector, SIZE: N x 3 with N is arbitrary number of modes
    + amplitude (np.array): amplitude of the vector potential for each mode, thus, it should
        have SIZE: N x 2 with N is the number of modes and should be consistent with the number
        of wavevector
    + V (float): volume
    + constant_c (float): speed of light constant
    + pol_vec (np.array, optional): polarization vector, SIZE N x 2 x 3 with N should be consistent
        with above arguments
    """
    def __init__(self, k_vector, amplitude, V, constant_c, pol_vec = None):

        super().__init__()

        self.number_modes = k_vector.shape[0]

        assert k_vector.shape == (self.number_modes, 3)
        assert isinstance(V, float)

        self.update_amplitude(amplitude)

        self.k_vector = k_vector
        self.V = V
        self.constant_c = constant_c

        if pol_vec is None:
            # if no polarization vector is provided, a set of orthonormal
            # vectors will be prepared
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

    def mode_function(self, t, R):
        """
        Mode function for free field (e.g. not confined)
        Args:
        + t (float): time
        + R (np.array): position. SIZE: N x 3 for N is number of points 
            where the field is evaluated.
        Returns:
        + np.array: SIZE: k x 2 x N x 3 with k is number of wavevector of the
            field, N is the number of points where the field are evaluated
        """

        k_vec = self.k_vector
        pol_vec = self.pol_vec

        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # free field mode function and c.c., a.k.a. exp(ikr) exp(-i \omega t) + c.c
        f_R = np.exp(
            1j * np.einsum("kj,mj->km",k_vec,R) - 1j * omega * t) # j = 3

        f_R = np.tile(f_R[:,np.newaxis,:,np.newaxis],(1,2,1,3))
        pol_vec = np.tile(pol_vec[:,:,np.newaxis,:], (1,1,len(R),1))

        f_R = f_R * pol_vec

        return f_R

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
    + constant_c (float): value of speed of light constant
    """
    def __init__(self, kappa, m, amplitude, S, L, constant_c):
        
        self.n_modes = len(kappa)
        assert kappa.shape == (self.n_modes, 2)
        assert len(m) == self.n_modes and len(m.shape) == 1
        assert amplitude.shape == (self.n_modes, 2)
        assert isinstance(S, float)
        assert isinstance(L, float)

        # kappa vector 
        self.kappa_vec = np.hstack([kappa, np.zeros((self.n_modes,1))])
        self.kappa = np.sqrt(
                np.einsum("ki,ki->k",self.kappa_vec,self.kappa_vec)
                )
        # unit vector along the kappa vector
        self.kappa_unit = self.kappa_vec / np.tile(self.kappa[:,np.newaxis], (1,3))

        # kz 
        self.kz = np.pi * m / L
        # z unit vector
        self.z_vec = np.tile(np.array([0,0,1])[np.newaxis,:], (len(self.kappa), 1))

        # k vector magnitude
        self.k = np.sqrt(self.kappa ** 2 + self.kz ** 2)
        self.omega = self.k * constant_c

        # value of amplitude/ C parameters
        self.C = amplitude

        # cavity geometry
        self.S = S
        self.L = L
        self.V = S * L

        self.constant_c = constant_c

        # constructing the TE mode polarization vector
        self.epsilon_TE = np.zeros((self.n_modes, 3))
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

        # evaluating the exp parts, j = 3, m = number of points of evaulation
        f_R = np.exp(
             1j * np.einsum("kj,mj->km",self.kappa_vec,R) - 1j * omega * t)

        fs_R = np.exp(
            -1j * np.einsum("kj,mj->km",self.kappa_vec,R) + 1j * omega * t)
        
        # TE mode
        C_TE = self.C[:,0] * np.sin(self.kz * R[:,-1])
        C_TE = np.tile(C_TE[:,np.newaxis],(1,3)) * self.epsilon_TE #k x 3

        # TM mode
        C_TM_coskz = self.C[:,1] * (self.kappa / self.k) * np.cos(self.kz * R[:,-1]) 
        C_TM_coskz = np.tile(C_TM_coskz[:,np.newaxis],(1,3)) * self.z_vec

        C_TM_sinkz = self.C[:,1] * 1j * (self.kz / self.k) * np.sin(self.kz * R[:,-1])
        C_TM_sinkz = np.tile(C_TM_sinkz[:,np.newaxis],(1,3)) * (self.kappa_unit)

        C_TM = (C_TM_coskz - C_TM_sinkz)

        C = C_TE + C_TM

        # i = 3
        A_R = np.einsum("ki,km->mi", C, f_R) \
            + np.einsum("ki,km->mi", np.conjugate(C), fs_R)

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

        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        C = self.C

        #grad_mu_r_dot = np.einsum("nij,ni->nj", gradD, R_dot) # i = j = 3

        exp_ikr = np.exp(np.einsum("ki,ni->kn",-1j * k_vec,R)) # i = 3

        Jk = np.einsum("nj,kn->kj",current,exp_ikr) # j = 3

        C_dot = np.einsum("kij,kj->ki",pol_vec, Jk) #i = 2, j = 3

        C_dot *= (2 * np.pi * 1j / k_val) * np.exp(1j * omega * t)

        return C_dot


if test:
    from utils import EM_mode_generate
    import reduced_parameter as red

    k_vec = EM_mode_generate(20)

    np.random.seed(20)
    """
    EXAMPLE CALCULATION FOR VECTOR POTENTIAL OF CAVITY FIELD
    """

    kappa = np.array([
        [1,0],
        #[0,1]
        ])

    m = np.array([1])
    print(m.shape)

    amplitude = np.array([
        np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
        #np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2)
        ])
    print(amplitude.shape)

    A = CavityFieldPotentialVector(
        kappa = kappa, m = m, amplitude = amplitude,
        S = 100.0, L = 10.0, constant_c = red.c
        )

    R = np.array([[1,1,0]])

    print(
        A(t = 0 , R = R)
        )







