import numpy as np
from utils import orthogonalize, timeit, repeat_x3

test = False

class BaseVectorPotential:
    """
    Base class for Vector Potential. 
    """
    def __init__(self, k_vector, amplitude,constant_c,V):

        self.n_modes = len(k_vector)

        self.update_amplitude(amplitude)

        self.constant_c = constant_c

        self.k_val = np.sqrt(
                np.einsum("ni,ni->n",k_vector,k_vector)) 

        self.omega = self.k_val * constant_c

        assert isinstance(V,float)
        self.V = V

    def update_amplitude(self, amplitude):
        assert amplitude.shape == (self.n_modes, 2)
        self.C = amplitude

    def mode_function(self, t, R):
        pass

    def __call__(self, t, R, amplitude = None):
        """
        Evaluate the vector potential at time t and multiple positions specified in R.
        The inherited class need to have the 'mode_function' method, which take in two
        variables 't' and 'R' and return numpy array of shape k x 2 x N x 3 for k and N are
        the number of field modes and the number of points where the field are evaluated
        respectively.
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
            projected by TE or TM mode function. In addition, it must have
            attribute 'r' for array of position vectors.

        """
        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        Jk = charge_assemble.current_mode_projection() # shape N x 3

        fs = np.conjugate(self.mode_function(t, charge_assemble.r))

        C_dot = np.einsum("mj,kimj->ki",Jk,fs)

        C_dot *= (2 * np.pi * 1j / k_val)

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
            projected by TE or TM mode function. In addition, it must have
            attribute 'r' for array of position vectors.
        """
        C_dot = self.dot_amplitude(t, charge_assembly)
        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        dA = self.__call__(
                t, charge_assembly.r, 
                amplitude = -1j * omega * self.C + C_dot)

        return dA

    def gradient(self, t, R):
        """
        Calcuate the gradient of the vector potential (gradA)_(ij) = dA_i/dr_j
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; 
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        k_vec = self.k_vector

        C = self.C
        
        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # evaluate mode function at time t and positions vector R
        f_R = self.mode_function(t, R)
        fs_R = np.conjugate(f_R)

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        kC = np.einsum("ki,kl->kil", C, 1j * k_vec)

        #
        gradA_R = np.einsum("kil,kimj->mjl",kC, f_R) \
            + np.einsum("kil,kimj->mjl",np.conjugate(kC), fs_R)

        return gradA_R

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

class FreeVectorPotential(BaseVectorPotential):
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

        super().__init__(k_vector, amplitude, constant_c,V)

        self.k_vector = k_vector
        self.mode_projection = None

        if pol_vec is None:
            # if no polarization vector is provided, a set of orthonormal
            # vectors will be prepared
            self.pol_vec = []

            for k_vec in self.k_vector:
                self.pol_vec.append(orthogonalize(k_vec)[1:,:])

            self.pol_vec = np.array(self.pol_vec)

        else:
            self.pol_vec = pol_vec

        assert self.pol_vec.shape == (self.n_modes, 2, 3)

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

class CavityVectorPotential(BaseVectorPotential):
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

        assert kappa.shape[0] == m.shape[0]
        assert kappa.shape[1] == 2 and len(m.shape) == 1

        # kappa vector 
        self.kappa_vec = np.hstack([kappa, np.zeros((len(kappa),1))])
        self.kappa = np.sqrt(
                np.einsum("ki,ki->k",self.kappa_vec,self.kappa_vec)
                )
        # unit vector along the kappa vector SIZE: k x 3
        self.kappa_unit = self.kappa_vec / np.tile(self.kappa[:,np.newaxis], (1,3))

        # kz 
        self.kz = 2 * np.pi * m / L
        # z unit vector
        self.z_vec = np.tile(np.array([0,0,1])[np.newaxis,:], (len(self.kappa), 1))

        #k_vector in general
        self.k_vector = np.hstack([kappa, self.kz[:,np.newaxis] ])
        
        # cavity geometry
        assert isinstance(S, float)
        assert isinstance(L, float)

        self.S = S
        self.L = L

        super().__init__(self.k_vector, amplitude, constant_c, V = 1.0)
        print("Warning, the volume is set to 1")

        # eta = kappa_unit x z_unit (x = cross product)
        # size = N_modes x 3
        self.eta = np.zeros((self.n_modes, 3))
        self.eta[:,0] = self.kappa_vec[:,1] / self.kappa
        self.eta[:,1] = -self.kappa_vec[:,0] / self.kappa

    def mode_function(self, t, R):
        omega = np.tile(self.omega[:,np.newaxis],(1,len(R)))

        expkz = np.exp(
            1j * np.einsum("kj,mj->km",self.kappa_vec,R) - 1j * omega * t
            )

        # TE mode  
        f_te = expkz * np.sin(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel()))

        f_te = np.tile(f_te[:,np.newaxis,:,np.newaxis], (1,1,1,3))
        eta = np.tile(self.eta[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        f_te = f_te * eta

        #TM mode
        coskz = np.cos(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel())
            ) * expkz
        coskz = np.tile(coskz[:,np.newaxis,:,np.newaxis],(1,1,1,3))

        sinkz = np.sin(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel())
            ) * expkz
        sinkz = np.tile(sinkz[:,np.newaxis,:,np.newaxis],(1,1,1,3))

        z_vec = self.z_vec * repeat_x3(self.kappa / self.k_val)
        z_vec = np.tile(z_vec[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        kappa = repeat_x3(self.kz / self.k_val) * self.kappa_unit
        kappa = np.tile(kappa[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        f_tm = coskz * z_vec - 1j * sinkz * kappa

        f_r = np.concatenate((f_te, f_tm), axis = 1)
        return f_r
         
def single_eval_cavityA(kappa_vec, kz, C, t, R, constant_c, V):

    kappa_vec = np.hstack([kappa_vec, [0]])
    kappa = np.sum(kappa_vec * kappa_vec)
    k_val = np.sqrt(kappa**2 + kz**2)
    omega = constant_c * k_val

    kappa_unit = kappa_vec / kappa
    eta = np.array([kappa_unit[1], -kappa_unit[0], 0])
    z_vec = np.array([0,0,1])
    
    exp_kr = np.exp(
        1j * np.sum(kappa_vec * R) - 1j * omega * t
        )

    #TE mode
    f_te = np.sin(kz * R[-1]) * exp_kr * eta
    print(f_te)
    fs_te = np.conjugate(f_te)

    #TM mode
    f_tm = kappa / k_val * np.cos(kz * R[-1]) * exp_kr * z_vec
    f_tm -= 1j * kz / k_val * np.sin(kz * R[-1]) * exp_kr * kappa_unit
    print(f_tm)
    fs_tm = np.conjugate(f_tm)

    A = C[0] * f_te + np.conjugate(C[0]) * fs_te
    A += C[1] * f_tm + np.conjugate(C[1]) * fs_tm

    return A / np.sqrt(V)

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

    amplitude = np.array([
        np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
        #np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2)
        ])
    print(amplitude)

    A = CavityPotentialVector(
        kappa = kappa, m = m, amplitude = amplitude,
        S = 100.0, L = 10.0, constant_c = red.c
        )

    """
    R = np.vstack([
        np.array([[1,1,0],[1,1,5]]),
        np.random.uniform(size = (3,3))
        ])
    """
    R = np.array([[1,1,1]])

    print(
        A(t = 0 , R = R)
        )







