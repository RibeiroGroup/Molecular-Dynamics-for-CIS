from copy import deepcopy
import numpy as np
from .utils import orthogonalize, timeit, repeat_x3

class BaseVectorPotential:
    """
    Base class for Vector Potential. 
    Attribute:
    + n_modes (int): number of modes
    + C (np.array of size n_modes x 2): ampltiude of all the modes
    + constant_c (float): value of speed of light constant
    """
    def __init__(self, k_vector, amplitude,constant_c, V, coupling_strength = 1):

        self.n_modes = len(k_vector)

        self.update_amplitude(amplitude)

        self.constant_c = constant_c

        self.k_val = np.sqrt(
                np.einsum("ni,ni->n",k_vector,k_vector)) 

        self.omega = self.k_val * constant_c

        assert isinstance(V,float)
        self.V = V

        self.history = {"t":[], "C":[], "energy":[]}

        self.coupling_strength = coupling_strength

    def update_amplitude(self, amplitude):
        assert amplitude.shape == (self.n_modes, 2)
        self.C = amplitude

    def record(self,t):

        self.history["t"].append(t)
        self.history["C"].append(deepcopy(self.C))

        self.history["energy"].append(
                self.hamiltonian(return_sum_only=False)
                )

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
        C /= np.sqrt(self.V)

        f_R = self.mode_function(t, R)

        fs_R = np.conjugate(f_R)

        A_R = np.einsum("ki,kimj->mj",C, f_R) \
            + np.einsum("ki,kimj->mj",np.conjugate(C), fs_R)

        return A_R 

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
        # change shape of k_val from K to K x 2 by tiling the last axis
        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        Jk = charge_assemble.current_mode_projection() # shape N x 3

        fs = np.conjugate(self.mode_function(t, charge_assemble.r))

        C_dot = np.einsum("mj,kimj->ki",Jk,fs)

        C_dot *= (2 * np.pi * 1j * self.coupling_strength / k_val)

        return C_dot

    def time_diff(self, t, charge_assembly, amplitude = None):
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
        C = self.C if amplitude is None else amplitude

        C_dot = self.dot_amplitude(t, charge_assembly)
        omega = np.tile(self.omega[:,np.newaxis], (1,2))

        dA = self.__call__(
                t, charge_assembly.r, 
                amplitude = -1j * omega * C + C_dot)

        return dA

    def gradient(self, t, R, amplitude = None):
        """
        Calculate the gradient of the vector potential (gradA)_(ij) = dA_i/dr_j
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; 
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        C = self.C if amplitude is None else amplitude
        C = C / np.sqrt(self.V)
        
        gradf_R = self.grad_mode_func(t, R)
        gradfs_R = np.conjugate(gradf_R)
        
        gradA_R = np.einsum("ki,lkimj->mjl", C, gradf_R) \
            + np.einsum("ki,lkimj->mjl", np.conjugate(C), gradfs_R) \

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

    def force(self, t, charge_assemble, amplitude = None):
        """
        Computing the force exerting by the electromagnetic field on 
        the charge particle/ dipole
        Args:
        + t (float): time
        + charge_assemble (Python object): object wrapper for charged particle
        or dipole. It should have the following attributes: 
            1/ r (np.array of size N x 3) for position
            2/ r_dot (np.array of size N x 3) for velocity
        and method "charge" which take no arguments and return np.array
        of size N x 3 x 3
        """

        dAdt = self.time_diff(t,charge_assemble, amplitude = amplitude)
        gradA = self.gradient(t,charge_assemble.r, amplitude = amplitude)

        r_dot = charge_assemble.r_dot
        q = charge_assemble.charge()

        force1 = np.einsum("nlj,nl->nj",q, r_dot)
        force1 = np.einsum("nj,nji->ni",force1,gradA)

        force2 = np.einsum("nj,nji->ni",dAdt, q)

        force3 = np.einsum("njl,nl->nj",gradA,r_dot)
        force3 = np.einsum("nj,nij->ni",force3,q)

        force = force1 - force2 - force3
        force *= self.coupling_strength
        force /= self.constant_c

        return force

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
    def __init__(self, k_vector, amplitude, V, constant_c, pol_vec = None, coupling_strength = 1):

        super().__init__(k_vector, amplitude, constant_c,V, coupling_strength)

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

    def grad_mode_func(self, t, R):

        k_vec = self.k_vector

        C = self.C / np.sqrt(self.V)
        
        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # evaluate mode function at time t and positions vector R
        f_R = self.mode_function(t, R)

        # l = 3
        gradf_R = np.einsum("kimj,kl->lkimj", f_R, 1j * self.k_vector )

        return gradf_R

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
    def __init__(self, kappa, m, amplitude, S, L, constant_c, coupling_strength = 1):

        assert kappa.shape[0] == m.shape[0]
        assert kappa.shape[1] == 2 and len(m.shape) == 1

        # kappa vector 
        self.kappa_vec = np.hstack([kappa, np.zeros((len(kappa),1))])
        self.kappa = np.sqrt(
                np.einsum("ki,ki->k",self.kappa_vec,self.kappa_vec)
                )
        # kz 
        self.kz = 2 * np.pi * m / L

        #k_vector in general
        self.k_vector = np.hstack([kappa, self.kz[:,np.newaxis] ])
        
        # cavity geometry
        assert isinstance(S, float)
        assert isinstance(L, float)

        self.S = S
        self.L = L

        super().__init__(self.k_vector, amplitude, constant_c, 
                V = S * L, coupling_strength = coupling_strength)
        #print("Warning, the volume is set to 1")

        # calculating unit vector
        # unit vector along the z axis
        self.z_vec = np.vstack([np.array([0,0,1]) for i in range(self.n_modes)])

        self.kappa_unit = [] # unit vector along kappa
        self.eta = []        # unit vector kappa x z_vec (cross product)

        for i, k in enumerate(self.kappa_vec):
            if self.kappa[i] == 0.0:
                self.kappa_unit.append(np.array([1,0,0]))
                self.eta.append(np.array([0,1,0]))
            else:
                self.kappa_unit.append(k / self.kappa[i])

                # eta = kappa_unit x z_unit (x = cross product)
                eta = np.zeros(3)
                eta[0] =  k[1] / self.kappa[i]
                eta[1] = -k[0] / self.kappa[i]

                self.eta.append(eta)

        self.kappa_unit = np.vstack(self.kappa_unit)
        self.eta = np.vstack(self.eta)

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

        # calculating the (kappa / k) . z_vec       and tiling the result
        z_vec = self.z_vec * repeat_x3(self.kappa / self.k_val)
        z_vec = np.tile(z_vec[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        # calculating the (   kz / k) . kappa_unit  and tiling the result
        kappa = repeat_x3(self.kz / self.k_val) * self.kappa_unit
        kappa = np.tile(kappa[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        f_tm = coskz * z_vec - 1j * sinkz * kappa

        f_r = np.concatenate((f_te, f_tm), axis = 1)
        return f_r

    def grad_mode_func(self, t, R):
        """
        Calcuate the gradient of the vector potential (gradA)_(ij) = dA_i/dr_j
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; 
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        k_vec = self.k_vector
        kz = np.tile(self.kz[:,np.newaxis], (1,len(R)))

        C = self.C / np.sqrt(self.V)
        
        omega = np.tile(self.omega[:,np.newaxis], (1, R.shape[0]))

        # evaluate mode function at time t and positions vector R
        gradf_R = self.mode_function(t, R)

        # calculating the gradient along the x and y axis (kappa vector) 
        # l = 2 as differentiating along the z axis is ommitted
        gradf_R = np.einsum("kimj,kl->lkimj", gradf_R, 1j * self.kappa_vec[:,:2] )

        #calculating the gradient along the z axis
        expkz = np.exp(
            1j * np.einsum("kj,mj->km",self.kappa_vec,R) - 1j * omega * t
            )

        # differentiating TE mode  
        gradf_te = expkz * kz * np.cos(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel()))

        gradf_te = np.tile(gradf_te[:,np.newaxis,:,np.newaxis], (1,1,1,3))
        eta = np.tile(self.eta[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        gradf_te = gradf_te * eta

        # differentiating TM mode
        # differentiating cos part
        grad_coskz = kz * np.sin(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel())
            ) * expkz
        grad_coskz = np.tile(grad_coskz[:,np.newaxis,:,np.newaxis],(1,1,1,3))

        # differentiating sin part
        grad_sinkz = kz * np.cos(
            np.einsum("k,m->km",self.kz,R[:,-1].ravel())
            ) * expkz
        grad_sinkz = np.tile(grad_sinkz[:,np.newaxis,:,np.newaxis],(1,1,1,3))

        # calculating the (kappa / k) . z_vec       and tiling the result
        z_vec = self.z_vec * repeat_x3(self.kappa / self.k_val)
        z_vec = np.tile(z_vec[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        # calculating the (   kz / k) . kappa_unit  and tiling the result
        kappa = repeat_x3(self.kz / self.k_val) * self.kappa_unit
        kappa = np.tile(kappa[:,np.newaxis,np.newaxis,:], (1,1,len(R),1))

        gradf_tm = (grad_coskz * z_vec - 1j * grad_sinkz * kappa)

        dz_fr = np.concatenate((gradf_te, gradf_tm), axis = 1)
        dz_fr = dz_fr[np.newaxis,:,:,:,:]

        gradf_R = np.concatenate((gradf_R, dz_fr), axis = 0)

        return gradf_R




