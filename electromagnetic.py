import numpy as np
import numpy.linalg as la

from utils import PBC_wrapping, orthogonalize
#import constants as constants

run_test = 1

class VectorPotential:
    def __init__(self, k_vector, amplitude, V, speed_of_light):# = self.v_light):
        """
        Class for computing potential vector A of the field, its amplitude derivative,
        and transverse force on (charged) atoms exert by the field.
        Args:
        + k_vector (np.array): should have shape N_modes x 3 x 3
            where the 1st dim -> mode, 2nd dim -> k vector, 1st & 2nd 
            polarization vector and the last dim is vector length (coordinate)
        """
        self.v_light = speed_of_light

        self.V = V

        self.N_modes = len(k_vector)

        if k_vector.shape == (self.N_modes, 3, 3):
            self.k_vector = k_vector
        else:
            assert k_vector.shape == (self.N_modes, 3)
            self.k_vector = np.array([
                orthogonalize(kvec) for kvec in k_vector
                ]) # each k_vector should have shape N_mode x 3 x 3

        assert amplitude.shape == (self.N_modes, 2)

        self.update_amplitude(amplitude=amplitude)

        k_vec = self.k_vector[:,0,:]

        self.k_val = np.sqrt(np.einsum("kj,kj->k",k_vec,k_vec))
        self.omega = np.array(self.k_val * self.v_light, dtype=np.complex128)

    def update_amplitude(self,amplitude = None, deltaC = None):

        assert amplitude is not None or deltaC is not None
        assert amplitude is None or deltaC is None

        if amplitude is not None:
            self.C = amplitude
        elif deltaC is not None:
            self.C += deltaC

    def __call__(self, R, C = None):
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]

        if C is None: C = self.C
        
        # free field mode function and c.c., a.k.a. exp(ikr)
        f_R = np.exp(
            1j * np.einsum("kj,nj->kn",k_vec,R))

        fs_R = np.exp(
            -1j * np.einsum("kj,nj->kn",k_vec,R))

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki" ,C ,pol_vec)

        #
        A_R = np.einsum("ki,kn->ni",C_epsilon_k, f_R) \
            + np.einsum("ki,kn->ni",np.conjugate(C_epsilon_k), fs_R)

        return A_R

    def gradient(self, R, C=None):
        """
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; (gradA)_(ij) = dA_i/dr_j
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]

        C = C if C is not None else self.C

        # free field mode function and c.c., a.k.a. exp(ikr)
        f_R = 1j *  np.exp(
            1j * np.einsum("kj,nj->kn",k_vec,R))

        fs_R = -1j * np.exp(
            -1j * np.einsum("kj,nj->kn",k_vec,R))

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki",C,pol_vec)

        k_o_C_epsilon_k = np.einsum("kj,ki->kij",k_vec,C_epsilon_k)

        #
        gradA_R = np.einsum("kij,kn->nij",k_o_C_epsilon_k, f_R) \
            + np.einsum("kij,kn->nij",np.conjugate(k_o_C_epsilon_k), fs_R)

        return gradA_R

    def Jk_perp(self, R, R_dot, gradD):
        """
        Args:
        + R (np.ndarray of shape N x 3):
        + R_dot (np.ndarray of shape N x 3):
        + gradD (np.array): shape N x 3 x 3 where the i-th 3x3 matrix is 
            the total gradient of dipole of all N atoms with the i-th atom
        """
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]
        omega = np.tile(self.omega[:,np.newaxis],(1,2))
        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        grad_mu_r_dot = np.einsum("nij,ni->nj", gradD, R_dot)

        exp_ikr = np.exp(np.einsum("ki,ni->kn",-1j * k_vec,R))

        grad_mu_eikr = np.einsum("nj,kn->kj",grad_mu_r_dot,exp_ikr)

        Jk_perp = np.einsum("kij,kj->ki",pol_vec,grad_mu_eikr)

        return Jk_perp

    def dot_C(self, R, R_dot, gradD, C = None):
        omega = np.tile(self.omega[:,np.newaxis],(1,2))
        k_val = np.tile(self.k_val[:,np.newaxis],(1,2))

        C = C if C is not None else self.C

        C_dot = 2 * np.pi * 1j / ( k_val * self.V ) \
                * self.Jk_perp(R, R_dot, gradD)

        C_dot -= 1j * omega * C

        return C_dot

    def mid_point_update(self, 
            R, R_new, R_dot, R_dot_new, gradD, gradD_new,
            h, C = None, inplace = True):

        omega = np.tile(self.omega[:,np.newaxis],(1,2))

        J = self.Jk_perp(R, R_dot, gradD)
        J_new = self.Jk_perp(R_new, R_dot_new, gradD_new)

        C_new = (h * np.pi * 1j / self.k) * (J_new + J)
        C_new += C
        C_new -= (h/2) * 1j * omega * C
        C_new /= (1 + 1j * omega * h/2)
        
        if inplace: self.C = C_new

        return C_new

    def time_diff(self, R, R_dot, C, gradD):

        C_dot = self.dot_C(R,R_dot,C=C,gradD=gradD)
        return  self.__call__(R, C_dot)

    def transv_force(self, R, R_dot, C, gradD):
        """
        Computing the transverse force exerting on the charged atoms by the field.
        The output have the shape (N x 3) 
        """
   
        gradA = self.gradient(R, C = C)

        dAdt = self.time_diff(R, R_dot, C=C, gradD=gradD)

        # S_l S_j (dA_j/dr_i) (dmu_j/dr_l) rdot_l
        # (gradA)_(ji) = dA_j/dr_i
        # (D_r mu)_(lj) = dmu_j / dr_l

        force1 = np.einsum("nlj,nl->nj",gradD, R_dot)
        force1 = np.einsum("nj,nji->ni",force1,gradA)

        # S_j (dA_j/dt) (dmu_j/dr_i)
        # (D_r mu)_(ij) = dmu_j / dr_i

        force2 = np.einsum("nj,nji->ni",dAdt, gradD)

        # S_l S_j (dA_j/dr_l) (dmu_j/dr_i) rdot_l
        # (gradA)_(jl) = dA_j/dr_l
        # (D_r mu)_(ij) = dmu_j / dr_i

        force3 = np.einsum("njl,nl->nj",gradA,R_dot)
        force3 = np.einsum("nj,nij->ni",force3,gradD)

        force1 /= self.v_light
        force2 /= self.v_light
        force3 /= self.v_light
        force = force1 - force2 - force3

        #return force1, force2, force3
        return np.real(force)

    def hamiltonian(self, return_sum_only = False):
        k_vector = self.k_vector[:,0,:]

        k_sum = np.einsum("ki,ki->k",k_vector,k_vector)
        c_sum = np.einsum("ki,ki->k",self.C,np.conjugate(self.C))

        H = (2*np.pi)**-1 * self.V * k_sum * c_sum

        if return_sum_only:
            return np.sum(H)

        return H

if run_test == True:
    import time
    from dipole import SimpleDipoleFunction
    from distance import DistanceCalculator
    from parameter import mu0, a, d0
    from test.electromagnetic import ExplicitTestVectorPotential
    from test.dipole import ExplicitTestDipoleFunction
    import input_dat
    from constants import c as speed_of_light

    np.random.seed(2)
    #k_vector = np.random.uniform(-5,5,(10,3)) 
    #k_vector = np.array([
    #    orthogonalize(kvec) for kvec in k_vector
    #    ]) 
    k_vector = input_dat.k_vec

    amplitudes = input_dat.C#np.random.rand(len(k_vector),2) + np.random.rand(len(k_vector),2) * 1j

    L = input_dat.L
    R = np.vstack([input_dat.r_xe,input_dat.r_ar])

    V = np.vstack([input_dat.v_xe,input_dat.v_ar])

    N_atoms = R.shape[0]

    XeIdx = np.hstack([np.ones(int(N_atoms/2)),np.zeros(int(N_atoms/2))])
    ArIdx = np.hstack([np.zeros(int(N_atoms/2)),np.ones(int(N_atoms/2))])

    AField = VectorPotential(k_vector, amplitudes, speed_of_light = speed_of_light, V = 1)
    
    AFieldTest = ExplicitTestVectorPotential(k_vector, amplitudes)

    print("+++ Difference between VectorPotential class and ExpliciTest for A evaluation +++")
    print(np.sum(abs(AFieldTest(R) - AField(R))))

    print("+++ Difference between VectorPotential class and ExpliciTest for A gradient +++")
    print(np.sum(abs(AFieldTest.gradient(R) - AField.gradient(R))))

    distance_calc = DistanceCalculator(n_points=N_atoms,box_length=L)
    dipole_func = SimpleDipoleFunction(distance_calc, XeIdx, ArIdx, mu0=mu0, a=a,d0=d0)

    dipole_func_ = ExplicitTestDipoleFunction(XeIdx ,ArIdx, mu0=mu0, a=a,d0=d0,L=L)

    gradD = dipole_func.gradient(R, return_all = True)
    gradD_sum = np.sum(gradD, axis = 1)

    print(gradD[0,15])

    C_dot = AField.dot_C(R, V, gradD_sum)
    C_dot_ = AFieldTest.dot_C(R,V,gradD)
    print(C_dot_)

    print("+++ Difference between VectorPotential class and ExpliciTest for time derivative of A +++")
    print(np.sum(abs(C_dot - C_dot_)))

    print("+++ Field Hamiltonian computation")
    Hem = AField.hamiltonian()
    Hem_ = AFieldTest.compute_Hem()

    print(Hem)

    print("+++ Transverse force computation test:")

    start = time.time()
    force = AField.transv_force(R,V, gradD=gradD_sum, C = AField.C)
    print("Runtime(s): ",time.time() - start)

    start = time.time()
    force_  = AFieldTest.transv_force(R,V, dipole_func_)
    print("Explicit computation runtime(s): ",time.time() - start)

    print("Sum of abs difference: ",np.sum(abs(force - force_)))

        
