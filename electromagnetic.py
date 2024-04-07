import numpy as np
import numpy.linalg as la

from utils import PBC_wrapping, orthogonalize
import constants

run_test = True

class VectorPotential:
    def __init__(self, k_vector, amplitudes):
        """
        Class for computing potential vector A of the field, its amplitude derivative,
        and transverse force on (charged) atoms exert by the field.
        Args:
        + k_vector (np.array): should have shape N_modes x 3 x 3
            where the 1st dim -> mode, 2nd dim -> k vector, 1st & 2nd 
            polarization vector and the last dim is vector length (coordinate)
        """

        self.N_modes = len(k_vector)

        if k_vector.shape == (self.N_modes, 3, 3):
            self.k_vector = k_vector
        else:
            assert k_vector.shape == (self.N_modes, 3)
            self.k_vector = np.array([
                orthogonalize(kvec) for kvec in k_vector
                ]) # each k_vector should have shape N_mode x 3 x 3

        assert amplitudes.shape == (self.N_modes, 2)

        self.C = amplitudes

        k_vec = self.k_vector[:,0,:]

        self.k_val = np.sqrt(np.einsum("kj,kj->k",k_vec,k_vec))
        self.omega = np.array(self.k_val * constants.c, dtype=np.complex128)

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

    def gradient(self, R):
        """
        |Ax.kx   Ax.ky   Ax.kz|    |dAx/dx   dAx/dy   dAx/dz|
        |Ay.kx   Ay.ky   Az.kz| == |dAy/dx   dAy/dy   dAz/dz|; (gradA)_(ij) = dA_i/dr_j
        |Az.kx   Az.ky   Az.kz|    |dAz/dx   dAz/dy   dAz/dz|
        """
        k_vec = self.k_vector[:,0,:]
        pol_vec = self.k_vector[:,1:,:]
        
        # free field mode function and c.c., a.k.a. exp(ikr)
        f_R = 1j *  np.exp(
            1j * np.einsum("kj,nj->kn",k_vec,R))

        fs_R = -1j * np.exp(
            -1j * np.einsum("kj,nj->kn",k_vec,R))

        #Multiply C and epsilon_k (pol_vec), the outcome shape is N_modes x 3
        # (sum over dim 2, which is the number of polarized vector)
        C_epsilon_k = np.einsum("kj,kji->ki",self.C,pol_vec)

        k_o_C_epsilon_k = np.einsum("kj,ki->kij",k_vec,C_epsilon_k)

        #
        gradA_R = np.einsum("kij,kn->nij",k_o_C_epsilon_k, f_R) \
            + np.einsum("kij,kn->nij",np.conjugate(k_o_C_epsilon_k), fs_R)

        return gradA_R

    def dot_C(self, R, R_dot, gradD):
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

        C_dot = np.einsum("kij,kj->ki",pol_vec,grad_mu_eikr)

        C_dot *= (2 * np.pi * 1j / k_val)
        C_dot -= 1j * omega * self.C

        return C_dot

    def time_diff(self, R, R_dot, gradD=None, C_dot=None):
        assert C_dot is not None or gradD is not None

        if C_dot is None:
            C_dot = self.dot_C(R, R_dot, gradD)

        return  self.__call__(R, C_dot)

    def transv_force(self, R, R_dot, gradD, C_dot = None):
        """
        Computing the transverse force exerting on the charged atoms by the field.
        The output have the shape (N x 3) 
        """
        gradA = self.gradient(R)
        dAdt = self.time_diff(R, R_dot, C_dot=C_dot, gradD=gradD)

        # S_l S_j (dA_j/dr_i) (dmu_j/dr_l) rdot_l
        # (gradA)_(ji) = dA_j/dr_i
        # (D_r mu)_(lj) = dmu_j / dr_l

        force1 = np.einsum("nlj,nl->nj",gradD, R_dot)
        force1 = np.einsum("nj,nji->ni",force1,gradA)

        # S_j (dA_j/dt) (dmu_j/dr_i)
        # (D_r mu)_(ij) = dmu_j / dr_i

        force2 = np.einsum("nj,nij->ni",dAdt, gradD)

        # S_l S_j (dA_j/dr_l) (dmu_j/dr_i) rdot_l
        # (gradA)_(jl) = dA_j/dr_l
        # (D_r mu)_(ij) = dmu_j / dr_i

        force3 = np.einsum("njl,nl->nj",gradA,R_dot)
        force3 = np.einsum("nj,nij->ni",force3,gradD)

        force1 /= constants.c
        force2 /= constants.c
        force3 /= constants.c
        force = force1 - force2 - force3

        #return force1, force2, force3
        return force

if run_test == True:
    from dipole import SimpleDipoleFunction
    from distance import DistanceCalculator
    from parameter import mu0_1, a1, d0_1
    from test.electromagnetic import ExplicitTestVectorPotential

    np.random.seed(2)

    k_vector = np.array([[[1.0,0.0,0.0], [0.0,1.0,0.0], [0.0,0.0,1.0]]])
    # np.random.uniform(-5,5,(10,3)) 
    #k_vector = np.array([
    #    orthogonalize(kvec) for kvec in k_vector
    #    ]) 

    amplitudes = np.random.rand(len(k_vector),2) + np.random.rand(len(k_vector),2) * 1j

    L = 40
    N_atoms = 20
    R = np.random.uniform(-L/2,L/2,size = (N_atoms, 3))
    V = np.random.uniform(-L/2,L/2,size = (N_atoms, 3))
    ArIdx = np.hstack([np.ones(int(N_atoms/2)),np.zeros(int(N_atoms/2))])
    XeIdx = np.hstack([np.zeros(int(N_atoms/2)),np.ones(int(N_atoms/2))])

    AField = VectorPotential(k_vector, amplitudes)
    
    AFieldTest = ExplicitTestVectorPotential(k_vector, amplitudes)

    print("+++ Difference between VectorPotential class and ExpliciTest for A evaluation +++")
    print(np.sum(abs(AFieldTest(R) - AField(R))))

    print("+++ Difference between VectorPotential class and ExpliciTest for A gradient +++")
    print(np.sum(abs(AFieldTest.gradient(R) - AField.gradient(R))))

    distance_calc = DistanceCalculator(n_points=N_atoms,box_length=L)
    dipole_func = SimpleDipoleFunction(distance_calc, ArIdx, XeIdx, mu0=mu0_1, a=a1,d0=d0_1)

    gradD = dipole_func.gradient(R)

    C_dot = AField.dot_C(R, V, gradD)
    C_dot_ = AFieldTest.dot_C(R,V,gradD)

    print("+++ Difference between VectorPotential class and ExpliciTest for time derivative of A +++")
    print(np.sum(abs(C_dot - C_dot_)))

    """
    f1,f2,f3 = AField.transv_force(R,V, gradD=gradD,C_dot = C_dot)
    f1_, f2_, f3_ = AFieldTest.transv_force(R,V, gradD=gradD)

    foo1 = f1 - f1_
    foo2 = f2 - f2_
    foo3 = f3 - f3_

    print(f3)
    print(f3_)
    """
    force = AField.transv_force(R,V, gradD=gradD,C_dot = C_dot)
    force_  = AFieldTest.transv_force(R,V, gradD=gradD)
    print(force - force_)

        
