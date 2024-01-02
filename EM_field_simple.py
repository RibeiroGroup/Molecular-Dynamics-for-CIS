import numpy as np
import matplotlib.pyplot as plt

import constants

"""
## !!!! ### 
Jk calculation for single charge only
"""

class SingleModeField:
    def __init__(self,C,k,epsilon):

        assert k.shape[-1] == 3
        self.k = k.reshape(3)

        assert epsilon.shape == (2,3)
        self.epsilon = epsilon.reshape(2,3)

        self.update(C)

        self.omega = constants.c * (self.k @ self.k)

    def update(self,C):
        assert C.shape[-1] == 2
        self.C = C.reshape(2)

    def get_exp_ikx(self,x):
        kx = self.k @ x
        return np.exp(1j * kx)

    def __call__(self, x, C = None):
        if C is not None:
            assert isinstance(C,np.ndarray) and C.shape[-1] == 3
        else: 
            C = self.C

        e_ikx = self.get_exp_ikx(x)

        Ckx_e1 = self.C[0] *  e_ikx\
            + np.conjugate(C[0]) * np.conjugate(e_ikx)
        
        Ckx_e2 = self.C[1] * e_ikx \
            + np.conjugate(C[1]) * np.conjugate(e_ikx)

        Ckx = Ckx_e1 * self.epsilon[0] + Ckx_e2 * self.epsilon[1]

        return Ckx

    def d_dx(self,x):
        return np.einsum("i,j->ij", self.__call__(x), self.k)

    def partial_partial_t(self, x, charge):
        C = -1j * self.omega * self.C
        jk = 2 * np.pi * 1j * constants.c * charge.get_jk(self) / self.omega 

        for i,epsilon in enumerate(self.epsilon):
            C[i] += epsilon * jk[i]

        return C

    def project_transverse(self

class ChargePoint:
    def __init__(self, m, q, r, v):
        self.m = m
        self.q = q
        self.r = r
        self.v = v

    def get_kinetic_energy(self):
        return 0.5 * self.m * self.v**2

    def update(self,r=None,v=None):
        if r is not None:
            self.r = r 
        if v is not None:
            self.v = v

    def get_jk(self, A, transverse = True):
        e_ikx = np.conjugate(A.get_exp_ikx(self.r))

        jk = (2 * np.pi)**(-1.5) * e_ikx * self.q * self.v

        if transverse: jk = A.project_transverse(jk)

        return jk

class ChargeCluster:
    def __init__(self, m, q, r, v):
        self.verify(m,1)
        self.verify(q,1)
        self.verify(r)
        self.verify(v)

        self.charge_points = []
        for i in range(self.n_charge_points):
            self.charge_points.append(
                ChargePoint(m[i],q[i],r[i],v[i])
            ) 

    def verify(self, array, size = 3):
        array = np.array(array)
        try:
            assert array.shape[1] == size
            assert array.shape[0] == self.n_charge_points
        except AttributeError:
            self.n_charge_points = array.shape[0]

    def get_jk(self,A, transverse = True):
        Jk = np.sum([
            point.get_jk(A,False) for point in self.charge_points]) 

        if transverse: Jk = A.project_transverse(Jk)

        return Jk

class EMHamiltonian:
    """
    Args:
    + A: SingleModeField or Python tuple of C, k, epsilon
    + charge_points: Python list of ChargePoint
    """
    def __init__(self, A, charge_points):
        if isinstance(A, tuple):
            self.A = SingleModeField(*A)

        assert isinstance(charge_points,ChargeCluster) \
            or isinstance(charge_points, ChargePoint)

        self.charge_points = charge_points
    
    def __call__(self):
        self.H_mat = 0
        for point in self.charge_points:
            self.H_mat += point.get_kinetic_energy()

        self.H_em = 1/(2*np.pi) * A.omega**2 / constant.c**2 \
            * float(A.C @ A.C)

        return self.H_mat, self.H_em

    def d_dq(self):
        result = 0
        for point in self.charge_points: 
            result += point.q * constants.c/point.m \
                * ( point.v @ A(point.r) )

        return k * result

            




