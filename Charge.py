import numpy as np

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

