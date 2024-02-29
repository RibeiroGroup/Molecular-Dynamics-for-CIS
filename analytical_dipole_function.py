import numpy as np
import matplotlib.pyplot as plt
import sympy as sm

from utils import timeitsm_Rax = sm.symbols("Rax")

sm_Ray = sm.symbols("Ray")
sm_Raz = sm.symbols("Raz")
sm_Ra = [sm_Rax, sm_Ray, sm_Raz]

sm_Rbx = sm.symbols("Rbx")
sm_Rby = sm.symbols("Rby")
sm_Rbz = sm.symbols("Rbz")
sm_Rb = [sm_Rbx, sm_Rby, sm_Rbz]

sm_mu0 = sm.symbols("mu0")
sm_a = sm.symbols("a")
sm_L = sm.symbols("L")
sm_d0 = sm.symbols("d0")
sm_d1 = sm.symbols("d1")
sm_d2 = sm.symbols("d2")
sm_d7 = sm.symbols("d7")

class BaseDipoleFunction:
    def __init__(self):
        
        self.dipole_function = [
            sm.lambdify(
            [sm_Rax, sm_Ray, sm_Raz, sm_Rbx, sm_Rby, sm_Rbz, 
             ] + self.parameters_sm, 
                dipole_exp
        ) for dipole_exp in self.dipole_exp]
        
        self.generate_Jacobi()
        #self.generate_Hessian()

    def generate_Jacobi(self):
        self.J_exp = [[],[],[]]
        self.J_func = [[],[],[]]

        for i,Ra in enumerate(sm_Ra):
            for j,dipole_f in enumerate(self.dipole_exp):
                d_mu = sm.diff(dipole_f, Ra)
                self.J_exp[i].append(d_mu)
                self.J_func[i].append(sm.lambdify(
                    [sm_Rax, sm_Ray, sm_Raz, sm_Rbx, sm_Rby, sm_Rbz, 
                    ]+ self.parameters_sm, d_mu
                ))

    def generate_Hessian(self):
        self.H_exp = [[[],[],[]],[[],[],[]],[[],[],[]]]
        self.H_function = [[[],[],[]],[[],[],[]],[[],[],[]]]
        for i, Ra_i in enumerate(sm_Ra):
            for j,mu_j in enumerate(self.dipole_exp):
                for k,Ra_k in enumerate(sm_Ra):
                    derivative = sm.diff(sm.diff(mu_j, Ra_i),Ra_k)
                    self.H_exp[i][j].append(derivative)
                    self.H_function[i][j].append(sm.lambdify(
                        [sm_Rax, sm_Ray, sm_Raz, sm_Rbx, sm_Rby, sm_Rbz, 
                        ] + self.parameters_sm, derivative))
    
    def __call__(self, ra, rb):
        ra = list(ra)
        rb = list(rb)

        args = ra + rb + self.parameters

        result = [dipole_f(*args) for dipole_f in self.dipole_function]
        
        return result

    def gradient(self,ra,rb):
        args = ra + rb + self.parameters
        result = np.zeros((3,3))
        for i, dmu_dRa in enumerate(self.J_func):
            for j, dmu_i_dRa in enumerate(dmu_dRa):
                result[i,j] = dmu_i_dRa(*args)

        return result

    def hessian(self,ra,rb):
        args = ra + rb + self.parameters 
        result = np.zeros((3,3,3))
        for i, d_dmu_dRai in enumerate(self.H_function):
            for j, d2muj_dRaidRa in enumerate(d_dmu_dRai):
                for k, d2muj_dRaidRak in enumerate(d2muj_dRaidRa):
                    result[i,j] = d2muj_dRaidRak(*args)
        return result

class LevineDipoleFunction(BaseDipoleFunction):
    def __init__(self, mu0, a, d0):
        sm_d = ((sm_Rax - sm_Rbx)**2 + (sm_Ray - sm_Rby)**2 \
             + (sm_Raz - sm_Rbz)**2)**(1/2)
        
        self.parameters = [mu0, a, d0]
        self.parameters_sm = [sm_mu0, sm_a, sm_d0]
        self.dipole_exp = [
            ((sm_Ra[i] - sm_Rb[i])/sm_d) * sm_mu0 * \
            sm.exp(2*sm_d0*a) * sm.exp(-2*sm_d*a)
            for i in range(3)
        ]
        
        super().__init__()

class GriegorievDipoleFunction(BaseDipoleFunction):
    def __init__(self, mu0, a, d0, d7):

        sm_d = ((sm_Rax - sm_Rbx)**2 + (sm_Ray - sm_Rby)**2 \
             + (sm_Raz - sm_Rbz)**2)**(1/2)
        
        self.parameters = [mu0, a, d0, d7]
        self.parameters_sm = [sm_mu0, sm_a, sm_d0, sm_d7]
        self.dipole_exp = [
            ((sm_Ra[i] - sm_Rb[i])/sm_d) *\
            (sm_mu0 * sm.exp(-sm_a*(sm_d-sm_d0)) - sm_d7/(sm_d**7))
            for i in range(3)
        ]
        
        super().__init__()

class MeuwlyDipoleFunction(BaseDipoleFunction):
    def __init__(self, mu0, d0, d1, d2, d7):
        sm_d = ((sm_Rax - sm_Rbx)**2 + (sm_Ray - sm_Rby)**2 \
             + (sm_Raz - sm_Rbz)**2)**(1/2)
        
        self.parameters = [mu0, d0, d1, d2, d7]
        self.parameters_sm = [sm_mu0, sm_d0, sm_d1, sm_d2, sm_d7]
        self.dipole_exp = [
            ((sm_Ra[i] - sm_Rb[i])/sm_d) * \
            (sm_mu0 * sm.exp(- (sm_d-sm_d0)/sm_d1 - (sm_d-sm_d0)**2/(sm_d2**2)) \
            - sm_d7/(sm_d**7))
            for i in range(3)
        ]

        super().__init__()
        
dipole_function1 = GriegorievDipoleFunction(mu0=0.0284, a=1.22522, d0=7.10, d7=14200)

print(dipole_function1([0,0,0],[5,0,0]))
print(dipole_function1.gradient([0,0,0],[5,0,0]))
