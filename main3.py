import numpy as np
import matplotlib.pyplot as plt

from utils import DistanceCalculator, \
    test_for_distance_vector, test_for_distance_matrix

########### BOX DIMENSION ##################

n_points = 500
L = 100

########### PARTICLES ##################

np.random.seed(100)
all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))
print(all_r.shape)
all_v = np.random.rand(n_points,3)
print(all_v.shape)


"""
0      , r1 - r2, r1 - r3, ...
r2 - r1, 0      , r2 - r3, ...
"""
def PBC_wrapping(r, L):
    r = np.where(r >= L/2, r - L, r)
    r = np.where(r < -L/2, r + L, r)
    return r

def PBC_decorator(L):
    def real_PBC_wrap(func):
        def PBC_wrapping_wrapper(*args,**kwargs):
            dist = func(*args, **kwargs)
            dist = PBC_wrapping(dist, L)

            return dist

        return PBC_wrapping_wrapper
    return real_PBC_wrap

def get_dist_matrix(distance_vector):

    distance_matrix = np.sqrt(
        np.sum(distance_vector**2, axis = -1))
    
    return distance_matrix

class MorsePotential:
    def __init__(self, n_points, De, Re, a, L):

        self.De = De
        self.Re = Re
        self.a = a

        self.distance_calc = DistanceCalculator(n_points,L)

    def get_potential(self, R):
        distance_vector = self.distance_calc(R)
        distance_matrix = get_dist_matrix(distance_vector)

        distance = distance_matrix[
            np.triu(np.ones(distance_matrix.shape, dtype=bool),k=1)]
        return self.De*(1 - np.exp(-self.a*(-self.Re + distance)))**2

    def get_force(self, R):
        
        distance_vector = self.distance_calc(R)
        distance_matrix = get_dist_matrix(distance_vector)

        f= -2*self.De*self.a*(1 - np.exp(-self.a*(-self.Re + distance_matrix)))\
            *np.exp(-self.a*(-self.Re + distance_matrix)) \
            /( distance_matrix + np.eye(len(distance_matrix)) )

        f = np.tile(f[:,:,np.newaxis], (1,1,3))

        f *= distance_vector# (-1.0*Rx + 1.0*x) 

        f = np.sum(f, axis = 1)

        return f


potential = MorsePotential(
    n_points = n_points,
    De =  1495 / 4.35975e-18 / 6.023e23,
    Re = 3.5e-10 / 5.29177e-11,
    a = 1/ ( (1/3 * 1e-10) / 5.29177e-11),
    L = L
)

r = all_r
v = all_v
h = 1e-3
n_steps = 5000

energy = {"T":[], "K":[], "H":[]}

T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T))
energy["T"].append(T)
K = np.sum(potential.get_potential(r) )
energy["K"].append(K)
H = T + K
energy["H"].append(H)
H0 = H

for i in range(1, n_steps + 1):
    k1v = potential.get_force(r)
    k1r = v

    k2v = potential.get_force(r + k1r*h/2)
    k2r = v + k1v*h/2

    k3v = potential.get_force(r + k2r*h/2)
    k3r = v + k2v*h/2

    k4v = potential.get_force(r + k3r*h)
    k4r = v + k3v*h

    r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    r = PBC_wrapping(r, L)
    v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)

    T = 0.5 * np.sum(np.einsum("ij,ji->i", v, v.T))
    energy["T"].append(T)
    K = np.sum(potential.get_potential(r) )
    energy["K"].append(K)
    H = T + K
    energy["H"].append(H)
    if i % 100 == 0:
        print("H = ",H, " K = ", K, " T = ", T)

print(H0 - H)

fig, ax = plt.subplots()
ax.plot(np.arange(n_steps+1), energy["H"])
fig.savefig("foo.jpeg")

fig, ax = plt.subplots()
ax.plot(np.arange(n_steps+1), energy["T"])
ax.plot(np.arange(n_steps+1), energy["K"])
fig.savefig("foo2.jpeg")
