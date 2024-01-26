import numpy as np

from utils import DistanceCalculator

De =  1495 / 4.35975e-18 / 6.023e23
Re = 3.5e-10 / 5.29177e-11
a = 1/ ( (1/3 * 1e-10) / 5.29177e-11)

class BoxBoundaryCondition:
    def __init__(self, Lx, Ly, Lz):
        self.L = np.array(
            [Lx, Ly, Lz]
                )

    def apply(self, r):
        r = np.where(
                r < self.L/2, r, r - self.L)

        r = np.where(
                r > -self.L/2, r, r + self.L)

        return r

np.random.seed(2024)

n_points = 5
r = np.random.rand(3,n_points) * 10
v = np.random.rand(3,n_points)

dist_calc = DistanceCalculator()

d_vec = 

print(r)
