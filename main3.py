import numpy as np
from utils import DistanceCalculator

n_points = 5
x = np.random.rand( n_points, 3)

dist_calc = DistanceCalculator(n_points)

d = dist_calc.get_distance_vector(x)

