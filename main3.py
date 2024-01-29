import numpy as np
from utils import DistanceCalculator, \
    test_for_distance_vector, verify_distant_matrix

n_points = 10
x = np.random.rand( n_points, 3) * 10

dist_calc = DistanceCalculator(n_points)

d = dist_calc(x)

d_ = verify_distant_matrix(x)

print(d - d_)

