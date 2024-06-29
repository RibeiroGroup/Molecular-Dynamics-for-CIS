import numpy as np
from utils import neighborlist_mask

########################
###### BOX LENGTH ######
########################

L = 100
cell_width = 10

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = int(L/10)
N_Xe = int(L/10)
N = N_Ar + N_Xe

# randomized initial coordinates
R_all = np.random.uniform(-L/2, L/2, (N, 3))

# Calculation from explicit test
true_distance_mat, true_distance_vec = explicit_test(R_all, L) 

############################################
##### Test without neighbor cell list. #####
############################################
print("##### Test without neighbor cell list. #####")

distance_calc = DistanceCalculator(N = N, box_length = L)

distance_mat = distance_calc.calculate_distance_matrix(R_all)
distance_vec = distance_calc.calculate_distance_vector_tensor(R_all)

print("+++ Difference between DistanceCalculator class and ExpliciTest for distance matrix +++")
print(np.sum(distance_mat - true_distance_mat))
    
print("+++ Difference between DistanceCalculator class and ExpliciTest for distance vector array +++")
print(np.sum(abs(distance_vec - true_distance_vec)))

#########################################
##### Test with neighbor cell list. #####
#########################################
print("##### Test with neighbor cell list. #####")

neighborlist = neighborlist_mask(R_all, L, cell_width)

distance_calc = DistanceCalculator(N, box_length = L)

distance_mat = distance_calc.calculate_distance_matrix(R_all, neighborlist = neighborlist)
distance_vec = distance_calc.calculate_distance_vector_tensor(R_all, neighborlist = neighborlist)

print("+++ Difference between DistanceCalculator class and ExpliciTest for distance matrix +++")
dist_diff = abs(distance_mat - true_distance_mat)
print(np.sum(np.where(dist_diff < 10, dist_diff, 0)))

print("+++ Difference between DistanceCalculator class and ExpliciTest for distance vector array +++")
dist_diff = distance_calc.repeat_x3(dist_diff)
dist_vec_diff = abs(distance_vec - true_distance_vec)
print(np.sum(np.where(dist_diff < cell_width, dist_vec_diff, 0)))

