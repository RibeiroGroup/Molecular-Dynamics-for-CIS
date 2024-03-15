import time
import numpy as np

from distance import DistanceCalculator

np.random.seed(10)

L = 5
cell_width = 1
#calculate the cell interval
L_bin = np.arange(-L/2,L/2+1,cell_width)
#calculate the center of the cell
cell_center_list = np.array(
        [(L + L_bin[i+1])/2 for i,L in enumerate(L_bin[:-1])]
        )
[print(i,center) for i, center in enumerate(cell_center_list)]

N_Ar = int(L*2/2)
N_Xe = int(L*2/2)

R_Ar = np.random.uniform(-L/2, L/2, (N_Ar,3))
R_Xe = np.random.uniform(-L/2, L/2, (N_Xe,3))

R_all = np.vstack([R_Ar, R_Xe])
print(R_all)

#BEGIN CALCULATING THE CELL NEIGHBOR LIST

start = time.time()

# Repeating R_all to get an array w dim: (N atoms, 3, num cell center)
foo = np.tile(R_all[:,:,np.newaxis],(1,1,len(cell_center_list)) )

# Repeating cell_center_list to get an array w dim: (N atoms, 3, num cell center)
# Assuming the cell centers coordinates are the same in x,y,z dim
foo2 = np.tile(cell_center_list[np.newaxis,np.newaxis,:],(R_all.shape[0],3,1))

# Calculating the distance (in either x, y, z dim) to corresponding cell center
# The smallest absolute distance => cell center index
cell_num = np.argmin(abs(foo2 - foo), axis = -1)

# Calculating the differences of cell center indices for all atoms in all
# 3 dim, cell center difference by one in either x, y, z => nearby cell
foo_mat = abs(
        np.tile(cell_num[:,np.newaxis,:],(1,len(cell_num),1)) \
        - np.tile(cell_num[np.newaxis,:,:],(len(cell_num),1,1))
        )

foo_mat = np.where(foo_mat == len(cell_center_list) - 1, 1, foo_mat)

cell_neighbor_mask = np.sum(foo_mat,axis = -1)
cell_neighbor_mask = np.where(cell_neighbor_mask <= 3, True, False) 

distance_calc = DistanceCalculator(
        n_points = len(R_all), mask = cell_neighbor_mask, 
        box_length = L)

distance_calc2 = DistanceCalculator(
        n_points = len(R_all),
        box_length = L)

dmat = distance_calc.get_all_distance_matrix(R_all)
dmat += np.eye(len(dmat)) * 10

dmat2 = distance_calc2.get_all_distance_matrix(R_all)
dmat2 += np.eye(len(dmat2)) * 10

def foofunc(X,De):
    return np.exp(- 2 * (X - De))

print(foofunc(dmat, cell_width) - foofunc(dmat2, cell_width) )
