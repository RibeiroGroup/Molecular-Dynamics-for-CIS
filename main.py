import time
import numpy as np

from distance import DistanceCalculator, explicit_test
from forcefield import LennardJonesPotential, explicit_test_LJ
from dipole import SimpleDipoleFunction

np.random.seed(10)

L = 1000
cell_width = 10

#calculate the cell interval
N_Ar = int(L/2)
N_Xe = int(L/2)
N = N_Ar + N_Xe

R_Ar = np.random.uniform(-L/2, L/2, (N_Ar,3))
R_Xe = np.random.uniform(-L/2, L/2, (N_Xe,3))
R_all = np.vstack([R_Ar, R_Xe])

idxAr = np.hstack(
    [np.ones(N_Ar), np.zeros(N_Xe)]
)

idxXe = np.hstack(
    [np.zeros(N_Ar), np.ones(N_Xe)]
)

epsilon = (np.outer(idxAr,idxAr) * 0.996 \
    + np.outer(idxAr, idxXe) * 1.377 \
    + np.outer(idxXe, idxAr) * 1.377 \
    + np.outer(idxXe, idxXe) * 1.904 ) * 1.59360e-3

sigma = (np.outer(idxAr,idxAr) * 3.41 \
    + np.outer(idxAr, idxXe) * 3.735 \
    + np.outer(idxXe, idxAr) * 3.735 \
    + np.outer(idxXe, idxXe) * 4.06) * (1e-10 / 5.29177e-11)

def neighbor_list_mask(R_all, L, cell_width):

    L_bin = np.arange(-L/2,L/2+1,cell_width)
    #calculate the center of the cell
    cell_center_list = np.array(
            [(L + L_bin[i+1])/2 for i,L in enumerate(L_bin[:-1])]
            )
    #[print(i,center) for i, center in enumerate(cell_center_list)]

    # Repeating R_all to get an array w dim: (N atoms, 3, num cell center)
    tiled_R_all = np.tile(R_all[:,:,np.newaxis],(1,1,len(cell_center_list)) )

    # Repeating cell_center_list to get an array w dim: (N atoms, 3, num cell center)
    # Assuming the cell centers coordinates are the same in x,y,z dim
    tiled_cell_center = np.tile(cell_center_list[np.newaxis,np.newaxis,:],(R_all.shape[0],3,1))

    # Calculating the distance (in either x, y, z dim) to corresponding cell center
    # The smallest absolute distance => cell center index/bin
    cell_bin = np.argmin(abs(tiled_cell_center - tiled_R_all), axis = -1)

    # Calculating the differences of cell center indices/bin for all atoms in all
    # 3 dim, cell center difference by one in either x, y, z => nearby cell
    R_bin_diff = abs(
            np.tile(cell_bin[:,np.newaxis,:],(1,len(cell_bin),1)) \
            - np.tile(cell_bin[np.newaxis,:,:],(len(cell_bin),1,1))
            )

    # Considering the Periodic Boundary condition
    R_bin_diff = np.where(R_bin_diff == len(cell_center_list) - 1, 1, R_bin_diff)

    mask = np.sum(R_bin_diff,axis = -1)
    mask = np.where(mask <= 3, True, False) 

    return mask

mask = neighbor_list_mask(R_all, cell_center_list)

start = time.time()
distance_calc = DistanceCalculator(
        n_points = len(R_all), mask = mask,
        box_length = L)

forcefield = LennardJonesPotential(
    sigma = sigma, epsilon = epsilon, distance_calc = distance_calc
)

dipole = SimpleDipoleFunction()

potential = forcefield.potential(R_all, return_matrix = True)
force = forcefield.force(R_all, return_matrix = True)
print(time.time() - start)

start = time.time()
distance_calc = DistanceCalculator(
        n_points = len(R_all),
        box_length = L)

forcefield = LennardJonesPotential(
    sigma = sigma, epsilon = epsilon, distance_calc = distance_calc
)

forcefield.potential(R_all, return_matrix = True)
forcefield.force(R_all, return_matrix = True)
print(time.time() - start)







