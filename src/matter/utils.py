import numpy as np

def PBC_wrapping(r, L):
    """
    Function for wrapping position or distance vector with periodic boundary condition
    Args:
    + r (np.array): array of position or vector
    + L (float): box length
    """
    if L is None:
        return r
    else:
        assert isinstance(L,float) or isinstance(L,int)
        r = np.where(r >= L/2, r - L, r)
        r = np.where(r < -L/2, r + L, r)
    return r

def neighborlist_mask(R_all, L, cell_width):
    """
    Class for generating neighborlist mask for accelerating calculation of distance
    Args:
    + R_all (np.array): position of all particles/atoms
    + L (float) dimension of the simulating cubic box
    + cell_width (float): dimension of cell
    """

    assert cell_width < L

    #binning the cubic box width to cells width
    L_bin = np.arange(-L/2,L/2+1,cell_width)

    #calculate the center of the cell
    cell_center_list = np.array(
            [(L + L_bin[i+1])/2 for i,L in enumerate(L_bin[:-1])]
            )

    # Repeating R_all to get an array w dim: (N atoms, 3, len(cell_center_list))
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

