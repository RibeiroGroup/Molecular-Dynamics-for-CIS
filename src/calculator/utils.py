import numpy as np

def PBC_wrapping(r, Lxy, Lz):
    """
    Function for wrapping position or distance vector with periodic boundary condition
    Args:
    + r (np.array): array of position or vector
    + L (float): box length
    """
    assert isinstance(Lxy,float) or isinstance(Lxy,int)
    assert isinstance(Lz,float) or isinstance(Lz,int)
    assert r.shape[-1] == 3

    r[:,:2] = np.where(r[:,:2] >= Lxy/2, r[:,:2] - Lxy, r[:,:2])
    r[:,:2] = np.where(r[:,:2] < -Lxy/2, r[:,:2] + Lxy, r[:,:2])

    r[:,-1] = np.where(r[:,-1] >= Lz/2, r[:,-1] - Lz, r[:,-1])
    r[:,-1] = np.where(r[:,-1] < -Lz/2, r[:,-1] + Lz, r[:,-1])

    return r

def neighborlist_mask(R_all, Lxy, Lz, cell_width):
    """
    Class for generating neighborlist mask for accelerating calculation of distance
    Args:
    + R_all (np.array): position of all particles/atoms
    + L (float) dimension of the simulating cubic box
    + cell_width (float): dimension of cell
    """

    assert cell_width < Lxy and cell_width < Lz

    #binning the cubic box width to cells width
    Lxy_bin = np.arange(-Lxy/2,Lxy/2+1,cell_width)
    Lz_bin = np.arange(-Lz/2,Lz/2+1,cell_width)

    #calculate the center of the cell
    cell_xycenter_list = np.array(
            [(L + Lxy_bin[i+1])/2 for i,L in enumerate(Lxy_bin[:-1])])
    cell_zcenter_list = np.array(
            [(L + Lz_bin[i+1])/2 for i,L in enumerate(Lz_bin[:-1])])
    print(cell_zcenter_list)

    # Repeating R_all to get an array w dim: (N atoms, 3, len(cell_center_list))
    tiled_R_all1 = np.tile( #shape N x 2 x len(cell_center...)
            R_all[:,:2][:,:,np.newaxis],(1,1,len(cell_xycenter_list)) ) 
    tiled_R_all2 = np.tile( #shape N x len(cell_center...)
            R_all[:,-1][:,np.newaxis],(1,len(cell_zcenter_list)) )

    # Repeating cell_center_list to get an array w dim: (N atoms, 3, num cell center)
    # Assuming the cell centers coordinates are the same in x,y,z dim
    tiled_cell_center_xy = np.tile( #shape N x 2 x len(cell_center...)
            cell_xycenter_list[np.newaxis,np.newaxis,:],(R_all.shape[0],2,1))
    tiled_cell_center_z = np.tile( #shape N x len(cell_center...)
            cell_zcenter_list[np.newaxis,:],(R_all.shape[0],1))

    # Calculating the distance (in either x, y, z dim) to corresponding cell center
    # The smallest absolute distance => cell center index/bin
    cell_bin_xy = np.argmin( # shape N x 2
            abs(tiled_cell_center_xy - tiled_R_all1), axis = -1) 
    cell_bin_z  = np.argmin( # shape N
            abs(tiled_cell_center_z - tiled_R_all2), axis = -1)
    print(cell_bin_z)
    cell_bin = np.hstack([cell_bin_xy, cell_bin_z.reshape(-1,1)])
    print(cell_bin)

    # Calculating the differences of cell center indices/bin for all atoms in all
    # 3 dim, cell center difference by one in either x, y, z => nearby cell
    R_bin_diff = abs(
            np.tile(cell_bin[:,np.newaxis,:],(1,len(cell_bin),1)) \
            - np.tile(cell_bin[np.newaxis,:,:],(len(cell_bin),1,1))
            )

    # Considering the Periodic Boundary condition
    R_bin_diff[:,:2] = np.where(
            R_bin_diff[:,:2] == len(cell_xycenter_list) - 1, 1, R_bin_diff[:,:2])
    R_bin_diff[:,-1] = np.where(
            R_bin_diff[:,-1] == len(cell_zcenter_list) - 1, 1, R_bin_diff[:,-1])

    mask = np.sum(R_bin_diff,axis = -1)
    mask = np.where(mask <= 3, True, False) 

    return mask

