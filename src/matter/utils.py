import numpy as np

from scipy.stats import maxwell

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

def neighborlist_mask(R_all, Lxy, Lz, cell_width_xy, cell_width_z):
    """
    Class for generating neighborlist mask for accelerating calculation of distance
    Args:
    + R_all (np.array): position of all particles/atoms
    + L (float) dimension of the simulating cubic box
    + cell_width (float): dimension of cell
    """

    assert cell_width_xy < Lxy and cell_width_z < Lz

    #binning the cubic box width to cells width
    Lxy_bin = np.arange(-Lxy/2,Lxy/2+1,cell_width_xy)
    Lz_bin = np.arange(-Lz/2,Lz/2+1,cell_width_z)

    #calculate the center of the cell
    cell_xycenter_list = np.array(
            [(L + Lxy_bin[i+1])/2 for i,L in enumerate(Lxy_bin[:-1])])
    cell_zcenter_list = np.array(
            [(L + Lz_bin[i+1])/2 for i,L in enumerate(Lz_bin[:-1])])

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
    cell_bin = np.hstack([cell_bin_xy, cell_bin_z.reshape(-1,1)])

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

def sample_velocity(N,max_velocity,min_velocity):
    r_dot = np.random.uniform(
                low = min_velocity, high = max_velocity,
                size = (N, 3))

    #calculate the magnitude of the velocity
    V = np.sqrt(np.einsum("ni,ni->n",r_dot,r_dot)) 
    #scaling the veclocity so that all veclocity magnitude is below the maximum
    scaler = np.where(max_velocity / V > 1 , 1, max_velocity / V)
    #if V > max_velocity, it will be scaled by -^
    scaler = np.tile(scaler[:,np.newaxis],(1,3))

    r_dot *= scaler

    return r_dot

class MaxwellSampler:
    """
    Sampling velocities' magnitudes based on MAXWELL distribution
    Args:
    + mass (np.array): mass of atoms whose velocities are sampled
    + red_temp_unit (float): unit of reduced temperature
    + K_temp (float): temperature in Kelvin
    """

    def __init__(self,mass, red_temp_unit, K_temp):

        red_temp = K_temp / red_temp_unit

        a = np.sqrt( red_temp / mass )

        self.distri = maxwell(scale = a)

    def __call__(self, N,repeat_x3 = True):

        r_dot_squared = [] 

        for i in range(N):
            r_dot_squared.append(self.distri.rvs())

        r_dot_squared = np.array(r_dot_squared)
        if repeat_x3:
            r_dot_squared = np.tile(r_dot_squared[:,np.newaxis],(1,3))

        return r_dot_squared

class AllInOneSampler:
    """
    Class for sampling everything
    Args:
    + N_atom_pairs (int): number of pairs of Argon and Xenon atoms
    + angle_range (float): is a where velocity angle are sampled in range of
        theta +/- a  and phi +/- a
    + L (float): length of the simulated box
    + d_ar_xe (float): initial distance between Argon and Xenon atoms
    + red_temp_unit (float): unit of reduced temperature
    + K_temp (float): temperature in Kelvin
    + ar_mass (float): (reduced) mass of Argon
    + xe_mass (float): (reduced) mass of Xenon
    """
    def __init__(
            self, N_atom_pairs, Lxy, Lz,
            d_ar_xe, d_impact, red_temp_unit, K_temp,
            ar_mass, xe_mass
            ):

        self.N_atom_pairs = N_atom_pairs

        self.Lxy = Lxy
        self.Lz = Lz
        self.d_ar_xe = d_ar_xe

        self.angle_range = np.arcsin(d_impact / d_ar_xe)

        self.sampler_ar = MaxwellSampler(
                mass = ar_mass,
                red_temp_unit = red_temp_unit, 
                K_temp = K_temp)

        self.sampler_xe = MaxwellSampler(
                mass = xe_mass,
                red_temp_unit = red_temp_unit, 
                K_temp = K_temp)

    def __call__(self):
        """
        Sampling and return position and velocity of all Xenons, Argons
        Return:
        + Python dictionary: 
            - "r" (tuple of 2 np.array): position array of Argon and Xenon
            - "r_dot" (tuple of 2 np.array): velocities array of Argon and Xenon
        """
        N_atom_pairs = self.N_atom_pairs
        offset = self.angle_range
        Lxy = self.Lxy
        Lz = self.Lz

        r_ar = np.hstack([
                np.random.uniform(-Lxy/2, Lxy/2,size = (N_atom_pairs,2)),
                np.random.uniform(-Lz/2 , Lz/2, size = (N_atom_pairs,1))
                ])

        phi = np.arccos(1 - 2 * np.random.uniform(0, 1, size = N_atom_pairs))
        theta = 2 * np.pi * np.random.uniform(0, 1, size = N_atom_pairs)
        r_xe = r_ar + np.array([
            self.d_ar_xe * np.sin(phi) * np.cos(theta),
            self.d_ar_xe * np.sin(phi) * np.sin(theta),
            self.d_ar_xe * np.cos(phi)
            ]).T

        r_dot_ar_sqrt = self.sampler_ar(N = N_atom_pairs)
        phi_ = np.random.uniform(phi-offset, phi+offset, size = N_atom_pairs)
        theta_ = np.random.uniform(theta-offset, theta+offset, size = N_atom_pairs)
        r_dot_ar = r_dot_ar_sqrt * np.array([
            np.sin(theta_) * np.cos(phi_),
            np.sin(theta_) * np.sin(phi_),
            np.cos(theta_)
            ]).T

        r_dot_xe_sqrt = self.sampler_xe(N = N_atom_pairs)
        phi_ = np.random.uniform(phi + np.pi - offset, phi + np.pi + offset, size = N_atom_pairs)
        theta_ = np.random.uniform(np.pi - theta - offset, np.pi - theta + offset, size = N_atom_pairs)
        r_dot_xe = r_dot_xe_sqrt * np.array([
            np.sin(theta_) * np.cos(phi_),
            np.sin(theta_) * np.sin(phi_),
            np.cos(theta_)
            ]).T

        return {"r":(r_ar,r_xe), "r_dot":(r_dot_ar, r_dot_xe)}


