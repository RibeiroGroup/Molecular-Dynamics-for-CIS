import numpy as np

from scipy.stats import maxwell

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
    def __init__(
            self, N_atom_pairs, angle_range, L,
            d_ar_xe, red_temp_unit, K_temp,
            ar_mass, xe_mass
            ):

        self.N_atom_pairs = N_atom_pairs

        self.angle_range = angle_range

        self.L = L
        self.d_ar_xe = d_ar_xe

        self.sampler_ar = MaxwellSampler(
                mass = ar_mass,
                red_temp_unit = red_temp_unit, 
                K_temp = K_temp)

        self.sampler_xe = MaxwellSampler(
                mass = xe_mass,
                red_temp_unit = red_temp_unit, 
                K_temp = K_temp)

    def __call__(self):
        N_atom_pairs = self.N_atom_pairs
        offset = self.angle_range
        L = self.L

        r_ar = np.random.uniform(-L/2,L/2,size = (N_atom_pairs,3))

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


