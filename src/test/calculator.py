from utils import neighborlist_mask
from forcefield import explicit_test_LJ
from dipole import explicit_test_dipole

def explicit_test_LJ(R, epsilon ,sigma, L):

    N = len(R)

    potential = np.zeros((N,N))
    force = np.zeros((N,N,3))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j: continue
            
            ep = epsilon[i,j]
            sig = sigma[i,j]
            
            dvec = ri - rj
            dvec = PBC_wrapping(dvec,L)

            d = np.sqrt(dvec @ dvec)

            potential[i,j] = 4 * ep * ( (sig/d)**12 - (sig/d)**6 )
            f = 4 * ep * (
                12 * (sig**12 / d**14) - 6 * (sig**6 / d**8)
            )

            force[i,j,:] = f * (dvec)

    return potential, force

def explicit_test_dipole(R, dipole_mask, mu0, a, d, L):

    dipole = np.zeros((len(R),len(R), 3))
    dipole_grad = np.zeros((len(R),len(R), 3, 3))

    for i, ri in enumerate(R):
        for j, rj in enumerate(R):
            if i == j : continue
            if not dipole_mask[i,j]: continue

            #calculating distance
            distance_vec = PBC_wrapping(ri - rj, L)
            distance = np.sqrt(distance_vec @ distance_vec)

            #calculating dipole vector
            dipole_ = mu0 * np.exp(-a * (distance - d)) 
            dipole_ *= distance_vec
            dipole[i,j,:] = dipole_

            # Calculating dipole gradient tensor
            exp_ad = np.exp(-a * (distance - d))

            distance_outer = np.einsum("j,k->jk", distance_vec, distance_vec)

            gradient = - a * mu0 * distance_outer * exp_ad / distance**2
            gradient -= mu0 * distance_outer * exp_ad / distance ** 3
            gradient += (mu0 * exp_ad / distance) * np.eye(3)

            dipole_grad[i,j,:,:] = gradient

    dipole += 1 * np.swapaxes(dipole,0,1)
    dipole_grad += -1 * np.swapaxes(dipole_grad,0,1)

    return dipole, dipole_grad

########################
###### BOX LENGTH ######
########################

L = 20
cell_width = 10

##########################
###### ATOMIC INPUT ######
##########################

# number of atoms
N_Ar = int(L/4)
N_Xe = int(L/4)
N = N_Ar + N_Xe

# randomized initial coordinates
R_all = np.random.uniform(-L/2, L/2, (N, 3))

N = R_all.shape[0]

idxXe = np.hstack([np.ones(int(N/2)),np.zeros(int(N/2))])
idxAr = np.hstack([np.zeros(int(N/2)),np.ones(int(N/2))])

######################################
###### FORCE-RELATED PARAMETERS ######
######################################

epsilon_Ar_Ar = 0.996 * 1.59360e-3
epsilon_Ar_Xe = 1.377 * 1.59360e-3
epsilon_Xe_Xe = 1.904 * 1.59360e-3

sigma_Ar_Ar = 3.41 * (1e-10 / 5.29177e-11)
sigma_Ar_Xe = 3.735* (1e-10 / 5.29177e-11)
sigma_Xe_Xe = 4.06 * (1e-10 / 5.29177e-11)

epsilon_mat = (np.outer(idxAr,idxAr) * epsilon_Ar_Ar \
    + np.outer(idxAr, idxXe) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxAr) * epsilon_Ar_Xe \
    + np.outer(idxXe, idxXe) * epsilon_Xe_Xe )

sigma_mat = (np.outer(idxAr,idxAr) * sigma_Ar_Ar \
    + np.outer(idxAr, idxXe) * sigma_Ar_Xe \
    + np.outer(idxXe, idxAr) * sigma_Ar_Xe \
    + np.outer(idxXe, idxXe) * sigma_Xe_Xe) 

############
### TEST ###
############
neighborlist = neighborlist_mask(R_all, L, cell_width)

dipole_mask = generate_dipole_mask(idxXe, idxAr)

calculator = Calculator(
    N, box_length = L, 
    epsilon = epsilon_mat, sigma=sigma_mat,
    positive_atom_idx = idxXe, negative_atom_idx = idxAr,
    mu0=0.0124 , a=1.5121, d=7.10,
    )

calculator.calculate_distance(R_all, neighborlist)

potential_, force_ = explicit_test_LJ(R_all, epsilon_mat, sigma_mat, L)

print("### Potential test ###")
potential = calculator.potential()
print(np.sum(abs(potential - potential_)))

print("### Force test ###")
force = calculator.force(return_matrix = True)
print(np.sum(abs(force - force_)))

print("### Dipole test ###")
dipole = calculator.dipole()
print(dipole)
dipole_, gradD_ = explicit_test_dipole(R_all, dipole_mask,
    mu0=0.0124 , a=1.5121, d=7.10, L = L)

print(np.sum(abs(dipole - dipole_)))

print("### Dipole gradient test ###")
gradD = calculator.dipole_grad()

print(np.sum(abs(gradD - gradD_)))
