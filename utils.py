import time
import numpy as np

def timeit(func):
    def inner(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print("Runtime by timeit: ",time.time() - start)
        return result
    return inner

def PBC_wrapping(r, L):
    if L is None:
        return r
    else:
        assert isinstance(L,float) or isinstance(L,int)
        r = np.where(r >= L/2, r - L, r)
        r = np.where(r < -L/2, r + L, r)
    return r

def get_dist_matrix(distance_vector):

    distance_matrix = np.sqrt(
        np.sum(distance_vector**2, axis = -1))
    
    return distance_matrix

"""
L = 100
n_points = 10
all_r = np.random.uniform(-L/2,L/2,size=(n_points,3))

dc = DistanceCalculator(5, box_length= L)
dc_ = DistanceCalculator(n_points, box_length= L)
rvec = dc.get_distance2(all_r[5:], all_r[:5])

foo = dc_(all_r)
foo = foo[:5,5:,:][dc.utriang_bool_mat_x3].reshape(-1,3)

print(rvec + foo)
"""

def test_for_distance_matrix(ra):
    n_points = ra.shape[0]
    dist_mat_ = np.zeros((n_points,n_points))
    for i, ri in enumerate(ra):
        for j,rj in enumerate(ra):
            if i == j: continue
            #j_ = j if j < i else j-1
            dist_mat_[i][j] = np.sqrt(np.sum((ri - rj)**2))

    return dist_mat_

def test_for_distance_vector(x):
    n_points = x.shape[0]

    d_ = np.zeros((n_points, n_points, 3))
    for i, x1 in enumerate(x):
        for j, x2 in enumerate(x):
            if i == j: continue
            d_[i,j,:] = x1 - x2

    return d_


"""
n_points = 10
all_r = np.random.rand(n_points,3)
distant_calc = DistanceCalculator(n_points)

rvec = distant_calc(all_r)

rvec_test = test_for_distance_vector(all_r)

print(rvec - rvec_test)

"""
"""
@PBC_decorator(L = L)
def get_dist_vector(R):
    rij_vec_tensor = np.zeros((n_points, n_points,3))
    for i in range(len(R)):
        for j in range(i, len(R)):
            
            ri = R[i]; rj = R[j];

            rij_vec = ri - rj
            #rij = np.sqrt(np.sum((ri - rj)**2))

            rij_vec_tensor[i,j,:] = rij_vec
            rij_vec_tensor[j,i,:] = - rij_vec

    return rij_vec_tensor

def PBC_wrapping(r, L):
    r = np.where(r >= L/2, r - L, r)
    r = np.where(r < -L/2, r + L, r)
    return r

def PBC_decorator(L):
    def real_PBC_wrap(func):
        def PBC_wrapping_wrapper(*args,**kwargs):
            dist = func(*args, **kwargs)
            dist = PBC_wrapping(dist, L)

            return dist

        return PBC_wrapping_wrapper
    return real_PBC_wrap


"""
