import numpy as np

def outer_along_0axis(array1, array2):
    return array1[:,:,np.newaxis] * array2[:,np.newaxis,:]
