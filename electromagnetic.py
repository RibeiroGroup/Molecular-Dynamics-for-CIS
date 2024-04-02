import numpy as np

class VectorPotential:
    def __init__(self, kvec, amplitudes, polarization_vector):

        assert kvec.shape[1] == 3 and len(kvec.shape) == 2
        self.n_modes = kvec.shape[0]
        self.kvec_list = np.array(kvec)

        assert amplitudes.shape == 
