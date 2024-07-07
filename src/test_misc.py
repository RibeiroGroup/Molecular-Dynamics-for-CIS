import numpy as np
from field.utils import EM_mode_generate_,EM_mode_generate, EM_mode_generate3
import utilities.reduced_parameter as red

possible_cavity_k = [0] + list(range(40,70)) 

k_vector2 = np.array(
        EM_mode_generate(possible_cavity_k, vector_per_kval = 3, max_kval = 70),
        dtype=np.float64)

print(len(k_vector2))
print(np.sqrt(np.einsum("ki,ki->k",k_vector2, k_vector2)) / (5e7*red.sigma))
