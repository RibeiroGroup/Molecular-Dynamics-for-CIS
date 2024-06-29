from field.utils import EM_mode_generate
import utilities.reduced_parameter as red

k_vec = EM_mode_generate(20)

np.random.seed(20)
"""
EXAMPLE CALCULATION FOR VECTOR POTENTIAL OF CAVITY FIELD
"""

kappa = np.array([
    [1,0],
    [0,1]
    ])

m = np.array([1,1])

amplitude = np.array([
    np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2),
    np.random.uniform(size = 2) + 1j * np.random.uniform(size = 2)
    ])
print(amplitude)

A = CavityVectorPotential(
    kappa = kappa, m = m, amplitude = amplitude,
    S = 100.0, L = 10.0, constant_c = red.c
    )

"""
R = np.vstack([
    np.array([[1,1,0],[1,1,5]]),
    np.random.uniform(size = (3,3))
    ])
"""
R = np.array([[1,1,1],[1,1,2],[1,2,1]])

print(
    A(t = 0 , R = R)
    )

print(A.gradient(0,R).shape)



