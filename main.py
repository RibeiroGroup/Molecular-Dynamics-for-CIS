import numpy as np

from MEfield import vector_potential
from test_cases import A_test_case

x = np.array([0.1,0.2,0.3]) # 3

C,k,epsilon = A_test_case()#True, 2022)
print(C,k,epsilon)

A = vector_potential(C=C, k=k, epsilon=epsilon)
print(A.diff_ra(x))
