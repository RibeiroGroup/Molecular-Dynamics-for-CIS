import numpy as np
import matplotlib.pyplot as plt

from EM_field_simple import SingleModeField, ChargePoint, EMHamiltonian
import constants

np.random.seed(20202)
# FIELD SPECS
A = SingleModeField(
    C=np.random.rand(2),
    k=np.array([1,0,0]),
    epsilon=np.array([[0,1,0], [0,0,1]])
    )

#PARTICLE SPECS 
charge_points = [
    ChargePoint(
        m = 1, q = 1, r = np.random.rand(3), 
        v = np.random.rand(3))
    ]

#CALCULATION
Hamiltonian = EMHamiltonian(A,charge_points)




