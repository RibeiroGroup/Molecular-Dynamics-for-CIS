import numpy as np

class RungeKutta4th:
    def __init__(self, f, const_args):
        self.f = f
        self.const_args = const_args

    def step(self, f_args):
         

"""
for i in range(int(1e4 + 1)):
    k1c = dot_C(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon = epsilon)

    k1v = compute_force(
        q=alpha.q, r=r, v=v, 
        k_vec=k_vec, C=C, epsilon=epsilon)

    k1r = v

    k2c = dot_C(
        q=alpha.q, r=r + h*k1r/2, v=v + h*k1v/2, 
        k_vec=k_vec, C=C + h*k1c/2, epsilon = epsilon)

    k2v = compute_force(
        q=alpha.q, r=r + h*k1r/2, v=v + h*k1v/2, 
        k_vec=k_vec, C=C + h*k1c/2, epsilon=epsilon)

    k2r = v + h*k1v/2

    k3c = dot_C(
        q=alpha.q, r=r + h*k2r/2, v=v + h*k2v/2, 
        k_vec=k_vec, C=C + h*k2c/2, epsilon = epsilon)

    k3v = compute_force(
        q=alpha.q, r=r + h*k2r/2, v=v + h*k2v/2, 
        k_vec=k_vec, C=C + h*k2c/2, epsilon=epsilon)

    k3r = v + h*k2v/2

    k4c = dot_C(
        q=alpha.q, r=r + h*k3r, v=v + h*k3v, 
        k_vec=k_vec, C=C + h*k3c, epsilon = epsilon)

    k4v = compute_force(
        q=alpha.q, r=r + h*k3r, v=v + h*k3v, 
        k_vec=k_vec, C=C + h*k3c, epsilon=epsilon)

    k4r = v + h*k3v

    r = r + (h/6) * (k1r + 2*k2r + 2*k3r + k4r)
    v = v + (h/6) * (k1v + 2*k2v + 2*k3v + k4v)
    C = C + (h/6) * (k1c + 2*k2c + 2*k3c + k4c)

    H_mat = compute_Hmat(v)# * constants.c
    mat_H_list.append(H_mat)

    H_em = compute_Hem(k_vec, C) 
    em_H_list.append(H_em)

    H_list.append(H_mat + H_em)

    steps_list.append(i)
    if i % 1e3 == 0:
        print("Step {}".format(i+1))
        print("r = ",r)
        print("v = ",v)
        print("H_mat = ",H_mat)
        print("C = ",C)
        print("H_em = ",H_em)
        print("total H = ",H_mat + H_em)
        print("delta H_em / delta H_mat = ", 
            (H_em - em_H_list[-2]) / (H_mat - mat_H_list[-2]) )

"""
