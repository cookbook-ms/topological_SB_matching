import numpy as np
import scipy.linalg as scila

def compute_fractional_matrix_power(A, alpha):
    if alpha<0:
        A = scila.fractional_matrix_power(A,-alpha).real
        A = np.linalg.pinv(A).real
    else:
        A = scila.fractional_matrix_power(A,alpha)
    return A


def bw_distance(cov1, cov2):
    return np.trace(cov1+cov2 - 2*compute_fractional_matrix_power(compute_fractional_matrix_power(cov1, 0.5) @ cov2 @ compute_fractional_matrix_power(cov1, 0.5), 0.5))