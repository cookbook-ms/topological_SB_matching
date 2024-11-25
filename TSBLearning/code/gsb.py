import numpy as np 
import warnings
import numpy as np
from scipy.sparse.linalg import eigsh
from ipdb import set_trace as debug


def get_eigendecomposition(
    lap_mat, tolerance: float = np.finfo(float).eps
) -> tuple:
    with warnings.catch_warnings(record=True):
        eigenvectors, eigenvalues = eigsh(lap_mat, k=lap_mat.shape[0])
        eigenvalues[np.abs(eigenvalues) < tolerance] = 0

    return eigenvectors, eigenvalues

class HodgeBasis:
    def __init__(self, b1, b2, eigenbasis=None, laplacians=None):
        self.b1 = b1
        self.b2 = b2
        if laplacians is not None:
            self.L1, self.L1_down, self.L1_up = laplacians
        else:
            self.L1_down = b1.T @ b1
            self.L1_up = b2 @ b2.T
            self.L1 = self.L1_down + self.L1_up
        if eigenbasis is not None:
            self.eig_evecs, self.eig_vals = eigenbasis
        else:
            self.eig_evecs, self.eig_vals = get_eigendecomposition(self.L1)

        self.grad_idx, self.curl_idx, self.harm_idx = self.get_hodge_idx

    @property
    def get_hodge_idx(self):
        try:
            return self.grad_idx, self.curl_idx, self.harm_idx
        except AttributeError:
            total_var, total_div, total_curl = [], [], []
            for i in range(self.eig_vals.shape[0]):
                total_var.append(self.eig_evecs[:,i].T@self.L1@self.eig_evecs[:,i])
                total_div.append(self.eig_evecs[:,i].T@self.L1_down@self.eig_evecs[:,i])
                total_curl.append(self.eig_evecs[:,i].T@self.L1_up@self.eig_evecs[:,i]) 

            eps = 1e-6
            grad_idx = np.where(np.array(total_div) >= eps)[0]
            curl_idx = np.where(np.array(total_curl) >= eps)[0]
            harm_idx = np.where(np.array(total_var) < eps)[0]
            assert len(grad_idx) + len(curl_idx) + len(harm_idx) == self.eig_vals.shape[0]
            return grad_idx, curl_idx, harm_idx

    @property
    def get_hodge_basis(self):
        grad_evecs = self.eig_evecs[:, self.grad_idx]
        curl_evecs = self.eig_evecs[:, self.curl_idx]
        harm_evecs = self.eig_evecs[:, self.harm_idx]
        return harm_evecs, grad_evecs, curl_evecs 
    
    @property
    def get_hodge_eig_vals(self):
        grad_evals = self.eig_vals[self.grad_idx]
        curl_evals = self.eig_vals[self.curl_idx]
        harm_evals = self.eig_vals[self.harm_idx]
        return harm_evals, grad_evals, curl_evals


    