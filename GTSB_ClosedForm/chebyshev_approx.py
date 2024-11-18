### the following code is a modified version of the code from the following repository:
# https://github.com/sibyllema/Fast-Multiscale-Diffusion-on-Graphs/blob/main/core.py
# the original paper is:
# Marcotte, Sibylle, et al. "Fast multiscale diffusion on graphs." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022.


# General imports
import numpy as np

# Maths
from scipy.special import ive # Bessel function
from scipy.special import factorial
from scipy.sparse.linalg import eigsh # Eigenvalues computation

################################################################################
### Theoretical bound definition ###############################################
################################################################################

def g(K, C):
    return 2 * np.exp((C**2.)/(K+2)-2*C) * (C**(K+1))/(factorial(K)*(K+1-C))

def get_bound_eps_generic(phi, x, tau, K):
    C = tau*phi/2.
    return g(K, C)**2.

def get_bound_eta_generic(phi, x, tau, K):
    C = tau*phi/2.
    assert(K > C-1)
    return g(K, C)**2. * np.exp(8*C)

def get_bound_eta_specific(phi, x, tau, K):
    C = tau*phi/2.
    n = len(x)
    if len(x.shape) == 1:
        # Case 1: X has shape (n,), it is one signal.
        a1 = np.sum(x)
        assert(a1 != 0.)
        return g(K, C)**2. * n * np.linalg.norm(x)**2. / (a1**2.)
    elif len(x.shape) == 2:
        # Case 2: X has shape (n,dim), it is multiple signals.
        # Take the maximum bound for every signal
        a1 = np.sum(x, axis=0)
        assert(not np.any(a1 == 0.))
        return g(K, C)**2. * n * np.amax(np.linalg.norm(x, axis=0)**2. / (a1**2.))

def E(K, C):
    b = 2 / (1 + np.sqrt(5))
    d = np.exp(b) / (2 + np.sqrt(5))
    if K <= 4*C:
        return np.exp((-b * (K+1)**2.) / (4*C)) * (1 + np.sqrt(C * np.pi / b)) + (d**(4*C)) / (1-d)
    else:
        return (d**K) / (1-d)

def get_bound_bergamaschi_generic(phi, x, tau, K):
    C = tau*phi/2.
    return (2*E(K, C)*np.exp(4*C))**2.

def get_bound_bergamaschi_specific(phi, x, tau, K):
    C = tau*phi/2.
    n = len(x)
    # Same branch as in get_bound_eta_specific()
    if len(x.shape) == 1:
        a1 = np.sum(x)
        assert(a1 != 0.)
        return 4 * E(K, C)**2. * n * np.linalg.norm(x)**2. / (a1**2.)
    elif len(x.shape) == 2:
        a1 = np.sum(x, axis=0)
        assert(not np.any(a1 == 0.))
        return 4 * E(K, C)**2. * n * np.amax(np.linalg.norm(x, axis=0)**2. / (a1**2.))

def reverse_bound(f, phi, x, tau, err):
    """ Returns the minimal K such that f(phi, x, tau, K) <= err. """
    # Starting value: C-1
    C = tau*phi/2.
    K_min = max(1, int(C))

    # Step 0: is C-1 enough?
    if f(phi, x, tau, K_min) <= err:
        return K_min

    # Step 1: searches a K such that f(*args) <= err, by doubling step size.
    K_max = 2 * K_min
    while f(phi, x, tau, K_max) > err:
        K_min = K_max
        K_max = 2 * K_min

    # Step 2: now we have f(...,K_max) <= err < f(...,K_min). Dichotomy!
    while K_max > 1+K_min:
        K_int = (K_max + K_min) // 2
        if f(phi, x, tau, K_int) <= err:
            K_max = K_int
        else:
            K_min = K_int
    return K_max

################################################################################
### method to compute the diffusion ########################################
################################################################################

def compute_chebychev_coeff_all(phi, tau, K):
    """ Compute the K+1 Chebychev coefficients for our functions. """
    return 2*ive(np.arange(0, K+1), -tau * phi)

def expm_multiply(L, tau, X=None, K=None, err=1e-32):
    # Get statistics
    phi = eigsh(L, k=1, return_eigenvectors=False)[0] / 2
    if X is None:
        X = np.eye(L.shape[0])
    N, d = X.shape
    # Case 1: tau is a single value
    if isinstance(tau, (float, int)):
        # Compute minimal K
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, np.amax(tau), err)
        # Compute coefficients (they should all fit in memory, no problem)
        coeff = compute_chebychev_coeff_all(phi, tau, K)
        # Initialize the accumulator with only the first coeff*polynomial
        T0 = X
        Y = .5 * coeff[0] * T0
        # Add the second coeff*polynomial to the accumulator
        T1 = (1 / phi) * L @ X - T0
        Y = Y + coeff[1] * T1
        # Recursively add the next coeff*polynomial
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            Y = Y + coeff[j] * T2
            T0 = T1
            T1 = T2
        return Y
    # Case 2: tau is, in fact, a list of tau
    # In this case, we return the list of the diffusions as these times
    elif isinstance(tau, list):
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, max(tau), err)
        coeff_list = [compute_chebychev_coeff_all(phi, t, K) for t in tau]
        T0 = X
        Y_list  = [.5 * coeff[0] * T0 for coeff in coeff_list]
        T1 = (1 / phi) * L @ X - T0
        Y_list  = [Y + coeff[1] * T1 for Y,coeff in zip(Y_list, coeff_list)]
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            Y_list = [Y + coeff[j] * T2 for Y,coeff in zip(Y_list, coeff_list)]
            T0 = T1
            T1 = T2
        return Y_list
    # Case 3: tau is a numpy array
    elif isinstance(tau, np.ndarray):
        # Compute the order K corresponding to the required error
        if K is None: K = reverse_bound(get_bound_eta_specific, phi, X, np.amax(tau), err)
        # Compute the coefficients for every tau
        coeff = np.empty(tau.shape+(K+1,), dtype=np.float64)
        for index, t in np.ndenumerate(tau):
            coeff[index] = compute_chebychev_coeff_all(phi, t, K)
        # Compute the output for just the first polynomial*coefficient
        T0 = X
        Y = np.empty(tau.shape+X.shape, dtype=X.dtype)
        for index, t in np.ndenumerate(tau):
            Y[index] = .5 * coeff[index][0] * T0
        # Add the second polynomial*coefficient
        T1 = (1 / phi) * L @ X - T0
        for index, t in np.ndenumerate(tau):
            Y[index] = Y[index] + coeff[index][1] * T1
        # Recursively add the others polynomials*coefficients
        for j in range(2, K + 1):
            T2 = (2 / phi) * L @ T1 - 2 * T1 - T0
            for index, t in np.ndenumerate(tau):
                Y[index] = Y[index] + coeff[index][j] * T2
            T0 = T1
            T1 = T2
        return Y
    else:
        print(f"expm_multiply(): unsupported data type for tau ({type(tau)})")