Here we provide a tutorial on how to implement the closed-form TSB solutions, driven by the THeatBM and THeatVE reference dynamics. That is, we corroborate the theoretical results in __Theorems 6 & 7__ in the paper.



**The procedure consists of the following steps**:
1. Create a topological structure, e.g., a graph or a simplicial 2-complex. (In the provided example, we consider a Swiss Roll graph.)
2. Specify the Gaussian boundary distributions. 
3. Choose the reference dynamic scheme from ```'tsb-bm'``` and ```'tsb-ve'```, and specify the heat diffusion coefficient $c$, as well as the noise diffusion coefficient $g$ (or $\sigma_{\max}, \sigma_{\min}$).
4. One can then obtain the intermediate samples from the closed-form solution by calling ```gtsb_closed_form()```. For example, the following function call returns the covariance matrix and the intermediate samples for the TSB-VE scheme:

```python
cov_t_tsb_ve, x_i_tsb_ve = gtsb_closed_form(scheme='tsb-ve', c=0.05, sigma_min=0.01, sigma_max=1)
```

**Chebyshev implementation**:
We additionally took the suggestion by the reviewer to implement the involved matrix exponentials using the Chebyshev polynomials. This is feasible due the closed-form expression of the polynomial coefficients for the matrix exponential [1, eq. 5].
In our Chebyshev-based re-implementation, we used the available code from the authors of [1] and we provide the code in the ```chebyshev_approx.py``` file. The function ```expm_multiply``` computes the matrix exponential using the Chebyshev polynomials.


[1] Marcotte, Sibylle, et al. "Fast multiscale diffusion on graphs." ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022.


**Padé and Chebyshev approximations comparisons**:
In our original computations, we applied Padé approximations for matrix exponentials in the smaller-scaled tests. Here we provide some comparisons between the two:

- compute the matrix exponential itself: ```expm(A)```

| Dimension | Chebyshev | Padé |
|-----------|-------|-------|
| 1000      | 0.28  | 0.36  |
| 2000      | 1.55 | 1.77  |
| 5000      | 12.80 | 11.96 |
| 10000     | 76.92 | 73.91 |


- compute the matrix exponential-vector product: ```expm_multiply(A, v)```

| Dimension | Chebyshev | Padé |
|-----------|-------|-------|
| 1000      | 0.02  | 0.002 |
| 2000      | 0.06  | 0.002 |
| 5000      | 0.26  | 0.003 |
| 10000     | 1.43  | 0.004 |
| 20000     | 5.37  | 0.01  |

** ```Padé``` denotes the implementation based on ```scipy.sparse.linalg```. 
