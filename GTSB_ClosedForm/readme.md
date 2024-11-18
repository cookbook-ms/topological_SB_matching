Here we provide a tutorial on how to implement the closed-form TSB solutions, driven by the THeatBM and THeatVE reference dynamics. That is, we corroborate the thereotical results in __Theorems 6 & 7__ in the paper.



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