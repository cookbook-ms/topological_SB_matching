import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch
import util
import data 

from base_sde import BaseSDE


def build(opt, p, q):
    return {
        'vp': VPSDE,
        've': VESDE,
        'bm': BMSDE,
        'tbm':TBMSDE,
        'tve': TVESDE,
        'tvp': TVPSDE,
    }.get(opt.sde_type)(opt, p, q)
    

class BMSDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(BMSDE, self).__init__(opt, p, q)
        self.var = opt.sigma

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return torch.Tensor([self.var])


class VPSDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VPSDE,self).__init__(opt, p, q)
        self.b_min=opt.beta_min
        self.b_max=opt.beta_max

    def _f(self, x, t):
        return compute_vp_drift_coef(t, self.b_min, self.b_max)*x

    def _g(self, t):
        return compute_vp_diffusion(t, self.b_min, self.b_max)


class VESDE(BaseSDE):
    def __init__(self, opt, p, q):
        super(VESDE,self).__init__(opt, p, q)
        self.s_min=opt.sigma_min
        self.s_max=opt.sigma_max

    def _f(self, x, t):
        return torch.zeros_like(x)

    def _g(self, t):
        return compute_ve_diffusion(t, self.s_min, self.s_max)


####################### topological diffusion SDE ############################
class TBMSDE(BaseSDE): 
    def __init__(self, opt, p, q):
        super(TBMSDE, self).__init__(opt, p, q)
        self.var = torch.tensor([opt.sigma]) 
        self.diffu = opt.diffu
        self.opt = opt 
        self.L = self._L

    @property
    def _L(self):
        try:
            return self.L
        except AttributeError:
            return  torch.Tensor(data.EarthquakeDataProcessor(problem_name=self.opt.problem_name).L)
        
    def _L_transform(self):
        return 1e-5 + self.L
    
    def _H_t(self):
        return -self.diffu * self.L
    
    def _f(self, x, t):
        drift = torch.matmul(self._H_t(), x.T)
        return drift.T
 
    def _g(self, t):
        return self.var   
    

####################### topological diffusion SDE +VE #########################
class TVESDE(TBMSDE): 
    def __init__(self, opt, p, q):
        super(TVESDE, self).__init__(opt, p, q)
        self.s_min = opt.sigma_min
        self.s_max = opt.sigma_max 
        
    def _f(self, x, t):
        drift = self._H_t()@ x.T
        return  torch.Tensor(drift.T)
 
    def _g(self, t):
        return torch.Tensor(compute_ve_diffusion(t, self.s_min, self.s_max))
    
    
####################### topological diffusion SDE +VE ############################
class TVPSDE(TBMSDE): 
    def __init__(self, opt, p, q):
        super(TVPSDE, self).__init__(opt, p, q)
        self.b_min = opt.beta_min
        self.b_max = opt.beta_max 
        
    def _f(self, x, t):
        drift = self._H_t()@ x.T
        return  torch.Tensor(drift.T + compute_vp_drift_coef(t, self.b_min, self.b_max)*x)
 
    def _g(self, t):
        return torch.Tensor(compute_vp_diffusion(t, self.b_min, self.b_max))


####################### functions for computing scalar coefficients ############################
def compute_sigmas(t, s_min, s_max):
    return s_min * (s_max/s_min)**t

def compute_ve_g_scale(s_min, s_max):
    return np.sqrt(2*np.log(s_max/s_min))

def compute_ve_diffusion(t, s_min, s_max):
    return compute_sigmas(t, s_min, s_max) * compute_ve_g_scale(s_min, s_max)

def compute_vp_diffusion(t, b_min, b_max):
    return torch.sqrt(b_min+t*(b_max-b_min))

def compute_vp_drift_coef(t, b_min, b_max):
    g = compute_vp_diffusion(t, b_min, b_max)
    return -0.5 * g**2
