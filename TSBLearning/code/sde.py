import numpy as np
import abc
from tqdm import tqdm
from functools import partial
import torch

import utils
import loss
import data 
from ipdb import set_trace as debug
import sys, math

def _assert_increasing(name, ts):
    assert (ts[1:] > ts[:-1]).all(), '{} must be strictly increasing'.format(name)

def build(opt, p, q):
    print(utils.magenta("build base sde....:{}".format(opt.sde_type)))
    return {
        'vp': VPSDE,
        've': VESDE,
        'bm': BMSDE,
        'tbm':TBMSDE,
        'tve': TVESDE,
        'tvp': TVPSDE,
    }.get(opt.sde_type)(opt, p, q)

class BaseSDE(metaclass=abc.ABCMeta):
    def __init__(self, opt, p, q):
        self.opt = opt
        self.dt=opt.T/opt.interval
        self.p = p # initial distribution
        self.q = q # final distribution

    @abc.abstractmethod
    def _f(self, x, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _g(self, x, t):
        raise NotImplementedError

    def f(self, x, t, direction):
        sign = 1. if direction=='forward' else -1.
        return sign * self._f(x,t)

    def g(self, t):
        return self._g(t)

    def dw(self, x, dt=None):
        dt = self.dt if dt is None else dt
        return torch.randn_like(x)*np.sqrt(dt)

    def propagate(self, t, x, z, direction, f=None, dw=None, dt=None):
        g = self.g(t)
        f = self.f(x,t,direction) if f is None else f
        dt = self.dt if dt is None else dt
        dw = self.dw(x,dt) if dw is None else dw
        return x + (f + g*z)*dt + g*dw

    def sample_traj(self, ts, policy, corrector=None, apply_trick=True, save_traj=True, if_seed=False):
        opt = self.opt
        direction = policy.direction
        assert direction in ['forward','backward']
        _assert_increasing('ts', ts)
        init_dist = self.p if direction=='forward' else self.q
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])
        x = init_dist.sample(if_seed)
        xs = torch.empty((x.shape[0], len(ts), *x.shape[1:])) if save_traj else None
        zs = torch.empty_like(xs) if save_traj else None
        _ts = tqdm(ts,desc=utils.yellow("Sample Traj: Propagating Dynamics..."+direction))
        for idx, t in enumerate(_ts):
            _t=t if idx==ts.shape[0]-1 else ts[idx+1]
            f = self.f(x,t,direction)
            z = policy(x,t)   
            dw = self.dw(x)
            t_idx = idx if direction=='forward' else len(ts)-idx-1 #
            if save_traj:
                xs[:,t_idx,...]=x
                zs[:,t_idx,...]=z
                
            x = self.propagate(t, x, z, direction, f=f, dw=dw)
            if corrector is not None and direction=='backward':
                denoise_xT = False 
                x  = self.corrector_langevin_update(_t ,x, corrector, denoise_xT)

        x_term = x
        res = [xs, zs, x_term]
        return res

    def corrector_langevin_update(self, t, x, corrector, denoise_xT):
        opt = self.opt
        batch = x.shape[0]
        g_t = self.g(t)
        for _ in range(opt.num_corrector):
            z =  corrector(x,t)
            z_avg_norm = z.reshape(batch,-1).norm(dim=1).mean()
            eps_temp = 2 * (opt.snr / z_avg_norm )**2
            noise=torch.randn_like(z)
            noise_avg_norm = noise.reshape(batch,-1).norm(dim=1).mean()
            eps = eps_temp * (noise_avg_norm**2)
            x = x + g_t*eps*z + g_t*torch.sqrt(2*eps)*noise
            
        if denoise_xT: x = x + g_t*z
        return x

    def compute_nll(self, samp_bs, ts, z_f, z_b, direction):
        assert z_f.direction == 'forward'
        assert z_b.direction == 'backward'
        opt = self.opt
        _assert_increasing('ts', ts)
        if direction=='forward':
            init_dist, term_dist = self.p, self.q
        else:
            init_dist, term_dist = self.q, self.p
            
        x = init_dist.sample() 
        ts = ts if direction=='forward' else torch.flip(ts,dims=[0])
        delta_logp = 0
        e = loss.sample_e(opt, x)

        for idx, t in enumerate(tqdm(ts,desc=utils.yellow("NLL Comp: Propagating Dynamics..."+direction))):
            with torch.set_grad_enabled(True):
                x.requires_grad_(True)
                g = self.g(t)
                f = self.f(x,t,direction)
                if direction == 'forward':
                    z = z_f(x,t)
                    z2 = z_b(x,t)
                    dx_dt = f + g * z - 0.5 * g * (z + z2)
                elif direction == 'backward':
                    z = z_b(x,t)
                    z2 = z_f(x,t)
                    dx_dt = -(f + g * z - 0.5 * g * (z + z2)) 
                divergence = divergence_approx(dx_dt, x, e=e)
                dlogp_x_dt = divergence.view(samp_bs, 1)

            del divergence, z2, g
            x, dx_dt, dlogp_x_dt = x.detach(), dx_dt.detach(), dlogp_x_dt.detach()
            z, f = z.detach(), f.detach()
            x = self.propagate(t, x, z, direction, f=f)
            if (idx == 0 and direction=='forward') or (idx == ts.shape[0]-1 and direction=='backward'): 
                continue
            delta_logp = delta_logp + dlogp_x_dt*self.dt
        
        logp_x =  delta_logp.view(-1)
        logpx_per_dim = - torch.sum(logp_x) / x.nelement() 
        
        return logpx_per_dim


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx

####################### SB-SDEs ############################
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
            return  torch.Tensor(data.GraphDataProcessor(problem_name=self.opt.problem_name).L).to_sparse()
        
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