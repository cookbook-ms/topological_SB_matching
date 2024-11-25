import numpy as np

import torch
import torch.distributions as td
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from prefetch_generator import BackgroundGenerator

import gsb
import utils
from ipdb import set_trace as debug
from django.utils.functional import cached_property


# build_boundary_distribution
def build_boundary_distribution(opt):
    print(utils.magenta("build boundary distribution..."))
    print("problem name: {}".format(opt.problem_name))
    opt.data_dim = get_data_dim(opt.problem_name)
    prior = build_final_sampler(opt, opt.samp_bs)
    pdata = build_init_sampler(opt, opt.samp_bs)
    return pdata, prior

def get_data_dim(problem_name):
    return {
        # 'ocean':       [SCDataProcessor(problem_name='ocean').get_data_dim],
        # 'plane':       [SCDataProcessor(problem_name='plane').get_data_dim],
        # 'earthquake':  [GraphDataProcessor(problem_name='earthquake').get_data_dim],
        # 'heatflow':    [GraphDataProcessor(problem_name='heatflow').get_data_dim],
        # 'traffic_pemsd': [GraphDataProcessor(problem_name='traffic_pemsd').get_data_dim],
        # 'brain':      [GraphDataProcessor(problem_name='brain').get_data_dim],
        'single_cell': [GraphDataProcessor(problem_name='single_cell').get_data_dim],
    }.get(problem_name)

def build_final_sampler(opt, batch_size): # FINAL
    if  utils.is_sc_dataset(opt):
        # for ocean task, the terminal distribution is a GP
        b1, b2, eigenbasis, laplacians = SCDataProcessor(opt.problem_name).get_data
        eigen_basis = gsb.HodgeBasis(b1, b2, eigenbasis, laplacians)
        mean = np.zeros(b1.shape[1])
        gp_params = {'gp_type': opt.term_gp}
        gp_params.update({
            'mean': mean,
            'harm_sigma': opt.term_harm_sigma,
            'grad_sigma': opt.term_grad_sigma,
            'curl_sigma': opt.term_curl_sigma,
            'harm_kappa': opt.term_harm_kappa,
            'grad_kappa': opt.term_grad_kappa,
            'curl_kappa': opt.term_curl_kappa,
            'alpha': opt.term_noise_scale,
        }) 
        if opt.term_gp == 'matern':
            gp_params.update({
                'harm_nu': opt.term_harm_nu,
                'grad_nu': opt.term_grad_nu,
                'curl_nu': opt.term_curl_nu,
            }) 
        return OceanGP(batch_size, eigen_basis, gp_params)
    elif utils.is_graph_dataset(opt):
        if opt.problem_name in ['heatflow','brain','single_cell']: # for matching
            return GraphDataProcessor(batch_size, opt.problem_name, opt.device, sample_gs0=False)
        else: # for generation
            return GraphDataProcessor(batch_size, opt.problem_name, opt.device)


def build_init_sampler(opt, batch_size): # INITIAL
    if utils.is_sc_dataset(opt):
        b1, b2, eigenbasis, laplacians = SCDataProcessor(opt.problem_name).get_data
        eigen_basis = gsb.HodgeBasis(b1, b2, eigenbasis, laplacians)
        mean = np.zeros(b1.shape[1])
        gp_params = {'gp_type': opt.init_gp}
        gp_params.update({
            'mean': mean,
            'harm_sigma': opt.init_harm_sigma,
            'grad_sigma': opt.init_grad_sigma,
            'curl_sigma': opt.init_curl_sigma,
            'harm_kappa': opt.init_harm_kappa,
            'grad_kappa': opt.init_grad_kappa,
            'curl_kappa': opt.init_curl_kappa,
            'alpha': opt.init_noise_scale,
        }) 
        if opt.init_gp == 'matern':
            gp_params.update({
                'harm_nu': opt.init_harm_nu,
                'grad_nu': opt.init_grad_nu,
                'curl_nu': opt.init_curl_nu,
            }) 
        return OceanGP(batch_size, eigen_basis, gp_params)
    
    elif utils.is_graph_dataset(opt):
        if opt.problem_name in ['heatflow','brain','single_cell']:
            return GraphDataProcessor(batch_size, opt.problem_name, opt.device,sample_gs0=True)
        else:
            return GraphDataProcessor(batch_size, opt.problem_name, opt.device)

    else:
        raise RuntimeError()

    
class OceanGP:
    def __init__(self, batch_size, eigen_basis, gp_params):
        self.harm_evals, self.grad_evals, self.curl_evals = eigen_basis.get_hodge_eig_vals
        self.harm_evecs, self.grad_evecs, self.curl_evecs = eigen_basis.get_hodge_basis
        self.batch_size = batch_size

        self.gp_type = gp_params['gp_type']
        self.mean = gp_params['mean']
        self.harm_sigma2 = gp_params['harm_sigma']
        self.grad_sigma2 = gp_params['grad_sigma']
        self.curl_sigma2 = gp_params['curl_sigma']
        self.harm_kappa = gp_params['harm_kappa']
        self.grad_kappa = gp_params['grad_kappa']
        self.curl_kappa = gp_params['curl_kappa']
        self.alpha = gp_params['alpha']
        if self.gp_type == 'matern':
               self.harm_nu = gp_params['harm_nu']
               self.grad_nu = gp_params['grad_nu']
               self.curl_nu = gp_params['curl_nu']   
               
        self.L = SCDataProcessor(problem_name='ocean').L
        
    def get_cov_spectral(self):
        if self.gp_type == 'diffusion':
            harm_cov = self.harm_sigma2 *  np.exp(- self.harm_kappa**2/2 * self.harm_evals)
            grad_cov = self.grad_sigma2 *  np.exp(- self.grad_kappa**2/2 * self.grad_evals)
            curl_cov = self.curl_sigma2 *  np.exp(- self.curl_kappa**2/2 * self.curl_evals)
        elif self.gp_type == 'matern':
            harm_cov = self.harm_sigma2 * (2*self.hamr_nu/self.harm_kappa**2 + self.harm_evals)**(self.harm_nu/2)
            grad_cov = self.grad_sigma2 * (2*self.grad_nu/self.grad_kappa**2 + self.grad_evals)**(self.grad_nu/2)
            curl_cov = self.curl_sigma2 * (2*self.curl_nu/self.curl_kappa**2 + self.curl_evals)**(self.curl_nu/2)
            
        return np.concatenate([harm_cov, grad_cov, curl_cov])
        
    def sample(self, if_seed=False):
        n = self.batch_size
        cov_spectral = self.get_cov_spectral()
        hodge_evecs = np.concatenate([self.harm_evecs, self.grad_evecs, self.curl_evecs], axis=1)
        ocean_samples = ocean_sampler_gp(self.mean, cov_spectral, hodge_evecs, n, if_seed)
        return torch.Tensor(ocean_samples) 


class SCDataProcessor:
    def __init__(self, problem_name):
        self.problem_name = problem_name
        if self.problem_name in ['ocean']:
            self.incidence_matrices, self.laplacians, self.eigenpairs, self.data = self._load_data
        elif self.problem_name == 'plane':
            self.incidence_matrices, self.laplacians, self.eigenpairs = self._load_data
        self.b1, self.b2 = self.incidence_matrices
        self.L = self.laplacians[0]
        self.L = torch.sparse_coo_tensor(self.L.nonzero(), self.L.data, self.L.shape, dtype=torch.float32)
        self.eigenbasis = self.get_eigen_basis
        self.eigen_vals = self.get_eigen_vals

    @cached_property
    def _load_data(self):
        if self.problem_name in ['ocean']:
            incidence_matrices, laplacians, _, _, data, eigenpairs = utils.load_ocean_dataset(
                self.problem_name, n_eigs=50)
            return incidence_matrices, laplacians, eigenpairs, data 
        elif self.problem_name == 'plane':
            incidence_matrices, laplacians, eigenpairs = utils.load_ocean_dataset(
                data_name=self.problem_name, n_eigs=50)
            return incidence_matrices, laplacians, eigenpairs

    @cached_property
    def get_laplacians(self):
        if self.laplacians is None:
            self.load_data()
        return self.L

    @cached_property
    def get_eigen_basis(self):
        try:
            return self.eigenbasis
        except AttributeError:
            self.harm_evecs, self.grad_evecs, self.curl_evecs = gsb.HodgeBasis(self.b1, self.b2, self.eigenpairs, self.laplacians).get_hodge_basis
            return np.concatenate([self.harm_evecs, self.grad_evecs, self.curl_evecs], axis=1)

    @cached_property
    def get_eigen_vals(self):
        try:
            return self.eigen_vals
        except AttributeError:
            self.harm_evals, self.grad_evals, self.curl_evals = gsb.HodgeBasis(self.b1, self.b2, self.eigenpairs, self.laplacians).get_hodge_eig_vals
            return np.concatenate([self.harm_evals, self.grad_evals, self.curl_evals])

    @cached_property
    def get_data_dim(self):
        return self.b1.shape[1]

    @cached_property
    def get_data(self):
        return self.b1, self.b2, self.eigenpairs, self.laplacians


class GraphDataProcessor:
    def __init__(self, batch_size=None, problem_name=None, device=None, sample_gs0=None):
        self.problem_name = problem_name 
        self.batch_size = batch_size
        if self.problem_name == 'earthquake':
            self.G, self.L, self.gs = self._load_earthquake_data
        elif self.problem_name == 'heatflow':
            self.G, self.L, self.gs, self.gs1, self.gs2 = self._load_heatflow_data
        elif self.problem_name == 'traffic_pemsd':
            self.b1, self.L, self.y, self.y_train, self.y_test = self._load_traffic_pemsd_data
        elif self.problem_name == 'brain':
            self.L, self.gs0, self.gs1, self.gs0_train, self.gs0_test, self.gs1_train, self.gs1_test = self._load_brain_data
        elif self.problem_name == 'single_cell':
            self.L, self.A, self.gs0, self.gs1 = self._load_single_cell_data
            
        self.device=device
        self.sample_gs0 = sample_gs0
        
    @cached_property    
    def _load_earthquake_data(self):
        G, L, gs = utils.load_earthquake_data()
        return G, L, gs
    
    @cached_property    
    def _load_heatflow_data(self):
        G, L, gs, gs1, gs2 = utils.load_heatflow_data()

        return G, L, gs, gs1, gs2
    
    @cached_property    
    def _load_single_cell_data(self):
        A, L, gs0, gs1 = utils.load_single_cell_data()
        return A, L, gs0, gs1
    
    @cached_property
    def _load_traffic_pemsd_data(self):
        b1, L, y, y_train, y_test = utils.load_traffic_pemsd_data()
        return b1, L, y, y_train, y_test
    
    @cached_property
    def _load_brain_data(self):
        L, gs0, gs1, gs0_train, gs0_test, gs1_train, gs1_test = utils.load_brain_data()
        return L, gs0, gs1, gs0_train, gs0_test, gs1_train, gs1_test
        
    @cached_property
    def get_data_dim(self):
        return self.L.shape[1]
    
    def sample(self, if_seed=False):
        n = self.batch_size
        if self.problem_name == 'earthquake':
            idx = np.random.choice(len(self.gs), n, replace=False)
            return torch.Tensor(self.gs[idx])
        
        elif self.problem_name == 'heatflow':
            assert self.sample_gs0 is not None
            if self.sample_gs0 == True:
                return torch.Tensor(np.tile(self.gs1, (n,1)))
            elif self.sample_gs0 == False:
                return torch.Tensor(np.tile(self.gs0, (n,1)))
            
        elif self.problem_name == 'single_cell':
            assert self.sample_gs0 is not None
            if self.sample_gs0 == True: # sample the initial
                return torch.Tensor(np.tile(self.gs0, (n,1)))
            elif self.sample_gs0 == False: # sample the final
                return torch.Tensor(np.tile(self.gs1, (n,1)))
            
        elif self.problem_name == 'traffic_pemsd':
            y_from = self.y_test if if_seed else self.y_train 
            idx = np.random.choice(len(y_from), n, replace=False)
            return torch.Tensor(y_from[idx])
        
        elif self.problem_name == 'brain':
            assert self.sample_gs0 is not None
            if self.sample_gs0 == True: # initial
                y_from = self.gs1_test if if_seed else self.gs1_train
            elif self.sample_gs0 == False: # final
                y_from = self.gs0_test if if_seed else self.gs0_train
            idx = np.random.choice(len(y_from), n, replace=False)
            return torch.Tensor(y_from[idx])
    
    
def ocean_sampler_gp(mean, cov_spectral, hodge_evecs, sample_size, if_seed=False, seed=23):
    '''use this to sample from the GP for initial distribution '''
    print(utils.magenta("sample from ocean GP"))
    n_spectrals = cov_spectral.shape[0]
    if if_seed:
        np.random.seed(seed)
        print("fix seed: {}".format(seed))
    v = np.random.multivariate_normal(np.zeros(n_spectrals), np.eye(n_spectrals), sample_size)
    assert v.shape == (sample_size,) + (n_spectrals,)
    ocean_flows = mean[:,None] + hodge_evecs @ (np.sqrt(cov_spectral)[:, None] * v.T)
    return ocean_flows.T