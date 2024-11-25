import os,sys,re

import numpy as np
import termcolor
import torch
import pickle
import scipy.io as sio 
from geomloss import SamplesLoss
from tqdm import tqdm
from ipdb import set_trace as debug


# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def is_sc_dataset(opt):
    return opt.problem_name in ['ocean','plane']

def is_graph_dataset(opt):
    return opt.problem_name in ['earthquake', 'heatflow', 'traffic_pemsd', 'brain', 'single_cell']

def get_time(sec):
    h = int(sec//3600)
    m = int((sec//60)%60)
    s = sec%60
    return h,m,s

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

def unflatten_dim01(x, dim01):
    # (dim0x1, *dim2) --> (dim0, dim1, *dim2)
    return x.reshape(*dim01, *x.shape[1:])

def save_nll(opt, values, stage, direction):
    fn = opt.ckpt_path + "/loss_{}.txt".format(direction)
    with open(fn, 'a') as f:
        # Write loss value with iteration to the file
        if stage == 1:
            f.write('num_iter {}, num_epoch {} \n'.format(opt.num_itr, opt.num_epoch))
            if opt.problem_name in ['ocean']: # allow for heterogeneous diffusion
                f.write('diffu {}, diffu_down {}, diffu_up {}, sigma {}, sigma_min {}, sigma_max {}, beta_min {}, beta_max {} \n'.format(opt.diffu, opt.diffu_down, opt.diffu_up, opt.sigma, opt.sigma_min, opt.sigma_max, opt.beta_min, opt.beta_max))
            else:
                f.write('diffu {}, sigma {}, sigma_min {}, sigma_max {}, beta_min {}, beta_max {} \n'.format(opt.diffu, opt.sigma, opt.sigma_min, opt.sigma_max, opt.beta_min, opt.beta_max))
            f.write('forward model: {}, backward model: {} \n'.format(opt.forward_net, opt.backward_net))
        f.write('stage {},  nll: {}\n'.format(stage,  values)) 
    print(green("nll saved: {}".format(fn)))    
    
def save_w_distance(opt, stage):
    for direction in ['forward', 'backward']:
        fn = opt.ckpt_path + "/loss_{}.txt".format(direction)
        with open(fn, 'a') as f:
            if opt.problem_name in ['brain','ocean','earthquake','traffic_pemsd','single_cell']:
                file_name = 'results/'+opt.log_fn
                W1, W2 = compute_w_distance(file_name, stage, direction, problem_name=opt.problem_name)
                f.write('stage {},  W1: {:.3f}, W2: {:.3f}\n'.format(stage, W1, W2))
            if stage == opt.num_stage:  
                f.write('\n') 

def compute_w_distance(file_name, stage, direction, problem_name=None):
    with open(file_name + '/stage{}-forward.npy'.format(stage), 'rb') as f:
        x = np.load(f)
    with open(file_name + '/stage{}-backward.npy'.format(stage), 'rb') as f:
        y = np.load(f)
    if direction == 'forward':
        x = torch.tensor(x[:,-1,:])
        y = torch.tensor(y[:,-1,:])
    elif direction == 'backward':
        x = torch.tensor(x[:,0,:])
        y = torch.tensor(y[:,0,:])
    
    loss_1 = SamplesLoss(loss="sinkhorn", p=1, blur=.05)
    loss_2 = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
    
    if problem_name in ['single_cell'] and direction == 'forward':
        coord = np.load('datasets/single_cell/coord.npy')
        label = np.load('datasets/single_cell/label.npy')
        x0 = coord[label==4, 0]
        x1 = coord[label==4, 1]
        x_true = torch.tensor(np.stack([x0, x1], axis=1))
        b = np.cumsum([2656, 4374, 3650, 3873, 3650])
        xx = x[0,:]
        yy = np.zeros(coord.shape[0])
        for t in range(5):
            if t==0:
                yy[xx.detach().cpu().numpy().argsort()[-b[t]:]] = t
            else:
                yy[xx.detach().cpu().numpy().argsort()[-b[t]:-b[t-1]]] = t   
                    
        y0 = coord[yy==4, 0]
        y1 = coord[yy==4, 1]
        y_pred = torch.tensor(np.stack([y0, y1], axis=1))
        L1 = loss_1(x_true, y_pred)
        L2 = torch.sqrt(loss_2(x_true, y_pred))
    
    else:
        L1 = loss_1(x, y)  
        L2 = torch.sqrt(loss_2(x, y)) 
    return L1.item(), L2.item()

def save_npy_traj(opt, fn, traj, direction=None):
    fn_npy = os.path.join(opt.eval_path, fn+'.npy')
    assert opt.problem_name in ['ocean', 'earthquake','heatflow','traffic_pemsd','brain','single_cell']
    np.save(fn_npy, traj)


def load_ocean_dataset(train_ratio=0.5, n_eigs=100):
    with open('datasets/ocean_flow/pacific_data.pkl', 'rb') as f:
        laplacians, eigenvectors, eigenvalues, (b1, b2), y = pickle.load(f)
        eigenvectors, eigenvalues = eigenvectors[:,:n_eigs], eigenvalues[:n_eigs]
    with open('datasets/ocean_flow/cochain1.pkl', 'rb') as f:
        y = pickle.load(f)
    n1 =  b1.shape[0], b1.shape[1]
    num_train = int(train_ratio * n1)
    x = np.arange(n1)
    random_perm = np.random.permutation(x)
    train_ids, test_ids = random_perm[:num_train], random_perm[num_train:]
    x_train, x_test = x[train_ids], x[test_ids]
    y_train, y_test = y[train_ids], y[test_ids]
    return (b1, b2), laplacians, (x_train, y_train), (x_test, y_test), (x, y), (eigenvectors, eigenvalues)

def load_earthquake_data():
    with open('datasets/earthquakes/eqs.pkl', 'rb') as f:
        G = pickle.load(f)
        L = pickle.load(f)
        gs = pickle.load(f)
        gs = gs[:-1,:]
    return G, L, gs

def load_heatflow_data():
    with open('datasets/heat_flows_aus/hfs.pkl', 'rb') as f:
        G = pickle.load(f)
        L = pickle.load(f)
        gs = pickle.load(f)
        gs1 = pickle.load(f)
        gs2 = pickle.load(f)        
    return G, L, gs, gs1, gs2 

def load_traffic_pemsd_data():
    y = np.load('datasets/PeMSD4 Datasets/PEMSD4_' + 'edge_features_matrix' + '.npz')['arr_0'].squeeze()
    laplacians = np.load('datasets/PeMSD4 Datasets/PEMSD4_' + 'hodge_Laplacian' + '.npz') ['arr_0']
    b1 = np.load('datasets/PeMSD4 Datasets/PEMSD4_' + 'B1' + '.npz')['arr_0']
    num_train = int(0.8 * y.shape[0])
    num_test = y.shape[0] - num_train
    x = np.arange(y.shape[0])
    random_perm = np.random.permutation(x)
    train_ids, test_ids = random_perm[:num_train], random_perm[num_train:]
    y_train, y_test = y[train_ids], y[test_ids]
    return b1, laplacians,  y, y_train, y_test

def load_brain_data():
    lap = sio.loadmat('datasets/brain_signals/lap.mat')['L']
    gs0 = sio.loadmat('datasets/brain_signals/aligned.mat')['Xa'].T
    gs1 = sio.loadmat('datasets/brain_signals/liberal.mat')['Xl'].T
    
    num_train = int(0.8 * gs0.shape[0])
    num_test = gs0.shape[0] - num_train
    x = np.arange(gs0.shape[0])
    random_perm = np.random.permutation(x)
    train_ids, test_ids = random_perm[:num_train], random_perm[num_train:]
    gs0_train, gs0_test = gs0[train_ids], gs0[test_ids]
    gs1_train, gs1_test = gs1[train_ids], gs1[test_ids]
    return lap, gs0, gs1, gs0_train, gs0_test, gs1_train, gs1_test

def load_single_cell_data():
    A = np.load('datasets/single_cell/A.npy')     
    L = np.load('datasets/single_cell/L.npy')
    gs0 = np.load('datasets/single_cell/gs0.npy') 
    gs1 = np.load('datasets/single_cell/gs4.npy') 
    return A, L, gs0, gs1 




