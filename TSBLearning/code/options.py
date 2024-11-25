import numpy as np
import os
import argparse
import random
import torch

import configs
import utils

from ipdb import set_trace as debug


def set():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-name",   type=str)
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--gpu",            type=int,   default=0,        help="GPU device")
    parser.add_argument("--load",           type=str,   default=None,     help="load the checkpoints")
    parser.add_argument("--dir",            type=str,   default=None,     help="directory name to save the experiments under results/")
    parser.add_argument("--name",           type=str,   default='debug',  help="son node of directionary for saving checkpoint")
    parser.add_argument("--log-fn",         type=str,   default=None,     help="name of tensorboard logging")
    parser.add_argument("--log-tb",         action="store_true",          help="logging with tensorboard")
    parser.add_argument("--cpu",            action="store_true",          help="use cpu device")

    # --------------- SB model ---------------
    parser.add_argument("--t0",             type=float, default=1e-2,     help="time integral start time")
    parser.add_argument("--T",              type=float, default=1.,       help="time integral end time")
    parser.add_argument("--interval",       type=int,   default=100,      help="number of interval")
    parser.add_argument("--forward-net",    type=str,   help="model class of forward nonlinear drift")
    parser.add_argument("--backward-net",   type=str,   help="model class of backward nonlinear drift")
    parser.add_argument("--num-res-block",  type=int,   default=1,        help="number of residual block for ResNet_FC")
    parser.add_argument("--sde-type",       type=str,   default='ve', choices=['ve', 'vp', 'bm', 'tbm', 'tve', 'tvp'],     help="type of SDE")
    parser.add_argument("--sigma",          type=float, default=0.01,     help="max diffusion for BMSDE and T_diffusion")
    parser.add_argument("--diffu",          type=float, default=10,       help="diffusion coefficient for T_diffusion")
    parser.add_argument("--diffu_down",     type=float, default=0,        help="[Only for heterogenous diffusion] diffusion coefficient for down Laplacian scaling")
    parser.add_argument("--diffu_up",       type=float, default=0,        help="[Only for heterogenous diffusion] diffusion coefficient for up Laplacian scaling")
    parser.add_argument("--sigma-max",      type=float, default=50,       help="max diffusion for VESDE")
    parser.add_argument("--sigma-min",      type=float, default=0.01,     help="min diffusion for VESDE")
    parser.add_argument("--beta-max",       type=float, default=20,       help="max diffusion for VPSDE")
    parser.add_argument("--beta-min",       type=float, default=0.1,      help="min diffusion for VPSDE")

    # --------------- SB training & sampling (corrector) ---------------
    parser.add_argument("--train-method",   type=str, default=None,       help="algorithm for training SB" )
    parser.add_argument("--use-arange-t",   action="store_true",          help="[sb alternate train] use full timesteps for training")
    parser.add_argument("--reuse-traj",     action="store_true",          help="[sb alternate train] reuse the trajectory from sampling")
    parser.add_argument("--use-corrector",  action="store_true",          help="[sb alternate train] enable corrector during sampling")
    parser.add_argument("--train-bs-x",     type=int,                     help="[sb alternate train] batch size for sampling data")
    parser.add_argument("--train-bs-t",     type=int,                     help="[sb alternate train] batch size for sampling timestep")
    parser.add_argument("--num-stage",      type=int,                     help="[sb alternate train] number of stage")
    parser.add_argument("--num-epoch",      type=int,                     help="[sb alternate train] number of training epoch in each stage")
    parser.add_argument("--num-corrector",  type=int, default=1,          help="[sb alternate train] number of corrector steps")
    parser.add_argument("--samp-bs",        type=int,                     help="[sb train] batch size for all trajectory sampling purposes")
    parser.add_argument("--num-itr",        type=int,                     help="[sb train] number of training iterations (for each epoch)")
    parser.add_argument("--test-bs-x",     type=int,  default=50,                help="[sb alternate test] batch size for sampling data")
    parser.add_argument("--test-bs-t",     type=int,  default=50,               help="[sb alternate test] batch size for sampling timestep")
    
    parser.add_argument("--snapshot-freq",  type=int,   default=0,        help="snapshot frequency w.r.t stages")
    parser.add_argument("--ckpt-freq",      type=int,   default=0,        help="checkpoint saving frequency w.r.t stages")
    
    # --------------- optimizer and loss ---------------
    parser.add_argument("--lr",             type=float,                   help="learning rate")
    parser.add_argument("--lr-f",           type=float, default=None,     help="learning rate for forward network")
    parser.add_argument("--lr-b",           type=float, default=None,     help="learning rate for backward network")
    parser.add_argument("--lr-gamma",       type=float, default=1.0,      help="learning rate decay ratio")
    parser.add_argument("--lr-step",        type=int,   default=1000,     help="learning rate decay step size")
    parser.add_argument("--l2-norm",        type=float, default=0.0,      help="weight decay rate")
    parser.add_argument("--optimizer",      type=str,   default='AdamW',  help="optmizer")
    parser.add_argument("--grad-clip",      type=float, default=None,     help="clip the gradient")
    parser.add_argument("--noise-type",     type=str,   default='gaussian', choices=['gaussian','rademacher'], help='choose noise type to approximate Trace term')

    
    # ----- ocean [ONLY] init and termial params ----------------
    parser.add_argument("--init-gp",     type=str,   default='diffusion',     help="initial GP type")
    parser.add_argument("--term-gp",     type=str,   default='diffusion',     help="terminal GP type")
    ## sigmas 
    parser.add_argument("--init-harm-sigma", type=float, default=0.1,              help="initial harmonic GP sigma")
    parser.add_argument("--init-grad-sigma", type=float, default=0.1,              help="initial gradient GP sigma")
    parser.add_argument("--init-curl-sigma", type=float, default=0.1,              help="initial curl GP sigma")
    
    parser.add_argument("--term-harm-sigma", type=float, default=0.1,              help="terminal harmonic GP sigma")
    parser.add_argument("--term-grad-sigma", type=float, default=0.1,              help="terminal gradient GP sigma")
    parser.add_argument("--term-curl-sigma", type=float, default=0.1,              help="terminal curl GP sigma")
    ## kappas
    parser.add_argument("--init-harm-kappa", type=float, default=1.0,              help="initial harmonic diffusion/matern GP kappa")
    parser.add_argument("--init-grad-kappa", type=float, default=1.0,              help="initial gradient diffusion/matern GP kappa")
    parser.add_argument("--init-curl-kappa", type=float, default=1.0,              help="initial curl diffusion/matern GP kappa")
    
    parser.add_argument("--term-harm-kappa", type=float, default=1.0,              help="terminal harmonic diffusion/matern GP kappa")
    parser.add_argument("--term-grad-kappa", type=float, default=1.0,              help="terminal gradient diffusion/matern GP kappa")
    parser.add_argument("--term-curl-kappa", type=float, default=1.0,              help="terminal curl diffusion/matern GP kappa")
    ## nus 
    parser.add_argument("--init-harm-nu", type=float, default=1.0,                 help="initial harmonic matern GP nu")
    parser.add_argument("--init-grad-nu", type=float, default=1.0,                 help="initial gradient matern GP nu")
    parser.add_argument("--init-curl-nu", type=float, default=1.0,                 help="initial curl matern GP nu")
    
    parser.add_argument("--term-harm-nu", type=float, default=1.0,                 help="terminal harmonic matern GP nu")
    parser.add_argument("--term-grad-nu", type=float, default=1.0,                 help="terminal gradient matern GP nu")
    parser.add_argument("--term-curl-nu", type=float, default=1.0,                 help="terminal curl matern GP nu")


    problem_name = parser.parse_args().problem_name
    default_config, model_configs = {
        # 'ocean':        configs.get_ocean_default_configs,
        # 'plane':        configs.get_plane_default_configs,
        # 'earthquake':   configs.get_eqs_default_configs,
        # 'heatflow':     configs.get_hfs_default_configs,
        # 'traffic_pemsd':      configs.get_traffic_default_configs,
        # 'brain':        configs.get_brain_default_configs,
        'single_cell':  configs.get_single_cell_default_configs,
    }.get(problem_name)()
    parser.set_defaults(**default_config)

    opt = parser.parse_args()

    # ========= seed & torch setup =========
    if opt.seed is not None:
        seed = opt.seed
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.set_default_tensor_type('torch.cuda.FloatTensor')    
    # ========= auto setup & path handle =========
    opt.device='cuda:'+str(opt.gpu)
    opt.model_configs = model_configs
    if opt.lr is not None:
        opt.lr_f, opt.lr_b = opt.lr, opt.lr

    if opt.use_arange_t and opt.train_bs_t != opt.interval:
        print('[warning] reset opt.train_bs_t to {} since use_arange_t is enabled'.format(opt.interval))
        opt.train_bs_t = opt.interval

    opt.ckpt_path = os.path.join('checkpoint', opt.dir) + '/' + opt.sde_type + '_' + opt.forward_net  # for check point
    opt.log_fn =  opt.dir  + '/' + opt.sde_type + '_' + opt.forward_net # for log-tb
    os.makedirs(opt.ckpt_path, exist_ok=True)
    if opt.snapshot_freq: # for snapshot, forward and backward
        opt.eval_path = os.path.join('results', opt.dir) + '/' + opt.sde_type + '_' + opt.forward_net 
        os.makedirs(os.path.join(opt.eval_path, 'forward'), exist_ok=True)
        os.makedirs(os.path.join(opt.eval_path, 'backward'), exist_ok=True)

    return opt
