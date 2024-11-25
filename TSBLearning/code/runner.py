
import os, time, gc

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD, RMSprop, Adagrad, AdamW, lr_scheduler, Adam
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

import policy
import sde
from loss import compute_sb_nll_alternate_train
import data
import utils

from ipdb import set_trace as debug


def build_optimizer_ema_sched(opt, policy):
    direction = policy.direction

    optim_name = {
        'Adam': Adam,
        'AdamW': AdamW,
        'Adagrad': Adagrad,
        'RMSprop': RMSprop,
        'SGD': SGD,
    }.get(opt.optimizer)

    optim_dict = {
            "lr": opt.lr_f if direction=='forward' else opt.lr_b,
            'weight_decay':opt.l2_norm,
    }
    if opt.optimizer == 'SGD':
        optim_dict['momentum'] = 0.9

    optimizer = optim_name(policy.parameters(), **optim_dict)
    ema = ExponentialMovingAverage(policy.parameters(), decay=0.99)
    if opt.lr_gamma < 1.0:
        sched = lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
    else:
        sched = None

    return optimizer, ema, sched

def freeze_policy(policy):
    for p in policy.parameters():
        p.requires_grad = False
    policy.eval()
    return policy

def activate_policy(policy):
    for p in policy.parameters():
        p.requires_grad = True
    policy.train()
    return policy


class Runner():
    def __init__(self,opt):
        super(Runner,self).__init__()
        
        self.start_time=time.time()
        self.ts = torch.linspace(opt.t0, opt.T, opt.interval)

        # build boundary distribution (p: init, q: final)
        self.p, self.q = data.build_boundary_distribution(opt)

        # build dynamics, forward (z_f) and backward (z_b) policies
        self.dyn = sde.build(opt, self.p, self.q)
        
        self.z_f = policy.build(opt, self.dyn, 'forward')  # p -> q
        self.z_b = policy.build(opt, self.dyn, 'backward') # q -> p

        self.optimizer_f, self.ema_f, self.sched_f = build_optimizer_ema_sched(opt, self.z_f)
        self.optimizer_b, self.ema_b, self.sched_b = build_optimizer_ema_sched(opt, self.z_b)

        if opt.load:
            utils.restore_checkpoint(opt, self, opt.load)

        if opt.log_tb: # tensorboard related things
            self.it_f = 0
            self.it_b = 0
            self.writer=SummaryWriter(
                log_dir=os.path.join('runs', opt.log_fn) if opt.log_fn is not None else None
            )

    def update_count(self, direction):
        if direction == 'forward':
            self.it_f += 1
            return self.it_f
        elif direction == 'backward':
            self.it_b += 1
            return self.it_b
        else:
            raise RuntimeError()

    def get_optimizer_ema_sched(self, z):
        if z == self.z_f:
            return self.optimizer_f, self.ema_f, self.sched_f
        elif z == self.z_b:
            return self.optimizer_b, self.ema_b, self.sched_b
        else:
            raise RuntimeError()

    @torch.no_grad()
    def sample_train_data(self, opt, policy_opt, policy_impt, reused_sampler):
        train_ts = self.ts
        # reuse or sample training xs and zs
        try:
            reused_traj = next(reused_sampler)
            train_xs, train_zs = reused_traj[:,0,...], reused_traj[:,1,...]
            print('generate train data from [{}]!'.format(utils.green('reused samper')))
        except:
            _, ema, _ = self.get_optimizer_ema_sched(policy_opt)
            _, ema_impt, _ = self.get_optimizer_ema_sched(policy_impt)
            with ema.average_parameters(), ema_impt.average_parameters():
                policy_impt = freeze_policy(policy_impt)
                policy_opt  = freeze_policy(policy_opt)
                corrector = (lambda x,t: policy_impt(x,t) + policy_opt(x,t)) if opt.use_corrector else None
                xs, zs, _ = self.dyn.sample_traj(train_ts, policy_impt, corrector=corrector)
                train_xs = xs; del xs
                train_zs = zs; del zs
            print('generate train data from [{}]!'.format(utils.red('sampling')))

        assert train_xs.shape[0] == opt.samp_bs
        assert train_xs.shape[1] == len(train_ts)
        assert train_xs.shape == train_zs.shape
        gc.collect()

        return train_xs, train_zs, train_ts
    
    
    def sb_alternate_train_stage(self, opt, stage, epoch, direction, reused_sampler=None):
        policy_opt, policy_impt = {
            'forward':  [self.z_f, self.z_b], # train forwad,   sample from backward
            'backward': [self.z_b, self.z_f], # train backward, sample from forward
        }.get(direction)

        for ep in range(epoch):
            train_xs, train_zs, train_ts = self.sample_train_data(
                opt, policy_opt, policy_impt, reused_sampler
            )
            policy_impt = freeze_policy(policy_impt)
            policy_opt = activate_policy(policy_opt)
            self.sb_alternate_train_ep(
                opt, ep, stage, direction, train_xs, train_zs, train_ts, policy_opt, epoch
            )
    
    def sb_alternate_train_ep(
        self, opt, ep, stage, direction, train_xs, train_zs, train_ts, policy, num_epoch
    ):
        assert train_xs.shape[0] == opt.samp_bs
        assert train_zs.shape[0] == opt.samp_bs
        assert train_ts.shape[0] == opt.interval
        assert direction == policy.direction

        optimizer, ema, sched = self.get_optimizer_ema_sched(policy)

        for it in range(opt.num_itr):
            samp_x_idx = torch.randint(opt.samp_bs,  (opt.train_bs_x,))
            samp_t_idx = torch.randint(opt.interval, (opt.train_bs_t,))
            if opt.use_arange_t: samp_t_idx = torch.arange(opt.interval)

            # -------- build sample --------
            ts = train_ts[samp_t_idx].detach()
            xs = train_xs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)
            zs_impt = train_zs[samp_x_idx][:, samp_t_idx, ...].to(opt.device)

            optimizer.zero_grad()
            xs.requires_grad_(True) # we need this due to the later autograd.grad

            # -------- handle for batch_x and batch_t ---------
            # (batch, T, xdim) --> (batch*T, xdim)
            xs      = utils.flatten_dim01(xs)
            zs_impt = utils.flatten_dim01(zs_impt)
            ts = ts.repeat(opt.train_bs_x)
            assert xs.shape[0] == ts.shape[0]
            assert zs_impt.shape[0] == ts.shape[0]

            # -------- compute loss and backprop --------
            loss, zs = compute_sb_nll_alternate_train(
                opt, self.dyn, ts, xs, zs_impt, policy, return_z=True
            )
            assert not torch.isnan(loss)

            loss.backward()

            if opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm(policy.parameters(), opt.grad_clip)
            optimizer.step()
            ema.update()
            if sched is not None: sched.step()

            # -------- logging --------
            zs = utils.unflatten_dim01(zs, [len(samp_x_idx), len(samp_t_idx)])
            zs_impt = zs_impt.reshape(zs.shape)
            self.log_sb_alternate_train(
                opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch
            )

    def sb_alternate_train(self, opt):
        for stage in range(opt.num_stage):
            forward_ep = backward_ep = opt.num_epoch

            # train backward policy;
            train_backward = not (stage == 0 and opt.load is not None) 
            if train_backward:
                self.sb_alternate_train_stage(opt, stage, backward_ep, 'backward')

            # evaluate backward policy;
            reused_sampler = self.evaluate(opt, stage+1)

            # train forward policy
            self.sb_alternate_train_stage(
                opt, stage, forward_ep, 'forward', reused_sampler=reused_sampler
            )

        if opt.log_tb: self.writer.close()

    @torch.no_grad()
    def evaluate(self, opt, stage): 
        for z in [self.z_f, self.z_b]:
            print('evaluating z: {}'.format(z.direction))
            z = freeze_policy(z)
            xs, zs, _ = self.dyn.sample_traj(self.ts, z, save_traj=True, if_seed=True)
            fn = "stage{}-{}".format(stage, z.direction)
            utils.save_npy_traj(
                opt, fn, xs.detach().cpu().numpy(), direction=z.direction
            )
            # compute the (NLL)
            nll = self.dyn.compute_nll(opt.samp_bs, self.ts, self.z_f, self.z_b, z.direction)
            utils.save_nll(opt, nll.cpu().numpy(), stage, z.direction)
        
        utils.save_w_distance(opt, stage)
                 

    def _print_train_itr(self, it, loss, optimizer, num_itr, name):
        time_elapsed = utils.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] train_it {1}/{2} | lr:{3} | loss:{4} | time:{5}"
            .format(
                utils.magenta(name),
                utils.cyan("{}".format(1+it)),
                num_itr,
                utils.yellow("{:.2e}".format(lr)),
                utils.red("{:.4f}".format(loss.item())),
                utils.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))

    def log_sb_alternate_train(self, opt, it, ep, stage, loss, zs, zs_impt, optimizer, direction, num_epoch):
        time_elapsed = utils.get_time(time.time()-self.start_time)
        lr = optimizer.param_groups[0]['lr']
        print("[{0}] stage {1}/{2} | ep {3}/{4} | train_it {5}/{6} | lr:{7} | loss:{8} | time:{9}"
            .format(
                utils.magenta("SB {}".format(direction)),
                utils.cyan("{}".format(1+stage)),
                opt.num_stage,
                utils.cyan("{}".format(1+ep)),
                num_epoch,
                utils.cyan("{}".format(1+it+opt.num_itr*ep)),
                opt.num_itr*num_epoch,
                utils.yellow("{:.2e}".format(lr)),
                utils.red("{:+.4f}".format(loss.item())),
                utils.green("{0}:{1:02d}:{2:05.2f}".format(*time_elapsed)),
        ))
        # save the time elapsed in the end of each stage 
        if 1+it+opt.num_itr*ep == opt.num_itr*num_epoch:
            fn = opt.ckpt_path + "/time.txt"
            with open(fn, 'a') as f:
                # save the time
                f.write("{0}:{1:02d}:{2:05.2f} \n".format(*time_elapsed))
            
        if opt.log_tb:
            step = self.update_count(direction)
            self.log_tb(step, loss.detach(), 'loss', 'SB_'+direction)

    def log_tb(self, step, val, name, tag):
        self.writer.add_scalar(os.path.join(tag,name), val, global_step=step)        
        