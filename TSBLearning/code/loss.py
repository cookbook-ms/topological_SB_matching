
import torch
import utils
from ipdb import set_trace as debug


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1

def sample_gaussian_like(y):
    return torch.randn_like(y)

def sample_e(opt, x):
    return {
        'gaussian': sample_gaussian_like,
        'rademacher': sample_rademacher_like,
    }.get(opt.noise_type)(x)

def compute_div_gz(opt, dyn, ts, xs, policy, return_zs=False):
    zs = policy(xs,ts)
    g_ts = dyn.g(ts)
    g_ts = g_ts[:,None]
    gzs = g_ts*zs
    e = sample_e(opt, xs)
    e_dzdx = torch.autograd.grad(gzs, xs, e, create_graph=True)[0]
    div_gz = e_dzdx * e

    return [div_gz, zs] if return_zs else div_gz

def compute_sb_nll_alternate_train(opt, dyn, ts, xs, zs_impt, policy_opt, return_z=False):
    assert opt.train_method == 'alternate'
    assert xs.requires_grad
    assert not zs_impt.requires_grad
    batch_x = opt.train_bs_x
    batch_t = opt.train_bs_t

    with torch.enable_grad():
        div_gz, zs = compute_div_gz(opt, dyn, ts, xs, policy_opt, return_zs=True)
        loss = zs*(0.5*zs + zs_impt) + div_gz
        loss = torch.sum(loss * dyn.dt) / batch_x / batch_t 
    return loss, zs if return_z else loss
