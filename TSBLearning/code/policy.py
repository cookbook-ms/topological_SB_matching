
import torch
import utils
import data

from ipdb import set_trace as debug

def build(opt, dyn, direction):
    print(utils.magenta("build {} policy...".format(direction)))

    net_name = getattr(opt, direction+'_net')
    net = _build_net(opt, net_name, direction)
    use_t_idx = (net_name in ['res', 'snn', 'gcn'])
    
    print('Using {} network'.format(net_name))
    if net_name == 'snn':  
        print('Using SNN policy')
        policy = SNNSchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, 
        lap_down=data.OceanDataProcessor(opt.problem_name).get_laplacians[1],
        lap_up=data.OceanDataProcessor(opt.problem_name).get_laplacians[2]
        )
    elif net_name == 'gcn':
        print('Using GCN policy')
        policy = GCNSchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx, 
        lap=data.GraphDataProcessor(problem_name=opt.problem_name).L
        )
    else:
        print('Using RES policy')
        policy = SchrodingerBridgePolicy(
        opt, direction, dyn, net, use_t_idx=use_t_idx
        )
    
    print(utils.red('number of parameters is {}'.format(utils.count_parameters(policy))))
    policy.to(opt.device)

    return policy

def _build_net(opt, net_name, direction=None): 
    if net_name == 'res':
        assert utils.is_sc_dataset(opt) or utils.is_graph_dataset(opt)
        from models.res.ResPolicy import build_res
        if utils.is_sc_dataset(opt):
            net = build_res(data.OceanDataProcessor(opt.problem_name).get_data_dim,opt.num_res_block)
        elif utils.is_graph_dataset(opt):
            net = build_res(data.GraphDataProcessor(problem_name=opt.problem_name).get_data_dim,opt.num_res_block)
    elif net_name == 'snn':
        from models.snn.SNNPolicy import build_scnn
        if utils.is_sc_dataset(opt):
            net = build_scnn(data.OceanDataProcessor(opt.problem_name).get_data_dim)
        elif utils.is_graph_dataset(opt):
            net = build_scnn(data.GraphDataProcessor(problem_name=opt.problem_name).get_data_dim)
    elif net_name == 'gcn':
        from models.gcn.GCNPolicy import build_gcn
        net = build_gcn(data.GraphDataProcessor(problem_name=opt.problem_name).get_data_dim)
    else:
        raise RuntimeError()
    return net

class SchrodingerBridgePolicy(torch.nn.Module):
    def __init__(self, opt, direction, dyn, net, use_t_idx=False):
        super(SchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx

    def forward(self, x, t):
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        out = self.net(x, t)
        return out


class SNNSchrodingerBridgePolicy(torch.nn.Module):
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, lap_down=None, lap_up=None):
        super(SNNSchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx

        if opt.problem_name in ['ocean','plane']:
            self.lap_down = torch.Tensor(lap_down.toarray()).to_sparse_csr()
            self.lap_up = torch.Tensor(lap_up.toarray()).to_sparse_csr()
        elif opt.problem_name in ['traffic_pemsd']:
            self.lap_down = torch.Tensor(lap_down).to_sparse_csr()
            self.lap_up = torch.Tensor(lap_up).to_sparse_csr()

    def forward(self, x, t):
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        out = self.net(x, t, self.lap_down, self.lap_up)
        return out
    
    
class GCNSchrodingerBridgePolicy(torch.nn.Module):
    def __init__(self, opt, direction, dyn, net, use_t_idx=False, lap=None,):
        super(GCNSchrodingerBridgePolicy,self).__init__()
        self.opt = opt
        self.direction = direction
        self.dyn = dyn
        self.net = net
        self.use_t_idx = use_t_idx
        self.lap = lap

    def forward(self, x, t):
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(x.shape[0])
        assert t.dim()==1 and t.shape[0] == x.shape[0]

        if self.use_t_idx:
            t = t / self.opt.T * self.opt.interval

        out = self.net(x, t, self.lap)
        return out
    