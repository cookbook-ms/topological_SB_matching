import torch
import torch.nn as nn
from models.utils import *
from models.gcn.gcn import GCN

def build_gcn(data_dim):
    return GCNPolicy(data_dim)

class GCNPolicy(torch.nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_embed_dim=128):
        super(GCNPolicy,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module1 = GCN(in_channels=1, hidden_channels=hidden_dim, n_layers=2)
        self.x_module2 = ResNet_FC(data_dim, hidden_dim, num_res_blocks=0)
        
        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, data_dim),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, t, lap):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        """

        # make sure t.shape = [T]
        if len(t.shape)==0:
            t=t[None]

        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x = x.unsqueeze(-1)
        x_out = torch.empty_like(x)
        for i in range(x.shape[0]):
            x_out[i] = self.x_module1(x[i], lap)
            
        x_out = x_out.squeeze(-1)    
        x_out = self.x_module2(x_out)
        out   = self.out_module(x_out+t_out)
        
        return out