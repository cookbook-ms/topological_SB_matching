import torch
import torch.nn as nn
from models.utils import *

def build_res(data_dim,num_res_block):
    return ResPolicy(data_dim, num_res_block=num_res_block)

class ResPolicy(torch.nn.Module):
    def __init__(self, data_dim, hidden_dim=256, time_embed_dim=128, num_res_block=1):
        super(ResPolicy,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )

        self.x_module = ResNet_FC(data_dim, hidden_dim, num_res_blocks=num_res_block)

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

    def forward(self,x, t):
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
        x_out = self.x_module(x)
        out   = self.out_module(x_out+t_out)
        return out