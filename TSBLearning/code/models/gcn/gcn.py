import torch
from models.gcn.gcn_layer import GCNLayer


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        conv_order=1,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # First layer -- initial layer has the in_channels as input, and inter_channels as the output
        self.layers = torch.nn.ModuleList(
            [
                GCNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    conv_order=conv_order
                )
            ]
        )

        for i in range(n_layers - 1):
            if i == n_layers -2: 
                out_channels = 1
                update_func = 'id'
            else: 
                out_channels = hidden_channels
                update_func = 'relu'
            self.layers.append(
                GCNLayer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    conv_order=conv_order,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )
            

    def forward(self, x, laplacian):
        
        for layer in self.layers:
            x = layer(x, laplacian)

        return x