"""Simplicial neural network implementation"""
import torch
from models.snn.snn_layer import SNNLayer


class SNN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        conv_order_down=1,
        conv_order_up=1,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # First layer -- initial layer has the in_channels as input, and inter_channels as the output
        self.layers = torch.nn.ModuleList(
            [
                SNNLayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    conv_order_down=conv_order_down,
                    conv_order_up=conv_order_up,
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
                SNNLayer(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    conv_order_down=conv_order_down,
                    conv_order_up=conv_order_up,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )
            

    def forward(self, x, laplacian_down, laplacian_up):
        
        for layer in self.layers:
            x = layer(x, laplacian_down, laplacian_up)
 

        return x