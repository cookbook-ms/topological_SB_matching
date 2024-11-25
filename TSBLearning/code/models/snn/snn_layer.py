"""Simplicial Neural Network Layer."""
import torch
from torch.nn.parameter import Parameter


class SNNLayer(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_order_down = conv_order_down
        self.conv_order_up = conv_order_up
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization
        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weight = Parameter(
            torch.Tensor(
                self.in_channels,
                self.out_channels,
                1 + self.conv_order_down + self.conv_order_up,
            )
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414) -> None:
       
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)

    def aggr_norm_func(self, conv_operator, x):

        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)
        if self.update_func == "id":
            return x
        
        return None

    def chebyshev_conv(self, conv_operator, conv_order, x):
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order))
        X[:, :, 0] = torch.mm(conv_operator, x)
        for k in range(1, conv_order):
            X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
            if self.aggr_norm:
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])

        return X

    def forward(self, x, laplacian_down, laplacian_up):
        num_simplices, _ = x.shape
        x_identity = torch.unsqueeze(x, 2)

        if self.conv_order_down > 0 and self.conv_order_up > 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_down, x_up), 2)
        elif self.conv_order_down > 0 and self.conv_order_up == 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x = torch.cat((x_identity, x_down), 2)
        elif self.conv_order_down == 0 and self.conv_order_up > 0:
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_up), 2)

        y = torch.einsum("nik,iok->no", x, self.weight)

        if self.update_func is None:
            return y

        return self.update(y)