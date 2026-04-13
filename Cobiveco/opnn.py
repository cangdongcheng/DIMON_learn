"""
Author: Minglang Yin, minglang_yin@brown.edu
Fast training version DeepONet
Fixed naming and nesting to match checkpoint keys.
"""
import torch
import torch.nn as nn

class opnn(nn.Module):
    def __init__(self, branch1_dim, branch2_dim, trunk_dim):
        super(opnn, self).__init__()
        self.z_dim = trunk_dim[-1]

        ## build branch net for Geometry (_branch_g)
        modules = []
        for i, h_dim in enumerate(branch1_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch_g = nn.Sequential(*modules)

        ## build branch net for Pacing (_branch_p)
        modules = []
        for i, h_dim in enumerate(branch2_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._branch_p = nn.Sequential(*modules)

        ## build trunk net
        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                    )
                )
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)

    def forward(self, f, f_bc, x):
        """
        f: M*dim_f
        x: N*dim_x
        y_br: M*dim_h
        y_tr: N*dim_h
        y_out: y_br(ij) tensorprod y_tr(kj)
        """
        y_br1 = self._branch_g(f)
        y_br2 = self._branch_p(f_bc)
        y_br = y_br1 * y_br2

        y_tr = self._trunk(x)
        y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
        return y_out
    
    def loss(self, f, f_bc, x, y):
        y_out = self.forward(f, f_bc, x)
        loss = ((y_out - y)**2).mean()
        return loss