"""
Geo-DeepONet: two-branch MIONet (geometry + trunk, no pacing branch).
Adapted from Cobiveco_with_fiber/opnn.py with pacing branch removed.
"""
import torch
import torch.nn as nn


class opnn(nn.Module):
    def __init__(self, branch_geo_dim, trunk_dim):
        super(opnn, self).__init__()
        self.z_dim = trunk_dim[-1]

        ## build branch net for Geometry (_branch_g)
        modules = []
        for i, h_dim in enumerate(branch_geo_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                ))
                in_channels = h_dim
        self._branch_g = nn.Sequential(*modules)

        ## build trunk net
        modules = []
        for i, h_dim in enumerate(trunk_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                ))
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)

    def forward(self, f_geo, x):
        """
        f_geo: (N, geo_modes) — PCA geometry coefficients
        x:     (M, 4)         — Cobiveco coordinates on reference
        returns: (N, M)        — predicted activation times
        """
        y_br = self._branch_g(f_geo)     # (N, p)
        y_tr = self._trunk(x)             # (M, p)
        y_out = torch.einsum("ij,kj->ik", y_br, y_tr)
        return y_out

    def loss(self, f_geo, x, y):
        y_out = self.forward(f_geo, x)
        return ((y_out - y) ** 2).mean()
