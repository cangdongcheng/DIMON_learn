"""
Geo-DONet: DeepONet for geometry-dependent V_m(x,t) prediction.

1 branch + 1 trunk:
  - Branch (geometry): theta (N, geo_modes) → (N, p)
  - Trunk (spatiotemporal): (ab, rt, tm, tv, t) = (Q, 5) → (Q, p)
  - Output: einsum + bias → (N, Q) transmembrane voltage

The trunk takes raw 5D input (4 Cobiveco + 1 normalized time).
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

        ## build trunk net (spatiotemporal)
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

        ## learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

    def encode_trunk(self, xt):
        """Run trunk MLP on raw (Q, 5) spatiotemporal input."""
        return self._trunk(xt)

    def forward(self, f_geo, xt):
        """
        f_geo: (N, geo_modes) — PCA geometry coefficients
        xt:    (Q, 5)         — query points (ab, rt, tm, tv, t)
        returns: (N, Q)        — predicted V_m at each query point
        """
        y_br = self._branch_g(f_geo)  # (N, p)
        y_tr = self._trunk(xt)         # (Q, p)
        y_out = torch.einsum("np,qp->nq", y_br, y_tr) + self.bias
        return y_out
