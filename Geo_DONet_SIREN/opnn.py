"""
Geo-DONet (SIREN trunk) — DeepONet for V_m(x,t) with sinusoidal trunk
activations to fit the sharp depolarisation upstroke that the Tanh trunk
smooths out.

Architecture:
  - Branch (geometry): theta (N, geo_modes) → (N, p), Tanh
        Geometry PCA modes are smooth; sine buys nothing here.
  - Trunk (spatiotemporal): (ab, rt, tm, tv, t) = (Q, 5) → (Q, p),
        sin(omega_0 * Wx + b) in every layer (SIREN-style).
  - Output: einsum + bias → (N, Q).

SIREN init (Sitzmann et al., NeurIPS 2020) — preserves N(0,1) pre-activations
across depth so the sin doesn't saturate at startup:
  First layer:        W ~ U(-1/n_in,                  1/n_in)
  Hidden + last:      W ~ U(-sqrt(6/n_in)/omega_0,    sqrt(6/n_in)/omega_0)

omega_0 controls the network's frequency bandwidth. Paper default is 30
(images, SDFs, audio). Lower → smoother (closer to a Tanh net); higher →
sharper but harder to train and more prone to overfit on small cohorts.
"""
import math
import torch
import torch.nn as nn


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features,
                                            1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class opnn(nn.Module):
    def __init__(self, branch_geo_dim, trunk_dim, omega_0=30.0):
        super().__init__()
        self.z_dim = trunk_dim[-1]
        self.omega_0 = omega_0

        ## Branch: Tanh (unchanged from baseline)
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

        ## Trunk: SIREN — sin activations on every layer including the last,
        ## mirroring the baseline's Tanh-on-every-layer trunk so the einsum
        ## sees a comparably bounded basis.
        layers = []
        for i in range(1, len(trunk_dim)):
            layers.append(SineLayer(
                trunk_dim[i - 1], trunk_dim[i],
                is_first=(i == 1), omega_0=omega_0,
            ))
        self._trunk = nn.Sequential(*layers)

        ## Learnable bias
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
        y_br = self._branch_g(f_geo)   # (N, p)
        y_tr = self._trunk(xt)          # (Q, p)
        return torch.einsum("np,qp->nq", y_br, y_tr) + self.bias
