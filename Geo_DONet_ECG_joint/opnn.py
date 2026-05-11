"""
Geo-DONet-ECG: end-to-end geometry → V_m(x,t) + ECG(10,t).

1 branch + 1 trunk + ECG head:
  - Branch (geometry): theta (N, 60) → (N, p)
  - Trunk (spatiotemporal): (ab, rt, tm, tv, t) = (Q, 5) → (Q, p)
  - V_m output: einsum(branch, trunk) + bias → (N, M, T)
  - ECG head: V_m(M, T) → ECG(10, T) via learned transfer

Joint training: loss = loss_vm + lambda_ecg * loss_ecg
"""
import torch
import torch.nn as nn


class GeoDONetECG(nn.Module):
    def __init__(self, branch_geo_dim, trunk_dim,
                 n_nodes=50797, n_electrodes=10, ecg_mode='linear'):
        super().__init__()
        self.z_dim = trunk_dim[-1]

        ## Branch: geometry
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

        ## Trunk: spatiotemporal (5D input)
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

        ## Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

        ## ECG head
        self.ecg_mode = ecg_mode
        if ecg_mode == 'linear':
            self.ecg_head = nn.Linear(n_nodes, n_electrodes, bias=True)
        elif ecg_mode == 'mlp':
            self.ecg_head = nn.Sequential(
                nn.Linear(n_nodes, 256),
                nn.Tanh(),
                nn.Linear(256, n_electrodes),
            )

    def encode_trunk(self, xt):
        """Run trunk MLP on raw (Q, 5) input."""
        return self._trunk(xt)

    def forward(self, f_geo, xt):
        """
        f_geo: (N, geo_modes)
        xt:    (Q, 5)
        returns: (N, Q) V_m prediction
        """
        y_br = self._branch_g(f_geo)
        y_tr = self._trunk(xt)
        return torch.einsum("np,qp->nq", y_br, y_tr) + self.bias

    def predict_ecg(self, vm):
        """
        vm: (N, M, T) — predicted V_m on reference mesh
        returns: (N, T, 10) — electrode potentials
        """
        return self.ecg_head(vm.transpose(1, 2))
