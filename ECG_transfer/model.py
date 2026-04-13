"""
ECG transfer model: V_m(x,t) → ECG(10,t).

Approach B: learn the mapping from transmembrane voltage on the
canonical reference mesh (50797 nodes) to 10 electrode potentials.

Three variants:
  1. Linear: W(10, 50797) applied per time step — neural transfer matrix
  2. MLP: per-time-step V_m(50797) → hidden → ECG(10)
  3. TemporalConv: 1D conv along time after spatial compression
"""
import torch
import torch.nn as nn


class LinearTransfer(nn.Module):
    """Learned linear transfer matrix. V_m(M) → ECG(10) per time step."""
    def __init__(self, n_nodes=50797, n_electrodes=10):
        super().__init__()
        self.W = nn.Linear(n_nodes, n_electrodes, bias=True)

    def forward(self, vm):
        """
        vm: (N, M, T) — V_m on reference mesh
        returns: (N, T, 10) — electrode potentials
        """
        # Transpose to (N, T, M), apply linear, get (N, T, 10)
        return self.W(vm.transpose(1, 2))


class MLPTransfer(nn.Module):
    """MLP transfer: V_m(M) → hidden → ECG(10) per time step."""
    def __init__(self, n_nodes=50797, n_electrodes=10, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_nodes, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_electrodes),
        )

    def forward(self, vm):
        """
        vm: (N, M, T)
        returns: (N, T, 10)
        """
        return self.net(vm.transpose(1, 2))


class TemporalConvTransfer(nn.Module):
    """Spatial compression + temporal 1D conv."""
    def __init__(self, n_nodes=50797, n_electrodes=10, spatial_hidden=64, n_conv_layers=3):
        super().__init__()
        # Spatial compression: V_m(M) → (spatial_hidden) per time step
        self.spatial = nn.Sequential(
            nn.Linear(n_nodes, spatial_hidden),
            nn.Tanh(),
        )
        # Temporal conv: (N, spatial_hidden, T) → (N, n_electrodes, T)
        layers = []
        in_ch = spatial_hidden
        for i in range(n_conv_layers - 1):
            layers.extend([
                nn.Conv1d(in_ch, spatial_hidden, kernel_size=5, padding=2),
                nn.Tanh(),
            ])
            in_ch = spatial_hidden
        layers.append(nn.Conv1d(in_ch, n_electrodes, kernel_size=5, padding=2))
        self.temporal = nn.Sequential(*layers)

    def forward(self, vm):
        """
        vm: (N, M, T)
        returns: (N, T, 10)
        """
        # Spatial: (N, T, M) → (N, T, H)
        h = self.spatial(vm.transpose(1, 2))
        # Temporal: (N, H, T) → (N, 10, T) → (N, T, 10)
        out = self.temporal(h.transpose(1, 2))
        return out.transpose(1, 2)
