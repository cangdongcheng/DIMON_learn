import torch
import torch.nn as nn

class opnn(nn.Module):
    def __init__(self, branch_g_dim, branch_s_dim, branch_p_dim, trunk_dim):
        super(opnn, self).__init__()

        ## build branch net geometry (60D PCA)
        modules = []
        for i, h_dim in enumerate(branch_g_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                ))
                in_channels = h_dim
        self._branch_g = nn.Sequential(*modules)

        ## build branch net scar (60D SDF PCA)
        modules = []
        for i, h_dim in enumerate(branch_s_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                ))
                in_channels = h_dim
        self._branch_s = nn.Sequential(*modules)

        ## build branch net pace (4D Pacing)
        modules = []
        for i, h_dim in enumerate(branch_p_dim):
            if i == 0:
                in_channels = h_dim
            else:
                modules.append(nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh()
                ))
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
                ))
                in_channels = h_dim
        self._trunk = nn.Sequential(*modules)

    def forward(self, f_g, f_s, x_pace, x):
        """
        f_g: M * dim_g (Geometry PCA)
        f_s: M * dim_s (Scar SDF PCA)
        x_pace: M * S * dim_p (Pacing parameters)
        x: N * dim_x (Trunk coordinates)
        """

        y_br_g = self._branch_g(f_g)     # [num_cases, dim_latent]
        y_br_s = self._branch_s(f_s)     # [num_cases, dim_latent]
        y_br_p = self._branch_p(x_pace)  # [num_cases, num_sims, dim_latent]
        
        # Add a dummy dimension to the geometry and scar outputs to broadcast properly
        # across the simulation dimension (S) of the pacing branch.
        # Element-wise Hadamard product of all three branches
        y_ = y_br_p * y_br_g.unsqueeze(1) * y_br_s.unsqueeze(1)

        y_tr = self._trunk(x)            # [num_nodes, dim_latent]
        
        # Inner product between merged branch outputs and trunk output
        y_out = torch.einsum("ijk,lk->ijl", y_, y_tr)
        
        return y_out