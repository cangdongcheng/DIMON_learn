"""Multi-output Geo-DeepONet for activation time + aligned-PCA coefficients."""
import torch
import torch.nn as nn


def build_mlp(input_dim, width, depth):
    layers = []
    current = input_dim
    for _ in range(depth):
        layers.append(nn.Sequential(nn.Linear(current, width), nn.Tanh()))
        current = width
    return nn.Sequential(*layers)


class FeatureDeepONet(nn.Module):
    """Predict D nodewise features from geometry theta and spatial coordinate x.

    Shared branch/trunk embeddings are combined multiplicatively.  Each output
    has its own learned reduction of that interaction:

        y[b,n,d] = sum_w branch[b,w] * trunk[n,w] * head[d,w] + bias[d]

    Output 0 is activation time; outputs 1..D-1 are aligned temporal-PCA
    coefficients.  Keeping the legacy ``_branch_g``/``_trunk`` names allows an
    old AT-only Geo_DeepONet checkpoint to initialise these two encoders.
    """

    def __init__(self, geo_dim=60, coord_dim=4, width=200, depth=4, output_dim=6):
        super().__init__()
        self.geo_dim = geo_dim
        self.coord_dim = coord_dim
        self.width = width
        self.depth = depth
        self.output_dim = output_dim
        self._branch_g = build_mlp(geo_dim, width, depth)
        self._trunk = build_mlp(coord_dim, width, depth)
        self.head = nn.Linear(width, output_dim)

    def forward(self, theta, coords):
        branch = self._branch_g(theta)       # (B, W)
        trunk = self._trunk(coords)          # (N, W)
        output = torch.einsum("bw,nw,dw->bnd", branch, trunk, self.head.weight)
        return output + self.head.bias

    def config(self):
        return dict(geo_dim=self.geo_dim, coord_dim=self.coord_dim,
                    width=self.width, depth=self.depth, output_dim=self.output_dim)


def initialise_from_at_checkpoint(model, checkpoint_path, map_location="cpu"):
    """Transfer the branch/trunk weights from the legacy scalar AT model.

    The legacy AT output was sum(branch*trunk), so the new AT head is initialised
    to all ones.  Its bias and the coefficient heads remain trainable.  Exact AT
    reproduction is not assumed because the new target uses z-score rather than
    the legacy min-shift normalisation.
    """
    raw = torch.load(checkpoint_path, map_location=map_location)
    state = raw.get("model_state_dict", raw)
    own = model.state_dict()
    copied = []
    for key, value in state.items():
        if key.startswith(("_branch_g.", "_trunk.")) and key in own:
            if own[key].shape != value.shape:
                raise ValueError(f"legacy weight shape mismatch for {key}: "
                                 f"{tuple(value.shape)} vs {tuple(own[key].shape)}")
            own[key] = value
            copied.append(key)
    if not copied:
        raise ValueError(f"{checkpoint_path} has no compatible _branch_g/_trunk weights")
    model.load_state_dict(own)
    with torch.no_grad():
        model.head.weight[0].fill_(1.0)
        model.head.bias[0].zero_()
    return len(copied)
