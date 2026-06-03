"""
Geo-DONet architecture  ──  functional MODULE 1: the network.

A geometry-conditioned DeepONet mapping a heart's PCA geometry code to its
transmembrane-voltage field V_m(x, t) on a fixed reference mesh:

    branch (geometry):   theta        (n_cases, geo_dim)         --MLP--> (n_cases, width)
    trunk  (space-time): query_points (n_query, coord_dim + 1)   --MLP--> (n_query, width)
    output:              einsum(branch, trunk)                        -> (n_cases, n_query)

Each trunk query point is a (space, time) pair, so the full field has
`n_query = n_nodes * n_frames`. That grid is large (~6M points), so
`chunked_forward` evaluates the trunk a few frames at a time while reusing the
single branch pass.

(The legacy code carried a learnable scalar bias its training path never added —
dropped here; the V_m offset is handled by normalization.)
"""
import re

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def build_mlp(input_dim, width, depth):
    """`depth` hidden layers of `width`, each Linear followed by Tanh."""
    layers, current_dim = [], input_dim
    for _ in range(depth):
        layers += [nn.Linear(current_dim, width), nn.Tanh()]
        current_dim = width
    return nn.Sequential(*layers)


class GeoDONet(nn.Module):
    def __init__(self, geo_dim=60, coord_dim=4, width=300, depth=4):
        super().__init__()
        self.branch = build_mlp(geo_dim, width, depth)        # theta        -> (n_cases, width)
        self.trunk = build_mlp(coord_dim + 1, width, depth)   # query_points -> (n_query, width)

    def forward(self, theta, query_points):
        """theta (n_cases, geo_dim); query_points (n_query, coord_dim+1)
        -> V_m (n_cases, n_query)."""
        branch_out = self.branch(theta)                       # (n_cases, width)
        trunk_out = self.trunk(query_points)                  # (n_query, width)
        return torch.einsum("nw,qw->nq", branch_out, trunk_out)


def build_trunk_chunks(coords, time, chunk_frames=25):
    """Pre-tile the (coords, t) trunk inputs once and split along time.

    `coords` (n_nodes, coord_dim) and `time` (n_frames,) must already be
    normalised and on the target device. Returns a list of
    (n_nodes * frames_in_chunk, coord_dim + 1) tensors.
    """
    n_nodes, coord_dim = coords.shape
    n_frames = time.shape[0]
    chunks = []
    for start in range(0, n_frames, chunk_frames):
        end = min(start + chunk_frames, n_frames)
        frames_in_chunk = end - start
        n_points = n_nodes * frames_in_chunk
        coords_tiled = coords.unsqueeze(1).expand(
            n_nodes, frames_in_chunk, coord_dim).reshape(n_points, coord_dim)
        time_tiled = time[start:end].unsqueeze(0).expand(
            n_nodes, frames_in_chunk).reshape(n_points, 1)
        chunks.append(torch.cat([coords_tiled, time_tiled], dim=1))
    return chunks


def chunked_forward(model, theta, trunk_chunks, n_nodes, use_checkpoint=False):
    """Evaluate V_m for all cases over the full grid, one time-chunk at a time.

    Returns (n_cases, n_nodes, n_frames). The branch is computed once and reused
    across chunks. `use_checkpoint` recomputes each trunk chunk during backward
    instead of storing its activations — those activations scale with the frame
    count (not the batch) and dominate training memory, so this is what keeps many
    frames (f301/f601) on a 48 GB GPU; ~1.3x slower, identical gradients. Has no
    effect under no_grad (val/inference).
    """
    n_cases = theta.shape[0]
    branch_out = model.branch(theta)                          # (n_cases, width)
    chunk_outputs = []
    for query_points in trunk_chunks:
        frames_in_chunk = query_points.shape[0] // n_nodes
        if use_checkpoint:
            trunk_out = checkpoint(model.trunk, query_points, use_reentrant=False)
        else:
            trunk_out = model.trunk(query_points)             # (n_points, width)
        vm_flat = torch.einsum("nw,qw->nq", branch_out, trunk_out)
        chunk_outputs.append(vm_flat.reshape(n_cases, n_nodes, frames_in_chunk))
    return torch.cat(chunk_outputs, dim=2)                    # (n_cases, n_nodes, n_frames)


# ════════════════════ loading checkpoints (any format) ════════════════════
# main.py infers a checkpoint's architecture from its weight shapes every run
# (config_from_state_dict) and rebuilds the normalizer from data — checkpoints are
# just weights. New (GeoDONet) checkpoints already use `branch`/`trunk` keys; the
# original `opnn` (Geo_DONet, Geo_DONet_ndiff) saved a structurally identical Tanh
# net under `_branch_g`/`_trunk` plus an unused scalar bias, which the helpers below
# translate. (SIREN checkpoints are NOT supported — their trunk is a sine layer.)

def is_legacy_state_dict(state_dict):
    """True for an original-opnn checkpoint (nested `_branch_g`/`_trunk` + scalar `bias`)."""
    return any(key == "bias" or key.startswith(("_branch_g.", "_trunk."))
               for key in state_dict)


def remap_legacy_state_dict(state_dict):
    """Translate an original-opnn state_dict to GeoDONet keys.

    Old `_branch_g.{i}.0.{weight,bias}` / `_trunk.{i}.0.{weight,bias}` (a Sequential
    of Sequential(Linear, Tanh)) -> flat `branch.{2i}.{...}` / `trunk.{2i}.{...}`
    (Linear, Tanh, Linear, ... with Tanh at the odd indices). The legacy scalar
    `bias` was added in the old forward but trained to ~0 and has no GeoDONet
    counterpart, so it is dropped; a non-negligible value is warned about.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key == "bias":
            if float(value.abs().max()) > 1e-6:
                print(f"warning: dropping nonzero legacy bias {float(value.flatten()[0]):.4g} "
                      f"— predictions shift by ~that amount in normalized V_m units")
            continue
        matched = re.fullmatch(r"(_branch_g|_trunk)\.(\d+)\.0\.(weight|bias)", key)
        if matched is None:
            raise KeyError(f"unrecognised legacy checkpoint key {key!r} "
                           f"(SIREN checkpoints are not supported)")
        net, layer, param = matched.groups()
        prefix = "branch" if net == "_branch_g" else "trunk"
        remapped[f"{prefix}.{2 * int(layer)}.{param}"] = value
    return remapped


def config_from_state_dict(state_dict):
    """Infer the GeoDONet(**config) kwargs from a GeoDONet-keyed state_dict
    (Linear weight is (out_features, in_features))."""
    if "branch.0.weight" not in state_dict or "trunk.0.weight" not in state_dict:
        raise KeyError("state_dict has no branch.0/trunk.0 — not a GeoDONet checkpoint")
    width, geo_dim = state_dict["branch.0.weight"].shape
    coord_dim = state_dict["trunk.0.weight"].shape[1] - 1            # trunk in = coords + time
    depth = sum(1 for key in state_dict if re.fullmatch(r"branch\.\d+\.weight", key))
    return {"geo_dim": int(geo_dim), "coord_dim": int(coord_dim),
            "width": int(width), "depth": int(depth)}


def build_model(config):
    """GeoDONet from a config dict (a checkpoint's `config` block or config_from_state_dict)."""
    return GeoDONet(geo_dim=config["geo_dim"], coord_dim=config["coord_dim"],
                    width=config["width"], depth=config["depth"])
