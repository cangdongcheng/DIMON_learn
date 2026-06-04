"""
Geo-DONet (SIREN trunk) architecture  ──  functional MODULE 1: the network.

Same geometry-conditioned DeepONet as the clean Geo_DONet, but the trunk's Tanh
activations are replaced by SIREN sinusoids (Sitzmann et al., NeurIPS 2020) to
fit the sharp depolarisation upstroke the smooth Tanh trunk smears:

    branch (geometry):   theta        (n_cases, geo_dim)         --Tanh MLP-->  (n_cases, width)
    trunk  (space-time): query_points (n_query, coord_dim + 1)   --SIREN MLP--> (n_query, width)
    output:              einsum(branch, trunk)                                -> (n_cases, n_query)

The branch stays Tanh — the PCA geometry modes are smooth, so sine buys nothing
there. Only the trunk is sinusoidal. As in the clean baseline the per-query grid
is `n_nodes * n_frames` (~6M points), so `chunked_forward` evaluates the trunk a
few frames at a time and reuses the single branch pass. The legacy learnable
scalar bias is dropped (it trained to ~0; the V_m offset is handled by
normalization) — exactly as the clean GeoDONet did.

`omega_0` (default 30, the paper recipe for images/SDFs) sets the trunk's frequency
bandwidth: lower -> smoother, closer to a Tanh net; higher -> sharper features but
harder to train and more prone to overfit on a small cohort. It is a forward-time
multiplier, NOT a stored weight, so a checkpoint must carry it to be reproducible —
which is why these checkpoints save a `config` block (unlike the shape-only
GeoDONet ones). Legacy original-opnn SIREN checkpoints did not, so they load with
omega_0 defaulting to LEGACY_OMEGA_0 (30, what they were trained with).
"""
import math
import re

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

LEGACY_OMEGA_0 = 30.0   # original-opnn SIREN runs were all trained at omega_0 = 30


class SineLayer(nn.Module):
    """Linear -> sin(omega_0 * .), with the SIREN init that keeps pre-activations
    ~N(0,1) across depth so the sine doesn't saturate at startup."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                bound = math.sqrt(6.0 / in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


def build_tanh_mlp(input_dim, width, depth):
    """`depth` hidden layers of `width`, each Linear followed by Tanh (the branch)."""
    layers, current_dim = [], input_dim
    for _ in range(depth):
        layers += [nn.Linear(current_dim, width), nn.Tanh()]
        current_dim = width
    return nn.Sequential(*layers)


def build_siren_mlp(input_dim, width, depth, omega_0):
    """`depth` SineLayers of `width` (the trunk); the first uses the SIREN
    first-layer init, the rest the hidden init."""
    layers, current_dim = [], input_dim
    for i in range(depth):
        layers.append(SineLayer(current_dim, width, is_first=(i == 0), omega_0=omega_0))
        current_dim = width
    return nn.Sequential(*layers)


class GeoDONetSIREN(nn.Module):
    def __init__(self, geo_dim=60, coord_dim=4, width=300, depth=4, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.branch = build_tanh_mlp(geo_dim, width, depth)              # theta        -> (n_cases, width)
        self.trunk = build_siren_mlp(coord_dim + 1, width, depth, omega_0)  # query_points -> (n_query, width)

    def forward(self, theta, query_points):
        """theta (n_cases, geo_dim); query_points (n_query, coord_dim+1)
        -> V_m (n_cases, n_query)."""
        branch_out = self.branch(theta)                                 # (n_cases, width)
        trunk_out = self.trunk(query_points)                            # (n_query, width)
        return torch.einsum("nw,qw->nq", branch_out, trunk_out)


def build_trunk_chunks(coords, time, chunk_frames=10):
    """Pre-tile the (coords, t) trunk inputs once and split along time.

    `coords` (n_nodes, coord_dim) and `time` (n_frames,) must already be
    normalised and on the target device. Returns a list of
    (n_nodes * frames_in_chunk, coord_dim + 1) tensors. Default chunk is smaller
    than the Tanh baseline's (10 vs 25): the SIREN trunk's checkpointed
    activations peak higher, so fewer frames per chunk keeps f301/f601 on a 48 GB GPU.
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
    count (not the batch) and dominate training memory, which on the SIREN trunk
    tips a 48 GB GPU over; ~1.3x slower, identical gradients. Has no effect under
    no_grad (val/inference).
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
# main.py rebuilds the model from a checkpoint's `config` block every run and the
# normalizer from data — weights are just weights. New (GeoDONetSIREN) checkpoints
# save `config` (incl. omega_0, which is NOT inferable from weights). The original
# `opnn` SIREN checkpoints (Geo_DONet_SIREN runs 1142478 / 1147941 / 1148776) saved
# a structurally identical net under `_branch_g`/`_trunk` plus an unused scalar bias,
# with no config — those are remapped and assigned LEGACY_OMEGA_0 (30).

def is_legacy_state_dict(state_dict):
    """True for an original-opnn SIREN checkpoint (nested `_branch_g`/`_trunk` +
    scalar `bias`)."""
    return any(key == "bias" or key.startswith(("_branch_g.", "_trunk."))
               for key in state_dict)


def remap_legacy_state_dict(state_dict):
    """Translate an original-opnn SIREN state_dict to GeoDONetSIREN keys.

    Old branch `_branch_g.{i}.0.{w,b}` (Sequential of Sequential(Linear, Tanh)) ->
    flat `branch.{2i}.{w,b}` (Linear at even indices, Tanh between). Old trunk
    `_trunk.{i}.linear.{w,b}` (Sequential of SineLayer) -> `trunk.{i}.linear.{w,b}`
    (same structure, just the attribute rename). The legacy scalar `bias` was added
    in the old forward but trained to ~0 and has no GeoDONetSIREN counterpart, so it
    is dropped; a non-negligible value is warned about.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key == "bias":
            if float(value.abs().max()) > 1e-6:
                print(f"warning: dropping nonzero legacy bias {float(value.flatten()[0]):.4g} "
                      f"— predictions shift by ~that amount in normalized V_m units")
            continue
        branch_match = re.fullmatch(r"_branch_g\.(\d+)\.0\.(weight|bias)", key)
        if branch_match is not None:
            layer, param = branch_match.groups()
            remapped[f"branch.{2 * int(layer)}.{param}"] = value
            continue
        trunk_match = re.fullmatch(r"_trunk\.(\d+)\.linear\.(weight|bias)", key)
        if trunk_match is not None:
            layer, param = trunk_match.groups()
            remapped[f"trunk.{int(layer)}.linear.{param}"] = value
            continue
        raise KeyError(f"unrecognised legacy SIREN checkpoint key {key!r}")
    return remapped


def config_from_state_dict(state_dict, omega_0=LEGACY_OMEGA_0):
    """Infer the GeoDONetSIREN(**config) kwargs from a GeoDONetSIREN-keyed state_dict
    (Linear weight is (out_features, in_features)). omega_0 is NOT stored in the
    weights, so it is supplied by the caller (the saved `config` block for new
    checkpoints, else LEGACY_OMEGA_0)."""
    if "branch.0.weight" not in state_dict or "trunk.0.linear.weight" not in state_dict:
        raise KeyError("state_dict has no branch.0/trunk.0.linear — not a GeoDONetSIREN checkpoint")
    width, geo_dim = state_dict["branch.0.weight"].shape
    coord_dim = state_dict["trunk.0.linear.weight"].shape[1] - 1     # trunk in = coords + time
    depth = sum(1 for key in state_dict if re.fullmatch(r"branch\.\d+\.weight", key))
    return {"geo_dim": int(geo_dim), "coord_dim": int(coord_dim),
            "width": int(width), "depth": int(depth), "omega_0": float(omega_0)}


def build_model(config):
    """GeoDONetSIREN from a config dict (a checkpoint's `config` block or
    config_from_state_dict)."""
    return GeoDONetSIREN(geo_dim=config["geo_dim"], coord_dim=config["coord_dim"],
                         width=config["width"], depth=config["depth"],
                         omega_0=config.get("omega_0", LEGACY_OMEGA_0))
