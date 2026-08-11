"""Vanilla conditional MLP baseline for geometry-conditioned V_m prediction.

Unlike Geo-DONet, this model has no branch/trunk decomposition. Every training
sample is one geometry/query pair:

    [theta_1, ..., theta_60, cobiveco_1, ..., cobiveco_4, time] -> V_m
"""

import re

import torch
import torch.nn as nn


class GeoMLP(nn.Module):
    def __init__(self, geo_dim=60, coord_dim=4, width=300, depth=4):
        super().__init__()
        input_dim = geo_dim + coord_dim + 1
        layers = []
        current = input_dim
        for _ in range(depth):
            layers.extend((nn.Linear(current, width), nn.Tanh()))
            current = width
        layers.append(nn.Linear(current, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs).squeeze(-1)


def paired_forward(model, theta, query):
    """Evaluate every case against a shared set of query points.

    theta: (B, geo_dim), query: (Q, coord_dim + 1) -> (B, Q).
    """
    batch, n_query = theta.shape[0], query.shape[0]
    theta_expanded = theta[:, None, :].expand(-1, n_query, -1)
    query_expanded = query[None, :, :].expand(batch, -1, -1)
    inputs = torch.cat((theta_expanded, query_expanded), dim=-1)
    return model(inputs.reshape(batch * n_query, -1)).reshape(batch, n_query)


def config_from_state_dict(state_dict):
    first = state_dict.get("net.0.weight")
    if first is None:
        raise KeyError("checkpoint has no net.0.weight; it is not a GeoMLP checkpoint")
    linear_indices = sorted(
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(r"net\.(\d+)\.weight", key))
    )
    if len(linear_indices) < 2:
        raise KeyError("GeoMLP checkpoint does not contain hidden and output layers")
    width, input_dim = first.shape
    depth = len(linear_indices) - 1
    output_width = state_dict[f"net.{linear_indices[-1]}.weight"].shape[0]
    if output_width != 1:
        raise ValueError(f"expected scalar GeoMLP output, got {output_width}")
    return {
        "input_dim": int(input_dim),
        "width": int(width),
        "depth": int(depth),
    }


def build_from_checkpoint_config(config, geo_dim, coord_dim):
    expected = geo_dim + coord_dim + 1
    if config["input_dim"] != expected:
        raise ValueError(
            f"checkpoint input dimension {config['input_dim']} does not match "
            f"data dimensions {geo_dim}+{coord_dim}+1={expected}"
        )
    return GeoMLP(
        geo_dim=geo_dim,
        coord_dim=coord_dim,
        width=config["width"],
        depth=config["depth"],
    )
