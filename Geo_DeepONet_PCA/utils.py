"""Data, normalisation, phase-decoder, and evaluation helpers."""
import numpy as np
import torch
import torch.nn as nn


def split_indices(n_cases, n_train=95, n_val=5):
    indices = np.arange(n_cases)
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def load_feature_data(path):
    npz = np.load(path, allow_pickle=True)
    required = {"theta", "coords", "targets", "target_names"}
    missing = required.difference(npz.files)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}; run prepare_data.py")
    return dict(theta=npz["theta"].astype(np.float32),
                coords=npz["coords"].astype(np.float32),
                targets=npz["targets"].astype(np.float32),
                target_names=npz["target_names"].astype(str),
                case_names=npz["case_names"] if "case_names" in npz.files else None,
                time=npz["time"].astype(np.float32) if "time" in npz.files else None)


class FeatureNormalizer:
    """Training-only normalisation for geometry, coordinates, and each target."""

    def __init__(self, theta_train, coords, target_train):
        self.theta_mean = theta_train.mean(axis=0).astype(np.float32)
        self.theta_std = theta_train.std(axis=0).astype(np.float32)
        self.theta_std[self.theta_std < 1e-8] = 1.0
        self.coord_min = coords.min(axis=0).astype(np.float32)
        self.coord_max = coords.max(axis=0).astype(np.float32)
        self.target_mean = target_train.mean(axis=(0, 1)).astype(np.float32)
        self.target_std = target_train.std(axis=(0, 1)).astype(np.float32)
        self.target_std[self.target_std < 1e-8] = 1.0

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        for key in ("theta_mean", "theta_std", "coord_min", "coord_max",
                    "target_mean", "target_std"):
            setattr(obj, key, np.asarray(state[key], dtype=np.float32))
        return obj

    def state(self):
        return {key: getattr(self, key) for key in
                ("theta_mean", "theta_std", "coord_min", "coord_max",
                 "target_mean", "target_std")}

    def theta(self, value):
        return (value - self.theta_mean) / self.theta_std

    def coords(self, value):
        return (value - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)

    def targets(self, value):
        return (value - self.target_mean) / self.target_std

    def targets_inverse(self, value):
        return value * self.target_std + self.target_mean


def activation_time(vm, time_ms, threshold=-10.0):
    crossed = (vm[:, :-1] < threshold) & (vm[:, 1:] >= threshold)
    nodes = np.where(crossed.any(axis=1))[0]
    first = np.argmax(crossed, axis=1)[nodes]
    result = np.full(vm.shape[0], np.nan, dtype=np.float32)
    v0, v1 = vm[nodes, first], vm[nodes, first + 1]
    t0, t1 = time_ms[first], time_ms[first + 1]
    result[nodes] = t0 + (threshold - v0) / (v1 - v0 + 1e-12) * (t1 - t0)
    return result


def shift_waveforms(waves, shift_ms, dt):
    """Linear temporal shift with constant edge padding; see phase analysis."""
    waves = np.asarray(waves, dtype=np.float32)
    shift_ms = np.asarray(shift_ms, dtype=np.float32)
    n_rows, n_frames = waves.shape
    position = (np.arange(n_frames, dtype=np.float32)[None, :]
                + shift_ms[:, None] / np.float32(dt))
    np.clip(position, 0.0, float(n_frames - 1), out=position)
    left = np.floor(position).astype(np.int32)
    np.minimum(left, n_frames - 2, out=left)
    fraction = position - left
    y0 = np.take_along_axis(waves, left, axis=1)
    y1 = np.take_along_axis(waves, left + 1, axis=1)
    return y0 + fraction * (y1 - y0)


def shifted_in_chunks(waves, shifts, dt, chunk_nodes=20_000):
    output = np.empty_like(waves, dtype=np.float32)
    for start in range(0, waves.shape[0], chunk_nodes):
        end = min(start + chunk_nodes, waves.shape[0])
        output[start:end] = shift_waveforms(waves[start:end], shifts[start:end], dt)
    return output


def load_decoder_basis(path, n_components=None):
    npz = np.load(path, allow_pickle=False)
    required = {"node_template", "residual_mean", "components", "time",
                "reference_at", "at_threshold"}
    missing = required.difference(npz.files)
    if missing:
        raise ValueError(f"decoder basis {path} is missing {sorted(missing)}")
    components = npz["components"].astype(np.float32)
    if n_components is not None:
        if components.shape[1] < n_components:
            raise ValueError(f"basis has {components.shape[1]} modes, need {n_components}")
        components = components[:, :n_components]
    return dict(node_template=npz["node_template"].astype(np.float32),
                residual_mean=npz["residual_mean"].astype(np.float32),
                components=components,
                time=npz["time"].astype(np.float32),
                reference_at=float(npz["reference_at"]),
                at_threshold=float(npz["at_threshold"]))


def decode_features(features, basis, chunk_nodes=20_000):
    """Decode physical [AT, coefficients...] for one heart to V_m(N,T)."""
    features = np.asarray(features, dtype=np.float32)
    n_components = features.shape[1] - 1
    components = basis["components"][:, :n_components]
    if features.shape[0] != basis["node_template"].shape[0]:
        raise ValueError("feature and lookup-table node counts differ")
    aligned = (basis["node_template"] + basis["residual_mean"]
               + features[:, 1:] @ components.T)
    shifts = np.float32(basis["reference_at"]) - features[:, 0]
    dt = float(np.median(np.diff(basis["time"])))
    return shifted_in_chunks(aligned, shifts, dt, chunk_nodes)


class DifferentiablePhaseDecoder(nn.Module):
    """PyTorch version of :func:`decode_features` for waveform supervision.

    The decoder has no trainable parameters.  It reconstructs aligned waveforms
    from node templates and PCA coefficients, then applies the activation-time
    shift with linear interpolation. Gradients flow to both predicted AT and PCA
    coefficients (except at the measure-zero interpolation-cell boundaries and
    where constant edge padding is active).

    ``forward`` intentionally accepts a node subset. This avoids materialising
    every heart x 50,797 nodes x 601 frames during training.
    """

    def __init__(self, basis, n_components):
        super().__init__()
        components = np.asarray(basis["components"][:, :n_components],
                                dtype=np.float32)
        node_template = np.asarray(basis["node_template"], dtype=np.float32)
        residual_mean = np.asarray(basis["residual_mean"], dtype=np.float32)
        time_ms = np.asarray(basis["time"], dtype=np.float32)
        if node_template.shape[1] != len(time_ms):
            raise ValueError("decoder node template and time grid disagree")
        if components.shape != (len(time_ms), n_components):
            raise ValueError("decoder components have an unexpected shape")
        self.register_buffer("node_template", torch.from_numpy(node_template))
        self.register_buffer("residual_mean", torch.from_numpy(residual_mean))
        self.register_buffer("components", torch.from_numpy(components))
        self.register_buffer("frame_index", torch.arange(len(time_ms),
                                                          dtype=torch.float32))
        self.reference_at = float(basis["reference_at"])
        self.dt = float(np.median(np.diff(time_ms)))
        self.n_components = int(n_components)

    def forward(self, physical_features, node_indices):
        """Decode ``(B,S,1+K)`` physical features to ``V_m(B,S,T)``.

        ``node_indices`` contains the S canonical mesh-node indices. The exact
        same constant-edge, linear temporal shift is used by the NumPy decoder.
        """
        if physical_features.ndim != 3:
            raise ValueError("physical_features must have shape (batch,nodes,channels)")
        if physical_features.shape[2] != self.n_components + 1:
            raise ValueError(f"decoder expects {self.n_components + 1} channels, "
                             f"got {physical_features.shape[2]}")
        node_indices = torch.as_tensor(node_indices, dtype=torch.long,
                                       device=physical_features.device)
        if node_indices.ndim != 1 or len(node_indices) != physical_features.shape[1]:
            raise ValueError("node_indices must be 1D and match the feature node axis")

        coefficients = physical_features[..., 1:]
        aligned = (self.node_template[node_indices][None, :, :]
                   + self.residual_mean[None, None, :]
                   + torch.einsum("bsk,tk->bst", coefficients, self.components))

        activation_time_ms = physical_features[..., 0]
        position = (self.frame_index[None, None, :]
                    + (self.reference_at - activation_time_ms[..., None]) / self.dt)
        position = position.clamp(0.0, float(aligned.shape[2] - 1))
        left = torch.floor(position).to(torch.long)
        left = left.clamp_max(aligned.shape[2] - 2)
        fraction = position - left.to(position.dtype)
        y0 = torch.gather(aligned, 2, left)
        y1 = torch.gather(aligned, 2, left + 1)
        return y0 + fraction * (y1 - y0)


def to_numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def vm_metrics(prediction, truth):
    difference = prediction - truth
    rel_l2 = float(np.linalg.norm(difference) / (np.linalg.norm(truth) + 1e-12))
    return rel_l2, float(np.abs(difference).mean())


def at_metrics(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    if not valid.any():
        return np.nan, np.nan
    difference = prediction[valid] - truth[valid]
    rel_l2 = float(np.linalg.norm(difference) / (np.linalg.norm(truth[valid]) + 1e-12))
    return rel_l2, float(np.abs(difference).mean())


def median_max_dvdt(vm, dt):
    return float(np.median(np.abs(np.gradient(vm, dt, axis=1)).max(axis=1)))
