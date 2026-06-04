"""
Geo-DONet (SIREN) helpers  ──  functional MODULE 2 (evaluation) and MODULE 3 (data).

A self-contained copy of the clean Geo_DONet utils (data + evaluation), plus the
signed-Laplacian neighbour-difference ("ndiff") loss used by the SIREN+ndiff
experiments. The run Configuration and the CLI both live in main.py; these
functions take their parameters explicitly so they stay free of global defaults.
"""
import numpy as np
import torch


# ═══════════════════════ MODULE 3 — data ═══════════════════════
def load_dataset(path):
    """Training npz -> float32 arrays: theta (n_cases, 60), coords (n_nodes, 4),
    vm (n_cases, n_nodes, n_frames), time (n_frames,), case_names (n_cases,) or None."""
    npz = np.load(path, allow_pickle=True)
    return {
        "theta": npz["theta"].astype(np.float32),
        "coords": npz["coords"].astype(np.float32),
        "vm": npz["vm"].astype(np.float32),
        "time": npz["time"].astype(np.float32),
        "case_names": npz["case_names"] if "case_names" in npz.files else None,
    }


def load_inputs(path):
    """Inference npz: at least `theta` and Cobiveco coords (key `coords` or
    `cobiveco`); optionally `vm` (GT), `time`, `case_names`. Returns the same
    dict shape as load_dataset, with vm/time/case_names = None when absent."""
    npz = np.load(path, allow_pickle=True)
    files = set(npz.files)
    if "theta" not in files:
        raise SystemExit(f"input {path} is missing 'theta'")
    coord_key = "coords" if "coords" in files else "cobiveco" if "cobiveco" in files else None
    if coord_key is None:
        raise SystemExit(f"input {path} is missing Cobiveco coords ('coords' or 'cobiveco')")
    return {
        "theta": npz["theta"].astype(np.float32),
        "coords": npz[coord_key].astype(np.float32),
        "vm": npz["vm"].astype(np.float32) if "vm" in files else None,
        "time": npz["time"].astype(np.float32) if "time" in files else None,
        "case_names": npz["case_names"] if "case_names" in files else None,
    }


def split_indices(n_cases, n_train, n_val):
    """Train / val / test by index order: first n_train, next n_val, rest test."""
    indices = np.arange(n_cases)
    return (indices[:n_train],
            indices[n_train:n_train + n_val],
            indices[n_train + n_val:])


class Normalizer:
    """Per-field normalization, fit on the TRAIN split only. Rebuilt from the
    training npz on every run (checkpoints store only weights), never serialized.

        theta : z-score              coords: per-feature min-max -> [0,1]
        vm    : (x - min) / std      time  : min-max -> [0,1]
    """

    def __init__(self, theta_train, vm_train, coords, time):
        self.theta_mean = theta_train.mean(0)
        self.theta_std = theta_train.std(0)
        self.theta_std[self.theta_std < 1e-8] = 1.0
        self.vm_shift = float(vm_train.min())
        self.vm_scale = float(vm_train.std())
        self.coord_min = coords.min(0)
        self.coord_max = coords.max(0)
        self.time_min = float(time.min())
        self.time_max = float(time.max())

    def theta(self, values):       return (values - self.theta_mean) / self.theta_std
    def vm(self, values):          return (values - self.vm_shift) / self.vm_scale
    def vm_inverse(self, values):  return values * self.vm_scale + self.vm_shift
    def coords(self, values):      return (values - self.coord_min) / (self.coord_max - self.coord_min + 1e-8)
    def time(self, values):        return (values - self.time_min) / (self.time_max - self.time_min + 1e-8)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)


# ════════════════════ MODULE 2 — evaluation ════════════════════
def mse(prediction, target):
    """Mean-squared error (torch) — the training loss / val monitoring metric."""
    return ((prediction - target) ** 2).mean()


def vm_rel_l2_mae(prediction, target):
    """Per-case relative L2 and MAE over (n_nodes, n_frames).
    prediction/target (n_cases, n_nodes, n_frames) numpy -> two (n_cases,) arrays."""
    n_cases = prediction.shape[0]
    diff = (prediction - target).reshape(n_cases, -1)
    rel_l2 = np.linalg.norm(diff, axis=1) / (np.linalg.norm(target.reshape(n_cases, -1), axis=1) + 1e-12)
    mae = np.abs(diff).mean(axis=1)
    return rel_l2, mae


def activation_time(vm, time_ms, threshold):
    """First upward crossing of `threshold` per node, linearly interpolated.
    vm (n_nodes, n_frames) physical mV; time_ms (n_frames,) -> (n_nodes,) ms,
    NaN where a node never crosses."""
    crossed = (vm[:, :-1] < threshold) & (vm[:, 1:] >= threshold)
    nodes = np.where(crossed.any(axis=1))[0]
    first = np.argmax(crossed, axis=1)[nodes]
    at = np.full(vm.shape[0], np.nan, dtype=np.float32)
    v0, v1 = vm[nodes, first], vm[nodes, first + 1]
    t0, t1 = time_ms[first], time_ms[first + 1]
    at[nodes] = t0 + (threshold - v0) / (v1 - v0 + 1e-12) * (t1 - t0)
    return at


def at_rel_l2_mae(at_pred, at_true):
    """Per-case AT relative L2 and MAE over nodes that activated in BOTH.
    at_* (n_cases, n_nodes) -> two (n_cases,) arrays (NaN if no shared node)."""
    n_cases = at_pred.shape[0]
    rel_l2 = np.full(n_cases, np.nan, np.float32)
    mae = np.full(n_cases, np.nan, np.float32)
    for c in range(n_cases):
        valid = np.isfinite(at_pred[c]) & np.isfinite(at_true[c])
        if not valid.any():
            continue
        diff = at_pred[c, valid] - at_true[c, valid]
        rel_l2[c] = np.linalg.norm(diff) / (np.linalg.norm(at_true[c, valid]) + 1e-12)
        mae[c] = np.abs(diff).mean()
    return rel_l2, mae


# ════════════════ ndiff — signed-Laplacian neighbour-difference loss ════════════════
# Auxiliary spatial loss (see Geo_DONet_ndiff): matches the signed neighbour
# difference (V_m minus the mean over its mesh neighbours) of the prediction to the
# ground truth. On the SIREN trunk this tests whether a high-frequency-capable net
# can actually drive the spatial neighbour-diff error down where the Tanh trunk could
# not. Total train loss = MSE + lambda * signed_laplacian_loss(pred - gt).

def build_laplacian_operator(adj_path, n_nodes, device):
    """Row-normalised adjacency Ahat = D^-1 A (sparse CSR) on the canonical-order
    mesh, so (Ahat @ vm)[i] = mean of vm over node i's neighbours. The signed
    neighbour difference is then L vm = vm - Ahat vm (a graph Laplacian). The adj
    npz must be in the SAME node order as the V_m data — see the repo's
    mesh_adjacency_data_order.npz (built from canonical.vtu)."""
    npz = np.load(adj_path)
    stored = int(npz["n_nodes"])
    if stored != n_nodes:
        raise SystemExit(f"adjacency n_nodes={stored} != data n_nodes={n_nodes} "
                         f"({adj_path}); is the adjacency in V_m node order?")
    src = npz["edge_src"].astype(np.int64)
    dst = npz["edge_dst"].astype(np.int64)
    rows = np.concatenate([src, dst])                  # both directions (undirected)
    cols = np.concatenate([dst, src])
    deg = np.bincount(rows, minlength=n_nodes).astype(np.float32)
    vals = (1.0 / deg[rows]).astype(np.float32)
    idx = torch.from_numpy(np.stack([rows, cols])).long()
    ahat = torch.sparse_coo_tensor(idx, torch.from_numpy(vals), (n_nodes, n_nodes)).coalesce()
    print(f"Laplacian operator: {n_nodes} nodes, {src.shape[0]} edges, "
          f"deg[min={int(deg.min())},max={int(deg.max())}]")
    return ahat.to_sparse_csr().to(device)


def signed_laplacian_loss(err, ahat):
    """Mean over (n_cases, n_nodes, n_frames) of (L err)^2 with L = I - Ahat,
    err = pred - target. Matches signed neighbour diff (V_m - mean neighbours)
    of pred vs GT."""
    n_cases, n_nodes, n_frames = err.shape
    e = err.permute(1, 0, 2).reshape(n_nodes, n_cases * n_frames)   # (n_nodes, n_cases*n_frames)
    le = e - torch.sparse.mm(ahat, e)
    return (le ** 2).mean()
