"""Create compact AT + aligned-PCA targets from the f601 V_m archive.

Run once on a CPU node.  The output is ~150 MB instead of the 11 GB compressed
V_m archive and is the only data file needed during feature-network training.
"""
import argparse
import os
import time

import numpy as np

from utils import activation_time, load_decoder_basis, shift_waveforms


DATA = "/home/svu/e1032484/scratch/geo_donet_data_f601.npz"
BASIS = "/home/svu/e1032484/scratch/pca_phase_aligned_basis_f601.npz"
OUTPUT = "/home/svu/e1032484/scratch/geo_deeponet_pca_f601_k5.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vm-data", default=DATA)
    parser.add_argument("--basis", default=BASIS)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--chunk-nodes", type=int, default=20_000)
    args = parser.parse_args()

    if args.n_components < 1:
        raise SystemExit("--n-components must be positive")
    basis = load_decoder_basis(args.basis, args.n_components)
    archive = np.load(args.vm_data, allow_pickle=True)
    theta = archive["theta"].astype(np.float32)
    coords = archive["coords"].astype(np.float32)
    time_ms = archive["time"].astype(np.float32)
    vm = archive["vm"]  # decompressed once; already float32 (~15 GB)
    case_names = archive["case_names"] if "case_names" in archive.files else None

    if not np.array_equal(time_ms, basis["time"]):
        raise SystemExit("V_m and decoder-basis time grids differ")
    if coords.shape[0] != basis["node_template"].shape[0]:
        raise SystemExit("V_m and decoder-basis node counts differ")

    n_cases, n_nodes, _ = vm.shape
    target_names = np.asarray(["activation_time_ms"]
                              + [f"pca_{k}" for k in range(1, args.n_components + 1)])
    targets = np.empty((n_cases, n_nodes, args.n_components + 1), dtype=np.float32)
    dt = float(np.median(np.diff(time_ms)))
    tic = time.time()
    print(f"raw V_m: {vm.shape}; preparing {targets.shape} [AT + {args.n_components} PCA]",
          flush=True)

    for case in range(n_cases):
        at = activation_time(vm[case], time_ms, basis["at_threshold"])
        invalid = ~np.isfinite(at)
        if invalid.any():
            raise SystemExit(f"case {case}: {invalid.sum()} nodes never cross "
                             f"{basis['at_threshold']:g} mV")
        targets[case, :, 0] = at
        shifts = at - np.float32(basis["reference_at"])
        for start in range(0, n_nodes, args.chunk_nodes):
            end = min(start + args.chunk_nodes, n_nodes)
            aligned = shift_waveforms(vm[case, start:end], shifts[start:end], dt)
            residual = (aligned - basis["node_template"][start:end]
                        - basis["residual_mean"])
            targets[case, start:end, 1:] = residual @ basis["components"]
        elapsed = time.time() - tic
        eta = elapsed / (case + 1) * (n_cases - case - 1)
        print(f"case {case + 1:3d}/{n_cases} | elapsed {elapsed / 60:.1f} min | "
              f"eta {eta / 60:.1f} min", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = dict(theta=theta, coords=coords, targets=targets,
                   target_names=target_names, time=time_ms,
                   basis_file=np.asarray(os.path.abspath(args.basis)))
    if case_names is not None:
        payload["case_names"] = case_names
    np.savez_compressed(args.output, **payload)
    size_mb = os.path.getsize(args.output) / 1024 ** 2
    print(f"saved -> {args.output} ({size_mb:.1f} MB)")
    n_summary = min(95, n_cases)
    print(f"target ranges (first {n_summary} hearts):")
    for d, name in enumerate(target_names):
        values = targets[:n_summary, :, d]
        print(f"  {name:>20}: mean {values.mean(): .5g}, std {values.std(): .5g}, "
              f"range [{values.min():.5g}, {values.max():.5g}]")


if __name__ == "__main__":
    main()
