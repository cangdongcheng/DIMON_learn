"""
Fix -1 (unsolved) activation times in DIMON_training_data_healthy.npz
by replacing them with the mean of valid mesh neighbours.

Uses canonical.vtu tet connectivity to build the adjacency graph.
Iterates until no -1 values remain (handles clusters where a -1 node's
only neighbours are also -1).

Usage:
    source ~/load_dataproc_env.sh
    cd DIMON
    python fix_neg1.py

Reads:  DIMON_training_data_healthy.npz
Writes: DIMON_training_data_healthy_fixed.npz (same keys, u_data patched)
"""
import os
from collections import defaultdict

import meshio
import numpy as np

DATA_BASE = "/home/users/nus/e1590340/scratch/Mengxiao_20260212_VTK_Merged_ED_CSV"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def build_adjacency(vtu_path):
    """Build node adjacency from tetrahedral connectivity."""
    m = meshio.read(vtu_path)
    tets = m.cells[0].data
    adj = defaultdict(set)
    for tet in tets:
        for i in range(4):
            for j in range(4):
                if i != j:
                    adj[tet[i]].add(tet[j])
    return adj


def fix_neg1_field(field, adj, label=""):
    """Replace -1 values with mean of valid neighbours. Iterates for clusters."""
    fixed = field.copy()
    iteration = 0
    while True:
        bad_mask = (fixed == -1)
        n_bad = bad_mask.sum()
        if n_bad == 0:
            break
        iteration += 1
        n_fixed_this_round = 0
        bad_nodes = np.where(bad_mask)[0]
        for node in bad_nodes:
            nbrs = list(adj[node])
            vals = fixed[nbrs]
            valid = vals[vals != -1]
            if len(valid) > 0:
                fixed[node] = valid.mean()
                n_fixed_this_round += 1
        if n_fixed_this_round == 0:
            print(f"  WARNING {label}: {n_bad} nodes still -1 with no valid neighbours")
            break
        if iteration > 10:
            print(f"  WARNING {label}: still {n_bad} after 10 iterations")
            break
    return fixed


def main():
    # Load data
    npz_path = os.path.join(SCRIPT_DIR, "DIMON_training_data_healthy.npz")
    data = np.load(npz_path)
    u_all = data['u_data'].copy()  # (125, 9, 50797)

    n_bad_before = (u_all == -1).sum()
    print(f"Before: {n_bad_before} values = -1")

    if n_bad_before == 0:
        print("Nothing to fix.")
        return

    # Build adjacency
    vtu_path = os.path.join(DATA_BASE, "reference", "canonical.vtu")
    print("Building adjacency from canonical.vtu...")
    adj = build_adjacency(vtu_path)
    print(f"  {len(adj)} nodes, avg {np.mean([len(v) for v in adj.values()]):.1f} neighbours")

    # Fix each heart × pacing
    total_fixed = 0
    for h in range(u_all.shape[0]):
        for s in range(u_all.shape[1]):
            n_bad = (u_all[h, s] == -1).sum()
            if n_bad > 0:
                u_all[h, s] = fix_neg1_field(u_all[h, s], adj,
                                              label=f"heart={h},pace={s}")
                n_still_bad = (u_all[h, s] == -1).sum()
                total_fixed += n_bad - n_still_bad

    n_bad_after = (u_all == -1).sum()
    print(f"\nAfter: {n_bad_after} values = -1 (fixed {total_fixed})")

    # Save
    out_path = os.path.join(SCRIPT_DIR, "DIMON_training_data_healthy_fixed.npz")
    save_dict = {k: data[k] for k in data.keys()}
    save_dict['u_data'] = u_all
    np.savez_compressed(out_path, **save_dict)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
