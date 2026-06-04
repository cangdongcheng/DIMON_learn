"""
Inspect reference.vtu + reference_cobiveco.npz and build node adjacency.

Goal: confirm the uploaded mesh gives the connectivity needed to replicate the
ground-truth "difference with all neighbours" visualization, and that its node
ordering matches the training data / cobiveco npz.

Run:
    source ~/load_dimon_env.sh
    python scratch_scripts/inspect_mesh.py
"""
import os
from collections import defaultdict

import numpy as np
import meshio

SCRATCH = "/home/svu/e1032484/scratch"
VTU = os.path.join(SCRATCH, "reference.vtu")
REF_COBI = os.path.join(SCRATCH, "reference_cobiveco.npz")
DATA = os.path.join(SCRATCH, "geo_donet_data_f121.npz")


def main():
    print("=== reading mesh ===", flush=True)
    m = meshio.read(VTU)
    pts = m.points
    print(f"points: shape={pts.shape} dtype={pts.dtype}")
    print(f"  bbox min={pts.min(0)}  max={pts.max(0)}")
    print("cell blocks:")
    tets = None
    for cb in m.cells:
        print(f"  {cb.type}: {cb.data.shape}")
        if cb.type == "tetra":
            tets = cb.data
    print("point_data keys:", list(m.point_data.keys()))

    # --- node ordering check vs reference_cobiveco.npz ---
    print("\n=== node-ordering check (vtu PointData vs reference_cobiveco.npz) ===")
    ref = np.load(REF_COBI, allow_pickle=True)
    cobi = ref["cobiveco"]
    labels = [str(x) for x in ref["labels"]]
    print("cobiveco npz labels:", labels)
    for i, lab in enumerate(labels):
        if lab in m.point_data:
            a = np.asarray(m.point_data[lab]).ravel().astype(np.float64)
            b = cobi[:, i].astype(np.float64)
            if a.shape == b.shape:
                print(f"  {lab:5s}: max|vtu-npz|={np.max(np.abs(a-b)):.3e}  "
                      f"(match={'YES' if np.allclose(a,b,atol=1e-5) else 'no'})")
            else:
                print(f"  {lab:5s}: shape mismatch vtu={a.shape} npz={b.shape}")
        else:
            print(f"  {lab:5s}: not in vtu point_data")

    # --- build undirected adjacency from tets ---
    print("\n=== adjacency from tet connectivity ===")
    assert tets is not None, "no tetra cells found"
    edges = set()
    for t in tets:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(t[i]), int(t[j])
                edges.add((a, b) if a < b else (b, a))
    n_nodes = pts.shape[0]
    deg = np.zeros(n_nodes, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    print(f"n_nodes={n_nodes}  unique_undirected_edges={len(edges)}")
    print(f"  (grad-loss operator had 320,286 edges — match={'YES' if len(edges)==320286 else 'DIFF'})")
    print(f"degree: min={deg.min()} max={deg.max()} mean={deg.mean():.2f}")
    iso = int((deg == 0).sum())
    print(f"isolated nodes (deg 0): {iso}")

    # save adjacency as edge arrays for reuse (CSR-friendly)
    out = os.path.join(SCRATCH, "reference", "mesh_adjacency.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ea = np.array(sorted(edges), dtype=np.int32)
    np.savez_compressed(out, edge_src=ea[:, 0], edge_dst=ea[:, 1],
                        n_nodes=np.int64(n_nodes), points=pts.astype(np.float32))
    print(f"\nsaved edges + 3D points -> {out} ({os.path.getsize(out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()