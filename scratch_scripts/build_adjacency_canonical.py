"""
Build node adjacency DIRECTLY from canonical.vtu, in Vm/training-data node order.

Provenance (confirmed by the user's build script):
  reference_cobiveco.npz  <-  canonical.vtu point_data (in order, float32 cast)
  geo_donet vm/coords     ==  reference_cobiveco.npz order
=> canonical.vtu point order IS the Vm node order, so its tets give adjacency
   in data order with ZERO matching.

Verifies that claim (canonical.vtu point_data == reference_cobiveco.npz, in order)
before trusting it, then emits the adjacency.

Run: source ~/load_dimon_env.sh && python scratch_scripts/build_adjacency_canonical.py
"""
import os
import numpy as np
import meshio

S = "/home/svu/e1032484/scratch"
VTU = os.path.join(S, "canonical.vtu")
OUT = os.path.join(S, "reference", "mesh_adjacency_data_order.npz")


def main():
    print("=== reading canonical.vtu ===", flush=True)
    m = meshio.read(VTU)
    pts = m.points.astype(np.float64)
    N = pts.shape[0]
    print(f"points: {pts.shape}  point_data: {list(m.point_data.keys())}")
    tets = next((cb.data for cb in m.cells if cb.type == "tetra"), None)
    assert tets is not None, "no tetra cells"
    print(f"tets: {tets.shape}")

    # --- verify ordering vs reference_cobiveco.npz (the npz was built from THIS file) ---
    ref = np.load(os.path.join(S, "reference_cobiveco.npz"))
    cobi = ref["cobiveco"]                                  # (N,5) float32
    labels = [str(x) for x in ref["labels"]]
    assert pts.shape[0] == cobi.shape[0], f"node count {pts.shape[0]} != {cobi.shape[0]}"
    print("\n=== in-order check: canonical.vtu point_data (float32) vs reference_cobiveco.npz ===")
    all_exact = True
    for i, lab in enumerate(labels):
        if lab not in m.point_data:
            print(f"  {lab:5s}: MISSING in vtu"); all_exact = False; continue
        v = np.asarray(m.point_data[lab]).ravel().astype(np.float32)
        md = float(np.max(np.abs(v - cobi[:, i])))
        exact = md == 0.0
        all_exact &= exact
        print(f"  {lab:5s}: max|vtu_f32 - npz| = {md:.3e}  {'EXACT' if exact else 'differs'}")

    # cross-check against the actual training data coords too
    data = np.load(os.path.join(S, "geo_donet_data_f121.npz"), mmap_mode="r")
    dc = np.asarray(data["coords"], dtype=np.float32)       # (N,4)
    vtu4 = np.stack([np.asarray(m.point_data[k]).ravel().astype(np.float32)
                     for k in ["ab", "rt", "tm", "tv"]], axis=1)
    md_data = float(np.max(np.abs(vtu4 - dc)))
    print(f"  vs geo_donet coords: max|diff| = {md_data:.3e}  "
          f"{'EXACT' if md_data == 0.0 else 'differs'}")
    print(f"\n=> canonical.vtu is in Vm order: {'YES (exact)' if all_exact and md_data==0.0 else 'NOT exact — investigate'}")

    # --- build undirected adjacency directly (canonical order == data order) ---
    print("\n=== building adjacency (no remapping) ===")
    edges = {}
    for t in tets:
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = int(t[a]), int(t[b])
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    edges[(u, v)] = np.linalg.norm(pts[u] - pts[v])
    E = len(edges)
    deg = np.zeros(N, dtype=np.int64)
    for (u, v) in edges:
        deg[u] += 1; deg[v] += 1
    print(f"n_nodes={N}  edges={E}  (grad-loss operator=320286 -> "
          f"{'MATCH' if E == 320286 else 'DIFF'})")
    print(f"degree: min={deg.min()} max={deg.max()} mean={deg.mean():.2f}  "
          f"isolated={int((deg==0).sum())}")
    ek = np.array(sorted(edges.keys()), dtype=np.int32)
    el = np.array([edges[(int(a), int(b))] for a, b in ek], dtype=np.float64)
    print(f"edge length (um): min={el.min():.1f} max={el.max():.1f} mean={el.mean():.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT,
                        edge_src=ek[:, 0], edge_dst=ek[:, 1],
                        inv_length=(1.0 / el).astype(np.float32),
                        points_xyz=pts.astype(np.float32),
                        n_nodes=np.int64(N))
    print(f"\nsaved -> {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()