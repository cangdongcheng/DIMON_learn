"""
Build mesh node adjacency in TRAINING-DATA node order.

The reference.vtu connectivity is in vtu node order; the vm data is in data
order (== reference_cobiveco.npz order, verified identical). The only shared
key is the 4D cobiveco coordinate. Data coords are float32(vtu_float64), so we
match on exact float32 cobiveco tuples (robust to rt periodicity), with a
periodicity-aware 5D nearest fallback for any unmatched/duplicate nodes.

Emits $SCRATCH/reference/mesh_adjacency_data_order.npz with:
  edge_src, edge_dst  (int32, data-order node indices, undirected i<j)
  inv_length          (float32, 1/||p_i - p_j|| in um) -- matches grad operator
  points_xyz          (float32, 50797x3, data order, um)
  n_nodes
Run: source ~/load_dimon_env.sh && python scratch_scripts/build_adjacency_data_order.py
"""
import os
import numpy as np
import meshio
from scipy.spatial import cKDTree

S = "/home/svu/e1032484/scratch"
OUT = os.path.join(S, "reference", "mesh_adjacency_data_order.npz")


def periodic5(c):
    """Map (ab,rt,tm,tv) -> 5D with rt unrolled to (cos,sin) so it's continuous."""
    ab, rt, tm, tv = c[:, 0], c[:, 1], c[:, 2], c[:, 3]
    two_pi = 2.0 * np.pi
    return np.stack([ab, np.cos(two_pi * rt), np.sin(two_pi * rt), tm, tv], axis=1)


def main():
    # canonical (data) order
    data = np.load(os.path.join(S, "geo_donet_data_f121.npz"), mmap_mode="r")
    A = np.asarray(data["coords"][:, :4], dtype=np.float32)   # (N,4) data order
    N = A.shape[0]

    # vtu: cobiveco point_data + 3D points + tets (all vtu order)
    m = meshio.read(os.path.join(S, "reference.vtu"))
    C64 = np.stack([np.asarray(m.point_data[k]).ravel()
                    for k in ["ab", "rt", "tm", "tv"]], axis=1)   # float64
    C = C64.astype(np.float32)
    pts = m.points.astype(np.float64)                              # (N,3) um, vtu order
    tets = None
    for cb in m.cells:
        if cb.type == "tetra":
            tets = cb.data
    assert tets is not None

    # --- periodicity-aware nearest match: vtu node -> data node ---
    # cobiveco is the only shared key; data vs vtu differ by ~4e-8 (float32 eps)
    # so we match by nearest in periodic-5D (rt seam removed), not exact tuples.
    A5 = periodic5(A.astype(np.float64))
    C5 = periodic5(C64)
    d, j = cKDTree(A5).query(C5, k=1)              # j[vi] = nearest data node
    print(f"vtu->data nearest (periodic-5D): max={d.max():.3e} "
          f"mean={d.mean():.3e} p99.9={np.percentile(d, 99.9):.3e}")
    vtu_to_data = j.astype(np.int64)

    # --- force a valid bijection vtu->data ---
    # pass 1: greedy by ascending match distance (claims the confident matches)
    order = np.argsort(d)
    used = np.zeros(N, dtype=bool)
    assigned = np.zeros(N, dtype=bool)              # over vtu nodes
    for vi in order:
        tgt = int(vtu_to_data[vi])
        if not used[tgt]:
            used[tgt] = True
            assigned[vi] = True
    orphan_vtu = np.where(~assigned)[0]
    free_data = np.where(~used)[0]
    print(f"clean greedy matches: {int(assigned.sum())}/{N}; "
          f"residual to assign: {len(orphan_vtu)}")
    # pass 2: optimal assignment on the small residual (periodic-5D cost)
    if len(orphan_vtu):
        from scipy.optimize import linear_sum_assignment
        Cmat = np.linalg.norm(
            C5[orphan_vtu][:, None, :] - A5[free_data][None, :, :], axis=2)
        ri, ci = linear_sum_assignment(Cmat)
        vtu_to_data[orphan_vtu[ri]] = free_data[ci]
        print(f"residual Hungarian: max dist={Cmat[ri, ci].max():.3e} "
              f"mean={Cmat[ri, ci].mean():.3e}")
    assert len(np.unique(vtu_to_data)) == N, "permutation not bijective"
    print("permutation vtu->data is a bijection: YES")

    # --- points in data order; sanity vs implied geometry not available (no data xyz) ---
    pts_data = np.empty_like(pts)
    pts_data[vtu_to_data] = pts            # data_index <- vtu point

    # --- remap tets to data indices, build undirected edges + lengths ---
    tets_d = vtu_to_data[tets]
    edges = {}
    for t in tets_d:
        for a in range(4):
            for b in range(a + 1, 4):
                u, v = int(t[a]), int(t[b])
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    edges[(u, v)] = np.linalg.norm(pts_data[u] - pts_data[v])
    E = len(edges)
    print(f"edges in data order: {E} (expect 320286 -> "
          f"{'MATCH' if E == 320286 else 'DIFF'})")
    ekeys = np.array(sorted(edges.keys()), dtype=np.int32)
    elen = np.array([edges[(int(a), int(b))] for a, b in ekeys], dtype=np.float64)
    print(f"edge length (um): min={elen.min():.1f} max={elen.max():.1f} "
          f"mean={elen.mean():.1f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        edge_src=ekeys[:, 0], edge_dst=ekeys[:, 1],
        inv_length=(1.0 / elen).astype(np.float32),
        points_xyz=pts_data.astype(np.float32),
        n_nodes=np.int64(N),
    )
    print(f"saved -> {OUT} ({os.path.getsize(OUT)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()