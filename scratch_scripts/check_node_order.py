"""
Determine the canonical node ordering and whether vtu <-> data is a clean
permutation, so the mesh adjacency can be remapped into training-data node order.

Compares three coordinate sources (all 50797 nodes, 4D cobiveco ab,rt,tm,tv):
  A) data    geo_donet_data_f121.npz['coords']           (vm node axis order)
  B) ref npz reference_cobiveco.npz['cobiveco'][:, :4]
  C) vtu     reference.vtu point_data ab,rt,tm,tv

Run: source ~/load_dimon_env.sh && python scratch_scripts/check_node_order.py
"""
import os
import numpy as np
import meshio
from scipy.spatial import cKDTree

S = "/home/svu/e1032484/scratch"


def load_all():
    data = np.load(os.path.join(S, "geo_donet_data_f121.npz"), mmap_mode="r")
    A = np.asarray(data["coords"][:, :4], dtype=np.float64)          # data coords
    ref = np.load(os.path.join(S, "reference_cobiveco.npz"))
    B = ref["cobiveco"][:, :4].astype(np.float64)                   # ref npz
    m = meshio.read(os.path.join(S, "reference.vtu"))
    C = np.stack([np.asarray(m.point_data[k]).ravel()
                  for k in ["ab", "rt", "tm", "tv"]], axis=1).astype(np.float64)
    return A, B, C, m


def in_order(name, X, Y):
    same = X.shape == Y.shape and np.allclose(X, Y, atol=1e-5)
    md = np.max(np.abs(X - Y)) if X.shape == Y.shape else float("nan")
    print(f"  {name}: in-order match={'YES' if same else 'no'}  max|diff|={md:.3e}")
    return same


def match(name, src, dst):
    """For each row of dst, find nearest row of src. Report distances + bijectivity."""
    tree = cKDTree(src)
    d, idx = tree.query(dst, k=1)
    uniq = len(np.unique(idx))
    print(f"  {name}: nearest-match dist  max={d.max():.3e} mean={d.mean():.3e} "
          f"p99={np.percentile(d,99):.3e} | unique targets={uniq}/{len(idx)} "
          f"(bijection={'YES' if uniq==len(idx) else 'NO'})")
    return d, idx


def main():
    A, B, C, m = load_all()
    print(f"shapes: data={A.shape} refnpz={B.shape} vtu={C.shape}")
    print("\n=== in-order comparisons ===")
    in_order("data  vs refnpz", A, B)
    in_order("data  vs vtu   ", A, C)
    in_order("refnpz vs vtu  ", B, C)

    print("\n=== permutation check (KDTree nearest in 4D cobiveco) ===")
    match("data  -> refnpz (find each data node in refnpz)", B, A)
    match("data  -> vtu    (find each data node in vtu)   ", C, A)
    match("refnpz-> vtu    (find each refnpz node in vtu) ", C, B)


if __name__ == "__main__":
    main()