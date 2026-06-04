"""
Extract canonical-reference Cartesian xyz (50797, 3) from canonical.vtu and
cache it as canonical_xyz.npz with key `cartesian_coords`, so DIMON eval can
render 3D snapshots on Vanda without the (missing) DIMON_training_data_healthy.npz.

Also VERIFIES node order: the VTU PointData (ab, rt, tm, tv) must match the
npz `coords` (Cobiveco ab, rt, tm, tv) row-for-row. If they don't, the xyz is
in a different node order and must NOT be used for the scatter plots.
"""
import numpy as np
import meshio

VTU = "/home/svu/e1032484/scratch/canonical.vtu"
NPZ = "/home/svu/e1032484/scratch/geo_donet_data_f121.npz"
OUT = "/home/svu/e1032484/scratch/canonical_xyz.npz"

mesh = meshio.read(VTU)
xyz = np.asarray(mesh.points, dtype=np.float32)
print(f"canonical.vtu: {xyz.shape[0]} points, xyz shape {xyz.shape}")

# Cobiveco from the VTU point data (order ab, rt, tm, tv)
pd = mesh.point_data
vtu_cob = np.column_stack([pd["ab"], pd["rt"], pd["tm"], pd["tv"]]).astype(np.float32)

# Cobiveco from the training npz
npz_cob = np.load(NPZ, allow_pickle=True)["coords"].astype(np.float32)  # (50797, 4)

assert vtu_cob.shape == npz_cob.shape, f"{vtu_cob.shape} vs {npz_cob.shape}"
max_abs = np.abs(vtu_cob - npz_cob).max()
print(f"max |VTU_cobiveco - npz_coords| = {max_abs:.3e}  (small => same node order)")

if max_abs < 1e-4:
    np.savez_compressed(OUT, cartesian_coords=xyz)
    print(f"VERIFIED same order. Wrote {OUT} (cartesian_coords {xyz.shape})")
else:
    print("MISMATCH — node order differs, NOT writing. Do not use this xyz blindly.")
