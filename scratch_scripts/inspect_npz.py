import numpy as np
import os

for path in [
    "/home/svu/e1032484/scratch/geo_donet_data_f121.npz",
    "/home/svu/e1032484/scratch/geo_donet_data.npz",
]:
    print(f"\n=== {path}  ({os.path.getsize(path)/1e9:.2f} GB) ===")
    z = np.load(path, mmap_mode="r")
    for k in z.files:
        a = z[k]
        info = f"  {k:30s}  shape={str(a.shape):28s}  dtype={a.dtype}"
        if a.size < 5_000_000 and np.issubdtype(a.dtype, np.number):
            info += f"  min={float(a.min()):.4g}  max={float(a.max()):.4g}"
        elif a.size <= 10:
            info += f"  values={a.tolist()}"
        print(info)
