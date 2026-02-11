import os
import numpy as np
import h5py

PATH = "data/raw/ascad-reference.npz"  # change me

if PATH.endswith(".npz"):
    d = np.load(PATH)
    print("NPZ keys:", d.files)
    for k in d.files:
        arr = d[k]
        if hasattr(arr, "shape"):
            print(k, arr.shape, arr.dtype)
elif PATH.endswith(".h5") or PATH.endswith(".hdf5"):
    with h5py.File(PATH, "r") as f:
        def walk(name, obj):
            if hasattr(obj, "shape"):
                print(name, obj.shape, obj.dtype)
        f.visititems(walk)
else:
    raise ValueError("Unsupported file format")

