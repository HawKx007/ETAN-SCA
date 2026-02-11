import sys, os

def is_mat_v73(path: str) -> bool:
    with open(path, "rb") as f:
        sig = f.read(8)
    return sig == b"\x89HDF\r\n\x1a\n"

def inspect(path: str):
    print("File:", path)
    print("Size (MB):", os.path.getsize(path)/1024/1024)

    if is_mat_v73(path):
        import h5py
        print("MATLAB v7.3 (HDF5)")
        with h5py.File(path, "r") as f:
            def walk(name, obj):
                if hasattr(obj, "shape"):
                    print(f"{name:45s} shape={obj.shape} dtype={obj.dtype}")
            f.visititems(walk)
    else:
        import scipy.io
        print("MATLAB <= v7.2")
        for name, shape, dtype in scipy.io.whosmat(path):
            print(f"{name:45s} shape={shape} dtype={dtype}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mat.py <path_to_mat>")
        raise SystemExit(1)
    inspect(sys.argv[1])
