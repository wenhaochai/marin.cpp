"""Paths, hashing and the fixture manifest shared by the levref modules."""
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time


def root() -> str:
    try:
        return os.environ["NANOMARIN_ROOT"]
    except KeyError:
        raise SystemExit("NANOMARIN_ROOT is not set: export it to the directory that holds project/marin and project/nanomarin")


def marin_dir() -> str:
    return os.path.join(root(), "project", "marin")


def nano_dir() -> str:
    return os.path.join(root(), "project", "nanomarin")


def marin_prefix() -> str:
    """The marin artifact store the baseline read its data from (the launcher's MARIN_PREFIX)."""
    return os.environ.get("MARIN_PREFIX") or os.path.join(root(), "marin_store_big")


def fixture_dir(size: str, name: str) -> str:
    d = os.path.join(nano_dir(), "cache", "levref", size, name)
    real = os.path.realpath(d)
    if real.startswith(os.path.realpath(marin_dir()) + os.sep) or real.startswith(os.path.realpath(marin_prefix()) + os.sep):
        raise SystemExit(f"refusing to write into the marin tree or store: {real}")
    os.makedirs(d, exist_ok=True)
    return d


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_array(a) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class TensorDir:
    """A fixture subdirectory of .npy tensors plus index.json (name -> file, dtype, shape, axes, sha256 of the raw bytes).

    bfloat16 has no numpy dtype, so a bf16 tensor is stored as its raw uint16 bit patterns in `<name>.bf16.npy`
    (load_tensor turns it back into exact float32 values; torch: torch.from_numpy(a).view(torch.bfloat16)).
    Every other dtype is stored as is in `<name>.npy`.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.index: dict = {}

    def save(self, name: str, arr, axes=None, **meta) -> "np.ndarray":
        import numpy as np

        a = np.asarray(arr)
        dtype = str(a.dtype)
        if dtype == "bfloat16":
            raw = a.view(np.uint16)
            fname = f"{name}.bf16.npy"
        else:
            raw = a
            fname = f"{name}.npy"
        np.save(os.path.join(self.path, fname), raw)
        self.index[name] = dict(file=fname, dtype=dtype, shape=list(a.shape), axes=list(axes) if axes is not None else None,
                                sha256=sha256_array(raw), **meta)
        return a

    def write_index(self, **extra) -> None:
        write_json(os.path.join(self.path, "index.json"), dict(**extra, tensors=self.index))


def bf16_bits_to_f32(u16):
    """Exact float32 values of bfloat16 bit patterns (numpy only)."""
    import numpy as np

    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_tensor(dirpath: str, name: str, *, as_float: bool = True):
    """Load one tensor of a TensorDir. bf16 comes back as exact float32 values (or raw uint16 with as_float=False)."""
    import numpy as np

    idx = json.load(open(os.path.join(dirpath, "index.json")))["tensors"][name]
    a = np.load(os.path.join(dirpath, idx["file"]))
    if idx["dtype"] == "bfloat16" and as_float:
        a = bf16_bits_to_f32(a)
    return a


def write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", marin_dir(), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception as e:  # provenance must not take the run down
        return f"ERROR {e!r}"


def environment() -> dict:
    """Software and hardware facts of this process (call after the JAX backend is up)."""
    import jax
    import jaxlib
    import levanter
    import haliax

    info: dict = dict(
        python=sys.version.split()[0],
        executable=sys.executable,
        jax=jax.__version__,
        jaxlib=jaxlib.__version__,
        levanter_file=levanter.__file__,
        haliax_file=haliax.__file__,
        marin_commit=_git("rev-parse", "HEAD"),
        marin_levanter_dirty=_git("status", "--porcelain", "--", "lib/levanter/src", "lib/haliax/src") or "clean",
        backend=jax.default_backend(),
        device_count=jax.device_count(),
        process_count=jax.process_count(),
        devices=[d.device_kind for d in jax.devices()],
    )
    try:
        info["platform_version"] = jax.devices()[0].client.platform_version
    except Exception as e:
        info["platform_version"] = f"ERROR {e!r}"
    pkgs = {}
    for dist in importlib.metadata.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name.startswith(("jax", "nvidia-", "tensorstore", "equinox", "optax")) or "cuda" in name:
            pkgs[name] = dist.version
    info["packages"] = dict(sorted(pkgs.items()))
    if jax.default_backend() == "gpu":
        try:
            info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                                capture_output=True, text=True, check=True).stdout.strip().splitlines()
        except Exception as e:
            info["nvidia_smi"] = f"ERROR {e!r}"
    return info


def write_manifest(out: str, *, checks: list[str], env: dict, extra: dict | None = None) -> dict:
    """manifest.json: provenance, the exact command, the Slurm job, and sha256 of every other file in `out`."""
    files = {}
    for dirpath, _, names in os.walk(out):
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            rel = os.path.relpath(p, out)
            if rel == "manifest.json" or rel.endswith(".tmp"):
                continue
            files[rel] = dict(bytes=os.path.getsize(p), sha256=sha256_file(p))
    keep_env = ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR", "CUDA_VISIBLE_DEVICES",
                "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "LEVANTER_PALLAS_CE_AUTOTUNE_ON_MISS",
                "MARIN_PREFIX", "HF_HUB_OFFLINE", "WANDB_MODE")
    man = dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        command=" ".join(shlex.quote(a) for a in [sys.executable, "-m", _module_name(), *sys.argv[1:]]),
        cwd=os.getcwd(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        hostname=os.uname().nodename,
        env={k: os.environ[k] for k in keep_env if k in os.environ},
        spec_checks=checks,
        environment=env,
        **(extra or {}),
        files=files,
    )
    write_json(os.path.join(out, "manifest.json"), man)
    return man


def loaded_libraries(patterns=("libcublas", "libcublasLt", "libcudnn", "libnccl", "libcuda.", "libcudart", "libnvJitLink",
                                "xla_cuda", "libcusolver", "libcufft", "libcusparse", "libnvrtc")) -> list[dict]:
    """Shared objects mapped into this process whose basename matches a pattern (from /proc/self/maps)."""
    seen = {}
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6 or not parts[5].startswith("/"):
                    continue
                p = parts[5]
                base = os.path.basename(p)
                if p not in seen and any(base.startswith(pat) or pat in base for pat in patterns):
                    try:
                        seen[p] = dict(path=p, realpath=os.path.realpath(p), bytes=os.path.getsize(p))
                    except OSError:
                        seen[p] = dict(path=p)
    except OSError as e:
        return [dict(error=repr(e))]
    return sorted(seen.values(), key=lambda d: d["path"])


def _module_name() -> str:
    main = sys.modules.get("__main__")
    spec = getattr(main, "__spec__", None)
    return spec.name if spec is not None else sys.argv[0]
