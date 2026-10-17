"""Content-addressed CUDA extensions with recoverable, per-build locking."""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils import cpp_extension


MODULES = {}
BUILD_LOCKS = {}
CACHE_LOCK = threading.Lock()
BUILD_TIMEOUT = 3600
COMPILER_LOCK = threading.Lock()


@contextmanager
def compilation_jobs():
    # MAX_JOBS is process-wide; keep concurrent builds from changing it.
    with COMPILER_LOCK:
        if "MAX_JOBS" in os.environ:
            yield
            return
        cpus = len(os.sched_getaffinity(0))
        if os.environ.get("SLURM_CPUS_PER_TASK"):
            cpus = min(cpus, int(os.environ["SLURM_CPUS_PER_TASK"]))
        os.environ["MAX_JOBS"] = str(max(1, min(cpus, 8)))
        try:
            yield
        finally:
            os.environ.pop("MAX_JOBS", None)


@lru_cache(maxsize=None)
def compiler_identity(command, search_path):
    executable = shutil.which(command, path=search_path)
    if executable is None:
        raise RuntimeError(f"Compiler not found: {command}")
    version = subprocess.check_output([executable, "--version"], text=True)
    return str(Path(executable).resolve()), version


def load_extension(
    name,
    *,
    cpp_sources="",
    cuda_sources="",
    extra_cflags=(),
    extra_cuda_cflags=(),
    extra_include_paths=(),
    no_implicit_headers=False,
):
    """Build once per configuration; failed or interrupted builds remain retryable.

    Reuse Torch's Ninja recipe, without its process-local version counter or
    lock-file polling. Only completed libraries become visible to other ranks.
    """
    cuda_flags = list(extra_cuda_cflags)
    if cuda_sources:
        # Resolve implicit targets before hashing, rather than inside Ninja.
        cuda_flags.extend(cpp_extension._get_cuda_arch_flags(cuda_flags))
    includes = [str(Path(path).resolve()) for path in extra_include_paths]
    compiler = cpp_extension.get_cxx_compiler()
    search_path = os.environ.get("PATH", "")
    toolchain = [compiler_identity(compiler, search_path)]
    if cuda_sources:
        if cpp_extension.CUDA_HOME is None:
            raise RuntimeError("CUDA toolkit not found; set CUDA_HOME to compile LunaEquivariance kernels")
        nvcc = os.environ.get("PYTORCH_NVCC", str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"))
        toolchain.append(compiler_identity(nvcc, search_path))
        if os.environ.get("CC"):
            toolchain.append(compiler_identity(os.environ["CC"], search_path))

    if not no_implicit_headers:
        cpp_sources = '#include <torch/extension.h>\n' + cpp_sources
        if cuda_sources:
            cuda_sources = '#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n' + cuda_sources
    build = (
        name, cpp_sources, cuda_sources, list(extra_cflags), cuda_flags, includes,
        torch.__version__, torch.version.cuda, torch.__file__,
        torch._C._GLIBCXX_USE_CXX11_ABI, sys.version, sys.prefix,
        sysconfig.get_config_var("SOABI"), cpp_extension.CUDA_HOME, toolchain,
        tuple((key, os.environ.get(key, "")) for key in (
            "CC", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
            "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
        )),
    )
    for include in includes:
        version_header = Path(include) / "cutlass/version.h"
        if version_header.is_file():
            build += ((str(version_header), version_header.read_text()),)
    digest = hashlib.sha256(json.dumps(build).encode()).hexdigest()[:24]
    name = f"{name}_{digest}"
    cache_dir = os.environ.get("LUNAEQ_CACHE_DIR")
    if not cache_dir:
        torch_cache = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
        cache_dir = str(Path(torch_cache) / "lunaeq")
    root = Path(cache_dir).expanduser().resolve()
    key = str(root), name
    cached = MODULES.get(key)
    if cached is not None:
        return cached

    with CACHE_LOCK:
        thread_lock = BUILD_LOCKS.setdefault(key, threading.Lock())
    timeout = float(os.environ.get("LUNAEQ_BUILD_TIMEOUT", BUILD_TIMEOUT))
    deadline = time.monotonic() + timeout
    if not thread_lock.acquire(timeout=max(0, timeout)):
        raise TimeoutError(f"Timed out waiting for LunaEquivariance build {name}")
    try:
        cached = MODULES.get(key)
        if cached is not None:
            return cached
        root.mkdir(parents=True, exist_ok=True)
        library = root / f"{name}{cpp_extension.LIB_EXT}"
        lock_path = root / f"{name}.lock"
        # Keep the lock inode: unlinking it could let two ranks lock different files.
        with lock_path.open("a") as lock_file:
            while True:
                try:
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for LunaEquivariance build lock {lock_path}")
                    time.sleep(0.05)

            if library.is_file():
                try:
                    module = cpp_extension._import_module_from_library(name, str(root), True)
                except (ImportError, OSError):
                    # An incomplete/external cache copy can be rebuilt under the lock.
                    library.unlink()
                else:
                    MODULES[key] = module
                    return module

            with tempfile.TemporaryDirectory(prefix=f".{name}.", dir=root) as directory:
                sources = []
                for filename, source in (("main.cpp", cpp_sources), ("cuda.cu", cuda_sources)):
                    if source or filename == "main.cpp":
                        path = Path(directory) / filename
                        path.write_text(source)
                        sources.append(str(path))
                with compilation_jobs():
                    cpp_extension._write_ninja_file_and_build_library(
                        name=name, sources=sources,
                        extra_cflags=list(extra_cflags), extra_cuda_cflags=cuda_flags,
                        extra_sycl_cflags=[], extra_ldflags=[], extra_include_paths=includes,
                        build_directory=directory, verbose=False,
                        with_cuda=bool(cuda_sources), with_sycl=False, is_standalone=False,
                    )
                os.replace(Path(directory) / library.name, library)
            try:
                module = cpp_extension._import_module_from_library(name, str(root), True)
            except (ImportError, OSError):
                library.unlink()
                raise
            MODULES[key] = module
            return module
    finally:
        thread_lock.release()
