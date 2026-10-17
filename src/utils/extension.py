"""CUDA source assembly and compiler options; build locking and caching live in jit."""

from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import torch

from .indexing import make_index_variants_source
from .jit import load_extension
from .streams import bind_stream_pool_source, install_stream_pool

# Torch bindings run in the main interpreter; avoid per-extension TSS keys.
BINDING_CFLAGS = ("-DPYBIND11_HAS_SUBINTERPRETER_SUPPORT=0",)


@lru_cache(maxsize=None)
def get_device_properties(device: torch.device):
    """Reuse immutable hardware properties for an explicit device."""
    return torch.cuda.get_device_properties(device)


@lru_cache(maxsize=None)
def get_cuda_arch(device: torch.device) -> str:
    """Cache the native compilation target for an explicit CUDA device."""
    properties = get_device_properties(device)
    major, minor = properties.major, properties.minor
    if major < 8:
        raise RuntimeError("LunaEquivariance CUDA kernels require compute capability 8.0 or newer")
    return f"{major}{minor}"


def load_tp_extension(
    name_prefix: str,
    source: str,
    *,
    max_channels: int,
    index_bits: int = 32,
    arch: str | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SO3 LunaTP.")
    if max_channels <= 0:
        raise ValueError("Generated SO3 LunaTP kernels require positive channels.")
    uses_streams = "LUNAEQ_AUXILIARY_STREAMS" in source
    source = bind_stream_pool_source(source)
    source = make_index_variants_source(source, name_prefix)
    cuda_flags = (
        ["-O3"]
        if "data_ptr<double>" in source or "torch::kFloat64" in source
        else ["-O3", "--use_fast_math"]
    )
    if arch is not None:
        cuda_flags += [
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
        ]
    extension = load_extension(
        name_prefix, cuda_sources=source,
        extra_cflags=["-O3", *BINDING_CFLAGS],
        extra_cuda_cflags=[*cuda_flags, *BINDING_CFLAGS],
    )
    if uses_streams:
        install_stream_pool(extension)
    return getattr(extension, f"i{index_bits}")


def find_cutlass_include() -> Path:
    """Find the CUTLASS header tree for either native compiler."""
    search_roots: list[Path] = []
    for name in ("LUNATP_CUTLASS_INCLUDE", "CUTLASS_INCLUDE_DIR", "CUTLASS_PATH"):
        value = os.environ.get(name)
        if value:
            search_roots.append(Path(value).expanduser())

    package_root = Path(__file__).resolve().parents[1]
    search_roots.extend(
        (
            package_root / "third_party" / "cutlass" / "include",
            package_root / "cutlass" / "include",
            Path(sys.prefix) / "include",
        )
    )
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        search_roots.append(Path(cuda_home).expanduser() / "include")
    for entry in sys.path:
        if entry:
            base = Path(entry)
            search_roots.extend((base / "include", base / "cutlass" / "include"))

    seen: set[Path] = set()
    for root in search_roots:
        for include_dir in (root, root / "include"):
            try:
                include_dir = include_dir.resolve()
            except OSError:
                continue
            if include_dir in seen:
                continue
            seen.add(include_dir)
            if (include_dir / "cutlass" / "cutlass.h").is_file() and (
                include_dir / "cutlass" / "gemm" / "device" / "gemm.h"
            ).is_file():
                break
        else:
            continue
        break
    else:
        raise RuntimeError(
            "CUTLASS headers were not found. Set LUNATP_CUTLASS_INCLUDE to the "
            "directory that contains cutlass/cutlass.h."
        )
    return include_dir


def combine_cuda_sources(sources: dict[str, str]) -> str:
    """Bind independent CUDA sources as submodules of one extension."""
    includes, bodies, bindings = [], [], []
    for name, source in sources.items():
        source = bind_stream_pool_source(source)
        includes.extend(re.findall(r"^\s*#include[^\n]*", source, flags=re.M))
        body = re.sub(r"^\s*#include[^\n]*", "", source, flags=re.M)
        body, count = re.subn(
            r"PYBIND11_MODULE\(TORCH_EXTENSION_NAME,\s*(\w+)\)",
            r"void bind(pybind11::module_ &\1)",
            body,
        )
        if count != 1:
            raise ValueError(f"Expected one binding block for {name}")

        # Macros are not scoped by C++ namespaces.
        macros = dict.fromkeys(re.findall(r"^\s*#define\s+(\w+)", body, flags=re.M))
        bodies.append(f"namespace {name} {{\n{body}\n}}\n" + "".join(
            f"#undef {macro}\n" for macro in macros
        ))
        bindings.append(f'auto {name}_module = m.def_submodule("{name}");\n'
                        f'{name}::bind({name}_module);')

    return ("\n".join(dict.fromkeys(includes)) + "\n" + "\n".join(bodies)
            + "\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
            + "\n".join(bindings) + "\n}\n")


def load_cuda_extension(
    name_prefix: str,
    source: str,
    *,
    arch: str,
    use_fast_math: bool = True,
    index_bits: int = 32,
    minimal_headers: bool = False,
):
    """Compile native CUDA, discovering CUTLASS only for sources that include it."""
    if not re.fullmatch(r"[0-9]{2,3}[af]?", arch) or int(arch.rstrip("af")) < 80:
        raise ValueError(f"unsupported CUDA architecture: {arch}")

    # New generated kernels use only Tensor, empty, and the Python caster.
    # Avoid parsing the complete C++ frontend in every CUDA specialization.
    if minimal_headers:
        source = (
            source.replace(
                "#include <torch/extension.h>",
                "#include <ATen/core/Tensor.h>\n#include <ATen/ops/empty.h>\n#include <torch/csrc/utils/pybind.h>",
            )
            .replace("torch::Tensor", "at::Tensor")
            .replace("torch::empty", "at::empty")
        )
    uses_streams = "LUNAEQ_AUXILIARY_STREAMS" in source
    source = bind_stream_pool_source(source)
    source = make_index_variants_source(source, name_prefix)

    # Inspect includes after stripping comments, before discovering headers.
    directives = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    uses_cutlass = re.search(
        r'^\s*#\s*include\s*[<"](?:cutlass|cute)/', directives, re.MULTILINE
    ) is not None
    include_paths = []
    if uses_cutlass:
        include_paths.append(str(find_cutlass_include()))

    cuda_cflags = [
        "-O3", *BINDING_CFLAGS,
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        f"-gencode=arch=compute_{arch},code=compute_{arch}",
    ]
    if use_fast_math:
        cuda_cflags.append("--use_fast_math")
    extension = load_extension(
        name_prefix, cuda_sources=source,
        extra_include_paths=include_paths,
        extra_cflags=["-O3", *BINDING_CFLAGS],
        extra_cuda_cflags=cuda_cflags,
        no_implicit_headers=minimal_headers,
    )
    if uses_streams:
        install_stream_pool(extension)
    return getattr(extension, f"i{index_bits}")
