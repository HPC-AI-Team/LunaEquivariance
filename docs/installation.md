# Installation

Install CUDA-enabled PyTorch first, then install LunaEquivariance:

```bash
pip install LunaEquivariance
```

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.10 or later |
| PyTorch | 2.8 or later, below 3.0 |
| GPU | NVIDIA GPU with compute capability 8.0 or later |
| CUDA toolchain | Linux, a CUDA toolkit with `nvcc`, and a compatible C++ compiler |
| Package dependencies | NumPy, Ninja and e3nn 0.6.x |
| CUTLASS headers | Version 3.8.0 included with the package |

See [Supported operations](operators.rst) for supported devices and data types.

## Check the installation

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from LunaEquivariance import o3, nn, TensorProduct, TensorProductConv, make_graph"
```

`torch.cuda.is_available()` should return `True`. You can then try the
[quick-start example](quickstart.md).

## First use and caching

LunaEquivariance compiles kernels as needed. The first call with a new kernel
configuration can take longer while the kernel compiles. Once compiled,
the kernel is cached and reused on later calls and future runs in the same environment.

The default cache directory on Linux is `~/.cache/torch_extensions/lunaeq/`.
You can choose another directory by setting `LUNAEQ_CACHE_DIR`.
Cache files are not removed automatically. To free disk space, delete the cache
directory when no LunaEquivariance jobs are running; kernels will be rebuilt as needed.

## Check your inputs

Pass a module and your inputs to `check_compatibility` to report numerical errors,
timings and speedup against e3nn or PyTorch.

```python
import torch
from LunaEquivariance import o3
from LunaEquivariance.utils.testing import check_compatibility

module = o3.Linear("8x0e + 4x1o", "16x0e + 4x1o").cuda()
batch_size = 128
x = torch.randn(batch_size, module.irreps_in.dim, device="cuda", requires_grad=True)

report = check_compatibility(module, (x,), derivative_order=2)
```

For a tensor-product convolution:

```python
from LunaEquivariance import TensorProductConv

conv = TensorProductConv(
    "2x1o", "2x1o", "2x0e", [(0, 0, 0, "uvu", True)],
    internal_weights=False, shared_weights=False,
).cuda()
num_atoms = 3
src = torch.tensor([1, 2, 0, 1], device="cuda")
dst = torch.tensor([0, 0, 1, 2], device="cuda")  # Sorted by destination.
graph = conv.create_graph(src, dst, num_atoms)
num_edges = src.numel()
x = torch.randn(num_atoms, conv.irreps_in1.dim, device="cuda", requires_grad=True)
edge_attr = torch.randn(num_edges, conv.irreps_in2.dim, device="cuda", requires_grad=True)
weight = torch.randn(num_edges, conv.weight_numel, device="cuda", requires_grad=True)

report = check_compatibility(
    conv, (x, edge_attr, weight), kwargs={"graph": graph},
)
```

Supported modules: `Linear`, `SphericalHarmonics`, `Norm`, `Activation`, `Gate`,
`NormActivation`, `Dropout`, `TensorProduct`, `TensorProductConv`,
`EquivariantNorm`, `GraphSoftmax`, `VarlenAttention` and `VarlenAttentionConv`.
