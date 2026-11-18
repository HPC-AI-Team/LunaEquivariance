# Changelog

## 0.3.0

- Add [TrimSO(2)](https://arxiv.org/abs/2610.20029) operators and `so2.models.optimize(model)` for eSEN, UMA, and EquiformerV3. ([#4](https://github.com/HPC-AI-Team/LunaEquivariance/pull/4))
- Fix CUDA 13.x build errors in graph construction and symmetric contraction. ([#3](https://github.com/HPC-AI-Team/LunaEquivariance/pull/3))
- Optimize CUDA kernels for `EquivariantNorm`, `GraphSoftmax`, and `SymmetricContraction`. ([#2](https://github.com/HPC-AI-Team/LunaEquivariance/pull/2))
- **API change:** `o3.SymmetricContraction` now requires 2D `mul_ir` input of shape `(num_atoms, irreps_in.dim)`. ([#2](https://github.com/HPC-AI-Team/LunaEquivariance/pull/2))
  
## 0.2.0

### New operator

- Add the `SymmetricContraction` operation with support for uniform and non-uniform irreps.

### Performance and memory

- Optimized FP32 Linear, UVW and UVWConv kernels. UVW contractions avoid large intermediate tensors and reduce repeated input reads in double-backward and triple-backward.
- Improved spherical harmonics, activation, gating and normalization kernels.
- Reworked fused variable-length attention and attention-convolution kernels using CUTLASS.

### Graph construction

- Added fused CUDA graph construction shared by UUU, UVU and UVW convolutions.
- **API change:** `make_graph(src, dst, num_nodes=None)` now generates CSR and traversal tables internally. Precomputed-table arguments have been removed.

### Documentation

- Added per-operator throughput and peak-memory benchmarks, covering forward through triple-backward.

## 0.1.0

- Initial release of SO(3)-equivariant CUDA operators: tensor products and convolutions, spherical harmonics, linear layers, activations, normalization, dropout and attention.
- FP32/FP64 support and derivatives up to third order for the core equivariant operators.
- Module and functional interfaces, e3nn integration, and `torch.compile` support.
