# Changelog

## 0.2.0

### Performance and memory

- Optimized FP32 Linear, UVW and UVWConv kernels. UVW contractions avoid large intermediate tensors and reduce repeated input reads in double-backward and triple-backward.
- Improved spherical harmonics, activation, gating and normalization kernels.
- Reworked fused variable-length attention and attention-convolution kernels using CUTLASS.

### Graph construction

- Added fused CUDA graph construction shared by UUU, UVU and UVW convolutions.
- **API change:** `make_graph(src, dst, num_nodes=None)` now generates CSR and traversal tables internally. Precomputed-table arguments have been removed.

### Documentation

- Added per-operator throughput and peak-memory benchmarks, covering forward through triple-backward.
- Added a NequIP AOTI example that saves and restores model-specific operator registrations.

### Compatibility

- Use a Tensor return for nonconstant spherical-harmonics backward operators so AOTInductor can export force models; constant harmonics retain their absent input gradient.

## 0.1.0

- Initial release of SO(3)-equivariant CUDA operators: tensor products and convolutions, spherical harmonics, linear layers, activations, normalization, dropout and attention.
- FP32/FP64 support and derivatives up to third order for the core equivariant operators.
- Module and functional interfaces, e3nn integration, and `torch.compile` support.
