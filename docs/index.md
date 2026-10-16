# Luna<wbr>Equivariance

High-performance SO(3)-equivariant operators for graph neural networks.

- **Performance:** order-of-magnitude speedups across a range of operators.
- **Operator coverage:** equivariant layers, fused graph convolutions and attention.
- **Differentiation and precision:** derivatives up to third order, with FP32/FP64 support.
- **PyTorch integration:** easy integration into PyTorch models, with `torch.compile` support.

## Installation

Install CUDA-enabled PyTorch first, then install LunaEquivariance:

```bash
pip install LunaEquivariance
```

See the [installation guide](installation.md) for requirements and setup.

## Documentation

- [Supported operations](operators.rst): available operators and supported configurations.
- [Public API](api.md): formulas, parameters and tensor shapes.
- [Usage](usage.md): examples and model integration.

```{toctree}
:hidden:
:maxdepth: 1

installation
operators
api
usage
```

## Citation

If you use LunaEquivariance in your research, please cite:

```bibtex
@misc{zhou2026lunatpenablingefficientscalable,
    title={LunaTP: Enabling Efficient and Scalable Tensor Product for Equivariant MLIPs},
    author={Yuanchang Zhou and Mingzhen Li and Chen Wang and Xiangyu Zhang and Guangming Tan and Weile Jia},
    year={2026},
    eprint={2610.18393},
    archivePrefix={arXiv},
    url={https://arxiv.org/abs/2610.18393},
}
```
