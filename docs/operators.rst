Supported operations
====================

The following operators are available as modules and functions. See the
:doc:`Public API <api>` for parameters, tensor shapes and examples.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Operator
     - Supported dtypes
     - Description
   * - ``o3.Linear``
     - FP32/FP64
     - Linear transformation with shared or external weights
   * - ``o3.SphericalHarmonics``
     - FP32/FP64
     - Vector input; degrees 0-12
   * - ``o3.Norm``
     - FP32/FP64
     - One norm per irrep copy
   * - ``nn.Activation``
     - FP32/FP64
     - Built-in scalar activations and their parameters
   * - ``nn.Gate``
     - FP32/FP64
     - Scalar activations and per-copy gates
   * - ``nn.NormActivation``
     - FP32/FP64
     - Activation applied to the norm of each irrep copy
   * - ``nn.Dropout``
     - FP32/FP64
     - Dropout applied to whole irrep copies
   * - ``nn.EquivariantNorm``
     - FP32/FP64
     - Layer or RMS normalization with learnable scale and bias
   * - ``nn.GraphSoftmax``
     - FP32/FP64
     - Segment-wise softmax with optional envelope
   * - ``nn.VarlenAttention``
     - FP16/BF16/FP32
     - Variable-length self-attention with optional bias and causal mask
   * - ``nn.VarlenAttentionConv``
     - FP16/BF16/FP32
     - Self-attention with envelope-weighted aggregation at atoms

   * - ``TensorProduct``
     - FP32/FP64
     - Tensor products with ``uuu``, ``uvu`` and ``uvw`` connections
   * - ``TensorProductConv``
     - FP32/FP64
     - Fused graph convolutions with ``uuu``, ``uvu`` and ``uvw`` connections

- Requires an NVIDIA GPU with compute capability 8.0 or later.
- All operators support derivatives up to third order.
