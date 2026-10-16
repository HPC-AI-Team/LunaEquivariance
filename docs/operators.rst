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
   * - :doc:`o3.Linear <operators/linear>`
     - FP32/FP64
     - Linear transformation with shared or external weights
   * - :doc:`o3.SphericalHarmonics <operators/spherical_harmonics>`
     - FP32/FP64
     - Vector input; degrees 0-12
   * - :doc:`o3.Norm <operators/norm>`
     - FP32/FP64
     - One norm per irrep copy
   * - :doc:`nn.Activation <operators/activation>`
     - FP32/FP64
     - Built-in scalar activations and their parameters
   * - :doc:`nn.Gate <operators/gate>`
     - FP32/FP64
     - Scalar activations and per-copy gates
   * - :doc:`nn.NormActivation <operators/norm_activation>`
     - FP32/FP64
     - Activation applied to the norm of each irrep copy
   * - :doc:`nn.Dropout <operators/dropout>`
     - FP32/FP64
     - Dropout applied to whole irrep copies
   * - :doc:`nn.EquivariantNorm <operators/equivariant_norm>`
     - FP32/FP64
     - Layer or RMS normalization with learnable scale and bias
   * - :doc:`nn.GraphSoftmax <operators/graph_softmax>`
     - FP32/FP64
     - Segment-wise softmax with optional envelope
   * - :doc:`nn.VarlenAttention <operators/varlen_attention>`
     - FP16/BF16/FP32
     - Variable-length self-attention with optional bias and causal mask
   * - :doc:`nn.VarlenAttentionConv <operators/varlen_attention_conv>`
     - FP16/BF16/FP32
     - Self-attention with envelope-weighted aggregation at atoms

   * - :doc:`TensorProduct <operators/tensor_product>`
     - FP32/FP64
     - Tensor products with ``uuu``, ``uvu`` and ``uvw`` connections
   * - :doc:`TensorProductConv <operators/tensor_product_conv>`
     - FP32/FP64
     - Fused graph convolutions with ``uuu``, ``uvu`` and ``uvw`` connections

- Requires an NVIDIA GPU with compute capability 8.0 or later.
- All operators support derivatives up to third order.
