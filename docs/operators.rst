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
   * - :doc:`o3.SymmetricContraction <operators/symmetric_contraction>`
     - FP32/FP64
     - Weighted contractions of repeated equivariant features
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
   * - :doc:`so2.rotation <operators/so2/rotation>`
     - FP32
     - Compact Wigner-D rotation rows from local coordinate frames
   * - :doc:`so2.rotate_gather <operators/so2/rotate_gather>`
     - FP32
     - Fused endpoint gather, rotation and radial scaling by angular order
   * - :doc:`so2.linear <operators/so2/linear>`
     - FP16/BF16/FP32/FP64
     - Order-wise linear projection on local-frame sectors
   * - :doc:`so2.recombine_activate <operators/so2/recombine_activate>`
     - FP32
     - Complex recombination with gated or spherical-grid activation
   * - :doc:`so2.inverse_rotate_reduce <operators/so2/inverse_rotate_reduce>`
     - FP32
     - Complex recombination, inverse rotation and receiver-node aggregation
   * - :doc:`so2.attention_scores <operators/so2/attention_scores>`
     - FP32
     - Fused LayerNorm, SmoothLeakyReLU and head-wise attention projection
   * - :doc:`so2.feed_forward <operators/so2/feed_forward>`
     - FP32
     - Gated coefficient or grid-factor equivariant feed-forward network
   * - :doc:`so2.radial_network <operators/so2/radial_network>`
     - FP32
     - Fused three-layer radial MLP with optional atom-type embeddings
   * - :doc:`so2.segmented_linear <operators/so2/segmented_linear>`
     - FP16/BF16/FP32/FP64
     - Segment-wise linear maps with mixed expert weights
   * - :doc:`so2.models.optimize <operators/so2/optimize>`
     - FP32
     - In-place SO(2) operator integration for eSEN, UMA and EquiformerV3

- Requires an NVIDIA GPU with compute capability 8.0 or later.
- All operators support derivatives up to third order.
