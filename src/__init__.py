"""Equivariant CUDA operators with e3nn-style o3/nn entries and tensor products."""

from .o3.irreps import Irrep, Irreps
from .lunatp.utils.layout import TensorProductConfig
from .lunatp.graph import LunaTPGraph, make_graph
from .lunatp import TensorProduct, TensorProductConv
from .lunatp.interface import tensor_product, tensor_product_conv, prepare_tensor_product
from .modules.linear import LunaEqLinear
from .modules.spherical_harmonics import LunaEqSphericalHarmonics
from .o3.spherical_harmonics.interface import spherical_harmonics
from .modules.activation import LunaEqActivation
from .modules.gate import LunaEqGate
from .modules.norm import LunaEqNorm
from .modules.norm_activation import LunaEqNormActivation
from .modules.dropout import LunaEqDropout
from .nn.dropout.interface import dropout
from .modules.varlen_attention import VarlenAttention, VarlenAttentionConv
from .nn.varlen_attention.interface import varlen_attention
from .nn.varlen_attention_conv.interface import varlen_attention_conv
from .modules.graph_softmax import GraphSoftmax
from .nn.graph_softmax.interface import graph_softmax, build_graph_softmax_row_ptr
from .modules.equ_norm import LunaEqEquivariantNorm
from .o3.linear.interface import linear
from .o3.norm.interface import norm
from .nn.activation.interface import activation
from .nn.gate.interface import gate
from .nn.norm_activation.interface import norm_activation
from .nn.equ_norm.interface import equivariant_norm
from . import nn, o3

# Match e3nn's spherical-harmonics function in the o3 namespace.
o3.spherical_harmonics = spherical_harmonics

Linear = LunaEqLinear
SphericalHarmonics = LunaEqSphericalHarmonics
Activation = LunaEqActivation
Gate = LunaEqGate
Norm = LunaEqNorm
NormActivation = LunaEqNormActivation
Dropout = LunaEqDropout
EquivariantNorm = LunaEqEquivariantNorm

class LunaEq:
    """Functional operators and optional parameter-owning modules."""

    Linear = Linear
    SphericalHarmonics = SphericalHarmonics
    Activation = Activation
    Gate = Gate
    Norm = Norm
    NormActivation = NormActivation
    Dropout = Dropout
    EquivariantNorm = EquivariantNorm
    linear = staticmethod(linear)
    spherical_harmonics = staticmethod(spherical_harmonics)
    activation = staticmethod(activation)
    gate = staticmethod(gate)
    norm = staticmethod(norm)
    norm_activation = staticmethod(norm_activation)
    dropout = staticmethod(dropout)
    equivariant_norm = staticmethod(equivariant_norm)
    TensorProduct = TensorProduct
    TensorProductConv = TensorProductConv
    LunaTPGraph = LunaTPGraph
    make_graph = staticmethod(make_graph)
    tensor_product = staticmethod(tensor_product)
    tensor_product_conv = staticmethod(tensor_product_conv)
    prepare_tensor_product = staticmethod(prepare_tensor_product)
    o3 = o3
    nn = nn


lunaeq = LunaEq
LunaEquivariance = LunaEq

__all__ = [
    "graph_softmax",
    "build_graph_softmax_row_ptr",
    "GraphSoftmax",
    "varlen_attention",
    "VarlenAttention",
    "VarlenAttentionConv",
    "varlen_attention_conv",
    "tensor_product",
    "tensor_product_conv",
    "prepare_tensor_product",
    "equivariant_norm",
    "TensorProduct",
    "TensorProductConv",
    "Irrep",
    "Irreps",
    "TensorProductConfig",
    "LunaTPGraph",
    "make_graph",
    "LunaEqLinear",
    "LunaEqSphericalHarmonics",
    "LunaEqActivation",
    "LunaEqGate",
    "LunaEqNorm",
    "LunaEqNormActivation",
    "LunaEqDropout",
    "Linear",
    "SphericalHarmonics",
    "spherical_harmonics",
    "Activation",
    "Gate",
    "Norm",
    "NormActivation",
    "Dropout",
    "linear",
    "activation",
    "gate",
    "norm",
    "norm_activation",
    "dropout",
    "o3",
    "nn",
    "LunaEq",
    "LunaEquivariance",
    "LunaEqEquivariantNorm",
    "EquivariantNorm",
]
