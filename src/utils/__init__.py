"""Shared CUDA compilation, indexing, precision and scalar-expression utilities."""

from threading import RLock

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.utils._python_dispatch import is_in_torch_dispatch_mode


# Serialize first registration of configuration-specific operators, never execution.
REGISTER_OPERATOR_LOCK = RLock()


def is_compiling():
    """Select registered CUDA calls while Dynamo or make_fx records a graph."""
    return torch.compiler.is_compiling() or (
        is_in_torch_dispatch_mode() and get_proxy_mode() is not None
    )
