"""Serializable static configuration for fixed-name compiled operators."""

import json

import torch


def _encode(value):
    if isinstance(value, tuple):
        return {"tuple": [_encode(item) for item in value]}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported operator configuration value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"tuple"}:
            return tuple(_decode(item) for item in value["tuple"])
        if set(value) == {"dtype"}:
            dtype = getattr(torch, value["dtype"])
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid operator dtype: {value['dtype']}")
            return dtype
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


@torch.compiler.assume_constant_result
def encode_config(value):
    """Keep configuration in the graph as a constant string, never a local ID."""
    return json.dumps(_encode(value), separators=(",", ":"))


def decode_config(value):
    """Decode on a configuration-cache miss; callers cache prepared operators."""
    return _decode(json.loads(value))
