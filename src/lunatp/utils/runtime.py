"""Shared tensor preparation and CUDA stream reuse for tensor products."""

from __future__ import annotations

import threading

import torch

from ...utils.streams import STREAM_SOURCE, get_stream_pool

DERIVATIVE_STREAM_SOURCE = STREAM_SOURCE + r"""
struct DerivativeFork {
    DerivativeStream& lane;
    cudaStream_t caller;
    bool joined = false;
    DerivativeFork(DerivativeStream& lane_, cudaStream_t caller_)
        : lane(lane_), caller(caller_) {
        // Default event flags create intra-graph dependency edges under capture.
        C10_CUDA_CHECK(cudaEventRecord(lane.ready, caller));
        C10_CUDA_CHECK(cudaStreamWaitEvent(lane.stream, lane.ready, 0));
    }
    void join() {
        C10_CUDA_CHECK(cudaEventRecord(lane.done, lane.stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(caller, lane.done, 0));
        joined = true;
    }
    ~DerivativeFork() {
        // Also preserve allocator ordering if a later allocation/launch throws.
        if (!joined) {
            C10_CUDA_CHECK_WARN(cudaEventRecord(lane.done, lane.stream));
            C10_CUDA_CHECK_WARN(cudaStreamWaitEvent(caller, lane.done, 0));
        }
    }
};

"""

_local = threading.local()


def prepare_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    templates: tuple[torch.Tensor, ...],
    *,
    alignment: int = 1,
) -> list[torch.Tensor]:
    """Prepare incoming derivatives without copying already valid tensors.

    Missing gradients become zeros with the matching template shape. Noncontiguous
    tensors are made contiguous. Only CUTLASS callers request 16-byte alignment;
    other callers retain contiguous offset views without cloning them.
    """
    result = []
    for gradient, template in zip(gradients, templates):
        gradient = prepare_gradient(gradient, template)
        if alignment > 1 and gradient.data_ptr() % alignment:
            gradient = gradient.clone()
        result.append(gradient)
    return result


def prepare_gradient(
    gradient: torch.Tensor | None,
    template: torch.Tensor,
) -> torch.Tensor:
    """Materialize a missing derivative or copy only a noncontiguous layout."""
    if gradient is None:
        return torch.zeros_like(template)
    return gradient.contiguous()


def auxiliary_streams(
    device: int | torch.device, count: int, *, priority: int = 0, scope=None
):
    """Reuse the native stream bank across operators and JIT configurations."""
    index = device if isinstance(device, int) else device.index
    if index is None:
        index = torch.cuda.current_device()
    # Callers that already obtained their stream pass its handle here.
    purpose = 0
    if isinstance(scope, tuple):
        purpose = 1 if scope[0] == "output_gradient" else 2
        parent = scope[1]
    else:
        parent = torch.cuda.current_stream(index).cuda_stream if scope is None else scope
    requests = getattr(_local, "requests", None)
    if requests is None:
        requests = _local.requests = {}
    key = (index, parent, priority, purpose, count)
    cached = requests.get(key)
    if cached is None:
        handles = get_stream_pool().get_handles(index, parent, count, priority, purpose)
        cached = tuple(torch.cuda.ExternalStream(handle, device=index) for handle in handles)
        requests[key] = cached
    return cached


class OverlapState:
    __slots__ = ("done", "start", "stream")

    def __init__(self, device: int) -> None:
        parent = torch.cuda.current_stream(device).cuda_stream
        self.stream = auxiliary_streams(
            device, 1, scope=("output_gradient", parent)
        )[0]
        self.start = torch.cuda.Event(enable_timing=False)
        self.done = torch.cuda.Event(enable_timing=False)


def get_overlap_state(op, device: torch.device, factory=OverlapState):
    """Cache an event owner per operator, thread, device and caller stream.

    Python ownership releases native events before CUDA runtime teardown; the
    factory key separates output-contraction overlap from second- and
    third-order stream states.
    """
    local = getattr(op, "_overlap_streams_local", None)
    if local is None:
        local = threading.local()
        op._overlap_streams_local = local
    states = getattr(local, "states", None)
    if states is None:
        states = {}
        local.states = states
    index = device.index
    parent = torch.cuda.current_stream(index).cuda_stream
    key = (index, parent, factory)
    state = states.get(key)
    if state is None:
        if factory is OverlapState:
            state = factory(index)
        else:
            streams = auxiliary_streams(
                index,
                factory.stream_count,
                priority=factory.stream_priority,
                scope=("convolution", parent),
            )
            state = factory(index, [stream.cuda_stream for stream in streams])
        states[key] = state
    return state
