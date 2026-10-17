"""One native auxiliary-stream bank shared by all generated extensions."""

from functools import lru_cache
import re
import types


STREAM_SOURCE = r"""
#include <cstdint>
// LUNAEQ_AUXILIARY_STREAMS
struct DerivativeStream {
    cudaStream_t stream;
    cudaEvent_t ready, done;
};
using StreamGetter = DerivativeStream* (*)(int, uintptr_t, int, int, int);
static StreamGetter stream_getter = nullptr;
void set_stream_pool(uintptr_t address) {
    stream_getter = reinterpret_cast<StreamGetter>(address);
}
DerivativeStream& derivative_stream(int device, cudaStream_t caller, int slot = 0) {
    return *stream_getter(device, reinterpret_cast<uintptr_t>(caller), 0, 0, slot);
}
"""

POOL_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#include <set>
#include <cstdint>
struct DerivativeStream {
    cudaStream_t stream;
    cudaEvent_t ready, done;
};
struct StreamBank {
    int device;
    bool exhausted = false;
    std::vector<std::unique_ptr<DerivativeStream>> lanes;
    explicit StreamBank(int device_) : device(device_) {}
    ~StreamBank() noexcept {
        int previous;
        if (cudaGetDevice(&previous) != cudaSuccess) return;
        if (previous != device && cudaSetDevice(device) != cudaSuccess) return;
        for (auto& lane : lanes) {
            cudaEventDestroy(lane->ready);
            cudaEventDestroy(lane->done);
        }
        if (previous != device) cudaSetDevice(previous);
    }
};
DerivativeStream* get_lane(int device, uintptr_t caller, int priority, int scope, int slot) {
    using Key = std::tuple<int, uintptr_t, int, int>;
    static thread_local std::map<Key, std::unique_ptr<StreamBank>> banks;
    auto& owner = banks[Key(device, caller, priority, scope)];
    if (!owner) owner = std::make_unique<StreamBank>(device);
    auto& bank = *owner;
    std::set<cudaStream_t> visited;
    while (!bank.exhausted && int(bank.lanes.size()) <= slot) {
        auto stream = c10::cuda::getStreamFromPool(priority, device).stream();
        if (!visited.insert(stream).second) {bank.exhausted = true; break;}
        if (reinterpret_cast<uintptr_t>(stream) == caller) continue;
        bool exists = false;
        for (auto& lane : bank.lanes) exists |= lane->stream == stream;
        if (exists) continue;
        const c10::cuda::CUDAGuard guard(device);
        auto lane = std::make_unique<DerivativeStream>();
        lane->stream = stream;
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&lane->ready, cudaEventDisableTiming));
        try {
            C10_CUDA_CHECK(cudaEventCreateWithFlags(&lane->done, cudaEventDisableTiming));
        } catch (...) {
            cudaEventDestroy(lane->ready);
            throw;
        }
        bank.lanes.push_back(std::move(lane));
    }
    return bank.lanes[slot % bank.lanes.size()].get();
}
std::vector<uintptr_t> get_handles(int device, uintptr_t caller, int count, int priority, int scope) {
    std::vector<uintptr_t> result;
    for (int i=0; i<count; ++i)
        result.push_back(reinterpret_cast<uintptr_t>(get_lane(device, caller, priority, scope, i)->stream));
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_handles", &get_handles);
    m.def("address", []() {return reinterpret_cast<uintptr_t>(&get_lane);});
}
"""


@lru_cache(maxsize=1)
def get_stream_pool():
    from .jit import load_extension

    return load_extension(
        "auxiliary_streams", cuda_sources=POOL_SOURCE,
        extra_cflags=["-O3", "-DPYBIND11_HAS_SUBINTERPRETER_SUPPORT=0"],
        extra_cuda_cflags=["-O3", "-DPYBIND11_HAS_SUBINTERPRETER_SUPPORT=0"],
    )


def install_stream_pool(module):
    setter = getattr(module, "set_stream_pool", None)
    if setter is not None:
        setter(get_stream_pool().address())
    for value in vars(module).values():
        if isinstance(value, types.ModuleType) and value.__name__.startswith(module.__name__ + "."):
            install_stream_pool(value)


def bind_stream_pool_source(source):
    if "LUNAEQ_AUXILIARY_STREAMS" not in source or '.def("set_stream_pool"' in source:
        return source
    return re.sub(
        r"(PYBIND11_MODULE\(TORCH_EXTENSION_NAME,\s*(\w+)\)\s*\{)",
        lambda match: match[1] + f'{match[2]}.def("set_stream_pool", &set_stream_pool);', source,
    )
