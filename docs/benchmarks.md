# Benchmarking

This example compares forward-pass time with e3nn. PyTorch's `Timer` handles
warmup and GPU synchronization; timings include CPU dispatch and GPU execution.
Both operators run once before timing to check correctness and compile kernels.

```python
import torch
from torch.utils.benchmark import Timer
from e3nn import o3 as e3nn_o3
from LunaEquivariance import o3

reference = e3nn_o3.Norm("64x0e + 32x1o + 16x2e + 8x3o").cuda()
model = o3.Norm(reference).cuda()
batch_size = 16384
x = torch.randn(batch_size, reference.irreps_in.dim, device="cuda")


def benchmark(model):
    timer = Timer("model(x)", globals={"model": model, "x": x})
    return timer.blocked_autorange(min_run_time=1).median * 1e6


with torch.no_grad():
    torch.testing.assert_close(model(x), reference(x), rtol=3e-5, atol=3e-5)
    reference_us = benchmark(reference)
    model_us = benchmark(model)

print(f"e3nn: {reference_us:.2f} us | LunaEquivariance: {model_us:.2f} us | "
      f"Speedup: {reference_us / model_us:.2f}x")
```

Compilation is excluded from these timings. For graph operators, build the graph
before timing as well.
