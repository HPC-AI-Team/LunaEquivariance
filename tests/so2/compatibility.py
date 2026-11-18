"""Derivative and timing comparisons for functions with external weights."""

from statistics import median
from time import perf_counter

import torch
from torch.utils._pytree import tree_flatten

from LunaEquivariance.utils.testing import error_metrics, evaluate, print_report


def check_compatibility(
    module, args, *, reference, kwargs=None, loss_fn=None, derivative_order=1,
    warmup=3, repeats=20, atol=1e-5, rtol=1e-4, seed=0, verbose=True,
):
    """Compare tensor functions and time the requested derivative order."""
    if not isinstance(derivative_order, int) or derivative_order < 0:
        raise ValueError("derivative_order should be a nonnegative integer")
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup should be nonnegative and repeats positive")
    kwargs = {} if kwargs is None else kwargs
    if loss_fn is None:
        def loss_fn(output):
            return sum(x.square().mean() if x.numel() else x.sum()
                       for x in tree_flatten(output)[0])
    tensors = [x for x in tree_flatten((args, kwargs))[0] if isinstance(x, torch.Tensor)]
    sample = next(x for x in tensors if x.is_floating_point())
    devices = sorted({x.device.index for x in tensors if x.is_cuda})

    def synchronize():
        for device in devices:
            torch.cuda.synchronize(device)

    def reset_seed():
        torch.random.default_generator.manual_seed(seed)
        for device in devices:
            torch.cuda.default_generators[device].manual_seed(seed)

    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        reset_seed()
        modules = (module, reference)
        targets = [list({id(x): x for x in tensors if x.requires_grad}.values())] * 2
        if derivative_order:
            if not targets[0]:
                raise ValueError("Set requires_grad=True on inputs or use trainable parameters")
        generator = torch.Generator().manual_seed(seed)
        directions = [
            tuple(
                torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device)
                for x in targets[0]
            )
            for _ in range(derivative_order - 1)
        ]
        results = []
        for candidate, module_targets in zip(modules, targets):
            reset_seed()
            results.append(evaluate(
                candidate, args, kwargs, loss_fn, module_targets,
                derivative_order, directions, collect=True,
            ))
        report = {
            "operator": getattr(module, "__name__", type(module).__name__), "reference": "PyTorch", "device": str(sample.device), "dtype": str(sample.dtype),
            "errors": {
                stage: error_metrics(value, results[1][stage], atol, rtol)
                for stage, value in results[0].items()
            },
            "loss": {
                "lunaeq": results[0]["loss"].item(),
                "reference": results[1]["loss"].item(),
            },
            "timings": {},
        }
        del results
        report["passed"] = all(error["passed"] for error in report["errors"].values())
        order = derivative_order
        samples = [[], []]
        for iteration in range(warmup + repeats):
            for index in ((0, 1) if iteration % 2 == 0 else (1, 0)):
                reset_seed()
                synchronize()
                start = perf_counter()
                if order == 0:
                    modules[index](*args, **kwargs)
                else:
                    evaluate(
                        modules[index], args, kwargs, loss_fn,
                        targets[index], order, directions,
                    )
                synchronize()
                if iteration >= warmup:
                    samples[index].append((perf_counter() - start) * 1000)
        luna_ms, reference_ms = map(median, samples)
        stage = "forward" if order == 0 else f"forward_loss_gradient_{order}"
        report["timings"][stage] = {
            "lunaeq_ms": luna_ms, "reference_ms": reference_ms,
            "speedup": reference_ms / luna_ms,
            "samples_ms": {"lunaeq": samples[0], "reference": samples[1]},
        }
    if verbose:
        print_report(report)
    return report
