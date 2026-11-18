"""Measure public SO(2) functions with Luna's compatibility checker.

Run from the repository root with:
    python -m tests.so2.benchmark --output /path/to/new/results.json
The small suite validates the measurement path without estimating performance.
"""

import argparse
import gc
import itertools
import json
import os
from pathlib import Path
import sys

import torch

from LunaEquivariance import so2
from tests.so2.compatibility import check_compatibility
from tests.so2.reference import (
    attention_score, ffn, gather_rotation, inverse_rotation_reduction,
    radial, recombination, rotation, segmented_linear,
)
from tests.so2 import bench_convolution, bench_transforms, bench_weights


def linear_reference(sectors, weights, bias=None, *, lmax, mmax):
    """Use one batched PyTorch GEMM per order, as in the native SO(2) layer."""
    edges = sectors[0].shape[0]
    result = [torch.nn.functional.linear(sectors[0].reshape(edges, -1), weights[0], bias)]
    for sector, weight in zip(sectors[1:], weights[1:], strict=True):
        output = torch.nn.functional.linear(sector.reshape(edges * 2, -1), weight)
        result.append(output.reshape(edges, 2, weight.shape[0]))
    return tuple(result)


REFERENCES = {
    so2.rotation: rotation.rotation,
    so2.rotate_gather: gather_rotation.rotate_gather,
    so2.linear: linear_reference,
    so2.recombine_activate: recombination.recombine_activate,
    so2.inverse_rotate_reduce: inverse_rotation_reduction.inverse_rotate_reduce,
    so2.feed_forward: ffn.feed_forward,
    so2.radial_network: radial.radial_network,
    so2.segmented_linear: segmented_linear.segmented_linear,
    so2.attention_scores: attention_score.attention_scores,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--small", action="store_true")
    names = [operator.__name__ for operator in REFERENCES]
    parser.add_argument("--operator", choices=names, nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("SO(2) performance checks require CUDA")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = {
        "device": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "dtype": "float32", "tf32": False,
        "kind": "measurement_path_check" if args.small else "performance",
        "warmup": 1 if args.small else 3, "repeats": 2 if args.small else 20,
        "statistic": "median_synchronized_wall_time", "cuda_graphs": False,
        "orders": {"0": "forward", "1": "forward_loss_backward",
                   "2": "forward_loss_backward_double_backward"},
        "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "node": os.uname().nodename, "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "argv": sys.argv,
        "cases": [],
    }
    problems = itertools.chain.from_iterable(module.cases(small=args.small, operators=args.operator)
                                             for module in (bench_convolution, bench_transforms, bench_weights))
    for name, function, inputs, kwargs, metadata in problems:
        if args.operator and metadata["operator"] not in args.operator:
            continue
        is_rotation = metadata["operator"] == "rotation"
        reference = (rotation.quaternion_wigner_from_edges_retained
                     if is_rotation and metadata["variant"] == "quaternion" else REFERENCES[function])
        print(f"\n{name}: {metadata}", flush=True)
        record = {"name": name, "configuration": metadata,
                  "reference": f"{reference.__module__}.{reference.__qualname__}",
                  "loss": "mean(sin(output))" if is_rotation else "sum(mean(output**2))",
                  "reports": []}
        result["cases"].append(record)
        for order in range(3):
            report = check_compatibility(
                function, inputs, reference=reference, kwargs=kwargs,
                derivative_order=order, warmup=result["warmup"], repeats=result["repeats"],
                # Squared Wigner row norms are constant on SO(3).
                loss_fn=(lambda output: output.sin().mean()) if is_rotation else None,
                atol=3e-5, rtol=3e-4,
            )
            record["reports"].append(report)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            if not report["passed"]:
                raise AssertionError(f"Numerical comparison failed for {name}, order {order}")
        del inputs, kwargs, report, record
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
