"""Shared numerical, compilation and export checks."""

import inspect
import os
from pathlib import Path
import subprocess
import sys

import torch
from e3nn import o3
from torch.utils._pytree import tree_leaves, tree_map, tree_map_only


REFERENCE_TOLERANCES = {
    torch.float16: (3e-3, 3e-3),
    torch.bfloat16: (3e-2, 3e-2),
    torch.float32: (2e-5, 2e-5),
    torch.float64: (1e-11, 1e-11),
}


def fp64(values):
    return tree_map_only(
        torch.Tensor,
        lambda value: value.detach().double().requires_grad_(value.requires_grad),
        values,
    )


def assert_reference_close(actual, reference, *, dtype=None):
    """Compare maximum error against FP64 using the computation dtype."""
    assert actual.shape == reference.shape
    assert reference.dtype == torch.float64
    if actual.numel() == 0:
        return
    atol, rtol = REFERENCE_TOLERANCES[dtype or actual.dtype]
    error = (actual.detach().double() - reference.detach()).abs().max()
    limit = atol + rtol * reference.detach().abs().max()
    assert (
        error <= limit
    ), f"max absolute error {error.item():.6e} exceeds {limit.item():.6e}"


def rotate_features(features, irreps, rotation):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        matrix = o3.Irreps(irreps).D_from_matrix(rotation.double().cpu())
    finally:
        torch.set_default_dtype(previous_dtype)
    return features @ matrix.to(features).T


def assert_compiled(module, samples, *, atol=3e-5, rtol=3e-4):
    """Compile the public call directly and compare its outputs and first gradients."""
    torch._dynamo.reset()
    compiled = torch.compile(
        module, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False}
    )
    for args, kwargs in samples:
        expected = module(*args, **kwargs)
        actual = compiled(*args, **kwargs)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        targets = tuple(
            value
            for value in tree_leaves((args, kwargs))
            if isinstance(value, torch.Tensor) and value.requires_grad
        ) + tuple(
            parameter for parameter in module.parameters() if parameter.requires_grad
        )
        wanted = tuple(value for value in tree_leaves(expected) if value.requires_grad)
        got = tuple(value for value in tree_leaves(actual) if value.requires_grad)
        if targets and wanted:
            seeds = tuple(torch.randn_like(value) for value in wanted)
            expected_grad = torch.autograd.grad(
                wanted, targets, seeds, allow_unused=True
            )
            actual_grad = torch.autograd.grad(got, targets, seeds, allow_unused=True)
            torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)


def dynamic_shapes(module, samples):
    """Share symbolic dimensions across operands and CSR lengths."""
    bound = [
        inspect.signature(module.forward).bind(*args, **kwargs).arguments
        for args, kwargs in samples
    ]
    leaves = [tree_leaves(arguments) for arguments in bound]
    dimensions = []

    def shape(value, index):
        if not isinstance(value, torch.Tensor):
            return None
        result = {}
        for axis in range(value.ndim):
            sizes = tuple(items[index].shape[axis] for items in leaves)
            if len(set(sizes)) == 1:
                continue
            for previous, dimension in dimensions:
                offsets = {current - old for current, old in zip(sizes, previous)}
                if len(offsets) == 1:
                    offset = offsets.pop()
                    result[axis] = dimension + offset if offset else dimension
                    break
            else:
                dimension = torch.export.Dim(f"size_{len(dimensions)}", min=2, max=4096)
                dimensions.append((sizes, dimension))
                result[axis] = dimension
        return result

    index = iter(range(len(leaves[0])))
    return tree_map(lambda tensor: shape(tensor, next(index)), bound[0])


def assert_aoti(cases, directory, *, atol=3e-5, rtol=3e-4):
    """Load two configurations together in a fresh interpreter, without constructing modules."""
    torch._dynamo.reset()
    directory = Path(directory)
    fixtures = []
    for index, (module, samples) in enumerate(cases):
        records = []
        for args, kwargs in samples:
            with torch.no_grad():
                output = module(*args, **kwargs)
            records.append(
                tree_map(
                    lambda value: value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else value,
                    (args, kwargs, output),
                )
            )
        args, kwargs = samples[0]
        with torch.no_grad():
            exported = torch.export.export(
                module,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes(module, samples),
                strict=True,
            )
        assert any(
            node.op == "call_function"
            and str(node.target).startswith(("lunaeq.", "lunaeq_graph.", "lunatp."))
            for node in exported.graph.nodes
        ), "Export bypassed the native operator"
        package = directory / f"operator_{index}.pt2"
        torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
        fixtures.append((str(package), records))
    fixture_path = directory / "inputs.pt"
    torch.save(fixtures, fixture_path)
    script = """
import sys
import torch
from torch.utils._pytree import tree_map
import LunaEquivariance

registry = {name for name in torch._C._dispatch_get_all_op_names()
            if name.startswith(('lunaeq::', 'lunaeq_graph::', 'lunatp::'))}
fixtures = torch.load(sys.argv[1], weights_only=True)
models = [torch._inductor.aoti_load_package(path) for path, _ in fixtures]
for index in (*range(len(models)), *reversed(range(len(models)))):
    for args, kwargs, expected in fixtures[index][1]:
        args, kwargs = tree_map(lambda value: value.cuda() if isinstance(value, torch.Tensor) else value, (args, kwargs))
        with torch.no_grad():
            actual = models[index](*args, **kwargs)
        actual = tree_map(lambda tensor: tensor.cpu(), actual)
        torch.testing.assert_close(actual, expected, atol=float(sys.argv[2]), rtol=float(sys.argv[3]))
assert registry == {name for name in torch._C._dispatch_get_all_op_names()
                    if name.startswith(('lunaeq::', 'lunaeq_graph::', 'lunatp::'))}
"""
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment["PYTHONPATH"] = os.pathsep.join(map(str, sys.path))
    result = subprocess.run(
        [sys.executable, "-c", script, str(fixture_path), str(atol), str(rtol)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class ConvolutionInputs(torch.nn.Module):
    """Expose prepared graph tensors as export inputs; topology construction stays outside."""

    def __init__(self, operator):
        super().__init__()
        self.operator = operator

    def forward(self, x, edge, weight, tables):
        from LunaEquivariance import LunaTPGraph
        from LunaEquivariance.lunatp.uvu_conv.graph import SourceSegmentTables

        graph = LunaTPGraph(
            src=tables[0],
            dst=tables[1],
            num_nodes=x.shape[0],
            dst_rowptr=tables[2],
            src_rowptr=tables[3],
            edge_ids_by_src=tables[4],
            edge_ids_by_dst=tables[5],
            node_order=tables[6],
            source_node_order=tables[7],
            node_tiers=tables[8],
            source_segments=SourceSegmentTables(*tables[9:]),
        )
        graph.store_topology()
        return self.operator(x, edge, weight, graph=graph)


def graph_tensors(graph):
    return (
        graph.src,
        graph.dst,
        graph.dst_rowptr,
        graph.src_rowptr,
        graph.edge_ids_by_src,
        graph.edge_ids_by_dst,
        graph.node_order,
        graph.source_node_order,
        graph.node_tiers,
        graph.source_segments.nodes,
        graph.source_segments.begin,
        graph.source_segments.end,
        graph.source_segments.rowptr,
        graph.source_segments.order,
    )
