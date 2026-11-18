"""Shared graph, reference and dependency contracts."""

import importlib
import inspect
import os
import subprocess
import sys
from importlib import import_module
from types import ModuleType

import pytest
import torch

from LunaEquivariance.so2.utils.tensor_contracts import allow_atomic_reduction, require_int32_offsets
from LunaEquivariance.so2.utils.graph import graph_tables, prefix_csr_metadata, prepare_graph
from LunaEquivariance.so2.rotation import interface as wigner
from LunaEquivariance.so2.rotation.kernels import backward as wigner_backward
from LunaEquivariance.so2.rotation.kernels import double_backward as wigner_double_backward
from tests.so2.checks import tiny_so2_graph


def test_csr_accepts_sorted_and_empty_graphs():
    edges = torch.tensor([[2, 0, 1, 0], [0, 1, 1, 2]])
    order, source, receiver = prefix_csr_metadata(edges, 3)
    assert order.tolist() == [1, 3, 2, 0]
    assert source.tolist() == [0, 2, 3, 4]
    assert receiver.tolist() == [0, 1, 3, 4]
    assert prefix_csr_metadata(torch.empty((2, 0), dtype=torch.long), 0)[
        1
    ].tolist() == [0]


@pytest.mark.parametrize(
    "edges,message",
    [
        ([[0, 1], [1, 0]], "sorted by receiver"),
        ([[-1, 0], [0, 1]], "source indices"),
        ([[0, 2], [0, 1]], "source indices"),
        ([[0, 1], [0, 2]], "receiver indices"),
    ],
)
def test_csr_rejects_invalid_graph_before_conversion(edges, message):
    with pytest.raises(RuntimeError, match=message):
        prefix_csr_metadata(torch.tensor(edges), 2)


def test_grid_major_int32_address_limit_without_allocating_workspace():
    grids, channels = 112, 64
    limit = 2**31 // (grids * channels)
    require_int32_offsets(
        (limit, grids, channels), (channels, limit * channels, 1), name="projection"
    )
    with pytest.raises(ValueError, match="32-bit element offset"):
        require_int32_offsets(
            (limit + 1, grids, channels),
            (channels, (limit + 1) * channels, 1),
            name="projection",
        )
    with pytest.raises(ValueError, match="32-bit element offset"):
        require_int32_offsets((2, 2), (2**31, 1), name="gapped input")


def test_atomic_contract_respects_strict_and_warn_only():
    previous = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        allow_atomic_reduction("probe")
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="not deterministic"):
            allow_atomic_reduction("probe")
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(RuntimeWarning, match="not deterministic"):
            allow_atomic_reduction("probe")
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn)


@pytest.mark.parametrize(
    "module,reference,names",
    [
        ("so2.rotate_gather", "gather_rotation", None),
        ("so2.inverse_rotate_reduce", "inverse_rotation_reduction", None),
        ("so2.utils.coefficient_gate", "gate", ("node_gate",)),
        ("so2.utils.degree_norm", "norm", ("fused_rms_norm",)),
        ("so2.feed_forward", "ffn", None),
        ("so2.radial_network", "radial", None),
        ("so2.inverse_rotate_reduce.interface", "inverse_rotation_reduction",
         ("fused_initial_atom", "fused_initial_scalar")),
        ("so2.radial_network.interface", "radial",
         ("fused_radial_input",)),
    ],
)
def test_public_tensor_operations_have_matching_reference_arguments(module, reference, names):
    operations = importlib.import_module(f"LunaEquivariance.{module}")
    definitions = importlib.import_module(f"tests.so2.reference.{reference}")
    for name in operations.__all__ if names is None else names:
        function = getattr(operations, name)
        if inspect.isclass(function):
            continue
        definition = getattr(definitions, name)
        actual = inspect.signature(function)
        expected = inspect.signature(definition)
        assert actual.parameters.keys() == expected.parameters.keys(), name
        assert [p.kind for p in actual.parameters.values()] == [
            p.kind for p in expected.parameters.values()
        ], name


@pytest.mark.parametrize("module,name,reference_module,definition", [
    ("so2.linear.interface", "raw_sector_linear", "projection", "raw_sector_linear"),
    ("so2.linear.interface", "project_sectors", "projection", "project_sectors"),
    ("so2.utils.sector_layout", "merge_sectors", "projection", "merge_sectors"),
    ("so2.recombine_activate.interface", "recombine_sectors",
     "recombination", "recombine_sectors"),
    ("so2.recombine_activate.interface", "gated_activation", "recombination", "activate_gated"),
    ("so2.recombine_activate.interface", "grid_activation", "recombination", "activate_grid"),
    ("so2.attention_scores", "attention_scores", "attention_score", "attention_scores"),
    ("so2.segmented_linear", "segmented_linear", "segmented_linear", "segmented_linear"),
])
def test_edge_reference_arguments(module, name, reference_module, definition):
    operations = importlib.import_module(f"LunaEquivariance.{module}")
    reference = importlib.import_module(f"tests.so2.reference.{reference_module}")
    actual = dict(inspect.signature(getattr(operations, name)).parameters)
    expected = inspect.signature(getattr(reference, definition)).parameters
    if name == "grid_activation":
        projected = actual.pop("projected")
        assert projected.kind is inspect.Parameter.KEYWORD_ONLY
        assert projected.default is None
    assert actual.keys() == expected.keys()
    assert [p.kind for p in actual.values()] == [p.kind for p in expected.values()]


def test_graph_tables_are_reused():
    topology = tiny_so2_graph("cpu")
    first = graph_tables(topology)
    assert graph_tables(topology) is first
    assert first[0].dtype == torch.int64
    assert all(value.dtype == torch.int32 for value in first[1])
    topology.src[0] = 0
    with pytest.raises(RuntimeError, match="topology changed"):
        graph_tables(topology)


def test_inference_graph_rebuilds_unversioned_tables():
    with torch.inference_mode():
        topology = tiny_so2_graph("cpu")
        first = graph_tables(topology)
        topology.src[0] = 0
        # A receiver-owned derivative table from a preceding call must not survive.
        topology._tables[("so2", "attention_csr")] = torch.empty(0, dtype=torch.int64)
        second = graph_tables(topology)
        assert second[0] is not first[0]
        assert second[0][0, 0] == 0
        assert first[0][0, 0] == 1
        assert ("so2", "attention_csr") not in topology._tables


@pytest.mark.parametrize(
    "call",
    [
        "wigner_vjp",
        "wigner_hvp",
        "tail_forward",
        "tail_initial",
        "tail_vjp",
        "tail_hvp",
    ],
)
def test_atomic_paths_reject_strict_before_launch_cpu(call):
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as forward
    from LunaEquivariance.so2.inverse_rotate_reduce.kernels import forward as forward_forward
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as derivatives
    from LunaEquivariance.so2.inverse_rotate_reduce.kernels import backward as derivatives_backward
    from LunaEquivariance.so2.inverse_rotate_reduce.kernels import double_backward as derivatives_double_backward
    operations = {
        "wigner_vjp": lambda: wigner_backward.wigner_backward(None, None, None, None, None, None),
        "wigner_hvp": lambda: wigner_double_backward.wigner_hvp(None, None, None, None, None, None, None),
        "tail_forward": lambda: forward_forward.sum_forward(
            None, None, None, None, None, lmax=2, mmax=2, wigner_layout=0
        ),
        "tail_initial": lambda: forward_forward.initial_forward(
            None, None, None, None, None, None, None, lmax=2, mmax=2, wigner_layout=0
        ),
        "tail_vjp": lambda: derivatives_backward.sum_backward(
            None, None, None, None, None, None, lmax=2, mmax=2, wigner_layout=0
        ),
        "tail_hvp": lambda: derivatives_double_backward.sum_double_backward(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            lmax=2,
            mmax=2,
            wigner_layout=0,
        ),
    }
    enabled, warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="(?i)determin"):
            operations[call]()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)


def test_atomic_warn_only_keeps_execution_cpu(monkeypatch):
    from LunaEquivariance.so2.inverse_rotate_reduce import interface as derivatives
    from LunaEquivariance.so2.inverse_rotate_reduce.kernels import backward as derivatives_backward
    seen = []

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                seen.append(kwargs["order"])

            return launch

    monkeypatch.setattr(derivatives_backward, "sector_message_backward_kernel", Kernel())
    monkeypatch.setattr(derivatives_backward, "sector_parameter_backward_kernel", Kernel())
    enabled, warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(RuntimeWarning, match="not deterministic"):
            derivatives_backward.sum_backward(
                torch.ones(2, 4, 2),
                (torch.ones(3, 2, 2),),
                torch.ones(3),
                torch.ones(3, 4, 4),
                torch.tensor([0, 0, 1]),
                2,
                lmax=1,
                mmax=0,
                wigner_layout=0,
            )
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    assert seen == [0, 0]


def test_public_so2_imports_without_model_packages():
    code = """
import importlib.abc
import sys
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fairchem', 'so2_ops'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectModels())
from LunaEquivariance import so2
assert set(so2.__all__) == {
    'rotation', 'rotate_gather', 'linear', 'recombine_activate', 'inverse_rotate_reduce',
    'attention_scores', 'feed_forward', 'radial_network', 'segmented_linear',
}
assert all(callable(getattr(so2, name)) for name in so2.__all__)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("module", [
    "so2",
    "so2.rotate_gather",
    "so2.inverse_rotate_reduce",
    "so2.linear",
    "so2.recombine_activate",
    "so2.rotation",
    "so2.attention_scores",
    "so2.feed_forward",
    "so2.radial_network",
    "so2.segmented_linear",
])
def test_operator_packages_export_only_their_public_api(module):
    operators = import_module(f"LunaEquivariance.{module}")
    exports = {
        name for name, value in vars(operators).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    }
    assert exports == set(operators.__all__)


def test_node_and_embedding_imports_do_not_load_model_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'fairchem' or fullname.startswith('LunaEquivariance.so2.models'):
            raise AssertionError(f'operator imported model dependency: {fullname}')

sys.meta_path.insert(0, RejectModelImports())
from LunaEquivariance.so2.utils.degree_norm import degree_norm
from LunaEquivariance.so2.utils.coefficient_gate import node_gate
from LunaEquivariance.so2.inverse_rotate_reduce.interface import fused_initial_atom, fused_initial_scalar
for function in (degree_norm, node_gate, fused_initial_atom, fused_initial_scalar):
    assert callable(function) and function.__doc__
""",
        ],
        check=True,
    )


def test_rotation_imports_without_model_packages():
    code = """
import sys
import importlib.abc
class RejectModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'fairchem' or fullname.startswith('LunaEquivariance.so2.models'):
            raise AssertionError(f'Unexpected model dependency: {fullname}')
sys.meta_path.insert(0, RejectModels())
from LunaEquivariance.so2.utils.layout import get_angular_layout
from LunaEquivariance.so2.rotation import rotation
assert callable(rotation)
assert not any(name == 'fairchem' or name.startswith('fairchem.') for name in sys.modules)
assert not any(name.startswith('LunaEquivariance.so2.models') for name in sys.modules)
assert get_angular_layout(lmax=3, mmax=2, channels=5).full_components == 16
from LunaEquivariance.so2.rotation.kernels.utils import branchless_edge_rotation
import torch
frames = branchless_edge_rotation(torch.tensor([[1., 0., 0.]]), torch.tensor([[.1, .7, .9]]))
assert frames.shape == (1, 3, 3)
from importlib import import_module
rotation = import_module("LunaEquivariance.so2.rotation")
for name in rotation.__all__:
    assert callable(getattr(rotation, name)), name
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_unsorted_graph_preserves_edge_tensor_alignment():
    edges = torch.tensor([[2, 0, 3, 1, 2], [3, 1, 3, 0, 1]])
    result = prepare_graph(edges, 5)
    assert result.edge_order.tolist() == [3, 1, 4, 0, 2]
    assert torch.equal(result.edge_index, edges[:, result.edge_order])
    source_order, source_ptr, receiver_ptr = result.metadata
    assert result.source_order is source_order
    assert result.source_row_ptr is source_ptr
    assert result.receiver_row_ptr is receiver_ptr
    assert source_order.dtype == source_ptr.dtype == receiver_ptr.dtype == torch.int32
    assert source_ptr.tolist() == [0, 1, 2, 4, 5, 5]
    assert receiver_ptr.tolist() == [0, 1, 3, 3, 5, 5]
    values = torch.arange(5.0).requires_grad_()
    sorted_values = values.index_select(0, result.edge_order)
    expected = torch.zeros(5).index_add(0, edges[1], values)
    actual = torch.zeros(5).index_add(0, result.edge_index[1], sorted_values)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(result.metadata, prefix_csr_metadata(result.edge_index, 5)):
        assert torch.equal(a, b)


@pytest.mark.parametrize("nodes", [0, 4])
def test_empty_graph(nodes):
    result = prepare_graph(torch.empty(2, 0, dtype=torch.int64), nodes)
    assert result.edge_order.numel() == 0
    assert (
        result.metadata[1].tolist() == result.metadata[2].tolist() == [0] * (nodes + 1)
    )


def test_reference_nodes_import_without_triton_or_model_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class RejectOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'fairchem'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectOptionalPackages())
from tests.so2.reference.ffn import grid_ffn
assert callable(grid_ffn)
""",
        ],
        check=True,
    )
