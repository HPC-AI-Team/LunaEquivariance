"""Graph ownership, ordering, prepared-table reuse and diagnostics."""

import pytest
import torch

from LunaEquivariance import make_graph


def test_destination_sorted_endpoints_are_preserved():
    src = torch.tensor([2, 0, 1])
    dst = torch.tensor([0, 1, 2])
    graph = make_graph(src, dst, 4)
    assert torch.equal(graph.dst, dst.to(graph.dst.dtype))
    assert torch.equal(graph.src, src.to(graph.src.dtype))
    assert graph.dst_rowptr.tolist() == [0, 1, 2, 3, 3]


def test_invalid_graph_includes_values():
    with pytest.raises(ValueError, match="num_nodes=2.*at least 4"):
        make_graph(torch.tensor([3]), torch.tensor([0]), 2)
    with pytest.raises(ValueError, match="nonnegative"):
        make_graph(torch.tensor([-1]), torch.tensor([0]), 2)


def test_source_edge_order_uses_sorted_input_edge_ids():
    src = torch.tensor([2, 0, 1, 0])
    dst = torch.tensor([0, 1, 1, 2])
    graph = make_graph(src, dst, 3, edge_ids_by_src=src.argsort())
    assert graph.dst.tolist() == [0, 1, 1, 2]
    assert graph.src[graph.edge_ids_by_src.long()].tolist() == [0, 0, 1, 2]
    assert graph.src_rowptr.tolist() == [0, 2, 3, 4]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_precomputed_graph_tables(dtype):
    src = torch.tensor([2, 0, 1, 0], dtype=dtype)
    dst = torch.tensor([0, 1, 1, 2], dtype=dtype)
    reference = make_graph(src, dst, 4)
    tables = {
        name: getattr(reference, name).to(dtype)
        for name in (
            "src_rowptr",
            "dst_rowptr",
            "edge_ids_by_src",
            "node_order",
            "source_node_order",
        )
    }

    graph = make_graph(src, dst, 4, **tables)
    for name, table in tables.items():
        actual = getattr(graph, name)
        torch.testing.assert_close(actual, getattr(reference, name), rtol=0, atol=0)
        assert actual.data_ptr() != table.data_ptr()
        table.add_(1)
        assert not torch.equal(actual, table)
