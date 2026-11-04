"""Graph construction, ordering and input ownership."""

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


def test_source_edge_order_is_generated():
    src = torch.tensor([2, 0, 1, 0])
    dst = torch.tensor([0, 1, 1, 2])
    graph = make_graph(src, dst, 3)
    assert graph.dst.tolist() == [0, 1, 1, 2]
    assert graph.src[graph.edge_ids_by_src.long()].tolist() == [0, 0, 1, 2]
    assert graph.src_rowptr.tolist() == [0, 2, 3, 4]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_generated_graph_tables_and_input_ownership(dtype):
    src = torch.tensor([2, 0, 1, 0], dtype=dtype)
    dst = torch.tensor([0, 1, 1, 2], dtype=dtype)
    graph = make_graph(src, dst, 4)
    assert graph.src_rowptr.tolist() == [0, 2, 3, 4, 4]
    assert graph.dst_rowptr.tolist() == [0, 1, 3, 4, 4]
    assert graph.edge_ids_by_src.sort().values.tolist() == [0, 1, 2, 3]
    assert graph.src[graph.edge_ids_by_src.long()].tolist() == [0, 0, 1, 2]
    assert graph.edge_ids_by_dst.tolist() == [0, 1, 2, 3]
    for rowptr, order in (
        (graph.dst_rowptr, graph.node_order),
        (graph.src_rowptr, graph.source_node_order),
    ):
        assert order.sort().values.tolist() == [0, 1, 2, 3]
        assert rowptr.diff()[order.long()].tolist() == [2, 1, 1, 0]
    assert graph.src.data_ptr() != src.data_ptr()
    assert graph.dst.data_ptr() != dst.data_ptr()
    src.zero_()
    dst.zero_()
    assert graph.src.tolist() == [2, 0, 1, 0]
    assert graph.dst.tolist() == [0, 1, 1, 2]
