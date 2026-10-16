"""UVU convolution with sorted edges, destination aggregation and autograd."""

import torch
from LunaEquivariance import TensorProductConv, TensorProductConfig


def main():
    torch.manual_seed(7)
    dtype = torch.float32  # Use torch.float64 for double precision.
    num_nodes, num_edges, channels = 3, 4, 32
    tp = TensorProductConfig(
        f"{channels}x1o",
        "1x1o",
        f"{channels}x0e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).to(device="cuda", dtype=dtype)
    op = TensorProductConv(tp, dtype=dtype)
    src = torch.tensor([0, 2, 1, 0], device="cuda")
    dst = torch.tensor([2, 0, 2, 1], device="cuda")
    x = torch.randn(
        num_nodes, tp.irreps_in1.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    edge_attr = torch.randn(num_edges, tp.irreps_in2.dim, device="cuda", dtype=dtype)
    weight = torch.randn(
        num_edges, tp.weight_numel, device="cuda", dtype=dtype, requires_grad=True
    )

    # Sort destinations once and apply the same permutation to all edge data.
    order = dst.argsort(stable=True)
    graph = op.create_graph(src[order], dst[order], num_nodes)
    out = op(x, edge_attr[order], weight[order], graph=graph)

    # For 1o x 1o -> 0e, each message is a normalized vector dot product.
    messages = (x[src].reshape(num_edges, channels, 3) * edge_attr[:, None, :]).sum(-1)
    messages = messages * weight / (3**0.5)
    reference = x.new_zeros(num_nodes, channels).index_add(0, dst, messages)
    atol, rtol = (1e-12, 1e-11) if dtype == torch.float64 else (5e-5, 5e-4)
    torch.testing.assert_close(out, reference, atol=atol, rtol=rtol)
    out.square().mean().backward()
    print(
        "node output:",
        tuple(out.shape),
        "maximum error:",
        (out - reference).abs().max().item(),
    )


if __name__ == "__main__":
    main()
