"""PyTorch module for CSR graph softmax."""

from torch import nn

from ..nn.graph_softmax.interface import graph_softmax


class GraphSoftmax(nn.Module):
    """Reuse softmax configuration across CSR graphs and model layers."""

    def __init__(self, *, eps=1e-8, smooth_out=False):
        super().__init__()
        self.eps = eps
        self.smooth_out = smooth_out

    def forward(self, data, csr, *, env=None):
        return graph_softmax(data, csr, env=env, eps=self.eps, smooth_out=self.smooth_out)

    def extra_repr(self):
        return f"eps={self.eps}, smooth_out={self.smooth_out}"


__all__ = ["GraphSoftmax"]
