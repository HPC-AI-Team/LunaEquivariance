"""Compose Linear, NormActivation and Dropout and backpropagate a loss."""

import torch
from LunaEquivariance import nn, o3


def main():
    linear = o3.Linear("8x0e + 4x1o", "16x0e + 4x1o", biases=True).cuda()
    activation = nn.NormActivation(linear.irreps_out, torch.sigmoid, bias=True).cuda()
    dropout = nn.Dropout(linear.irreps_out, p=0.1).cuda()
    x = torch.randn(1024, linear.irreps_in.dim, device="cuda", requires_grad=True)
    y = dropout(activation(linear(x)))
    loss = y.square().mean()
    loss.backward()
    print("output:", tuple(y.shape), "loss:", loss.item())


if __name__ == "__main__":
    main()
