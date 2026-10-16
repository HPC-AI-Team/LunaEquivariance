"""Feature rotations shared by equivariance tests."""

import torch
from e3nn import o3


def rotate_features(features, irreps, rotation):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        matrix = o3.Irreps(irreps).D_from_matrix(rotation.double().cpu())
    finally:
        torch.set_default_dtype(previous_dtype)
    return features @ matrix.to(features).T
