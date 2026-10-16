"""Compare a LunaEquivariance operator with e3nn."""

import torch

from LunaEquivariance.utils.testing import check_compatibility


def main():
    from LunaEquivariance import o3

    module = o3.Linear("8x0e + 4x1o", "16x0e + 4x1o", biases=True).cuda()
    batch_size = 128
    x = torch.randn(batch_size, module.irreps_in.dim, device="cuda", requires_grad=True)
    target = torch.randn(batch_size, module.irreps_out.dim, device="cuda")
    report = check_compatibility(
        module, (x,),
        loss_fn=lambda output: (output - target).square().mean(),
        derivative_order=2,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
