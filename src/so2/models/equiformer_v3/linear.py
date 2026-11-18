"""EquiformerV3 parameter binding for SO(2) sector projection."""


import torch
from LunaEquivariance.so2 import linear
from LunaEquivariance.so2.linear.interface import project_recombined_sectors
from LunaEquivariance.so2.recombine_activate.interface import recombine_sectors


def project_raw_sectors(module, sectors: tuple[torch.Tensor, ...]):
    """Bind SO2Linear weights without recombining the two product planes."""
    lmax, mmax = module.lmax, module.mmax
    if not 0 <= mmax <= lmax or len(sectors) != mmax + 1:
        raise ValueError("SO(2) channel maps do not match the retained angular orders")
    edges = sectors[0].shape[0]
    input_channels = module.num_in_channels
    expected = [(edges, (lmax + 1) * input_channels)]
    expected.extend(
        (edges, 2, (lmax - order + 1) * input_channels) for order in range(1, mmax + 1)
    )
    actual = tuple(tuple(sector.shape) for sector in sectors)
    if actual != tuple(expected):
        raise ValueError(f"unexpected SO2Linear sectors: {actual=}")
    return linear(
        sectors,
        (module.fc_m0.weight, *(layer.fc.weight for layer in module.so2_m_linear)),
        module.fc_m0.bias, lmax=lmax, mmax=mmax,
    )


def sector_linear(module, sectors: tuple[torch.Tensor, ...], *, raw_output=False):
    """Run cuBLAS, optionally leaving recombination to the receiver reduction."""
    raw = project_raw_sectors(module, sectors)
    edges = sectors[0].shape[0]
    lmax, mmax = module.lmax, module.mmax
    if raw_output:
        if module.extra_m0_out_channels not in (None, 0):
            raise ValueError("raw attention values cannot contain scalar side outputs")
        return (raw[0].view(edges, lmax + 1, module.num_out_channels), *raw[1:])
    output, extra = recombine_sectors(
        raw,
        lmax=lmax,
        mmax=mmax,
        channels=module.num_out_channels,
        extra_channels=module.extra_m0_out_channels or 0,
    )
    if module.extra_m0_out_channels is None:
        return output
    return output, extra


def project_attention_values(module, sectors):
    # The wider real GEMM writes one value per signed coefficient. The
    # receiver reads those sectors without assembling a common edge buffer.
    if (
        (module.lmax, module.mmax, module.num_in_channels, module.num_out_channels)
        != (4, 2, 32, 128)
        or module.extra_m0_out_channels not in (None, 0)
        or len(sectors) != 3
        or torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.autograd.forward_ad._current_level >= 0
        or torch.is_autocast_enabled("cuda")
        or torch.are_deterministic_algorithms_enabled()
        or torch.get_float32_matmul_precision() != "highest"
    ):
        return sector_linear(module, sectors, raw_output=True)

    edges = sectors[0].shape[0]
    device = sectors[0].device
    if (
        not 8192 <= edges <= 32768
        or tuple(value.shape for value in sectors) != ((edges, 160), (edges, 2, 128), (edges, 2, 96))
        or device.type != "cuda"
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(device) not in ((8, 0), (9, 0))
        or not all(value.dtype == torch.float32 and value.device == device and value.is_contiguous()
                   for value in sectors)
    ):
        return sector_linear(module, sectors, raw_output=True)

    weights = (module.fc_m0.weight, *(layer.fc.weight for layer in module.so2_m_linear))
    scalar, *values = project_recombined_sectors(sectors, weights, module.fc_m0.bias)
    return scalar.reshape(edges, 5, 128), *values
