"""PyTorch module for spherical harmonics."""

from collections import OrderedDict

import torch

from ..o3.spherical_harmonics.interface import register_ops, UNSET, SphericalHarmonicsConfig, spherical_harmonics, parse_irreps


class LunaEqSphericalHarmonics(torch.nn.Module):
    normalize: bool
    normalization: str
    _ls_list: list[int]
    _lmax: int

    def __init__(
        self,
        irreps_out,
        normalize=UNSET,
        normalization='integral',
        irreps_in=None,
        *,
        dtype=None,
    ):
        super().__init__()
        if hasattr(irreps_out, '_ls_list'):
            official = irreps_out
            self.training = official.training
            irreps_out, normalize, normalization, irreps_in = (
                tuple(official._ls_list),
                official.normalize,
                official.normalization,
                official.irreps_in,
            )

        if normalize is UNSET:
            raise TypeError('SphericalHarmonics requires normalize=True or False')
        if normalization not in ('component', 'norm', 'integral'):
            raise ValueError('normalization must be component, norm or integral')
        if dtype is None:
            dtype = torch.get_default_dtype()
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError('SphericalHarmonics requires a floating dtype')

        irreps_in, output, degrees = parse_irreps(irreps_out, irreps_in)
        self.irreps_in, self.irreps_out = irreps_in, output
        self._ls_list = list(degrees)
        self._lmax = max(degrees)
        self.normalize, self.normalization = bool(normalize), normalization
        self.compute_dtype = dtype
        self.meta = dict(
            degrees=degrees,
            normalize=self.normalize,
            normalization=normalization,
            dtype=dtype,
            out_dim=self.irreps_out.dim,
        )
        if self.irreps_out.dim > (1 << 31) - 1:
            raise OverflowError('SphericalHarmonics feature dimension must fit int32')

        # Precision bookkeeping is not model state: the reference has no buffers.
        self._dtype_anchor = torch.empty(0, dtype=dtype)
        self.kernels: OrderedDict[tuple, dict] = OrderedDict()
        self._dtype_ops = {}
        self.last_execution_backend = 'uninitialized'
        self.backward_backend = None

    def __getstate__(self):
        state = super().__getstate__().copy()
        state['kernels'] = OrderedDict()
        state['_dtype_ops'] = {}
        return state

    def _apply(self, fn, recurse=True):
        anchor = fn(self._dtype_anchor)
        dtype = anchor.dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError('SphericalHarmonics requires a floating dtype')
        result = super()._apply(fn, recurse=recurse)
        self._dtype_anchor = anchor
        if dtype != self.compute_dtype:
            self.compute_dtype = dtype
            self.meta = {**self.meta, 'dtype': dtype}
            # Saved autograd contexts keep their old specialization/cache.
            self.kernels = OrderedDict()
            self._dtype_ops = {}
        return result

    def specialize(self, dtype):
        normalize = bool(self.normalize)
        if normalize != self.meta['normalize'] or self.normalization != self.meta['normalization']:
            if self.normalization not in ('component', 'norm', 'integral'):
                raise ValueError('normalization must be component, norm or integral')
            self.meta = {**self.meta, 'normalize': normalize, 'normalization': self.normalization}
            self.kernels = OrderedDict()
            self._dtype_ops = {}
        op = self._dtype_ops.get(dtype)
        if op is None:
            op = SphericalHarmonicsConfig(
                self._ls_list, self.normalize, self.normalization, dtype,
                irreps_in=self.irreps_in,
            )
            self._dtype_ops[dtype] = op
            # Specializations snapshot configuration and dtype for saved backward
            # contexts. Only the primary dtype shares the module's cache.
            if dtype == self.compute_dtype:
                self.kernels = op.kernels
        return op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            return torch.ops.lunaeq.spherical_harmonics(
                x, self._ls_list, self.normalize, self.normalization
            )
        if torch.compiler.is_compiling() and x.is_cuda and x.dtype in (torch.float32, torch.float64) and x.shape[-1] == 3:
            name = register_ops(tuple(self._ls_list), self.normalize, self.normalization, 32, x.dtype)
            return getattr(torch.ops.lunaeq, name).default(x)
        config = self.specialize(x.dtype)
        result = spherical_harmonics(
            self._ls_list, x, self.normalize, self.normalization, config=config
        )
        self.last_execution_backend = config.last_execution_backend
        self.backward_backend = config.backward_backend
        return result


__all__ = ['LunaEqSphericalHarmonics']
