"""Irrep types adapted from e3nn 0.6.0: https://github.com/e3nn/e3nn/tree/0.6.0."""

from __future__ import annotations

from typing import List, Union

import torch


def irreps_key(irreps):
    """Keep configuration keys independent of Irreps tuple subclasses."""
    if isinstance(irreps, str):
        return irreps
    return tuple((mul, ir if isinstance(ir, str) else (ir[0], ir[1])) for mul, ir in irreps)


class Irrep(tuple):
    def __new__(cls, l: Union[int, "Irrep", str, tuple], p=None):
        if p is None:
            if isinstance(l, Irrep):
                return l
            if isinstance(l, _MulIr):
                return l.ir.l
            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-1])
                    assert l >= 0
                    p = {"e": 1, "o": -1, "y": (-1) ** l}[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p = l
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be on of (-1, 1), got {p}")
        return super().__new__(cls, (l, p))

    @property
    def l(self) -> int:
        return self[0]

    @property
    def p(self) -> int:
        return self[1]

    def __repr__(self) -> str:
        p = {+1: "e", -1: "o"}[self.p]
        return f"{self.l}{p}"

    @property
    def dim(self) -> int:
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        return self.l == 0 and self.p == 1

    def __mul__(self, other):
        other = Irrep(other)
        p = self.p * other.p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield Irrep(l, p)

    def __rmul__(self, other):
        assert isinstance(other, int)
        return Irreps([(other, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul
        assert isinstance(mul, int)
        assert isinstance(ir, Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self) -> str:
        return f"{self.mul}x{self.ir}"


class Irreps(tuple):
    def __new__(cls, irreps=None) -> Union[_MulIr, "Irreps"]:
        if isinstance(irreps, Irreps):
            return super().__new__(cls, irreps)
        out = []
        if isinstance(irreps, Irrep):
            out.append(_MulIr(1, Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = Irrep(ir)
                        else:
                            mul = 1
                            ir = Irrep(mul_ir)
                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None
                if isinstance(mul_ir, str):
                    mul = 1
                    ir = Irrep(mul_ir)
                elif isinstance(mul_ir, Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = Irrep(ir)
                if not (isinstance(mul, int) and mul >= 0 and (ir is not None)):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an irrep.')
                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    @staticmethod
    def spherical_harmonics(lmax: int, p: int = -1) -> "Irreps":
        return Irreps([(1, (l, p ** l)) for l in range(lmax + 1)])

    def slices(self):
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(
        self,
        *size: int,
        normalization: str = "component",
        requires_grad: bool = False,
        dtype=None,
        device=None,
    ) -> torch.Tensor:
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]
        if normalization == "component":
            return torch.randn(
                *lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device,
            )
        elif normalization == "norm":
            x = torch.zeros(
                *lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device,
            )
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        ir = Irrep(ir)
        return sum((mul for mul, irrep in self if ir == irrep))

    def __add__(self, irreps) -> "Irreps":
        irreps = Irreps(irreps)
        return Irreps(super().__add__(irreps))

    def __mul__(self, other) -> "Irreps":
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return Irreps(super().__mul__(other))

    def __rmul__(self, other) -> "Irreps":
        return Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return Irreps(out)

    def remove_zero_multiplicities(self) -> "Irreps":
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return Irreps(out)

    @property
    def dim(self) -> int:
        return sum((mul * ir.dim for mul, ir in self))

    @property
    def num_irreps(self) -> int:
        return sum((mul for mul, _ in self))

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self) -> str:
        return "+".join((f"{mul_ir}" for mul_ir in self))


# Public modules use the installed e3nn class when available; standalone
# tensor-product configuration keeps the local implementation above.
try:
    from e3nn.o3 import Irreps as PublicIrreps
except ModuleNotFoundError as error:
    if error.name != "e3nn":
        raise
    PublicIrreps = Irreps
