"""Static layouts for external features, radial weights, and rotation rows.

Full features use degree-major rows ``l*l + l + m``. Reduced features use
sectors ``m=0, +1, -1, +2, -2, ...`` with ascending degrees in each sector.
``AngularLayout.dense_rows`` maps these rows to the full representation.
For width ``w=L+1-m``, physical SO(2) sectors have logical shape
``(E, w, C)`` at order zero and ``(E, 2*w, C)`` otherwise, with positive
rows first. Their GEMM layouts are ``(E, w*C)`` and ``(E, 2, w*C)``.
Both layouts store the same coefficients. Raw nonzero-order matrix
products instead have shape ``(E, 2, 2*w*C)`` and must be recombined
before they represent physical coefficients.

Order-degree radial weights store one ``2*C`` endpoint block per (m, l),
sharing the positive/negative order pair; ``radial_group_offsets`` gives
each block's offset in the flattened per-edge weight tensor.

Retained rotations concatenate degree blocks, with local rows ordered
``0, +1, -1, ...`` and all ``2*l+1`` columns. Their per-sector-row offsets
are ``RetainedRotationLayout.compact_row_offsets``. Cached layouts contain
only immutable Python metadata and do not allocate device tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class AngularLayout:
    """Immutable coefficient and radial layouts, built by ``get_angular_layout``.

    ``dense_rows`` maps retained order-major coefficients to full rows
    ``l*l+l+m``. ``sector_widths`` and ``sector_row_offsets`` locate each
    absolute-order group. ``radial_group_offsets`` locates the corresponding
    endpoint-channel weights. Remaining fields provide degree reductions and
    mappings used by kernels. This record contains Python values, not tensors.
    """

    lmax: int
    mmax: int
    channels: int
    full_components: int
    reduced_components: int
    reduced_degrees: tuple[int, ...]
    dense_rows: tuple[int, ...]
    radial_row_offsets: tuple[int, ...]
    radial_group_rows0: tuple[int, ...]
    radial_group_rows1: tuple[int, ...]
    radial_group_dense_rows0: tuple[int, ...]
    radial_group_dense_rows1: tuple[int, ...]
    radial_group_degrees: tuple[int, ...]
    radial_group_offsets: tuple[int, ...]
    sector_widths: tuple[int, ...]
    sector_row_offsets: tuple[int, ...]
    radial_elements: int
    degree_reduced_counts: tuple[int, ...]
    degree_reduced_offsets: tuple[int, ...]
    degree_dense_starts: tuple[int, ...]
    lmajor_degrees: tuple[int, ...]
    lmajor_to_mmajor: tuple[int, ...]

    @classmethod
    def create(cls, *, lmax: int, mmax: int, channels: int) -> "AngularLayout":
        if lmax < 0 or not 0 <= mmax <= lmax or channels <= 0:
            raise ValueError(f"invalid prefix dimensions: {lmax=} {mmax=} {channels=}")

        degrees: list[int] = list(range(lmax + 1))
        dense_rows: list[int] = [degree * degree + degree for degree in degrees]
        radial_offsets: list[int] = [degree * 2 * channels for degree in degrees]
        sector_widths = tuple(lmax - order + 1 for order in range(mmax + 1))
        sector_row_offsets = [0]
        radial_sector_offset = (lmax + 1) * 2 * channels
        row_offset = lmax + 1

        # Every m>0 sector stores [+m rows, -m rows]. The radial weights are
        # shared by the real/imaginary pair but not across m.
        for order in range(1, mmax + 1):
            width = lmax - order + 1
            sector_row_offsets.append(row_offset)
            local_offsets = [
                radial_sector_offset + local * 2 * channels for local in range(width)
            ]
            for sign in (1, -1):
                for degree in range(order, lmax + 1):
                    degrees.append(degree)
                    dense_rows.append(degree * degree + degree + sign * order)
                radial_offsets.extend(local_offsets)
            row_offset += 2 * width
            radial_sector_offset += width * 2 * channels

        group_rows0 = list(range(lmax + 1))
        group_rows1 = [-1] * (lmax + 1)
        group_degrees = list(range(lmax + 1))
        group_offsets = [degree * 2 * channels for degree in range(lmax + 1)]
        positive_start = lmax + 1
        radial_sector_offset = (lmax + 1) * 2 * channels
        for order in range(1, mmax + 1):
            width = lmax - order + 1
            negative_start = positive_start + width
            for local, degree in enumerate(range(order, lmax + 1)):
                group_rows0.append(positive_start + local)
                group_rows1.append(negative_start + local)
                group_degrees.append(degree)
                group_offsets.append(radial_sector_offset + local * 2 * channels)
            positive_start += 2 * width
            radial_sector_offset += width * 2 * channels

        degree_dense_starts = tuple(
            degree * degree + max(0, degree - mmax) for degree in range(lmax + 1)
        )
        lmajor_dense_rows = tuple(
            dense_start + local
            for degree, dense_start in enumerate(degree_dense_starts)
            for local in range(min(2 * degree + 1, 2 * mmax + 1))
        )
        dense_to_mmajor = {dense: row for row, dense in enumerate(dense_rows)}

        return cls(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            full_components=(lmax + 1) ** 2,
            reduced_components=len(degrees),
            reduced_degrees=tuple(degrees),
            dense_rows=tuple(dense_rows),
            radial_row_offsets=tuple(radial_offsets),
            radial_group_rows0=tuple(group_rows0),
            radial_group_rows1=tuple(group_rows1),
            radial_group_dense_rows0=tuple(dense_rows[row] for row in group_rows0),
            radial_group_dense_rows1=tuple(
                dense_rows[row] if row >= 0 else -1 for row in group_rows1
            ),
            radial_group_degrees=tuple(group_degrees),
            radial_group_offsets=tuple(group_offsets),
            sector_widths=sector_widths,
            sector_row_offsets=tuple(sector_row_offsets),
            radial_elements=radial_sector_offset,
            degree_reduced_counts=tuple(
                min(2 * degree + 1, 2 * mmax + 1) for degree in range(lmax + 1)
            ),
            degree_reduced_offsets=tuple(
                sum(min(2 * previous + 1, 2 * mmax + 1) for previous in range(degree))
                for degree in range(lmax + 1)
            ),
            degree_dense_starts=degree_dense_starts,
            lmajor_degrees=tuple(
                degree
                for degree in range(lmax + 1)
                for _ in range(min(2 * degree + 1, 2 * mmax + 1))
            ),
            lmajor_to_mmajor=tuple(
                dense_to_mmajor[dense] for dense in lmajor_dense_rows
            ),
        )


@lru_cache(maxsize=None)
def get_angular_layout(*, lmax: int, mmax: int, channels: int) -> AngularLayout:
    """Return cached layout metadata for ``0 <= mmax <= lmax`` and channels > 0.

    Full node coefficients have (L+1)^2 rows. Retained rows are grouped by
    0,+1,-1,..., increasing degree inside each signed order. Nonzero sectors
    contain positive then negative planes, with L+1-m degrees per plane.
    """
    return AngularLayout.create(lmax=lmax, mmax=mmax, channels=channels)


@dataclass(frozen=True)
class RetainedRotationLayout:
    """Compile-time dimensions and row mappings for retained degree blocks."""

    lmax: int
    mmax: int
    channels: int
    full_components: int
    reduced_components: int
    reduced_degrees: tuple[int, ...]
    degree_rows: tuple[tuple[int, ...], ...]
    degree_settings: tuple[tuple[int, int, int, int, int], ...]
    compact_row_offsets: tuple[int, ...]
    compact_elements: int

    @classmethod
    def create(cls, *, lmax: int, mmax: int, channels: int) -> "RetainedRotationLayout":
        sectors = get_angular_layout(lmax=lmax, mmax=mmax, channels=channels)
        reduced_degrees = sectors.reduced_degrees

        degree_rows = tuple(
            tuple(
                row for row, degree_at_row in enumerate(reduced_degrees)
                if degree_at_row == degree
            )
            for degree in range(lmax + 1)
        )
        degree_settings = tuple(
            (
                degree, len(rows), degree * degree,
                2 * degree + 1, next_power_of_two(2 * degree + 1),
            )
            for degree, rows in enumerate(degree_rows)
        )
        compact_row_offsets = [0] * len(reduced_degrees)
        compact_elements = 0
        for degree, rows in enumerate(degree_rows):
            for row in rows:
                compact_row_offsets[row] = compact_elements
                compact_elements += 2 * degree + 1
        return cls(
            lmax=lmax,
            mmax=mmax,
            channels=channels,
            full_components=sectors.full_components,
            reduced_components=sectors.reduced_components,
            reduced_degrees=reduced_degrees,
            degree_rows=degree_rows,
            degree_settings=degree_settings,
            compact_row_offsets=tuple(compact_row_offsets),
            compact_elements=compact_elements,
        )

    @property
    def degree_count(self) -> int:
        return self.lmax + 1

    @property
    def degree_row_counts(self) -> tuple[int, ...]:
        return tuple(len(rows) for rows in self.degree_rows)

    @property
    def degree_row_offsets(self) -> tuple[int, ...]:
        offset = 0
        offsets = []
        for rows in self.degree_rows:
            offsets.append(offset)
            offset += len(rows)
        return tuple(offsets)

    @property
    def degree_component_blocks(self) -> tuple[int, ...]:
        """Smallest power-of-two reduction tile for each irreducible degree."""

        return tuple(setting[4] for setting in self.degree_settings)

    @property
    def sector_widths(self) -> tuple[int, ...]:
        """Number of degree rows in m=0,1,...,M sectors."""

        return self.sectors.sector_widths

    @property
    def sector_row_offsets(self) -> tuple[int, ...]:
        """First reduced-layout row for every nonnegative m sector."""

        return self.sectors.sector_row_offsets

    @property
    def sectors(self):
        return get_angular_layout(lmax=self.lmax, mmax=self.mmax, channels=self.channels)


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


@lru_cache(maxsize=None)
def get_retained_layout(*, lmax: int, mmax: int, channels: int) -> RetainedRotationLayout:
    """Return cached immutable dimensions for ``0 <= mmax <= lmax`` and C > 0.

    ``full_components`` is (L+1)^2, ``reduced_components`` counts retained
    angular rows, and ``compact_elements`` counts stored rotation entries.
    ``degree_rows[l]`` lists degree l's positions in the order-major feature
    layout. ``compact_row_offsets`` locates their 2*l+1 entries in packed
    rotation storage. The record contains Python metadata and no tensors.
    """

    return RetainedRotationLayout.create(lmax=lmax, mmax=mmax, channels=channels)


def check_orders(lmax, mmax, values):
    if lmax < 0 or not 0 <= mmax <= lmax:
        raise ValueError("SO(2) degrees require 0 <= mmax <= lmax")
    if len(values) != mmax + 1:
        raise ValueError("one feature sector is required for each order 0..mmax")
