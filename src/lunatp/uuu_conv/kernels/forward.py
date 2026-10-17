"""UUU convolution forward CUDA."""

from __future__ import annotations
import torch
from ....utils.precision import float_literal
from ...utils.layout import build_signed_ij_classes


def emit_contraction(
    op,
    spec,
    targets,
    lines,
    *,
    with_env=False,
):
    dtype = op.compute_dtype
    scalar = "double" if dtype == torch.float64 else "float"
    dims = {"x": int(spec["d1"]), "y": int(spec["d2"]), "g": int(spec["d3"])}
    # Scale the shorter input before CG and accumulate directly into the
    # output owner, avoiding a live angular Q vector for every path.
    axis = "x" if dims["x"] <= dims["y"] else "y"
    factor = "w * e" if with_env else "w"
    input_loads = "\n".join(lines[1:])
    scaled_features = "\n".join(
        f"const {scalar} scaled_{component} = {axis}{component} * scale;"
        for component in range(dims[axis])
    )
    lines = []
    classes = build_signed_ij_classes(spec["terms"], spec_y_offset=0, y_name_base=0)
    for cls, pairs in enumerate(classes):
        additions = []
        for member in pairs.members:
            value = (
                f"scaled_{member.i_idx} * y{member.j_name}"
                if axis == "x"
                else f"x{member.i_idx} * scaled_{member.j_name}"
            )
            additions.append(("+" if member.relative_sign > 0 else "-") + f" ({value})")
        lines.append(f"""const {scalar} pair{cls} = {" ".join(additions)};""")
        for _i, _j, k, coefficient in pairs.members[0].terms:
            target = (
                targets["gg"][k]
                if not isinstance(targets["gg"], str)
                else targets["gg"]
            )
            lines.append(
                f"{target} += {float_literal(coefficient, dtype)} * pair{cls};"
            )
    return r"""{{
{input_loads}
const {scalar} scale = {weight_envelope};
{scaled_features}{pair_contractions}
}}""".format(
        input_loads=input_loads,
        scalar=scalar,
        weight_envelope=factor,
        scaled_features=scaled_features,
        pair_contractions="\n" + "\n".join(lines) if lines else "",
    )


def make_forward_node_planes(op):
    """Transpose bounded node features once; preserve all edge attributes."""
    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    slices = sorted(
        {
            (int(spec["x_offset"]), int(spec["d1"]), int(spec["channels"]))
            for spec in op.generated_specs
        }
    )
    total_tiles = sum((channels + 31) // 32 for _, _, channels in slices)
    cases, decode, base = [], [], 0
    for group, (offset, dim, channels) in enumerate(slices):
        tiles = (channels + 31) // 32
        decode += [
            f"""{("if" if group == 0 else "else if")} (tile < {base + tiles}) {{""",
            f"group = {group}; channel_base = (tile - {base}) * 32;",
            "}",
        ]
        base += tiles
        cases += [
            r"""case {group}: {{
#pragma unroll
for (int j = 0; j < 32; j += 8) {{
if (channel_base + ty + j < {channels} && tx < {dim})
planes[ty + j][tx] = __ldg(original_x + Index(node) * {in1_dim} + {offset} + (channel_base + ty + j) * {dim} + tx);
}}
__syncthreads();
#pragma unroll
for (int j = 0; j < 32; j += 8) {{
if (channel_base + tx < {channels} && ty + j < {dim})
plane_x[Index(node) * {in1_dim} + {offset} + (ty + j) * {channels} + channel_base + tx] = planes[tx][ty + j];
}}
break;
}}""".format(
                group=group,
                channels=channels,
                dim=dim,
                in1_dim=op.in1_dim,
                offset=offset,
            )
        ]
    setup = [r"""const Index task = blockIdx.x;
if (task >= Index(num_nodes) * {total_tiles}) return;
const int node = int(task / {total_tiles});
const int tile = int(task % {total_tiles});
const int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
int group = 0, channel_base = 0;
__shared__ {scalar} planes[32][33];""".format(total_tiles=total_tiles, scalar=scalar)]
    name = "lunatp_uuuconv_forward_node_planes"
    kernel = r"""__global__ __launch_bounds__(256)
void {name}(const {scalar}* original_x, {scalar}* plane_x, int num_nodes) {{
{setup}
switch (group) {{
{cases}
}}
}}""".format(
        name=name,
        scalar=scalar,
        setup="\n".join(setup + decode),
        cases="\n".join(cases),
    )
    calls = [
        r"""if (num_nodes > 0) {{
dim3 grid(Index(num_nodes) * {total_tiles});
{name}<<<grid, 256, 0, stream>>>(x, packed_forward_x.data_ptr<{scalar}>(), num_nodes);
}}
x = packed_forward_x.data_ptr<{scalar}>();""".format(total_tiles=total_tiles, name=name, scalar=scalar)
    ]
    return kernel, calls


def make_grouped_forward_partition(
    op,
    parameters,
    arguments,
    groups,
    partition,
    *,
    emit_path_contractions,
    with_env=False,
    group_axis="x",
    packed_x=False,
    share_basis=False,
):
    """Bound instruction footprint and register state to one input-irrep family."""
    scalar = "double" if op.compute_dtype == torch.float64 else "float"
    workers, threads = 2, 64
    total_tiles = sum((channels + 31) // 32 for (_, _, _, channels), _ in groups)
    max_live = max(sum(int(spec["d3"]) for _, spec in paths) for _, paths in groups)
    cases, decode, base = [], [], 0
    for group, ((x_offset, _, d1, channels), paths) in enumerate(groups):
        width = (channels + 31) // 32
        decode += [
            f"""{("if" if group == 0 else "else if")} (tile < {base + width}) {{""",
            f"group = {group}; channel_base = (tile - {base}) * 32;",
            "}",
        ]
        base += width
        lines = [f"case {group}: {{", "const int u = channel_base + lane;"]
        for path, spec in paths:
            for k in range(int(spec["d3"])):
                lines.append(f"{scalar} acc_{path}_{k} = 0.0;")
        lines.append(r"""if (u < {channels}) {{
for (Index edge = dst_rowptr[node] + neighbor_worker; edge < dst_rowptr[node + 1]; edge += {workers}) {{
const int s = __ldg(src + edge);
""".format(channels=channels, workers=workers))
        if group_axis == "x":
            for i in range(d1):
                address = (
                    f"{x_offset} + {i * channels} + u"
                    if packed_x
                    else f"{x_offset} + u * {d1} + {i}"
                )
                lines.append(
                    f"const {scalar} resident_x{i} = __ldg(x + Index(s) * {op.in1_dim} + {address});"
                )
        shared_source_input = (
            share_basis
            and group_axis == "y"
            and len(
                {(spec["x_offset"], spec["d1"], spec["channels"]) for _, spec in paths}
            )
            == 1
        )
        if shared_source_input:
            first = paths[0][1]
            for i in range(int(first["d1"])):
                address = (
                    f"""{int(first["x_offset"]) + i * channels} + u"""
                    if packed_x
                    else f"""{int(first["x_offset"])} + u * {int(first["d1"])} + {i}"""
                )
                lines.append(
                    f"const {scalar} resident_shared_x{i} = __ldg(x + Index(s) * {op.in1_dim} + {address});"
                )
        y_slices = (
            {(x_offset, d1)}
            if group_axis == "y"
            else {(int(spec["y_offset"]), int(spec["d2"])) for _, spec in paths}
        )
        for y_offset, d2 in sorted(y_slices):
            for j in range(d2):
                lines.append(
                    f"const {scalar} resident_y{y_offset}_{j} = __ldg(y + Index(edge) * {op.in2_dim} + {y_offset} + u * {d2} + {j});"
                )
        shared_basis = None
        if shared_source_input:
            if len(paths) < 2:
                shared_basis = None
            else:
                basis_pool, basis_products = {}, 0
                for path, spec in paths:
                    for pairs in build_signed_ij_classes(
                        spec["terms"], spec_y_offset=0, y_name_base=0
                    ):
                        key = tuple(
                            (member.i_idx, member.j_name, member.relative_sign)
                            for member in pairs.members
                        )
                        basis_products += len(key)
                        basis_pool.setdefault(key, []).extend(
                            (path, int(k), coefficient)
                            for _, _, k, coefficient in pairs.members[0].terms
                        )
                if (
                    basis_products == 0
                    or sum(len(key) for key in basis_pool) > 0.9 * basis_products
                ):
                    shared_basis = None
                else:
                    basis_scalar = (
                        "double" if op.compute_dtype == torch.float64 else "float"
                    )
                    basis_lines = ["{"]
                    for path, spec in paths:
                        basis_value = (
                            f"""__ldg(weight + Index(edge) * {op.weight_numel} + {int(spec["weight_offset"])} + u)"""
                            if spec["has_weight"]
                            else "1.0"
                        )
                        if with_env:
                            basis_value = f"({basis_value} * __ldg(env + edge))"
                        basis_lines.append(
                            f"const {basis_scalar} basis_weight_{path} = {basis_value};"
                        )
                        for k in range(int(spec["d3"])):
                            basis_lines.append(
                                f"{basis_scalar} basis_q_{path}_{k} = 0.0;"
                            )
                    for cls, (members, basis_consumers) in enumerate(
                        basis_pool.items()
                    ):
                        additions = [
                            f"""{("+" if sign > 0 else "-")} (resident_shared_x{i} * resident_y{int(paths[0][1]["y_offset"])}_{j})"""
                            for i, j, sign in members
                        ]
                        basis_lines.append(
                            f"""const {basis_scalar} basis_pair_{cls} = {" ".join(additions)};"""
                        )
                        for path, k, coefficient in basis_consumers:
                            basis_lines.append(
                                f"basis_q_{path}_{k} += {float_literal(coefficient, op.compute_dtype)} * basis_pair_{cls};"
                            )
                    for path, spec in paths:
                        for k in range(int(spec["d3"])):
                            basis_lines.append(
                                f"acc_{path}_{k} += basis_weight_{path} * basis_q_{path}_{k};"
                            )
                    basis_lines.append("}")
                    shared_basis = "\n".join(basis_lines)
        if shared_basis is not None:
            lines.append(shared_basis)
        else:
            for path, spec in paths:
                resident = (
                    {f"x{i}": f"resident_x{i}" for i in range(d1)}
                    if group_axis == "x"
                    else {}
                )
                if shared_source_input:
                    resident.update(
                        {
                            f"x{i}": f"resident_shared_x{i}"
                            for i in range(int(spec["d1"]))
                        }
                    )
                elif packed_x and group_axis == "y":
                    resident.update(
                        {
                            f"x{i}": f"""__ldg(x + Index(s) * {op.in1_dim} + {int(spec["x_offset"]) + i * int(spec["channels"])} + u)"""
                            for i in range(int(spec["d1"]))
                        }
                    )
                resident.update(
                    {
                        f"y{j}": f"""resident_y{int(spec["y_offset"])}_{j}"""
                        for j in range(int(spec["d2"]))
                    }
                )
                lines.append(
                    emit_path_contractions(
                        op,
                        spec,
                        path,
                        0,
                        {"gg": [f"acc_{path}_{k}" for k in range(int(spec["d3"]))]},
                        with_env=with_env,
                        resident_values=resident,
                    )
                )
        lines += ["}", "}"]
        slot = 0
        for path, spec in paths:
            for k in range(int(spec["d3"])):
                lines.append(
                    f"neighbor_sums[threadIdx.x * {max_live} + {slot}] = acc_{path}_{k};"
                )
                slot += 1
        lines += ["__syncthreads();", f"if (neighbor_worker == 0 && u < {channels}) {{"]
        slot = 0
        for path, spec in paths:
            d3, offset = int(spec["d3"]), int(spec["out_offset"])
            for k in range(d3):
                lines += [
                    r"""{scalar} value_{slot} = 0.0;
#pragma unroll
for (int worker = 0; worker < {workers}; ++worker) value_{slot} += neighbor_sums[(worker * 32 + lane) * {max_live} + {slot}];
out_g[Index(node) * {out_dim} + {offset} + u * {d3} + {k}] = value_{slot};""".format(
                        scalar=scalar,
                        slot=slot,
                        workers=workers,
                        max_live=max_live,
                        out_dim=op.out_dim,
                        offset=offset,
                        d3=d3,
                        k=k,
                    )
                ]
                slot += 1
        lines += ["}", "break;", "}"]
        cases.append("\n".join(lines))

    setup = [
        r"""const Index task = blockIdx.x;
if (task >= Index(num_nodes) * {total_tiles}) return;
const int lane = threadIdx.x & 31;
const int item = int(task / {total_tiles});
const int tile = int(task % {total_tiles});
const int node = node_order ? __ldg(node_order + item) : item;
int group = 0, channel_base = 0;
const int neighbor_worker = threadIdx.x >> 5;
__shared__ {scalar} neighbor_sums[{shared_width}];""".format(
            total_tiles=total_tiles,
            scalar=scalar,
            shared_width=threads * max_live,
        )
    ]
    name = "lunatp_uuuconv_fwd_input_groups" + (
        f"_{partition}" if partition is not None else ""
    )
    kernel = r"""__global__ __launch_bounds__({threads}, 2)
void {name}({parameters}) {{
{setup}
switch (group) {{
{cases}
}}
}}""".format(
        threads=threads,
        name=name,
        parameters=parameters,
        setup="\n".join(setup + decode),
        cases="\n".join(cases),
    )
    launches = [
        r"""if (num_nodes > 0) {{
dim3 grid(Index(num_nodes) * {total_tiles});
{name}<<<grid, {threads}, 0, stream>>>({arguments});
}}""".format(
            total_tiles=total_tiles,
            name=name,
            threads=threads,
            arguments=arguments,
        )
    ]
    return kernel, launches
