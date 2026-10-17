"""Fused Linear double-backward and triple-backward CUDA kernels and bindings."""

from __future__ import annotations

import torch

from ....utils.precision import float_literal
from .utils import TASK_ALIGNMENT


def emit_weight_group(meta, targets, *, path_major=True):
    weights = meta["weight_numel"]
    shared = meta["shared_weights"]
    item_major = shared or not path_major
    task_weights = sum(
        ((spec["in_channels"] * spec["out_channels"] + TASK_ALIGNMENT - 1) // TASK_ALIGNMENT)
        * TASK_ALIGNMENT for spec in meta["generated_specs"]
    )
    lines = [("int q=local%{task_weights}; Index item=local/{task_weights};").format(task_weights=task_weights)] if item_major else []
    sources = tuple(dict.fromkeys(term[0] for terms in targets.values() for term in terms))
    seeds = tuple(dict.fromkeys(term[1] for terms in targets.values() for term in terms))
    task_offset, previous = 0, "Index(0)"
    extents = []
    for path, spec in enumerate(meta["generated_specs"]):
        cin, cout, d, wi = (
            spec[key] for key in ("in_channels", "out_channels", "components", "weight_offset")
        )
        if item_major:
            lines.append((r"""if(q>={task_offset} && q<{end}) {{
int u=(q-{task_offset})/{columns},v=(q-{task_offset})%{columns};
Index destination=item*{weights}+{offset}+q-{task_offset};""").format(
                task_offset=task_offset, end=task_offset+cin*cout, columns=cout, weights=weights, offset=wi,
            ))
        else:
            # Round tasks once per path to keep small external matrices' warps occupied.
            extent = f"((Index(num_items)*{cin*cout}+{TASK_ALIGNMENT-1})/{TASK_ALIGNMENT})*{TASK_ALIGNMENT}"
            end = f"weight_path_end_{path}"
            lines.append((r"""const Index {end_name}={previous}+({extent});
if(local>={previous} && local<{previous}+Index(num_items)*{count}) {{
Index pair=local-{previous}; Index item=pair/{count}; int q=pair%{count};
int u=q/{columns},v=q%{columns};
Index destination=item*{weights}+{offset}+q;""").format(
                end_name=end, previous=previous, extent=extent, count=cin*cout,
                columns=cout, weights=weights, offset=wi,
            ))
            previous = end
            extents.append(f"({extent})")
        begin, end = ("0", "num_items") if shared else ("item", "item+1")
        alpha = float_literal(spec["path_weight"], meta["dtype"])
        lines.append((r"""{accumulators}
for(Index n={begin};n<{end};++n) {{
#pragma unroll
for(int m=0;m<{components};++m) {{
{source_loads}
{seed_loads}
{products}
}}
}}
{stores}
}}""").format(
            accumulators="\n".join(("Element part_{target}=Element(0);").format(target=target) for target in targets),
            begin=begin, end=end, components=d,
            source_loads="\n".join(
                ("Element source_{role}={role}[n*{dimension}+{offset}+u*{components}+m];").format(role=role, dimension=meta['in_dim'], offset=spec['x_offset'], components=d)
                for role in sources
            ),
            seed_loads="\n".join(
                ("Element seed_{role}={role}[n*{dimension}+{offset}+v*{components}+m];").format(role=role, dimension=meta['out_dim'], offset=spec['out_offset'], components=d)
                for role in seeds
            ),
            products="\n".join(
                ("part_{target}=fma(source_{a},seed_{g},part_{target});").format(target=target, a=a, g=g)
                for target, terms in targets.items() for a, g in terms
            ),
            stores="\n".join(("out_{target}[destination]=part_{target}*Element({alpha});").format(target=target, alpha=alpha) for target in targets),
        ))
        task_offset += ((cin * cout + TASK_ALIGNMENT - 1) // TASK_ALIGNMENT) * TASK_ALIGNMENT
    count = (
        f"Index({task_weights})" if shared
        else "Index(0)+" + "+".join(extents) if path_major else f"Index(num_items)*{task_weights}"
    )
    return "\n".join(lines), count


def emit_bias_group(meta, targets):
    lines = ["int q=local;"]
    lines.extend(("Element value_{target}=Element(0);").format(target=target) for target in targets)
    for spec in meta["bias_specs"]:
        start, count = spec["bias_offset"], spec["channels"]
        lines.append((r"""if(q>={start} && q<{end}) {{
{sums}
}}""").format(
            start=start, end=start+count,
            sums="\n".join(
                ("for(Index n=0;n<num_items;++n) value_{target}+={g}[n*{dimension}+q-{start}+{offset}];").format(target=target, g=g, dimension=meta['out_dim'], start=start, offset=spec['out_offset'])
                for target, terms in targets.items() for (g,) in terms
            ),
        ))
    lines.extend(("out_{target}[local]=value_{target};").format(target=target) for target in targets)
    return "\n".join(lines)


def emit_feature_group(meta, targets, broadcast, *, backward):
    dimension = meta["in_dim" if backward else "out_dim"]
    source_dim = meta["out_dim" if backward else "in_dim"]
    groups = {}
    for spec in meta["generated_specs"]:
        key = (
            spec["x_offset" if backward else "out_offset"],
            spec["in_channels" if backward else "out_channels"],
            spec["components"],
        )
        groups.setdefault(key, []).append(spec)
    lines = [("Index n=local/{dimension}; int q=local%{dimension};").format(dimension=dimension)]
    lines.extend(("Element value_{target}=Element(0);").format(target=target) for target in targets)
    for (offset, channels, d), specs in groups.items():
        lines.append((r"""if(q>={offset} && q<{end}) {{
int channel=(q-{offset})/{components},component=(q-{offset})%{components};""").format(offset=offset, end=offset+channels*d, components=d))
        for spec in specs:
            source_offset = spec["out_offset" if backward else "x_offset"]
            width = spec["out_channels" if backward else "in_channels"]
            windex = (
                f"channel*{spec['out_channels']}+k"
                if backward else f"k*{spec['out_channels']}+channel"
            )
            batch = "" if meta["shared_weights"] else f"n*{meta['weight_numel']}+"
            sources = tuple(dict.fromkeys(term[0] for terms in targets.values() for term in terms))
            weights = tuple(dict.fromkeys(term[1] for terms in targets.values() for term in terms))
            alpha = float_literal(spec["path_weight"], meta["dtype"])
            lines.append((r"""{{
{accumulators}
#pragma unroll {unroll}
for(int k=0;k<{width};++k) {{
{source_loads}
{matrix_loads}
{products}
}}
{path_sums}
}}""").format(
                accumulators="\n".join(("Element part_{target}=Element(0);").format(target=target) for target in targets),
                unroll=4 if width <= 32 else 1, width=width,
                source_loads="\n".join(
                    ("Element source_{role}={role}[n*{dimension}+{offset}+k*{components}+component];").format(role=role, dimension=source_dim, offset=source_offset, components=d)
                    for role in sources
                ),
                matrix_loads="\n".join(
                    ("Element matrix_{role}={role}[{batch}{offset}+{index}];").format(role=role, batch=batch, offset=spec['weight_offset'], index=windex)
                    for role in weights
                ),
                products="\n".join(
                    ("part_{target}=fma(source_{a},matrix_{w},part_{target});").format(target=target, a=a, w=w)
                    for target, terms in targets.items() for a, w in terms
                ),
                path_sums="\n".join(("value_{target}=fma(part_{target},Element({alpha}),value_{target});").format(target=target, alpha=alpha) for target in targets),
            ))
        lines.append("}")
    if not backward:
        for spec in meta["bias_specs"]:
            start, count = spec["out_offset"], spec["channels"]
            for target, terms in broadcast.items():
                for (bias,) in terms:
                    lines.append(("if(q>={start} && q<{end}) value_{target}+={bias}[q-{start}+{offset}];").format(
                        start=start, end=start+count, target=target, bias=bias, offset=spec['bias_offset'],
                    ))
    lines.extend(("out_{target}[local]=value_{target};").format(target=target) for target in targets)
    return "\n".join(lines)


def make_dx_dw_dg_db_source(meta, program):
    """Own each input/weight/output/bias slot once in a single launch.

    Multiple targets with the same layout share live operand loads. Missing
    terms disappear from generated CUDA. The double-backward and triple-backward
    requests share this lowering, including optional bias broadcasts. Role
    boundaries and individual weight paths begin on a warp boundary so a
    warp does not serialize different reduction lengths. All work runs on
    the caller's current stream.
    """
    groups = []
    kinds = {}
    padded_weights = sum(
        ((spec["in_channels"] * spec["out_channels"] + TASK_ALIGNMENT - 1) // TASK_ALIGNMENT)
        * TASK_ALIGNMENT
        for spec in meta["generated_specs"]
    )
    dispatch_weights = (
        not meta["shared_weights"]
        and bool(program.get("weight"))
        and meta["weight_numel"] > 0
        and padded_weights >= 2 * meta["weight_numel"]
    )
    path_major_counts = []
    for kind in ("input", "weight", "output", "bias"):
        targets = dict(program.get(kind, {}))
        broadcast = program.get("broadcast", {}) if kind == "output" else {}
        targets.update({target: targets.get(target, ()) for target in broadcast})
        kinds.update({target: kind for target in targets})
        dimension = (
            meta["in_dim" if kind == "input" else "out_dim"]
            if kind in ("input", "output")
            else meta["weight_numel" if kind == "weight" else "bias_numel"]
        )
        if not targets or not dimension:
            continue
        count = (
            f"Index(num_items)*{dimension}"
            if kind in ("input", "output") or kind == "weight" and not meta["shared_weights"]
            else f"Index({dimension})"
        )
        if kind in ("input", "output"):
            body = emit_feature_group(meta, targets, broadcast, backward=kind == "input")
        elif kind == "weight":
            body, count = emit_weight_group(meta, targets, path_major=False)
            if dispatch_weights:
                body_new, count_new = emit_weight_group(meta, targets, path_major=True)
                body = f"if constexpr(use_path_major) {{\n{body_new}\n}} else {{\n{body}\n}}"
                count = f"(use_path_major?({count_new}):({count}))"
        else:
            body = emit_bias_group(meta, targets)
        if kind != "weight":
            body = f"if(local<({count})) {{\n{body}\n}}"
            count = f"((({count})+{TASK_ALIGNMENT-1})/{TASK_ALIGNMENT})*{TASK_ALIGNMENT}"
        path_major_counts.append(count_new if kind == "weight" and dispatch_weights else count)
        groups.append((kind, count, body))

    names = tuple(program["active_outputs"])
    if not program["inputs"]:
        raise ValueError("A native Linear program needs a tensor dependency")
    declarations = []
    branches = []
    previous = "Index(0)"
    for kind, count, body in groups:
        end = f"end_{kind}"
        declarations.append(f"const Index {end}={previous}+({count});")
        branches.append(
            (r"""if(index>={previous} && index<{end}) {{ Index local=index-{previous};
{body}
}}""").format(previous=previous, end=end, body=body)
        )
        previous = end

    parameters = [f"const Element* __restrict__ {role}" for role in program["inputs"]]
    parameters.extend(f"Element* __restrict__ out_{name}" for name in names)
    arguments = [f"{role}.data_ptr<Element>()" for role in program["inputs"]]
    arguments.extend(f"result_{name}.data_ptr<Element>()" for name in names)
    allocations = []
    for name in names:
        kind = kinds.get(name)
        if kind in ("input", "output"):
            dimension = meta["in_dim" if kind == "input" else "out_dim"]
            shape = f"{{num_items,int64_t({dimension})}}"
        elif kind == "weight":
            shape = (
                f"{{int64_t({meta['weight_numel']})}}"
                if meta["shared_weights"]
                else f"{{num_items,int64_t({meta['weight_numel']})}}"
            )
        elif kind == "bias":
            shape = f"{{int64_t({meta['bias_numel']})}}"
        else:
            # Requested empty irreps retain their correct zero-sized shape.
            shape = f"{{int64_t(0)}}"
        initialization = (
            "\n".join(
                f"result_{name}.narrow(-1,{offset},{count}).zero_();"
                for offset, count in meta.get("weight_gaps", ())
            ) if kind == "weight" else ""
        )
        allocations.append(
            f"auto result_{name}=torch::empty({shape},options);"
            + ("\n" + initialization if initialization else "")
        )
    returns = [
        f"pybind11::cast(result_{name})" if name in names else "pybind11::none()"
        for name in program["outputs"]
    ]

    first = program["inputs"][0]
    scalar = "double" if meta["dtype"] == torch.float64 else "float"
    phase = 'double_backward' if program['order'] == 2 else 'triple_backward'
    kernel = f"linear_{phase}_dx_dw_dg_db"
    template_prefix = "template<bool PathMajor>\n" if dispatch_weights else ""
    template_flag = "\n    constexpr bool use_path_major=PathMajor;" if dispatch_weights else ""
    dispatch_statement = (
        "\n    const bool use_path_major=("
        + "+".join(f"({count})" for count in path_major_counts)
        + "+127)/128>=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;"
        if dispatch_weights
        else ""
    )
    launch_grid = f"std::min<int64_t>(65535,({previous}+127)/128)"
    launch_arguments = f"{','.join(arguments)},num_items"
    launch_statement = (
        f"if({previous}>0) {kernel}<<<{launch_grid},128,0,stream>>>({launch_arguments});"
    )
    if dispatch_weights:
        launch_statement = (r"""if({previous}>0) {{
        if(use_path_major) {kernel}<true><<<{launch_grid},128,0,stream>>>({launch_arguments});
        else {kernel}<false><<<{launch_grid},128,0,stream>>>({launch_arguments});
    }}""").format(
            previous=previous,
            kernel=kernel,
            launch_grid=launch_grid,
            launch_arguments=launch_arguments,
        )

    return (r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
using Element={scalar};
{template_prefix}__global__ __launch_bounds__(128,4) void {kernel}(
    {parameters},int64_t num_items) {{{template_flag}
    {group_extents}
    for(Index index=Index(blockIdx.x)*128+threadIdx.x;index<{previous};index+=Index(gridDim.x)*128) {{
        {contractions}
    }}
}}
pybind11::tuple launch({launch_parameters},int64_t num_items) {{
    c10::cuda::CUDAGuard guard({first}.device());
    auto options={first}.options();{dispatch_statement}
    {allocations}
    {group_extents}
    cudaStream_t stream=at::cuda::getCurrentCUDAStream();
    {launch_statement}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return pybind11::make_tuple({results});
}}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {{m.def("launch",&launch);}}
""").format(
        scalar=scalar,
        template_prefix=template_prefix,
        kernel=kernel,
        parameters=','.join(parameters),
        template_flag=template_flag,
        group_extents="\n".join(declarations),
        previous=previous,
        contractions="\n".join(branches),
        launch_parameters=','.join((f'torch::Tensor {role}' for role in program['inputs'])),
        first=first,
        dispatch_statement=dispatch_statement,
        allocations="\n".join(allocations),
        launch_statement=launch_statement,
        results=','.join(returns),
    )
