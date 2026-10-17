"""Build both index widths from one CUDA implementation.

Only counters that grow with the workload use Index. Channel indices, node IDs,
thread indices and tile-local arithmetic retain their original bounded types.
"""

import re

LAYOUT_SOURCE = r"""
struct ForwardLayout {
    int rank;
    int compact;
    int64_t feature_stride;
    int64_t batch_step;
    int64_t sizes[63];
    int64_t strides[63];
};
__device__ __forceinline__ int64_t forward_row_offset(
    Index row, const ForwardLayout& layout) {
    if(layout.compact) return int64_t(row)*layout.batch_step;
    int64_t rest=row,offset=0;
    for(int axis=layout.rank-1;axis>=0;--axis) {
        offset+=(rest%layout.sizes[axis])*layout.strides[axis];
        rest/=layout.sizes[axis];
    }
    return offset;
}
"""

LAYOUT_HOST = r"""
    TORCH_CHECK(x.dim()>=1 && x.dim()<=64,"FWD layout rank must be 1..64");
    ForwardLayout x_layout{};
    x_layout.rank=int(x.dim()-1);
    x_layout.feature_stride=x.stride(-1);
    x_layout.compact=1;
    int64_t span=1;
    bool found=false;
    for(int axis=x_layout.rank-1;axis>=0;--axis) {
        x_layout.sizes[axis]=x.size(axis);
        x_layout.strides[axis]=x.stride(axis);
        if(x.size(axis)<=1) continue;
        if(!found) {x_layout.batch_step=x.stride(axis);found=true;}
        else if(x.stride(axis)!=x_layout.batch_step*span) x_layout.compact=0;
        span*=x.size(axis);
    }
"""


def name_cuda_kernels(
    source: str, operator: str, *, index_bits: int | None = None, precision: str | None = None,
) -> str:
    """Give generated kernel symbols readable names, without changing bindings."""
    context = operator
    operator = re.sub(r"^(?:lunaeq|lunatp)_", "", operator)
    operator = re.sub(r"(uuu|uvu|uvw)_conv", r"\1conv", operator)
    operators = (
        "spherical_harmonics", "norm_activation", "varlen_attention_conv",
        "attention_conv", "varlen_attention", "graph_softmax",
        "equ_norm", "uuuconv", "uvuconv", "uvwconv", "activation", "dropout",
        "linear", "norm", "gate", "uuu", "uvu", "uvw", "attention",
    )
    operator = next((name for name in operators if operator.startswith(name)), operator)
    if operator in ("attention", "attention_conv"):
        operator = "varlen_" + operator
    if precision is None:
        double_precision = re.search(
            r"(?:using\s+(?:Element|Acc)\s*=\s*double|data_ptr<double>)", source,
        )
        precision = "fp64" if double_precision else "fp32"
        if operator.startswith("varlen_attention"):
            storage = re.search(r"using\s+Storage\s*=\s*at::(Half|BFloat16)", source)
            if storage:
                precision = "fp16" if storage.group(1) == "Half" else "bf16"
            elif operator == "varlen_attention_conv":
                lowp = re.search(r"#define\s+LOWP_BF16\s+([01])\b", source)
                if lowp:
                    precision = "bf16" if lowp.group(1) == "1" else "fp16"
    kernels = re.findall(
        r"__global__[^{};]*?\bvoid\s+(?:__launch_bounds__[^{};]*?\)\s*)?([A-Za-z]\w*)\s*\(",
        source,
    )
    names = {}
    for kernel in dict.fromkeys(kernels):
        name = re.sub(r"^(?:lunaeq|lunatp)_", "", kernel)
        name = re.sub(r"(uuu|uvu|uvw)_conv", r"\1conv", name)
        name = re.sub(r"(?:^|_)(?:fp32|fp64|i32|i64)(?=_|$)", "", name)
        name = re.sub(r"(?:^|_)kernel(?=_|$)", "", name).strip("_")
        for prefix in (operator, operator.replace("varlen_", "")):
            if name.startswith(prefix + "_"):
                name = name[len(prefix) + 1:]
                break
        phase = re.search(
            r"(?:^|_)(triple_backward|double_backward|triple_bwd|double_bwd|"
            r"triplebwd|doublebwd|backward|forward|fwd|bwd)(?=_|$)", name,
        )
        if phase:
            stage = phase.group(1).replace("backward", "bwd").replace("forward", "fwd")
            stage = stage.replace("triple_bwd", "triplebwd").replace("double_bwd", "doublebwd")
            name = (name[:phase.start()] + "_" + name[phase.end():]).strip("_")
        else:
            # Helpers inherit the extension's derivative stage.
            if "triple" in context:
                stage = "triplebwd"
            elif "double" in context:
                stage = "doublebwd"
            else:
                stage = "bwd" if "backward" in context or "_bwd" in context else "fwd"
        suffix = (f"_{precision}" if precision else "") + (f"_i{index_bits}" if index_bits is not None else "")
        names[kernel] = f"lunaeq_{operator}_{stage}" + (f"_{name}" if name else "") + suffix
    if len(set(names.values())) != len(names):
        raise ValueError("Kernel names collide after normalization")
    # Leave pybind names, diagnostics, include paths and comments untouched.
    tokens = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\b\w+\b'
    return re.sub(tokens, lambda match: names.get(match.group(), match.group()), source, flags=re.S)


def make_index_variants_source(source: str, name: str) -> str:
    """Expose two submodules in one extension, with independently typed bindings."""
    includes = re.findall(r"^\s*#include[^\n]*", source, flags=re.M)
    body = re.sub(r"^\s*#include[^\n]*", "", source, flags=re.M)
    body, count = re.subn(
        r"PYBIND11_MODULE\(TORCH_EXTENSION_NAME,\s*(\w+)\)",
        r"void bind(pybind11::module_ &\1)",
        body,
    )
    if count != 1:
        raise ValueError("Expected one binding block in generated CUDA source")

    fast = (
        "using Index = int; using EdgeIndex = int32_t; using WeightIndex = uint32_t;\n"
        + name_cuda_kernels(body, name, index_bits=32)
    )

    wide_body = body
    wide_body = re.sub(
        r"(__global__[^{};]*?\bvoid\s+\w+\s*\()",
        r"\1int64_t base_x, int64_t base_y, int64_t base_z, int64_t logical_x, int64_t logical_y, int64_t logical_z, ",
        wide_body,
    )
    for axis in "xyz":
        wide_body = wide_body.replace(
            f"blockIdx.{axis}", f"(base_{axis} + int64_t(blockIdx.{axis}))"
        )
        wide_body = wide_body.replace(f"gridDim.{axis}", f"logical_{axis}")
    wide_body = re.sub(r"\bdim3 (\w*grid)\b", r"LogicalGrid \1", wide_body)
    wide_body = wide_body.replace("static_cast<unsigned int>", "static_cast<int64_t>")

    pattern = re.compile(r"(\b\w+(?:<[^;{}\n]*?>)?)<<<(.*?)>>>\(", re.S)
    for match in reversed(list(pattern.finditer(wide_body))):
        kernel, config = match.groups()
        parts = []
        start = depth = 0
        for i, char in enumerate(config):
            depth += (char == "(") - (char == ")")
            if char == "," and depth == 0:
                parts.append(config[start:i].strip())
                start = i + 1
        parts.append(config[start:].strip())
        if len(parts) != 4:
            raise ValueError(
                f"CUDA launch must specify grid, block, shared memory and stream: {config}"
            )
        grid, block, shared, stream = parts
        grid = re.sub(r"\bdim3\(", "LogicalGrid(", grid)
        depth, pos = 1, match.end()
        while depth:
            depth += (wide_body[pos] == "(") - (wide_body[pos] == ")")
            pos += 1
        args = wide_body[match.end() : pos - 1]
        if wide_body[pos] != ";":
            raise ValueError("Expected a complete CUDA launch statement")

        # Keep a launch on one line so it also fits a CUDA preprocessor macro.
        launch = (
            "{ const LogicalGrid batch_extent = " + grid + "; "
            "for (int64_t batch_z = 0; batch_z < batch_extent.z;) { "
            "const int64_t batch_nz = std::min<int64_t>(batch_extent.z-batch_z, 65535); "
            "for (int64_t batch_y = 0; batch_y < batch_extent.y;) { "
            "const int64_t batch_ny = std::min<int64_t>(batch_extent.y-batch_y, 65535); "
            "for (int64_t batch_x = 0; batch_x < batch_extent.x;) { "
            "const int64_t batch_nx = std::min<int64_t>(batch_extent.x-batch_x, 2147483647); "
            + kernel
            + "<<<dim3(batch_nx,batch_ny,batch_nz), "
            + block
            + ", "
            + shared
            + ", "
            + stream
            + ">>>(batch_x,batch_y,batch_z,batch_extent.x,batch_extent.y,batch_extent.z, "
            + args
            + "); batch_x += batch_nx; } batch_y += batch_ny; } batch_z += batch_nz; } }"
        )
        wide_body = wide_body[: match.start()] + launch + wide_body[pos + 1 :]

    wide_body = (
        """struct LogicalGrid {
    int64_t x, y, z;
    LogicalGrid(int64_t x_, int64_t y_=1, int64_t z_=1): x(x_), y(y_), z(z_) {}
};
"""
        + wide_body
    )
    wide = (
        "using Index = int64_t; using EdgeIndex = int64_t; using WeightIndex = int64_t;\n"
        + name_cuda_kernels(wide_body, name, index_bits=64)
    )
    return (
        "#include <cstdint>\n#include <algorithm>\n"
        + "\n".join(dict.fromkeys(includes))
        + "\nnamespace compact {\n"
        + fast
        + "\n}\n"
        + "\nnamespace wide {\n"
        + wide
        + "\n}\n"
        + """PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    auto fast = m.def_submodule("i32");
    auto wide = m.def_submodule("i64");
    compact::bind(fast);
    wide::bind(wide);
}
"""
    )
