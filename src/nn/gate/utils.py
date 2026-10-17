"""Static gate layout preparation, shared by function calls across inputs."""

from collections import OrderedDict
from functools import lru_cache

import torch

from ...o3.irreps import PublicIrreps as Irreps
from ..activation.interface import prepare_activation


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def prepare_gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
    scalars, gates, gated = map(Irreps, (irreps_scalars, irreps_gates, irreps_gated))
    if any(ir.l != 0 for _, ir in scalars + gates):
        raise ValueError("Gate scalars and gates must be scalar irreps")
    if gates.num_irreps != gated.num_irreps:
        raise ValueError("The number of gate scalars must equal the number of gated irreps")
    scalar_activation, scalar_out, _ = prepare_activation(str(scalars), act_scalars)
    gate_activation, gate_out, _ = prepare_activation(str(gates), act_gates)
    fields = tuple(irreps.simplify() for irreps in (scalars, gates, gated))
    unsorted = fields[0] + fields[1] + fields[2]
    sorted_irreps, permutation, _ = unsorted.sort()
    slices = sorted_irreps.slices()
    extract, instructions, field_index = [], [], 0
    for irreps in fields:
        indices = permutation[field_index:field_index + len(irreps)]
        instructions.append(indices)
        spans = []
        for section, index in zip(irreps.slices(), indices):
            spans.append(dict(input_offset=slices[index].start, output_offset=section.start,
                              width=section.stop - section.start))
        extract.append(tuple(spans))
        field_index += len(irreps)
    scalar_extract, gate_extract, gated_extract = extract
    scalar_paths, offset = [], 0
    for width, spec in scalar_activation:
        for span in scalar_extract:
            begin = max(offset, span['output_offset'])
            end = min(offset + width, span['output_offset'] + span['width'])
            if begin < end:
                scalar_paths.append(dict(width=end - begin,
                    input_offset=span['input_offset'] + begin - span['output_offset'],
                    output_offset=begin, activation=spec))
        offset += width
    scalar_dim = offset
    gate_specs = [spec for width, spec in gate_activation for _ in range(width)]
    # Match ElementwiseTensorProduct's multiplicity splitting in order.
    vectors, weights = list(gated.simplify()), list(gate_out.simplify())
    index = 0
    while index < len(vectors):
        vmul, vir = vectors[index]
        gmul, gir = weights[index]
        if vmul < gmul:
            weights[index] = (vmul, gir)
            weights.insert(index + 1, (gmul - vmul, gir))
        if gmul < vmul:
            vectors[index] = (gmul, vir)
            vectors.insert(index + 1, (vmul - gmul, vir))
        index += 1
    gated_paths, output_irreps = [], []
    vector_column = gate_column = output_column = 0
    for (copies, ir), (_, gate_ir) in zip(vectors, weights):
        output_irreps.append((copies, (ir.l, ir.p * gate_ir.p)))
        for copy_index in range(copies):
            vcol = vector_column + copy_index * ir.dim
            gcol = gate_column + copy_index
            vspan = next(span for span in gated_extract if span['output_offset'] <= vcol < span['output_offset'] + span['width'])
            gspan = next(span for span in gate_extract if span['output_offset'] <= gcol < span['output_offset'] + span['width'])
            path = dict(copies=1, components=ir.dim,
                input_offset=vspan['input_offset'] + vcol - vspan['output_offset'],
                gate_offset=gspan['input_offset'] + gcol - gspan['output_offset'],
                output_offset=scalar_dim + output_column + copy_index * ir.dim,
                activation=gate_specs[gcol])
            if gated_paths:
                previous = gated_paths[-1]
                n, components = previous['copies'], previous['components']
                if (path['components'] == components and path['activation'] == previous['activation']
                    and path['input_offset'] == previous['input_offset'] + n * components
                    and path['gate_offset'] == previous['gate_offset'] + n
                    and path['output_offset'] == previous['output_offset'] + n * components):
                    previous['copies'] += 1
                    continue
            gated_paths.append(path)
        vector_column += copies * ir.dim
        gate_column += copies
        output_column += copies * ir.dim
    irreps_out = scalar_out + Irreps(output_irreps)
    return dict(
        in_dim=sorted_irreps.dim, out_dim=irreps_out.dim,
        irreps_in=sorted_irreps.simplify(), irreps_out=irreps_out,
        scalar_paths=tuple(scalar_paths), gated_paths=tuple(gated_paths),
        extract_specs=tuple(extract),
        extract_identity=tuple(tuple(indices) == tuple(range(len(sorted_irreps))) for indices in instructions),
        extract_dims=tuple(irreps.dim for irreps in fields),
        scalar_activation=scalar_activation, gate_activation=gate_activation,
        tp_irreps1=Irreps(vectors), tp_irreps2=Irreps(weights),
        has_grad=bool(scalar_activation or gate_column),
    ), OrderedDict()
