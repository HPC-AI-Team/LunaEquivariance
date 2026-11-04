"""Sparse Clebsch–Gordan paths and a shared polynomial basis."""

from collections import Counter, defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import factorial, sqrt
from typing import NamedTuple

import numpy as np
import torch
from e3nn import o3


class Input(NamedTuple):
    tensor: int
    start: int
    stop: int


class Coupling(NamedTuple):
    op: tuple
    args: tuple


def path_tree(path):
    if isinstance(path, Input):
        return dict(input=path.tensor, start=path.start, stop=path.stop)
    return dict(op=tuple(str(irrep) for irrep in path.op),
                args=tuple(path_tree(child) for child in path.args))


def reverse_work(terms):
    work = 0
    for component in {term[0] for term in terms}:
        edges, internal = set(), set()
        for output, inputs, _ in terms:
            if output == component:
                edges.update(inputs[:length] for length in range(1, len(inputs) + 1))
                internal.update(inputs[:length] for length in range(1, len(inputs)))
        work += 2 * len(edges) + len(internal) + 2
    return work


def symmetric_multiplicities(irreps, correlation):
    bound = max(irrep.l for irrep in irreps) * correlation
    counts = np.zeros((correlation + 1, 2 * bound + 1, 2), dtype=np.int64)
    counts[0, bound, 0] = 1
    for irrep in irreps:
        parity = int(irrep.p == -1)
        for magnetic in range(-irrep.l, irrep.l + 1):
            for degree in range(1, correlation + 1):
                for total in range(-bound, bound + 1):
                    previous = total - magnetic
                    if -bound <= previous <= bound:
                        for sign in range(2):
                            counts[degree, total + bound, sign] += counts[degree - 1, previous + bound, sign ^ parity]
    return {(degree, angular, parity): int(counts[degree, bound + angular, int(parity == -1)]
                                         - counts[degree, bound + angular + 1, int(parity == -1)]
                                         if angular < bound else counts[degree, bound + angular, int(parity == -1)])
            for degree in range(1, correlation + 1)
            for angular in range(bound + 1) for parity in (1, -1)}


def rational_coefficients(values, max_error=None):
    result = np.empty_like(values)
    for index in np.ndindex(values.shape):
        value = float(values[index])
        square = Fraction(value * value).limit_denominator(1 << 40)
        result[index] = np.copysign(sqrt(square.numerator / square.denominator), value)
    if max_error is not None:
        # Keep coefficients that cannot be rounded within the error budget.
        keep = np.abs(values - result) > max_error
        result[keep] = values[keep]
    return result


@lru_cache(maxsize=128)
def build_couplings(input_irreps, output_irreps, correlation, dtype_name, use_reduced_cg=False):
    if use_reduced_cg or correlation > 3:
        result = reduced_couplings(input_irreps, output_irreps, correlation, dtype_name)
        if not use_reduced_cg:
            prefix = dict(result["cg_dag"], inputs=tuple((o3.Irrep(irrep).l, o3.Irrep(irrep).p) for irrep in input_irreps),
                          outputs=tuple((o3.Irrep(irrep).l, o3.Irrep(irrep).p) for irrep in output_irreps),
                          correlation=correlation)
            projection = original_projection(prefix)
            result.update(num_paths=projection["num_paths"], path_counts=projection["path_counts"])
            result["cg_dag"]["weight_members"] = projection["weight_members"]
        return result
    return build_polynomial_couplings(input_irreps, output_irreps, correlation, dtype_name)


@lru_cache(maxsize=128)
def build_polynomial_couplings(input_irreps, output_irreps, correlation, dtype_name):
    """Contract the original ordered CG paths in the requested construction dtype."""
    inputs = tuple(o3.Irrep(irrep) for irrep in input_irreps)
    outputs = tuple(o3.Irrep(irrep) for irrep in output_irreps)
    dtype = getattr(torch, dtype_name)
    input_dim = sum(irrep.dim for irrep in inputs)

    @lru_cache(None)
    def coefficients(op, precision):
        left, right, output = op
        return o3.wigner_3j(output.l, left.l, right.l, dtype=precision, device="cpu") * output.dim ** .5

    @lru_cache(None)
    def polynomial(path, precision):
        if isinstance(path, Input):
            return tuple({(component,): 1.0} for component in range(path.start, path.stop))
        left, right = path.args
        previous = polynomial(left, precision)
        columns = sorted({indices for row in previous for indices in row})
        matrix = torch.tensor([[row.get(indices, 0.0) for indices in columns] for row in previous],
                              dtype=precision, device="cpu")
        tensor = torch.einsum("jk,ijl->ikl", matrix, coefficients(path.op, precision))
        result = tuple({} for _ in range(path.op[-1].dim))
        for component, column, right_component in tensor.nonzero().tolist():
            result[component][(*columns[column], right.start + right_component)] = tensor[component, column, right_component].item()
        return result

    first, offset = [], 0
    for irrep in inputs:
        first.append((irrep, Input(0, offset, offset + irrep.dim)))
        offset += irrep.dim
    def extend_paths(previous_paths, degree, natural, requested=None):
        paths = []
        for left, previous in previous_paths:
            offset = 0
            for right in inputs:
                for output in left * right:
                    if natural and (output.l >= 12 or output.p != (-1) ** output.l):
                        continue
                    if requested is not None and output not in requested:
                        continue
                    paths.append((output, Coupling((left, right, output),
                                                   (previous, Input(degree - 1, offset, offset + right.dim)))))
                offset += right.dim
        return tuple(sorted(paths, key=lambda item: item[0]))

    @lru_cache(None)
    def intermediate_paths(degree, natural):
        if degree == 1:
            return tuple(first)
        return extend_paths(intermediate_paths(degree - 1, natural), degree, natural)

    by_degree = {1: first}
    for degree in range(2, correlation + 1):
        natural = degree == 4
        by_degree[degree] = extend_paths(intermediate_paths(degree - 1, natural), degree, natural, outputs)

    rows, ideal_rows, paths, path_counts = [], [], [], []
    component_offset = 0
    for irrep in outputs:
        for degree in range(correlation, 0, -1):
            selected = [path for output, path in by_degree[degree] if output == irrep]
            path_counts.append(max(1, len(selected)))
            for path in selected or [None]:
                polynomials = []
                for precision in (dtype, torch.float64):
                    terms = defaultdict(float)
                    if path is not None:
                        for component, values in enumerate(polynomial(path, precision)):
                            for indices, value in values.items():
                                terms[component_offset + component, tuple(sorted(indices))] += value
                    polynomials.append({key: value for key, value in terms.items() if value != 0})
                rows.append(polynomials[0])
                ideal_rows.append(polynomials[1])
                paths.append(dict(tree=path, degree=degree, output_component=component_offset,
                                  output_components=irrep.dim, output=str(irrep)))
        component_offset += irrep.dim
    terms = tuple((component, path, indices, value)
                  for path, row in enumerate(rows) for (component, indices), value in sorted(row.items()))
    result = dict(input_dim=input_dim, num_paths=len(rows), correlation=correlation,
                  path_counts=tuple(path_counts), terms=tuple(sorted(terms)))
    columns = sorted({key for row in rows for key in row})
    original = np.array([[row.get(key, 0.0) for key in columns] for row in rows])
    ideal = np.array([[row.get(key, 0.0) for key in columns] for row in ideal_rows])
    multiplicities = symmetric_multiplicities(inputs, correlation)
    rank = sum(multiplicities.get((degree, irrep.l, irrep.p), 0)
               for irrep in outputs for degree in range(1, correlation + 1))
    if rank == 0:
        return result

    @lru_cache(None)
    def pair_info(path):
        tensor = coefficients(path.op, dtype)
        left, right = path.args
        values = defaultdict(float)
        for component, a, b in tensor.nonzero().tolist():
            values[component, tuple(sorted((left.start + a, right.start + b)))] += tensor[component, a, b].item()
        pair_terms = tuple((*key, value) for key, value in sorted(values.items()) if value != 0)
        sign = -1 if pair_terms and pair_terms[0][-1] < 0 else 1
        signature = tuple((component, indices, value * sign) for component, indices, value in pair_terms)
        return tensor, pair_terms, signature, sign

    pairs, outer_terms = [], []
    for info in paths:
        path, degree, output_offset = info["tree"], info["degree"], info["output_component"]
        outer = defaultdict(float)
        if correlation > 3:
            signature = None
            outer.update(rows[len(pairs)])
        elif path is None:
            signature = None
        elif degree == 1:
            signature = None
            for component in range(info["output_components"]):
                outer[output_offset + component, (path.start + component,)] = 1.0
        else:
            _, _, signature, sign = pair_info(path.args[0] if degree == 3 else path)
            available = {component for component, _, _ in signature}
            if degree == 2:
                for component in available:
                    outer[output_offset + component, (input_dim + component,)] = float(sign)
            else:
                tensor = coefficients(path.op, dtype)
                for component, inner, right_component in tensor.nonzero().tolist():
                    if inner in available:
                        indices = (path.args[1].start + right_component, input_dim + inner)
                        outer[output_offset + component, indices] += sign * tensor[component, inner, right_component].item()
        pairs.append(signature)
        outer_terms.append(tuple((*key, value) for key, value in sorted(outer.items()) if value != 0))

    norm = np.linalg.norm(ideal, axis=1)
    normalized = np.divide(ideal, norm[:, None], out=np.zeros_like(ideal), where=norm[:, None] != 0)
    inner_costs = {pair: reverse_work(pair) for pair in pairs if pair}
    outer_costs = [reverse_work(values) for values in outer_terms]
    orthogonal, pivots, selected_pairs = [], [], set()
    for _ in range(rank):
        q = np.array(orthogonal)
        candidates = []
        for path, vector in enumerate(normalized):
            if path in pivots or norm[path] < 1e-12:
                continue
            residual = vector.copy()
            if len(q):
                for _ in range(2):
                    residual -= q.T @ (q @ residual)
            remaining = np.linalg.norm(residual)
            if remaining <= 1e-10:
                continue
            pair = pairs[path]
            cost = outer_costs[path] + (inner_costs.get(pair, 0) if pair not in selected_pairs else 0)
            candidates.append((cost, path, remaining, residual))
        if not candidates:
            if correlation > 3:
                break
            raise RuntimeError("could not construct the symmetric coupling basis")
        _, path, remaining, residual = min(candidates, key=lambda item: item[:2])
        pivots.append(path)
        orthogonal.append(residual / remaining)
        selected_pairs.add(pairs[path])

    if not pivots:
        return result
    mapping = np.linalg.lstsq(original[pivots].T, original.T, rcond=None)[0].T
    mapping = rational_coefficients(mapping)
    residual = np.max(np.abs(original - mapping @ original[pivots]), initial=0)
    tolerance = 32 * torch.finfo(dtype).eps * max(1.0, float(np.max(np.abs(original), initial=0)))
    if residual > tolerance:
        return result

    rank = len(pivots)
    weight_paths = tuple(tuple((int(path), float(mapping[path, basis]))
                                for path in np.flatnonzero(mapping[:, basis])) for basis in range(rank))
    if correlation > 3:
        result["polynomials"] = dict(
            terms=tuple((component, basis, indices, value)
                        for basis, path in enumerate(pivots)
                        for (component, indices), value in sorted(rows[path].items())),
            compute_weight_paths=weight_paths,
        )
        return result

    selected_paths, local_blocks, consumers = [], {}, defaultdict(list)
    for basis, original_path in enumerate(pivots):
        info = paths[original_path]
        path, degree = info["tree"], info["degree"]
        selected_paths.append(dict(path=original_path, basis_path=basis, **{
            key: value for key, value in info.items() if key != "tree"}, tree=path_tree(path)))
        if degree == 1:
            continue
        pair = path.args[0] if degree == 3 else path
        tensor, pair_terms, signature, sign = pair_info(pair)
        key = repr(pair)
        consumers[signature].append((basis, original_path, key, sign))
        local_blocks[key] = dict(tree=path_tree(pair), tensor=tensor, canonical_terms=pair_terms, sign=sign)
        if degree == 3:
            local_blocks[repr(path)] = dict(tree=path_tree(path), tensor=coefficients(path.op, dtype))
    result["binary"] = dict(
        rank=rank, original_num_paths=len(paths), input_dim=input_dim,
        output_components=component_offset, selected_paths=tuple(selected_paths),
        compute_weight_paths=weight_paths,
        local_cg_blocks=local_blocks,
        pair_groups=tuple(dict(terms=signature, consumers=tuple(members))
                          for signature, members in consumers.items()),
    )
    return result


def polynomial_dag(couplings):
    """Represent already contracted coefficients as a shared scalar-product CG DAG."""
    input_dim = couplings["input_dim"]
    nodes, prefixes = [], {}
    for component in range(input_dim):
        prefixes[component,] = len(nodes)
        nodes.append(dict(kind="input", input_start=component, input_width=1,
                          irrep=(0, 1), irrep_dim=1, degree=1,
                          composition=tuple(int(index == component) for index in range(input_dim))))
    members = defaultdict(list)
    for output, path, indices, coefficient in couplings["terms"]:
        indices = tuple(indices)
        for degree in range(2, len(indices) + 1):
            prefix = indices[:degree]
            if prefix not in prefixes:
                prefixes[prefix] = len(nodes)
                nodes.append(dict(kind="coupling", left_id=prefixes[prefix[:-1]],
                                  right_id=prefixes[prefix[-1],], cg_id=0,
                                  irrep=(0, 1), irrep_dim=1, degree=degree,
                                  composition=tuple(prefix.count(index) for index in range(input_dim))))
        members[output, indices].append((path, coefficient))
    roots, weight_members = [], []
    for basis, ((output, indices), paths) in enumerate(sorted(members.items())):
        roots.append(dict(node_id=prefixes[indices], degree=len(indices),
                          composition=tuple(indices.count(index) for index in range(input_dim)),
                          irrep=(0, 1), output_block=output, output_component_offset=output,
                          selected_basis_id=basis))
        weight_members.append(tuple(paths))
    local_cg = (dict(shape=(1, 1, 1), output_offsets=(0, 1), left_components=(0,),
                     right_components=(0,), coefficients=(1.,)),) if len(nodes) > input_dim else ()
    return dict(nodes=tuple(nodes), roots=tuple(roots), local_cg=local_cg,
                weight_members=tuple(weight_members))


@lru_cache(None)
def symmetric_character(angular, degree):
    bound = angular * degree
    counts = [defaultdict(int) for _ in range(degree + 1)]
    counts[0][0] = 1
    for magnetic in range(-angular, angular + 1):
        for order in range(1, degree + 1):
            for total, value in tuple(counts[order - 1].items()):
                counts[order][total + magnetic] += value
    return tuple(sorted(counts[degree].items()))


@lru_cache(None)
def composition_multiplicities(inputs, composition):
    character = {0: 1}
    parity = 1
    for (angular, sign), degree in zip(inputs, composition):
        parity *= sign ** degree
        result = defaultdict(int)
        for left, a in character.items():
            for right, b in symmetric_character(angular, degree):
                result[left + right] += a * b
        character = result
    return {(angular, parity): character.get(angular, 0) - character.get(angular + 1, 0)
            for angular in range(max(character) + 1)
            if character.get(angular, 0) != character.get(angular + 1, 0)}


def compositions(degree, width):
    if width == 1:
        yield (degree,)
    else:
        for first in range(degree + 1):
            for tail in compositions(degree - first, width - 1):
                yield (first, *tail)


@lru_cache(None)
def monomials(inputs, composition):
    offset = 0
    blocks = []
    for (angular, _), degree in zip(inputs, composition):
        width = 2 * angular + 1
        blocks.append(tuple(combinations_with_replacement(range(offset, offset + width), degree)))
        offset += width
    columns = tuple(tuple(index for piece in pieces for index in piece) for pieces in product(*blocks))
    return columns


@lru_cache(None)
def cg_tensor(left, right, output, dtype_name):
    dtype = getattr(torch, dtype_name)
    return o3.wigner_3j(output[0], left[0], right[0], dtype=dtype) * sqrt(2 * output[0] + 1)


def coupled_polynomial(previous, right_start, coefficients):
    rows = [defaultdict(float) for _ in range(coefficients.shape[0])]
    for output, left, right in coefficients.nonzero().tolist():
        coefficient = coefficients[output, left, right].item()
        for indices, value in previous[left].items():
            rows[output][tuple(sorted((*indices, right_start + right)))] += coefficient * value
    return tuple({key: value for key, value in row.items() if value != 0} for row in rows)


def reduced_couplings(input_irreps, output_irreps, correlation, dtype_name):
    irreps = tuple(o3.Irrep(irrep) for irrep in input_irreps)
    outputs = tuple(o3.Irrep(irrep) for irrep in output_irreps)
    inputs = tuple((irrep.l, irrep.p) for irrep in irreps)
    largest_input = max(irrep.l for irrep in irreps)
    largest_output = max(irrep.l for irrep in outputs)
    nodes, roots = [], []
    local_cg = {}
    previous, previous_polynomials = {}, {}
    offsets, offset = [], 0
    for block, irrep in enumerate(irreps):
        offsets.append(offset)
        composition = tuple(int(index == block) for index in range(len(irreps)))
        node = len(nodes)
        nodes.append(dict(kind="input", input_start=offset, input_width=irrep.dim,
                          irrep=(irrep.l, irrep.p), irrep_dim=irrep.dim, degree=1, composition=composition))
        previous[composition] = {(irrep.l, irrep.p): [node]}
        previous_polynomials[node] = tuple({(offset + component,): 1.0} for component in range(irrep.dim))
        offset += irrep.dim
    for degree in range(1, correlation + 1):
        if degree > 1:
            current, current_polynomials = {}, {}
            for composition in compositions(degree, len(irreps)):
                last = max(index for index, count in enumerate(composition) if count)
                parent = list(composition)
                parent[last] -= 1
                parent = tuple(parent)
                expected = composition_multiplicities(inputs, composition)
                columns = monomials(inputs, composition)
                locations = {indices: index for index, indices in enumerate(columns)}
                weights = np.array([sqrt(np.prod([factorial(count) for count in Counter(indices).values()]) / factorial(degree))
                                    for indices in columns])
                candidates = defaultdict(list)
                for left, prefix_nodes in previous.get(parent, {}).items():
                    for irrep in o3.Irrep(*left) * irreps[last]:
                        output = (irrep.l, irrep.p)
                        if not expected.get(output):
                            continue
                        # Necessary triangle bound: any contributing prefix can gain at most
                        # (correlation-degree)*largest_input angular momentum afterwards.
                        if irrep.l > largest_output + (correlation - degree) * largest_input:
                            continue
                        if degree == correlation and irrep not in outputs:
                            continue
                        tensor = cg_tensor(left, inputs[last], output, "float64")
                        for prefix in prefix_nodes:
                            polynomial = coupled_polynomial(previous_polynomials[prefix], offsets[last], tensor)
                            vector = np.zeros((irrep.dim, len(columns)), dtype=np.float64)
                            for component, row in enumerate(polynomial):
                                for indices, value in row.items():
                                    vector[component, locations[indices]] = value
                            vector *= weights
                            vector = vector.ravel()
                            norm = np.linalg.norm(vector)
                            if norm > 1e-12:
                                candidates[output].append((prefix, polynomial, vector / norm, float(norm)))
                selected_by_irrep = {}
                for output, count in sorted(expected.items()):
                    irrep = o3.Irrep(*output)
                    if (irrep.l > largest_output + (correlation - degree) * largest_input
                                  or (degree == correlation and irrep not in outputs)):
                        continue
                    entries = candidates.get(output, [])
                    residuals = np.array([entry[2] for entry in entries])
                    selected = []
                    for _ in range(count):
                        norms = np.linalg.norm(residuals, axis=1)
                        maximum = float(norms.max()) if len(norms) else 0.0
                        # Roundoff-level norm ties must not change the selected CG tree.
                        tied = np.flatnonzero(norms >= maximum * (1 - 1e-12)) if maximum else []
                        pivot = int(tied[0]) if len(tied) else -1
                        remaining = float(norms[pivot]) if pivot >= 0 else 0.0
                        if remaining <= 1e-10:
                            raise RuntimeError((degree, composition, output, "rank deficit", len(selected), count, remaining))
                        prefix, polynomial, _, norm = entries[pivot]
                        node = len(nodes)
                        op = (tuple(nodes[prefix]["irrep"]), inputs[last], output)
                        if op not in local_cg:
                            local_cg[op] = cg_tensor(*op, dtype_name)
                        nodes.append(dict(kind="coupling", left_id=prefix, right_id=last,
                                          irrep=output, irrep_dim=irrep.dim, degree=degree,
                                          composition=composition, op=op))
                        current_polynomials[node] = polynomial
                        selected.append(node)
                        direction = residuals[pivot] / remaining
                        for _ in range(2):
                            residuals -= np.outer(residuals @ direction, direction)
                    selected_by_irrep[output] = selected
                current[composition] = selected_by_irrep
            previous, previous_polynomials = current, current_polynomials
        for composition, by_irrep in previous.items():
            for output_block, output in enumerate(outputs):
                for node in by_irrep.get((output.l, output.p), []):
                    roots.append(dict(node_id=node, degree=degree, composition=composition,
                                      irrep=(output.l, output.p), output_block=output_block))
    # Match the public output-block then descending-degree weight grouping.
    roots.sort(key=lambda root: (root["output_block"], -root["degree"], root["composition"], root["node_id"]))
    output_offsets, offset = [], 0
    for output in outputs:
        output_offsets.append(offset)
        offset += output.dim
    for basis, root in enumerate(roots):
        root.update(selected_basis_id=basis, output_component_offset=output_offsets[root["output_block"]])
    cg_ids = {op: index for index, op in enumerate(local_cg)}
    local_records = []
    for op, tensor in local_cg.items():
        indices = tensor.nonzero().tolist()
        counts = [0] * tensor.shape[0]
        for component, _, _ in indices:
            counts[component] += 1
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        local_records.append(dict(
            shape=tuple(tensor.shape), output_offsets=tuple(offsets),
            left_components=tuple(left for _, left, _ in indices),
            right_components=tuple(right for _, _, right in indices),
            coefficients=tuple(tensor[component, left, right].item() for component, left, right in indices),
        ))
    for node in nodes:
        if node["kind"] == "coupling":
            node["cg_id"] = cg_ids[node.pop("op")]
    path_counts = tuple(sum(root["degree"] == degree and root["output_block"] == block for root in roots)
                        for block, _ in enumerate(outputs) for degree in range(correlation, 0, -1))
    return dict(
        input_dim=sum(irrep.dim for irrep in irreps), num_paths=len(roots), correlation=correlation,
        path_counts=path_counts, terms=(),
        cg_dag=dict(nodes=tuple(nodes), roots=tuple(roots), local_cg=tuple(local_records),
                    weight_members=tuple(((basis, 1.0),) for basis in range(len(roots)))),
    )


def original_projection(prefix):
    nodes, roots = prefix["nodes"], prefix["roots"]
    inputs = prefix["inputs"]
    outputs = prefix["outputs"]
    correlation = prefix["correlation"]
    largest_input = max(angular for angular, _ in inputs)
    largest_output = max(angular for angular, _ in outputs)
    groups = defaultdict(list)
    for node_id, node in enumerate(nodes):
        groups[node["degree"], tuple(node["composition"]), tuple(node["irrep"])].append(node_id)
    polynomials, matrices, tables = {}, {}, {}
    offsets = [node["input_start"] for node in nodes if node["kind"] == "input"]

    def polynomial(node_id):
        if node_id not in polynomials:
            node = nodes[node_id]
            if node["kind"] == "input":
                polynomials[node_id] = tuple({(node["input_start"] + component,): 1.0}
                                            for component in range(node["input_width"]))
            else:
                left = node["left_id"]
                right = node["right_id"]
                tensor = cg_tensor(tuple(nodes[left]["irrep"]), tuple(nodes[right]["irrep"]), tuple(node["irrep"]), "float64")
                polynomials[node_id] = coupled_polynomial(polynomial(left), nodes[right]["input_start"], tensor)
        return polynomials[node_id]

    def matrix(key):
        if key not in matrices:
            degree, composition, output = key
            columns = monomials(inputs, composition)
            locations = {indices: column for column, indices in enumerate(columns)}
            value = np.zeros((len(groups[key]), (2 * output[0] + 1) * len(columns)))
            for row, node_id in enumerate(groups[key]):
                for component, entries in enumerate(polynomial(node_id)):
                    for indices, coefficient in entries.items():
                        value[row, component * len(columns) + locations[indices]] = coefficient
            matrices[key] = value, locations
        return matrices[key]

    def multiplication(key, right, output):
        table_key = (key, right, output)
        if table_key not in tables:
            degree, composition, left = key
            child = list(composition)
            child[right] += 1
            child_key = degree + 1, tuple(child), output
            if not groups[child_key]:
                tables[table_key] = None
                return None
            target, locations = matrix(child_key)
            source = np.zeros((len(groups[key]), target.shape[1]))
            tensor = cg_tensor(left, inputs[right], output, "float64")
            for row, node_id in enumerate(groups[key]):
                candidate = coupled_polynomial(polynomial(node_id), offsets[right], tensor)
                for component, entries in enumerate(candidate):
                    for indices, coefficient in entries.items():
                        source[row, component * len(locations) + locations[indices]] = coefficient
            transform = rational_coefficients(np.linalg.lstsq(target.T, source.T, rcond=None)[0].T, max_error=1e-10)
            error = float(np.max(np.abs(source - transform @ target), initial=0))
            if error > 1e-10:
                raise RuntimeError(("prefix multiplication span failure", table_key, error))
            tables[table_key] = transform
        return tables[table_key]

    output_records = {}
    for natural, maximum in ((False, correlation), (True, 4)):
        if natural and correlation < 4:
            continue
        previous = [(tuple(node["irrep"]), tuple(node["composition"]), np.ones(1))
                    for node in nodes if node["kind"] == "input"]
        if not natural:
            output_records[1] = previous
        for degree in range(2, maximum + 1):
            current = []
            for left, composition, coefficients in previous:
                parent = degree - 1, composition, left
                for right, input_irrep in enumerate(inputs):
                    child = list(composition)
                    child[right] += 1
                    child = tuple(child)
                    for irrep in o3.Irrep(*left) * o3.Irrep(*input_irrep):
                        output = irrep.l, irrep.p
                        if natural and (irrep.l >= 12 or irrep.p != (-1) ** irrep.l):
                            continue
                        if irrep.l > largest_output + (correlation - degree) * largest_input:
                            continue
                        if degree == correlation and output not in outputs:
                            continue
                        transform = multiplication(parent, right, output) if coefficients is not None else None
                        value = rational_coefficients(coefficients @ transform, max_error=1e-10) if transform is not None else None
                        current.append((output, child, value))
            current.sort(key=lambda record: o3.Irrep(*record[0]))
            previous = current
            if not natural or degree == 4:
                output_records[degree] = current
    weight_members = [[] for _ in roots]
    root_basis = {(root["output_block"], root["node_id"]): root["selected_basis_id"] for root in roots}
    counts, paths = [], 0
    for output_block, output in enumerate(outputs):
        for degree in range(correlation, 0, -1):
            selected = [record for record in output_records[degree] if record[0] == output]
            counts.append(max(1, len(selected)))
            for _, composition, coefficients in selected:
                key = degree, composition, output
                if coefficients is not None:
                    for node_id, coefficient in zip(groups[key], coefficients, strict=True):
                        if coefficient != 0:
                            weight_members[root_basis[output_block, node_id]].append((paths, float(coefficient)))
                paths += 1
            if not selected:
                paths += 1
    return dict(path_counts=tuple(counts), num_paths=paths,
                weight_members=tuple(tuple(members) for members in weight_members))
