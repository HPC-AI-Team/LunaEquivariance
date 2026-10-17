"""Backward, double-backward and triple-backward contractions for S = CG * x * y * w * g.

x/y/w are node features, edge attributes and weights; g is the gradient of the
forward output. rx/ry/rw are gradients supplied to the first backward outputs.
bx/by/bw/bg are gradients supplied to the second backward outputs.

Each output is a sum of the listed products, in tuple order. For example,
backward uses q0 = CG(x, y), z0 = w * g, and gw = q0 * g.
The x/y entries contract the remaining feature with CG and the named z term.

Each order lists products needed for its outputs. q terms contract CG with
feature pairs; z terms contract weights with output gradients. gx/gy/gw/gg
follow x/y/w/g order; grx/gry/grw follow rx/ry/rw. Tuple order fixes the generated
floating-point accumulation order and must be preserved.
"""

INPUTS = ("x", "y", "w", "g", "rx", "ry", "rw", "bx", "by", "bw", "bg")


OUTPUT_SHAPES = {
    "gx": "x",
    "gy": "y",
    "gw": "w",
    "gg": "g",
    "grx": "x",
    "gry": "y",
    "grw": "w",
}


PROGRAMS = {
    1: {
        "cg": {"q0": (("x", "y"),)},
        "z": {"z0": (("w", "g"),)},
        "x": {"gx": (("y", "z0"),)},
        "y": {"gy": (("x", "z0"),)},
        "w": {"gw": (("q0", "g"),)},
        "forward": (),
        "outputs": ("gx", "gy", "gw"),
    },
    2: {
        "cg": {"q0": (("x", "y"),), "q1": (("rx", "y"), ("x", "ry"))},
        "z": {"z0": (("w", "g"),), "z1": (("rw", "g"),)},
        "x": {"gx": (("ry", "z0"), ("y", "z1"))},
        "y": {"gy": (("rx", "z0"), ("x", "z1"))},
        "w": {"gw": (("q1", "g"),)},
        "forward": (("q1", "w"), ("q0", "rw")),
        "outputs": ("gx", "gy", "gw", "gg"),
    },
    3: {
        "cg": {
            "q0": (("x", "y"),),
            "q1": (("rx", "y"), ("x", "ry")),
            "q2": (("rx", "by"), ("bx", "ry")),
            "q3": (("bx", "y"), ("x", "by")),
        },
        "z": {
            "z0": (("w", "g"),),
            "z1": (("bw", "g"), ("w", "bg")),
            "z2": (("rw", "g"),),
            "z3": (("rw", "bg"),),
        },
        "x": {
            "gx": (("ry", "z1"), ("by", "z2"), ("y", "z3")),
            "grx": (("by", "z0"), ("y", "z1")),
        },
        "y": {
            "gy": (("rx", "z1"), ("bx", "z2"), ("x", "z3")),
            "gry": (("bx", "z0"), ("x", "z1")),
        },
        "w": {
            "gw": (("q2", "g"), ("q1", "bg")),
            "grw": (("q3", "g"), ("q0", "bg")),
        },
        "forward": (("q2", "w"), ("q1", "bw"), ("q3", "rw")),
        "outputs": ("gx", "gy", "gw", "gg", "grx", "gry", "grw"),
    },
}


ENV_INPUTS = ("x", "y", "w", "g", "e", "rx", "ry", "rw", "re", "bx", "by", "bw", "bg", "be")
ENV_OUTPUT_SHAPES = {**OUTPUT_SHAPES, "ge": "e", "gre": "e"}


# The envelope e multiplies each edge message. None means no envelope factor;
# re/be are gradients supplied to its backward/double-backward outputs.
# Keep the same explicit product and accumulation ordering as PROGRAMS.
ENV_PROGRAMS = {
    1: {
        "cg": {
            "q0": (("x", "y", "e"),),
            "q1": (("x", "y", None),),
        },
        "z": {
            "z0": (("w", "g"),),
        },
        "x": {
            "gx": (("y", "z0", "e"),),
        },
        "y": {
            "gy": (("x", "z0", "e"),),
        },
        "w": {
            "gw": (("q0", "g"),),
        },
        "e": {
            "ge": (("q1", "z0"),),
        },
        "forward": (),
        "outputs": ("gx", "gy", "gw", "ge"),
    },
    2: {
        "cg": {
            "q0": (("rx", "y", "e"), ("x", "ry", "e"), ("x", "y", "re")),
            "q1": (("x", "y", "e"),),
            "q2": (("rx", "y", None),),
            "q3": (("x", "ry", None),),
            "q4": (("x", "y", None),),
        },
        "z": {
            "z0": (("w", "g"),),
            "z1": (("rw", "g"),),
        },
        "x": {
            "gx": (("ry", "z0", "e"), ("y", "z1", "e"), ("y", "z0", "re")),
        },
        "y": {
            "gy": (("rx", "z0", "e"), ("x", "z1", "e"), ("x", "z0", "re")),
        },
        "w": {
            "gw": (("q0", "g"),),
        },
        "e": {
            "ge": (("q2", "z0"), ("q3", "z0"), ("q4", "z1")),
        },
        "forward": (("q0", "w"), ("q1", "rw")),
        "outputs": ("gx", "gy", "gw", "gg", "ge"),
    },
    3: {
        "cg": {
            "q0": (
                ("bx", "ry", "e"),
                ("bx", "y", "re"),
                ("rx", "by", "e"),
                ("x", "by", "re"),
                ("rx", "y", "be"),
                ("x", "ry", "be"),
            ),
            "q1": (("rx", "y", "e"), ("x", "ry", "e"), ("x", "y", "re")),
            "q2": (("bx", "y", "e"), ("x", "by", "e"), ("x", "y", "be")),
            "q3": (("bx", "ry", None),),
            "q4": (("bx", "y", None),),
            "q5": (("rx", "by", None),),
            "q6": (("x", "by", None),),
            "q7": (("rx", "y", None),),
            "q8": (("x", "ry", None),),
            "q9": (("x", "y", None),),
            "q10": (("x", "y", "e"),),
        },
        "z": {
            "z0": (("rw", "g"),),
            "z1": (("w", "g"),),
            "z2": (("bw", "g"), ("w", "bg")),
            "z3": (("rw", "bg"),),
        },
        "x": {
            "gx": (
                ("by", "z0", "e"),
                ("by", "z1", "re"),
                ("ry", "z2", "e"),
                ("y", "z2", "re"),
                ("y", "z3", "e"),
                ("ry", "z1", "be"),
                ("y", "z0", "be"),
            ),
            "grx": (("by", "z1", "e"), ("y", "z2", "e"), ("y", "z1", "be")),
        },
        "y": {
            "gy": (
                ("bx", "z0", "e"),
                ("bx", "z1", "re"),
                ("rx", "z2", "e"),
                ("x", "z2", "re"),
                ("x", "z3", "e"),
                ("rx", "z1", "be"),
                ("x", "z0", "be"),
            ),
            "gry": (("bx", "z1", "e"), ("x", "z2", "e"), ("x", "z1", "be")),
        },
        "w": {
            "gw": (("q0", "g"), ("q1", "bg")),
            "grw": (("q2", "g"), ("q10", "bg")),
        },
        "e": {
            "ge": (
                ("q3", "z1"),
                ("q4", "z0"),
                ("q5", "z1"),
                ("q6", "z0"),
                ("q7", "z2"),
                ("q8", "z2"),
                ("q9", "z3"),
            ),
            "gre": (("q4", "z1"), ("q6", "z1"), ("q9", "z2")),
        },
        "forward": (("q0", "w"), ("q2", "rw"), ("q1", "bw")),
        "outputs": ("gx", "gy", "gw", "gg", "ge", "grx", "gry", "grw", "gre"),
    },
}
