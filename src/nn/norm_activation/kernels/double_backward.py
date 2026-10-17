"""NormActivation double-backward radial CUDA partials."""

from .backward import emit_radial_geometry


def emit_radial_double_backward(meta, program, components, input_address, output_address):
    """Emit the requested second radial partials without changing their arithmetic."""
    input_at = lambda role, component=0: (
        input_address(role, component) if role in program["inputs"] else "Element(0)"
    )
    outputs = program["active_outputs"]
    lines = emit_radial_geometry(meta, program, components, input_address)
    if "sb" in outputs:
        lines.append(f"{output_address('sb', 0)}=ab*h+abb*U*p;")
    for component in range(components):
        if "sx" in outputs:
            lines.append(f"{output_address('sx', component)}=ab*U*{input_at('grad_out', component)};")
        if "sg" in outputs:
            lines.append(f"{output_address('sg', component)}=a*{input_at('rx', component)}+ab*U*{input_at('x', component)};")
    lines += ["} else {"]
    if components == 1:
        lines += f"""if(isfinite(radius) && radius==fabsf({input_at('x')})) {{
const Element sign=({input_at('x')}<Element(0))?Element(-1):Element(1);
const Element direction={input_at('rx')}+sign*U;
const Element gradient={input_at('bx')}+sign*Z;""".splitlines()
        if "sx" in outputs:
            lines.append(f"{output_address('sx', 0)}=sign*F2*{input_at('grad_out', 0)}*direction;")
        if "sb" in outputs:
            lines.append(f"{output_address('sb', 0)}=F2*{input_at('grad_out', 0)}*direction;")
        if "sg" in outputs:
            lines.append(f"{output_address('sg', 0)}=F1*direction;")
        lines += ["} else {"]
    lines += ["const Element eta=p*iv,c=F1-a,d=c*iv;"]
    lines += r"""const Element alpha=tr*iv;
const Element A=d*alpha+ab*U;
const Element B=F2*eta*(alpha+U)+d*h-(Element(3)*d*alpha+ab*U)*eta;""".splitlines()
    if "sb" in outputs:
        lines.append(f"{output_address('sb', 0)}=F2*(alpha+U)*eta+ab*(h-eta*alpha);")
    for component in range(components):
        if "sx" in outputs:
            lines.append(
                f"{output_address('sx', component)}=A*{input_at('grad_out', component)}"
                f"+B*iv*{input_at('x', component)}+d*eta*{input_at('rx', component)};"
            )
        if "sg" in outputs:
            lines.append(
                f"{output_address('sg', component)}=a*{input_at('rx', component)}"
                f"+(c*alpha+F1*U)*iv*{input_at('x', component)};"
            )
    if components == 1:
        lines += ["}"]
    lines += ["}"]
    return lines, [(meta["activation"]["kind"], {})]
