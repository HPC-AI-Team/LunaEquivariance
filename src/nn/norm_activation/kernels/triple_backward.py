"""NormActivation triple-backward radial CUDA partials."""

from .backward import emit_radial_geometry


def emit_radial_triple_backward(meta, program, components, input_address, output_address):
    """Emit the requested third radial partials without changing their arithmetic."""
    input_at = lambda role, component=0: (
        input_address(role, component) if role in program["inputs"] else "Element(0)"
    )
    outputs = program["active_outputs"]
    lines = emit_radial_geometry(meta, program, components, input_address)
    if "tb" in outputs:
        lines.append(f"{output_address('tb', 0)}=ab*ff+abb*(U*(m+e)+Z*h)+abbb*Z*U*p;")
    if "trb" in outputs:
        lines.append(f"{output_address('trb', 0)}=ab*(m+e)+abb*Z*p;")
    for component in range(components):
        if "tx" in outputs:
            lines.append(f"{output_address('tx', component)}=abb*Z*U*{input_at('grad_out', component)}+ab*U*{input_at('bg', component)};")
        if "tg" in outputs:
            lines.append(f"{output_address('tg', component)}=abb*Z*U*{input_at('x', component)}+ab*Z*{input_at('rx', component)}+ab*U*{input_at('bx', component)};")
        if "trx" in outputs:
            lines.append(f"{output_address('trx', component)}=ab*Z*{input_at('grad_out', component)}+a*{input_at('bg', component)};")
    lines += ["} else {"]
    if components == 1:
        lines += f"""if(isfinite(radius) && radius==fabsf({input_at('x')})) {{
const Element sign=({input_at('x')}<Element(0))?Element(-1):Element(1);
const Element direction={input_at('rx')}+sign*U;
const Element gradient={input_at('bx')}+sign*Z;""".splitlines()
        if "tx" in outputs:
            lines.append(f"{output_address('tx', 0)}=gradient*F3*{input_at('grad_out', 0)}*direction+sign*F2*{input_at('bg', 0)}*direction;")
        if "tb" in outputs:
            lines.append(f"{output_address('tb', 0)}=sign*gradient*F3*{input_at('grad_out', 0)}*direction+F2*{input_at('bg', 0)}*direction;")
        if "tg" in outputs:
            lines.append(f"{output_address('tg', 0)}=sign*gradient*F2*direction;")
        if "trx" in outputs:
            lines.append(f"{output_address('trx', 0)}=sign*gradient*F2*{input_at('grad_out', 0)}+F1*{input_at('bg', 0)};")
        if "trb" in outputs:
            lines.append(f"{output_address('trb', 0)}=gradient*F2*{input_at('grad_out', 0)}+sign*F1*{input_at('bg', 0)};")
        lines += ["} else {"]
    lines += ["const Element eta=p*iv,c=F1-a,d=c*iv;"]
    lines += r"""const Element alpha=tr*iv,beta=j*iv,kappa=e*iv;
const Element C=m+e;
const Element D=alpha*C+beta*h+eta*kk-Element(3)*eta*alpha*beta;
const Element E=U*C+Z*h-eta*(beta*U+Z*alpha);
const Element V=eta*(alpha+U)*(beta+Z);
const Element arr=F2*iv-Element(2)*d*iv,arb=F2*iv-ab*iv;
const Element wr=d*ff+arr*D+d*alpha*kappa+arb*E+ab*U*kappa+F3*V;
const Element we=d*(kk-Element(3)*alpha*beta)-ab*(beta*U+Z*alpha)+F2*(alpha+U)*(beta+Z);
const Element wa=d*(C-Element(3)*eta*beta)-ab*eta*Z+F2*eta*(beta+Z);
const Element wb=d*(h-Element(3)*eta*alpha)-ab*eta*U+F2*eta*(alpha+U);
const Element wm=d*alpha+ab*U,wh=d*beta+ab*Z,wk=d*eta;
const Element wn=wr-iv*(we*eta+wa*alpha+wb*beta+radius*wm*kappa);""".splitlines()
    if "tb" in outputs:
        lines.append(f"{output_address('tb', 0)}=ab*ff+arb*D+F2*iv*E+F3*V;")
    if "trb" in outputs:
        lines.append(f"{output_address('trb', 0)}=ab*(C-eta*beta)+F2*eta*(beta+Z);")
    for component in range(components):
        if "tx" in outputs:
            lines.append(
                f"{output_address('tx', component)}=wn*iv*{input_at('x', component)}"
                f"+we*iv*{input_at('grad_out', component)}+wa*iv*{input_at('rx', component)}"
                f"+wb*iv*{input_at('bx', component)}+wm*{input_at('bg', component)};"
            )
        if "tg" in outputs:
            lines.append(
                f"{output_address('tg', component)}=we*iv*{input_at('x', component)}"
                f"+wh*{input_at('rx', component)}+wm*{input_at('bx', component)};"
            )
        if "trx" in outputs:
            lines.append(
                f"{output_address('trx', component)}=wa*iv*{input_at('x', component)}"
                f"+wh*{input_at('grad_out', component)}+wk*{input_at('bx', component)}"
                f"+a*{input_at('bg', component)};"
            )
    if components == 1:
        lines += ["}"]
    lines += ["}"]
    return lines, [(meta["activation"]["kind"], {})]
