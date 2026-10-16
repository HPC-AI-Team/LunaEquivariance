from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from LunaEquivariance import o3 as luna_o3, nn as luna_nn

from LunaEquivariance.utils.testing import check_compatibility, TensorProductConvReference


@pytest.mark.parametrize("existing_module", [False, True])
@pytest.mark.parametrize("derivative_order", [0, 1, 2, 3])
def test_loss_gradients_and_timings(existing_module, derivative_order, capsys):
    reference = o3.Linear("2x0e", "3x0e").double()
    module = luna_o3.Linear(reference) if existing_module else luna_o3.Linear("2x0e", "3x0e").double()
    with torch.no_grad():
        module.weight.add_(0.3)
    module.weight.grad = torch.ones_like(module.weight)
    old_grad = module.weight.grad.clone()
    old_weight = module.weight.detach().clone()
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    report = check_compatibility(
        module, (x,),
        loss_fn=lambda output: output.sin().sum(),
        derivative_order=derivative_order,
        warmup=0,
        repeats=1,
        atol=1e-9,
        rtol=1e-8,
    )
    assert report["passed"], report
    if derivative_order:
        assert report["errors"][f"gradient_{derivative_order}"]["passed"]
    stage = "forward" if derivative_order == 0 else f"forward_loss_gradient_{derivative_order}"
    assert list(report["timings"]) == [stage]
    assert all(timing["speedup"] > 0 for timing in report["timings"].values())
    assert capsys.readouterr().out.count("speedup") == 1
    assert x.grad is None
    assert report["loss"]["reference"] == pytest.approx(module(x).sin().sum().item())
    torch.testing.assert_close(module.weight, old_weight)
    torch.testing.assert_close(module.weight.grad, old_grad)


def test_norm_activation_string_irreps_with_bias():
    report = check_compatibility(
        luna_nn.NormActivation("2x0e+1x1o", torch.sigmoid, bias=True),
        (torch.randn(3, 5, requires_grad=True),),
        warmup=0,
        repeats=1,
        verbose=False,
    )
    assert report["passed"], report


def test_invalid_input_raises():
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        check_compatibility(
            luna_o3.Linear("2x0e", "2x0e"), (torch.randn(3, 7),),
            verbose=False,
        )


def test_nonfinite_loss_is_not_a_pass():
    report = check_compatibility(
        luna_o3.Norm("1x1o"), (torch.randn(4, 3),),
        loss_fn=lambda x: x.sum() * float("nan"),
        derivative_order=0,
        repeats=1,
        warmup=0,
        verbose=False,
    )
    assert not report["passed"]
    assert not report["errors"]["loss"]["passed"]


def test_conv_reference_preserves_separate_scalar_paths():
    tp = o3.TensorProduct(
        "2x0e+2x1o",
        "1x0e+1x1o",
        "2x0e",
        [(0, 0, 0, "uvu", True), (1, 1, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    ).double()
    graph = SimpleNamespace(
        src=torch.tensor([1, 0, 1]), dst=torch.tensor([0, 1, 1]), num_nodes=3
    )
    x, y, w = (
        torch.randn(3, 8).double(),
        torch.randn(3, 4).double(),
        torch.randn(3, 4).double(),
    )
    env = torch.randn(3, 1).double()
    node, scalar = TensorProductConvReference(tp, scalar_outputs=True)(
        x, y, w, graph=graph, env=env, return_l0_feat=True
    )
    edge = tp(x[graph.src], y, w)
    torch.testing.assert_close(
        node, torch.zeros(3, 2).double().index_add(0, graph.dst, edge * env)
    )
    assert scalar.shape == (3, 4)
    torch.testing.assert_close(scalar[:, :2] + scalar[:, 2:], edge)


class OffsetLinear(luna_o3.Linear):
    def forward(self, x):
        return super().forward(x) + 1


class DetachedLinear(luna_o3.Linear):
    def forward(self, x):
        return super().forward(x).detach() + x * 0


@pytest.mark.parametrize(
    "reference_type,error_key", [(OffsetLinear, "loss"), (DetachedLinear, "gradient_1")]
)
def test_mismatch_is_reported(reference_type, error_key):
    reference = reference_type("2x0e", "2x0e").double()
    x = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    report = check_compatibility(
        reference, (x,), warmup=0, repeats=1, verbose=False
    )
    assert not report["passed"]
    assert not report["errors"][error_key]["passed"]
    assert "speedup" in report["timings"]["forward_loss_gradient_1"]


def test_unweighted_conv_scalar_reference():
    tp = o3.TensorProduct(
        "2x0e+2x1o",
        "2x0e+2x1o",
        "2x0e",
        [(0, 0, 0, "uuu", False), (1, 1, 0, "uuu", False)],
        internal_weights=False,
    ).double()
    graph = SimpleNamespace(
        src=torch.tensor([1, 0]), dst=torch.tensor([0, 1]), num_nodes=2
    )
    x, y = torch.randn(2, 8).double(), torch.randn(2, 8).double()
    node, scalar = TensorProductConvReference(tp, scalar_outputs=True)(
        x, y, graph=graph, return_l0_feat=True
    )
    edge = tp(x[graph.src], y)
    torch.testing.assert_close(scalar[:, :2] + scalar[:, 2:], edge)
    torch.testing.assert_close(node, edge)


@pytest.mark.parametrize("kind", ["sh", "norm", "activation", "gate", "norm_activation", "dropout"])
def test_module_reference(kind):
    if kind == "sh":
        module = luna_o3.SphericalHarmonics("0e + 1o", normalize=True)
        width = 3
    elif kind == "norm":
        module = luna_o3.Norm("2x0e + 1x1o", squared=True)
        width = 5
    elif kind == "activation":
        module = luna_nn.Activation("2x0o + 1x1o", [torch.abs, None])
        width = 5
    elif kind == "gate":
        module = luna_nn.Gate("2x0e", [torch.tanh], "1x0o", [torch.abs], "1x1o")
        width = module.irreps_in.dim
    elif kind == "norm_activation":
        module = luna_nn.NormActivation("2x0e + 1x1o", torch.sigmoid, bias=True)
        with torch.no_grad():
            module.biases.fill_(0.2)
        width = 5
    else:
        module = luna_nn.Dropout("2x0e + 1x1o", p=0.3).eval()
        width = 5
    x = torch.randn(4, width, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["linear", "sh", "norm", "activation", "gate", "norm_activation", "dropout"])
def test_cuda_module_three_derivatives(kind, dtype):
    from LunaEquivariance import Linear, SphericalHarmonics, Norm, Activation, Gate, NormActivation, Dropout

    torch.manual_seed(42)
    modules = {
        "linear": lambda: Linear("2x0e+1x1o", "3x0e+2x1o", biases=True),
        "sh": lambda: SphericalHarmonics("0e+1o+2e", normalize=True),
        "norm": lambda: Norm("2x0e+1x1o"),
        "activation": lambda: Activation("2x0e+1x1o", [torch.tanh, None]),
        "gate": lambda: Gate("2x0e", [torch.tanh], "1x0e", [torch.sigmoid], "1x1o"),
        "norm_activation": lambda: NormActivation("2x0e+1x1o", torch.sigmoid, bias=True),
        "dropout": lambda: Dropout("2x0e+1x1o", p=0.3),
    }
    module = modules[kind]().to(device="cuda", dtype=dtype)
    width = 3 if kind == "sh" else 6 if kind == "gate" else 5
    x = torch.randn(5, width, device="cuda", dtype=dtype).requires_grad_()
    assert_cuda_report(module, (x,))


def assert_cuda_report(module, args, **kwargs):
    tolerance = 1e-9 if args[0].dtype == torch.float64 else 3e-4
    report = check_compatibility(
        module, args, kwargs=kwargs, derivative_order=3,
        loss_fn=lambda output: sum(x.sin().mean() for x in (output if isinstance(output, tuple) else (output,))),
        warmup=1, repeats=2, atol=tolerance, rtol=tolerance, verbose=False,
    )
    assert report["passed"], report
    assert report["operator"] == type(module).__name__
    assert report["errors"]["gradient_3"]["passed"]
    assert all(item["speedup"] > 0 for item in report["timings"].values())


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("conv,mode,internal,shared,env_shape", [
    (conv, mode, internal, shared, env_shape)
    for conv in (False, True) for mode in ("uuu", "uvu", "uvw")
    for internal, shared in ((False, False), (False, True), (True, True))
    for env_shape in (((4,), (4, 1)) if conv else (None,))
    if not (conv and mode == "uvu" and shared)
])
def test_cuda_tensor_product_modes(dtype, conv, mode, internal, shared, env_shape):
    from LunaEquivariance import TensorProduct, TensorProductConv

    torch.manual_seed(42)
    cls = TensorProductConv if conv else TensorProduct
    module = cls("2x1o", "2x1o", "2x0e", [(0, 0, 0, mode, True)],
                 internal_weights=internal, shared_weights=shared).to(device="cuda", dtype=dtype)
    x = torch.randn(3 if conv else 4, 6, device="cuda", dtype=dtype).requires_grad_()
    y = torch.randn(4, 6, device="cuda", dtype=dtype).requires_grad_()
    args = (x, y)
    if not internal:
        shape = (module.weight_numel,) if shared else (4, module.weight_numel)
        args += (torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(),)
    kwargs = {}
    if conv:
        src = torch.tensor([1, 2, 0, 1], device="cuda")
        dst = torch.tensor([0, 0, 1, 2], device="cuda")
        kwargs = dict(graph=module.create_graph(src, dst, 3),
                      env=torch.rand(env_shape, device="cuda", dtype=dtype).requires_grad_())
        if mode == "uvu":
            kwargs["return_l0_feat"] = True
    assert_cuda_report(module, args, **kwargs)


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("grouping,norm,weighting,layout", [
    ("merged", "layer", "component", "mul_ir"),
    ("separable", "layer", "norm", "ir_mul"),
    ("per_irrep", "rms", "inverse_dim_squared", "mul_ir"),
    ("merged", "rms", "norm", "ir_mul"),
    ("separable", "rms", "component", "mul_ir"),
    ("per_irrep", "layer", "inverse_dim_squared", "ir_mul"),
    ("merged", "layer", "custom", "ir_mul"),
    ("separable", "rms", "custom", "mul_ir"),
])
def test_cuda_equivariant_norm(dtype, grouping, norm, weighting, layout):
    from LunaEquivariance import EquivariantNorm

    torch.manual_seed(42)
    options = dict(stat_weights=[0.2, 0.3, 0.1]) if weighting == "custom" else dict(weighting=weighting)
    module = EquivariantNorm("3x0e+2x1o+2x2e", grouping=grouping, norm=norm,
                             layout=layout, bias=True, device="cuda", dtype=dtype, **options)
    with torch.no_grad():
        module.weight.uniform_(0.8, 1.2)
        module.bias.uniform_(-0.1, 0.1)
    x = torch.randn(4, 19, device="cuda", dtype=dtype).requires_grad_()
    assert_cuda_report(module, (x,))


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("env_shape,smooth", [
    (shape, smooth) for shape in (None, (7,), (7, 1), (7, 3))
    for smooth in (False, True) if shape is not None or not smooth
])
def test_cuda_graph_softmax(dtype, env_shape, smooth):
    from LunaEquivariance import GraphSoftmax

    torch.manual_seed(42)
    x = torch.randn(7, 3, device="cuda", dtype=dtype).requires_grad_()
    csr = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    kwargs = {} if env_shape is None else dict(env=(torch.rand(env_shape, device="cuda", dtype=dtype) + 0.2).requires_grad_())
    assert_cuda_report(GraphSoftmax(smooth_out=smooth), (x, csr), **kwargs)


@pytest.mark.cuda
@pytest.mark.parametrize("conv,env_shape", [(False, None), (True, (7,)), (True, (7, 1))])
@pytest.mark.parametrize("bias_kind,causal,flatten", [(0, False, False), (1, False, True), (2, True, False)])
def test_cuda_attention(conv, env_shape, bias_kind, causal, flatten):
    from LunaEquivariance import VarlenAttention, VarlenAttentionConv

    torch.manual_seed(42)
    cls = VarlenAttentionConv if conv else VarlenAttention
    module = cls(num_heads=2, causal=causal, softmax_scale=0.4)
    args = tuple((torch.randn(7, 2, d, device="cuda") * 0.2).requires_grad_() for d in (8, 8, 5))
    if flatten:
        args = tuple(x.flatten(1) for x in args)
    csr = torch.tensor([0, 3, 3, 7], device="cuda", dtype=torch.int32)
    kwargs = {}
    if bias_kind:
        shape = (7, 2) if bias_kind == 1 else (3, 2, 4, 4)
        kwargs["bias"] = (torch.randn(shape, device="cuda") * 0.1).requires_grad_()
    if conv:
        kwargs.update(envelope=torch.rand(env_shape, device="cuda").requires_grad_(), return_attn_out=True)
    assert_cuda_report(module, (*args, csr), **kwargs)


def test_frozen_parameters():
    module = luna_o3.Linear("2x0e", "3x0e").requires_grad_(False)
    x = torch.randn(3, 2, requires_grad=True)
    report = check_compatibility(module, (x,), warmup=0, repeats=1, verbose=False)
    assert report["passed"], report
    assert all(not p.requires_grad for p in module.parameters())


@pytest.mark.cuda
def test_attention_conv_default_tuple_loss():
    from LunaEquivariance import VarlenAttentionConv

    args = tuple(torch.randn(3, 1, 8, device="cuda").requires_grad_() for _ in range(3))
    csr = torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32)
    report = check_compatibility(
        VarlenAttentionConv(), (*args, csr), kwargs={"return_attn_out": True},
        derivative_order=3, warmup=0, repeats=1, verbose=False,
    )
    assert report["reference"] == "PyTorch"
    assert report["passed"], report


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["norm", "softmax"])
def test_low_precision_pytorch_reference(kind, dtype):
    from LunaEquivariance import EquivariantNorm, GraphSoftmax

    torch.manual_seed(12)
    if kind == "norm":
        module = EquivariantNorm("3x0e+2x1o", dtype=dtype)
        args = (torch.randn(4, 9, dtype=dtype, requires_grad=True),)
    else:
        module = GraphSoftmax()
        args = (torch.randn(4, 3, dtype=dtype, requires_grad=True), torch.tensor([0, 2, 4]))
    report = check_compatibility(module, args, derivative_order=3, warmup=0, repeats=1,
                                 atol=0.03, rtol=0.03, verbose=False)
    assert report["reference"] == "PyTorch"
    assert report["passed"], report


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["uuu", "uvu", "uvw"])
@pytest.mark.parametrize("env_shape", [(4,), (4, 1)])
def test_tensor_product_conv_envelope_compile(mode, env_shape):
    from LunaEquivariance import TensorProductConv

    torch.manual_seed(42)
    module = TensorProductConv(
        "2x1o", "2x1o", "2x0e", [(0, 0, 0, mode, True)],
        internal_weights=False, shared_weights=False,
    ).to(device="cuda", dtype=torch.float64)
    graph = module.create_graph(
        torch.tensor([1, 2, 0, 1], device="cuda"),
        torch.tensor([0, 0, 1, 2], device="cuda"), 3,
    )
    args = tuple(torch.randn(shape, device="cuda", dtype=torch.float64).requires_grad_()
                 for shape in ((3, 6), (4, 6), (4, module.weight_numel), env_shape))

    def forward(x, edge_attr, weight, env):
        return module(x, edge_attr, weight, graph=graph, env=env)

    expected = forward(*args)
    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
    actual = compiled(*args)
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected.sin().sum(), args)
    actual_grads = torch.autograd.grad(actual.sin().sum(), args)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grads[-1].shape == env_shape
