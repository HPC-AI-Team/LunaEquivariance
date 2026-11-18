import torch

from tests.so2.compatibility import check_compatibility


def test_function_with_external_parameters_and_second_derivatives():
    def operator(x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias).tanh()

    def reference(x, weight, bias):
        return (x @ weight.T + bias).tanh()

    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in ((3, 5), (4, 5), (4,))
    )
    report = check_compatibility(
        operator, inputs, reference=reference, derivative_order=2,
        warmup=0, repeats=1, verbose=False,
    )
    assert report["passed"]
    assert set(report["errors"]) == {"output", "loss", "gradient_1", "gradient_2"}
    assert report["operator"] == "operator"
    timing = report["timings"]["forward_loss_gradient_2"]
    assert timing["samples_ms"]["lunaeq"] == [timing["lunaeq_ms"]]
    assert timing["samples_ms"]["reference"] == [timing["reference_ms"]]
    assert timing["speedup"] == timing["reference_ms"] / timing["lunaeq_ms"]


def test_function_report_detects_wrong_parameter_gradient():
    def operator(x, weight):
        return x @ weight.detach().T

    def reference(x, weight):
        return x @ weight.T

    x = torch.ones(2, 3, requires_grad=True)
    weight = torch.ones(4, 3, requires_grad=True)
    report = check_compatibility(
        operator, (x, weight), reference=reference,
        warmup=0, repeats=1, verbose=False,
    )
    assert report["errors"]["output"]["passed"]
    assert not report["errors"]["gradient_1"]["passed"]
    assert not report["passed"]


def test_so2_function_uses_explicit_reference():
    from LunaEquivariance import so2
    from tests.so2.reference.projection import linear

    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(2, dtype=torch.float64, requires_grad=True)
    report = check_compatibility(
        so2.linear, ((x,), (weight,), bias), reference=linear, kwargs={"lmax": 0, "mmax": 0},
        derivative_order=2, warmup=0, repeats=1, verbose=False,
    )
    assert report["passed"]
    assert report["reference"] == "PyTorch"
