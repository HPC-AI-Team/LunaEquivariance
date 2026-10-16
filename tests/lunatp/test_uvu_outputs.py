"""Run with: python -m unittest discover -s tests -p test_uvu_outputs.py."""

import unittest
import pytest
import torch
from e3nn.o3 import Irreps, TensorProduct as E3nnTensorProduct
from LunaEquivariance import TensorProduct
from tests.lunatp.test_uvu_shapes import derivative_chain


@pytest.mark.cuda
@unittest.skipUnless(torch.cuda.is_available(), "UVU requires CUDA")
class UVUOutputTests(unittest.TestCase):
    def test_overlapping_outputs_and_derivatives(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                torch.manual_seed(939)
                inputs = Irreps("32x0e + 32x1o")
                edges = Irreps("1x0e + 1x1o")
                outputs = inputs + Irreps("32x2e")
                instructions = [
                    (i, j, k, "uvu", True)
                    for i, (_, a) in enumerate(inputs)
                    for j, (_, b) in enumerate(edges)
                    for k, (_, c) in enumerate(inputs)
                    if c in a * b
                ]
                tp = E3nnTensorProduct(
                    inputs,
                    edges,
                    outputs,
                    instructions,
                    internal_weights=shared,
                    shared_weights=shared,
                ).cuda()
                op = TensorProduct(tp)
                x = torch.randn(inputs.dim, 19, device="cuda").t().requires_grad_()
                y = torch.randn(edges.dim, 19, device="cuda").t().requires_grad_()
                weight = (
                    tp.weight
                    if shared
                    else torch.randn(
                        19, tp.weight_numel, device="cuda", requires_grad=True
                    )
                )
                grad_out = torch.randn(
                    19, outputs.dim, device="cuda", requires_grad=True
                )
                operands = (x, y, weight, grad_out)
                first_directions = tuple(
                    (torch.randn_like(t, requires_grad=True) for t in operands[:3])
                )
                second_directions = tuple((torch.randn_like(t) for t in operands))
                expected = derivative_chain(
                    tp(x, y, weight), operands, first_directions, second_directions
                )
                actual = derivative_chain(
                    op(x, y, None if shared else weight),
                    operands,
                    first_directions,
                    second_directions,
                )
                for order, (reference, result) in enumerate(zip(expected, actual)):
                    with self.subTest(order=order):
                        for left, right in zip(reference, result):
                            self.assertTrue(bool(torch.isfinite(right).all()))
                            error = (left - right).double().norm()
                            scale = left.double().norm().clamp_min(1e-20)
                            self.assertLess((error / scale).item(), 2e-05)
                for result in (actual[0][0], actual[2][3], actual[3][3]):
                    self.assertEqual(
                        torch.count_nonzero(result[:, inputs.dim :]).item(), 0
                    )
                with torch.no_grad():
                    inference = op(x, y, None if shared else weight)
                torch.testing.assert_close(
                    inference, expected[0][0], rtol=0.0005, atol=5e-05
                )


if __name__ == "__main__":
    unittest.main()
