"""Backend-scalar regressions for the flat contraction facade."""

import pytest
import quimb.tensor as qtn

import pepsy


@pytest.mark.parametrize("strip_exponent", (False, True))
def test_contract_flat_preserve_backend_keeps_torch_gradient(strip_exponent):
    """The opt-in raw result must retain its Torch autograd graph."""
    torch = pytest.importorskip("torch")
    value = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    class _FlatTN:
        Lx = 1
        Ly = 1

        def contract(self, *args, **kwargs):
            del args
            if kwargs["strip_exponent"]:
                return value, torch.zeros((), dtype=value.dtype)
            return value

    result = pepsy.contract_flat(
        _FlatTN(),
        method="exact",
        strip_exponent=strip_exponent,
        preserve_backend=True,
    )
    mantissa = result[0] if strip_exponent else result
    assert mantissa is value

    mantissa.square().backward()
    assert value.grad.item() == pytest.approx(4.0)


def test_contract_flat_default_still_formats_backend_scalar():
    """The compatibility default remains a reporting-friendly Python value."""
    torch = pytest.importorskip("torch")

    class _FlatTN:
        Lx = 1
        Ly = 1

        def contract(self, *args, **kwargs):
            del args, kwargs
            return torch.tensor(2.0, dtype=torch.float64, requires_grad=True)

    result = pepsy.contract_flat(_FlatTN(), method="exact")
    assert isinstance(result, complex)
    assert result == pytest.approx(2.0)


def test_contract_flat_middle_out_preserves_torch_gradient():
    """Direct outside-to-middle compression must retain Torch autograd."""
    torch = pytest.importorskip("torch")
    tn = qtn.TN2D_rand(3, 3, D=2, seed=271, dtype="float64")
    source_arrays = []
    for tensor in tn:
        data = torch.tensor(tensor.data, dtype=torch.float64, requires_grad=True)
        tensor.modify(data=data)
        source_arrays.append(data)

    mantissa, exponent = pepsy.contract_flat(
        tn,
        method="mps",
        chi=16,
        compression_mode="direct",
        boundary_direction="middle-out-x",
        cutoff=0.0,
        contraction_opt="greedy",
        strip_exponent=True,
        preserve_backend=True,
    )

    assert isinstance(mantissa, torch.Tensor)
    assert isinstance(exponent, torch.Tensor)
    mantissa.square().backward()
    assert all(array.grad is not None for array in source_arrays)
    assert all(torch.isfinite(array.grad).all() for array in source_arrays)
