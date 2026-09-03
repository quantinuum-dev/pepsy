"""Tests for public backend conversion helpers."""

import warnings

import numpy as np
import pytest
import quimb.tensor as qtn

import pepsy


def _available_torch_devices():
    """Return Torch devices available to backend integration tests."""
    try:
        import torch
    except ImportError:
        return ["cpu"]

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            devices.append("mps")
    return devices


def test_to_float_is_public_backend_helper():
    assert pepsy.to_float is pepsy.backends.to_float


def test_backend_infer_is_available_at_the_high_level_and_for_mps():
    """The shared backend contract accepts arrays and tensor networks."""
    assert pepsy.backend_infer is pepsy.backends.backend_infer

    array_info = pepsy.backend_infer(np.ones(2, dtype=np.complex128))
    assert array_info["backend"] == "numpy"
    assert array_info["dtype"] == "complex128"

    mps_info = pepsy.backend_infer(
        qtn.MPS_computational_state("00", dtype="complex128")
    )
    assert mps_info["backend"] == "numpy"
    assert mps_info["dtype"] == "complex128"


@pytest.mark.parametrize("device", _available_torch_devices())
def test_torch_backend_and_linalg_on_available_devices(device):
    """Backend inference and native linalg agree on CPU/CUDA/MPS devices."""
    torch = pytest.importorskip("torch")

    dtype = torch.float64 if device == "cpu" else torch.float32
    to_backend = pepsy.backend_torch(device=device, dtype=dtype)
    matrix = to_backend(np.arange(12, dtype=np.float64).reshape(4, 3))
    info = pepsy.backend_infer(matrix)
    assert info == {
        "backend": "torch",
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(matrix.device),
    }

    try:
        q, r = torch.linalg.qr(matrix)
        _, sigma, _ = torch.linalg.svd(matrix, full_matrices=False)
    except (RuntimeError, NotImplementedError) as exc:
        if device == "cpu":
            raise
        pytest.skip(f"Torch {device} linalg is unavailable: {exc}")

    assert q.device == matrix.device
    assert r.device == matrix.device
    assert sigma.device == matrix.device


def test_torch_float_backend_drops_only_zero_imaginary_part_without_warning():
    """Real Symmray blocks with complex containers stay cleanly float64."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(dtype=torch.float64, device="cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = to_backend(np.array([1.0 + 0.0j, -2.0 + 0.0j]))
    assert out.dtype is torch.float64
    assert not [warning for warning in caught if "discards the imaginary part" in str(warning.message)]


def test_torch_backend_preserves_graph_when_dtype_casting_existing_tensor():
    """The default converter mode keeps graph-bearing tensors connected."""
    torch = pytest.importorskip("torch")
    to_backend = pepsy.backend_torch(dtype=torch.complex128, device="cpu")
    value = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)

    converted = to_backend(value)
    loss = converted.real.square().sum()
    loss.backward()

    assert converted.dtype is torch.complex128
    torch.testing.assert_close(
        value.grad,
        torch.tensor([4.0, 6.0], dtype=torch.float64),
    )


def test_to_float_handles_backend_scalar_without_numpy_coercion():
    class BackendScalar:
        shape = ()

        def detach(self):
            return self

        def cpu(self):
            return self

        def item(self):
            return 1.25

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("NumPy conversion should not be used.")

    assert pepsy.to_float(BackendScalar()) == pytest.approx(1.25)


def test_register_torch_svd_for_autoray():
    """The opt-in Torch SVD registration enables Pepsy's robust autoray SVD."""
    torch = pytest.importorskip("torch")
    import autoray as ar

    pepsy.reg_rel_svd_torch()
    svd_fn = ar.get_lib_fn("torch", "linalg.svd")
    assert svd_fn.__module__ == "pepsy.backends.linalg_torch"

    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    u, s, vh = ar.do("linalg.svd", matrix)
    assert u.shape == (2, 2)
    assert s.shape == (2,)
    assert vh.shape == (2, 2)


def test_stabilized_svd_autoray_accepts_thin_keyword_and_rejects_full():
    """The custom SVD adapter accepts normal thin-SVD Autoray calls."""
    torch = pytest.importorskip("torch")
    import autoray as ar

    try:
        pepsy.TorchLinalgConfig(stabilized=True).register()
        matrix = torch.randn(4, 3, dtype=torch.float64)
        u, s, vh = ar.do("linalg.svd", matrix, full_matrices=False)
        assert u.shape == (4, 3)
        assert s.shape == (3,)
        assert vh.shape == (3, 3)
        with pytest.raises(NotImplementedError, match="thin SVD"):
            ar.do("linalg.svd", matrix, full_matrices=True)
        pepsy.TorchLinalgConfig(mode="real", stabilized=True).register()
        with pytest.raises(TypeError, match="real Torch tensor"):
            ar.do(
                "linalg.svd",
                torch.randn(4, 3, dtype=torch.complex128),
                full_matrices=False,
            )
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_torch_linalg_config_is_public_and_reports_runtime():
    """The structured Torch policy exposes resolved and runtime information."""
    torch = pytest.importorskip("torch")
    config = pepsy.TorchLinalgConfig(
        mode="complex",
        stabilized=False,
        cpu_svd="torch",
    )
    assert config.svd_driver == "gesvd"
    assert config.resolved_svd_fallback == "none"
    assert config.approximate is False
    assert config.exact is True
    info = config.describe()
    assert info["torch_version"] == torch.__version__
    assert info["cuda_available"] is False or isinstance(
        info["cuda_available"], bool
    )
    assert info["cpu_svd"] == "torch"


def test_torch_linalg_config_rejects_unacknowledged_approximate_driver():
    """The approximate CUDA driver must be an explicit user decision."""
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="approximate"):
        pepsy.TorchLinalgConfig(svd_driver="gesvda")

    config = pepsy.TorchLinalgConfig(
        svd_driver="gesvda",
        allow_approximate=True,
    )
    assert config.approximate is True
    assert config.exact is False


@pytest.mark.parametrize("cpu_svd", ["scipy_gesdd", "scipy_gesvd"])
def test_torch_linalg_config_cpu_exact_scipy_complex64(cpu_svd):
    """Exact SciPy CPU choices preserve complex64 reconstruction."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import autoray as ar

    config = pepsy.TorchLinalgConfig(
        stabilized=False,
        cpu_svd=cpu_svd,
    )
    assert config.exact is True
    try:
        config.register()
        matrix = torch.randn(7, 4, dtype=torch.complex64)
        u, s, vh = ar.do("linalg.svd", matrix)
        assert u.dtype is torch.complex64
        assert s.dtype is torch.float32
        assert vh.dtype is torch.complex64
        reconstructed = u @ torch.diag_embed(s.to(matrix.dtype)) @ vh
        torch.testing.assert_close(reconstructed, matrix, rtol=2e-5, atol=2e-5)
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_native_svd_policy_passes_explicit_cuda_driver(monkeypatch):
    """The configured CUDA driver reaches Torch without requiring a GPU test."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch

    calls = []

    class FakeCudaTensor:
        device = torch.device("cuda")
        is_cuda = True
        requires_grad = False

    def fake_svd(array, *args, **kwargs):
        calls.append((array, args, kwargs))
        return "u", "s", "vh"

    monkeypatch.setattr(torch.linalg, "svd", fake_svd)
    result = linalg_torch._native_svd_configured(  # pylint: disable=protected-access
        FakeCudaTensor(),
        driver="gesvdj",
    )
    assert result == ("u", "s", "vh")
    assert calls[0][2]["driver"] == "gesvdj"
    assert calls[0][2]["full_matrices"] is False


def test_torch_linalg_config_cpu_scipy_stabilized_path_keeps_backward():
    """An explicit SciPy CPU forward can still use the stabilized VJP."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import autoray as ar

    config = pepsy.TorchLinalgConfig(
        mode="real",
        stabilized=True,
        cpu_svd="scipy_gesdd",
    )
    try:
        config.register()
        matrix = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        u, s, vh = ar.do("linalg.svd", matrix)
        assert u.shape == (5, 3)
        assert s.shape == (3,)
        assert vh.shape == (3, 3)
        (u.square().sum() + s.sum() + vh.square().sum()).backward()
        assert torch.isfinite(matrix.grad).all()
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_torch_linalg_config_native_scipy_path_rejects_autodiff():
    """Native registration never silently drops gradients on SciPy output."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import autoray as ar

    config = pepsy.TorchLinalgConfig(
        stabilized=False,
        cpu_svd="scipy_gesvd",
    )
    try:
        config.register()
        matrix = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        with pytest.raises(RuntimeError, match="forward-only"):
            ar.do("linalg.svd", matrix)
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_torch_linalg_config_activated_restores_previous_policy():
    """Scoped policy activation restores the prior native configuration."""
    pytest.importorskip("torch")
    native = pepsy.TorchLinalgConfig()
    stabilized = pepsy.TorchLinalgConfig(stabilized=True)
    try:
        native.register()
        with stabilized.activated():
            assert pepsy.get_torch_linalg_config() == stabilized
        assert pepsy.get_torch_linalg_config() == native
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_torch_linalg_registration_is_idempotent(monkeypatch):
    """Repeated public/backend registration does not re-patch Autoray."""
    pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.svd", None)
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        linalg_torch.reg_real_svd_torch()
        linalg_torch.reg_real_svd_torch()
    finally:
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)

    assert len(calls) == 1


def test_torch_linalg_registration_can_switch_svd_modes(monkeypatch):
    """Real and relative SVD modes can intentionally replace one another."""
    pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.svd", None)
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        linalg_torch.reg_real_svd_torch()
        linalg_torch.reg_rel_svd_torch()
        linalg_torch.reg_rel_svd_torch()
    finally:
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)

    assert len(calls) == 2


@pytest.mark.parametrize("shape", ((4, 4), (5, 3), (3, 5), (2, 5, 3)))
def test_torch_real_qr_backward_matches_native(shape):
    """The validated real QR rule matches Torch for all reduced shapes."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_real

    torch.manual_seed(100 + sum(shape))
    matrix = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    q, r = QR_real.apply(matrix)
    dq = torch.randn_like(q)
    dr = torch.randn_like(r)
    actual = torch.autograd.grad((q * dq).sum() + (r * dr).sum(), matrix)[0]

    native_matrix = matrix.detach().clone().requires_grad_()
    native_q, native_r = torch.linalg.qr(native_matrix)
    expected = torch.autograd.grad(
        (native_q * dq).sum() + (native_r * dr).sum(),
        native_matrix,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("shape", ((4, 4), (5, 3), (3, 5), (2, 5, 3)))
def test_torch_complex_qr_backward_matches_native(shape):
    """The explicit complex QR wrapper preserves Torch's conjugate VJP."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_complex

    torch.manual_seed(200 + sum(shape))
    matrix = torch.randn(*shape, dtype=torch.complex128)
    matrix = matrix + 1j * torch.randn(*shape, dtype=torch.complex128)
    matrix.requires_grad_()
    q, r = QR_complex.apply(matrix)
    dq = torch.randn_like(q) + 1j * torch.randn_like(q)
    dr = torch.randn_like(r) + 1j * torch.randn_like(r)
    actual = torch.autograd.grad(
        (q.conj() * dq).real.sum() + (r.conj() * dr).real.sum(),
        matrix,
    )[0]

    native_matrix = matrix.detach().clone().requires_grad_()
    native_q, native_r = torch.linalg.qr(native_matrix)
    expected = torch.autograd.grad(
        (native_q.conj() * dq).real.sum() + (native_r.conj() * dr).real.sum(),
        native_matrix,
    )[0]

    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert torch.isfinite(actual).all()


def test_torch_complex_qr_wrapper_passes_gradcheck():
    """Complex QR remains locally differentiable away from rank loss."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_complex

    torch.manual_seed(250)
    matrix = torch.randn(3, 2, dtype=torch.complex128)
    matrix = (matrix + 1j * torch.randn_like(matrix)).requires_grad_()
    dq = torch.randn(3, 2, dtype=torch.complex128)
    dr = torch.randn(2, 2, dtype=torch.complex128)

    def loss(value):
        q, r = QR_complex.apply(value)
        return (q.conj() * dq).real.sum() + (r.conj() * dr).real.sum()

    assert torch.autograd.gradcheck(
        loss,
        (matrix,),
        eps=1.0e-6,
        atol=1.0e-5,
        rtol=1.0e-4,
    )


def test_torch_real_qr_rank_deficient_uses_regularized_backward():
    """Rank-deficient real QR warns and keeps a finite useful VJP."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_real

    torch.manual_seed(275)
    matrix = torch.randn(4, 3, dtype=torch.float64)
    matrix[:, 1] = matrix[:, 0]
    matrix.requires_grad_()

    with pytest.warns(RuntimeWarning, match="rank-deficient"):
        q, r = QR_real.apply(matrix)
        dq = torch.randn_like(q)
        dr = torch.randn_like(r)
        actual = torch.autograd.grad((q * dq).sum() + (r * dr).sum(), matrix)[0]
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual) > 0


def test_stabilized_qr_autoray_accepts_reduced_keyword_and_guards_dtype():
    """The custom QR adapter accepts reduced mode and rejects complex inputs."""
    torch = pytest.importorskip("torch")
    import autoray as ar

    try:
        pepsy.TorchLinalgConfig(mode="real", stabilized=True).register()
        matrix = torch.randn(4, 3, dtype=torch.float64)
        q, r = ar.do("linalg.qr", matrix, mode="reduced")
        assert q.shape == (4, 3)
        assert r.shape == (3, 3)
        with pytest.raises(TypeError, match="real Torch tensor"):
            ar.do(
                "linalg.qr",
                torch.randn(4, 3, dtype=torch.complex128),
                mode="reduced",
            )
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_stabilized_complex_qr_autoray_regularizes_rank_deficiency():
    """The stabilized complex Autoray QR path uses a finite safe VJP."""
    torch = pytest.importorskip("torch")
    import autoray as ar

    try:
        pepsy.TorchLinalgConfig(mode="complex", stabilized=True).register()
        matrix = torch.tensor(
            ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
            dtype=torch.complex128,
            requires_grad=True,
        )
        with pytest.warns(RuntimeWarning, match="rank-deficient"):
            q, r = ar.do("linalg.qr", matrix, mode="reduced")
        gradient = torch.autograd.grad(
            (q.conj() * torch.ones_like(q)).real.sum()
            + (r.conj() * torch.ones_like(r)).real.sum(),
            matrix,
        )[0]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    finally:
        pepsy.reset_linalg_registrations(backend="torch")


def test_torch_real_qr_safe_regularizes_rank_deficient_gauge():
    """The PEPS block-QR rule keeps a useful finite singular-pivot VJP."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_real_safe

    matrix = torch.randn(4, 3, dtype=torch.float64)
    matrix[:, 1] = matrix[:, 0]
    matrix.requires_grad_()
    q, r = QR_real_safe.apply(matrix)
    gradient = torch.autograd.grad(q.sum() + r.sum(), matrix)[0]

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("complex_input", (False, True))
def test_torch_qr_safe_regularizes_small_nonzero_pivot(monkeypatch, complex_input):
    """The relative QR epsilon also protects a near-singular nonzero pivot."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch

    dtype = torch.complex128 if complex_input else torch.float64
    qr = (
        linalg_torch.QR_complex_safe
        if complex_input
        else linalg_torch.QR_real_safe
    )
    matrix = torch.tensor(
        ((1.0, 1.0, 1.0), (0.0, 1.0e-8, 1.0)),
        dtype=dtype,
        requires_grad=True,
    )
    calls = []
    original_backward = linalg_torch._regularized_qr_backward

    def traced_backward(*args):
        calls.append(args[-1].detach().clone())
        return original_backward(*args)

    monkeypatch.setattr(linalg_torch, "_regularized_qr_backward", traced_backward)
    q, r = qr.apply(matrix)
    gradient = torch.autograd.grad(
        (q.conj() * torch.ones_like(q)).real.sum()
        + (r.conj() * torch.ones_like(r)).real.sum(),
        matrix,
    )[0]

    assert len(calls) == 1
    assert bool(calls[0].all())
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("complex_input", (False, True))
def test_torch_qr_safe_zero_block_has_explicit_zero_vjp(complex_input):
    """An all-zero QR block has no intrinsic scale or preferred gauge."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_complex_safe, QR_real_safe

    dtype = torch.complex128 if complex_input else torch.float64
    qr = QR_complex_safe if complex_input else QR_real_safe
    matrix = torch.zeros(4, 3, dtype=dtype, requires_grad=True)
    q, r = qr.apply(matrix)
    dq = torch.ones_like(q)
    dr = torch.ones_like(r)
    gradient = torch.autograd.grad(
        (q.conj() * dq).real.sum() + (r.conj() * dr).real.sum(),
        matrix,
    )[0]

    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("complex_input", (False, True))
def test_torch_qr_safe_regularized_vjp_preserves_zero_pivot_reconstruction(
    complex_input,
):
    """A zero unpivoted QR diagonal retains the gradient of ``Q @ R``."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_complex_safe, QR_real_safe

    dtype = torch.complex128 if complex_input else torch.float64
    qr = QR_complex_safe if complex_input else QR_real_safe
    matrix = torch.tensor(
        ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=dtype,
        requires_grad=True,
    )
    if complex_input:
        with torch.no_grad():
            matrix[1, 2] += 0.25j
    q, r = qr.apply(matrix)
    torch.testing.assert_close(q @ r, matrix)
    expected = torch.randn_like(matrix)
    loss = (expected.conj() * (q @ r)).real.sum()
    gradient = torch.autograd.grad(loss, matrix)[0]

    torch.testing.assert_close(gradient, expected, rtol=1.0e-5, atol=1.0e-7)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("complex_input", (False, True))
def test_torch_qr_safe_regularizes_mixed_batched_zero_pivots(complex_input):
    """Regularizing one batch member does not change its full-rank neighbor."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import QR_complex_safe, QR_real_safe

    dtype = torch.complex128 if complex_input else torch.float64
    qr = QR_complex_safe if complex_input else QR_real_safe
    matrix = torch.tensor(
        (
            ((1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
            ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        ),
        dtype=dtype,
        requires_grad=True,
    )
    if complex_input:
        with torch.no_grad():
            matrix[1, 1, 2] += 0.25j
    q, r = qr.apply(matrix)
    expected = torch.randn_like(matrix)
    loss = (expected.conj() * (q @ r)).real.sum()
    gradient = torch.autograd.grad(loss, matrix)[0]

    torch.testing.assert_close(gradient, expected, rtol=1.0e-5, atol=1.0e-7)
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("complex_input", (False, True))
@pytest.mark.parametrize("case", ("zero", "repeated", "rank_deficient"))
def test_torch_svd_degenerate_inputs_have_finite_gradients(case, complex_input):
    """Regularized SVD gradients stay finite for singular edge cases."""
    torch = pytest.importorskip("torch")
    from pepsy.backends.linalg_torch import SVD, SVD_real

    real_dtype = torch.float64
    if case == "zero":
        matrix = torch.zeros(3, 3, dtype=real_dtype)
    elif case == "repeated":
        matrix = torch.diag(torch.tensor((2.0, 2.0, 0.5), dtype=real_dtype))
    else:
        matrix = torch.tensor(
            ((1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 1.0, 1.0)),
            dtype=real_dtype,
        )
    if complex_input:
        matrix = matrix.to(torch.complex128)
        matrix = matrix + 0.1j * torch.eye(3, dtype=torch.complex128)
    matrix.requires_grad_()

    torch.manual_seed(300 + len(case) + int(complex_input))
    svd = SVD if complex_input else SVD_real
    u, sigma, vh = svd.apply(matrix)
    gu = torch.randn_like(u)
    gsigma = torch.randn_like(sigma)
    gvh = torch.randn_like(vh)
    loss = (
        (u.conj() * gu).real.sum()
        + (sigma * gsigma).real.sum()
        + (vh.conj() * gvh).real.sum()
    )
    gradient = torch.autograd.grad(loss, matrix)[0]

    assert torch.isfinite(gradient).all()


def test_quimb_torch_split_drivers_stabilize_real_symmray_blocks():
    """Quimb's composed Torch split path is stable and lossless."""
    torch = pytest.importorskip("torch")
    qd = pytest.importorskip("quimb.tensor.decomp")
    from pepsy.backends import config

    # The second diagonal of R is structurally zero, while its row has a
    # nonzero later component.  The stabilized phase must be one, rather than
    # zero, at that position or QR no longer reconstructs this matrix.
    matrix = torch.tensor(
        ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=torch.float64,
        requires_grad=True,
    )
    try:
        config.register_torch_linalg(
            mode="real",
            stabilized=True,
            quimb_split_drivers=True,
        )
        q, _, r = qd.qr_stabilized(matrix)
        torch.testing.assert_close(q @ r, matrix)
        qr_gradient = torch.autograd.grad(q.sum() + r.sum(), matrix, retain_graph=True)[0]
        expected_gradient = torch.randn_like(matrix)
        reconstruction_gradient = torch.autograd.grad(
            (expected_gradient * (q @ r)).sum(),
            matrix,
            retain_graph=True,
        )[0]
        u, s, vh = qd.svd_truncated(matrix, absorb=None)
        svd_gradient = torch.autograd.grad(
            u.sum() + s.sum() + vh.sum(),
            matrix,
        )[0]
        assert torch.isfinite(qr_gradient).all()
        assert torch.count_nonzero(qr_gradient) > 0
        torch.testing.assert_close(reconstruction_gradient, expected_gradient)
        assert torch.isfinite(svd_gradient).all()
    finally:
        qd.qr_stabilized.register("torch", qd.qr_stabilized._default_fn)
        qd.svd_truncated.register("torch", qd.svd_truncated._default_fn)


def test_quimb_torch_split_drivers_stabilize_complex_rank_deficient_block():
    """The complex split path is stable and exact for a zero QR diagonal."""
    torch = pytest.importorskip("torch")
    qd = pytest.importorskip("quimb.tensor.decomp")
    from pepsy.backends import config

    matrix = torch.tensor(
        ((1.0, 1.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=torch.complex128,
        requires_grad=True,
    )
    try:
        config.register_torch_linalg(
            mode="complex",
            stabilized=True,
            quimb_split_drivers=True,
        )
        q, _, r = qd.qr_stabilized(matrix)
        torch.testing.assert_close(q @ r, matrix)
        qr_gradient = torch.autograd.grad(
            (q.conj() * torch.ones_like(q)).real.sum()
            + (r.conj() * torch.ones_like(r)).real.sum(),
            matrix,
            retain_graph=True,
        )[0]
        expected_gradient = torch.randn_like(matrix)
        reconstruction_gradient = torch.autograd.grad(
            (expected_gradient.conj() * (q @ r)).real.sum(),
            matrix,
            retain_graph=True,
        )[0]
        u, s, vh = qd.svd_truncated(matrix, absorb=None)
        svd_gradient = torch.autograd.grad(
            (u.conj() * torch.ones_like(u)).real.sum()
            + s.sum()
            + (vh.conj() * torch.ones_like(vh)).real.sum(),
            matrix,
        )[0]
        assert torch.isfinite(qr_gradient).all()
        assert torch.count_nonzero(qr_gradient) > 0
        torch.testing.assert_close(reconstruction_gradient, expected_gradient)
        assert torch.isfinite(svd_gradient).all()
    finally:
        qd.qr_stabilized.register("torch", qd.qr_stabilized._default_fn)
        qd.svd_truncated.register("torch", qd.svd_truncated._default_fn)


def test_torch_complex_qr_registration_uses_native_fallback(monkeypatch):
    """The analytical complex QR rule is not registered through Autoray."""
    torch = pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.qr", None)
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        linalg_torch.reg_complex_qr_torch()
    finally:
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)

    assert len(calls) == 1
    assert calls[0][0][0:2] == ("torch", "linalg.qr")
    assert linalg_torch._same_callable(calls[0][0][2], torch.linalg.qr)


def test_register_torch_linalg_complex_uses_native_defaults(monkeypatch):
    """The default complex umbrella registration keeps native linalg."""
    torch = pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import config, linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.svd", None)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.qr", None)
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        config.register_torch_linalg(mode="complex")
    finally:
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)

    registered = {args[1]: args[2] for args, _kwargs in calls}
    assert linalg_torch._same_callable(registered["linalg.qr"], torch.linalg.qr)
    assert registered["linalg.svd"].__name__ == "native_svd_gesvd_torch"


def test_register_torch_linalg_stabilized_real_is_opt_in(monkeypatch):
    """Stabilized real SVD/QR rules require an explicit opt-in."""
    pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import config, linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    original_policy = linalg_torch._QR_RANK_POLICY
    original_factor = linalg_torch._QR_RANK_TOL_FACTOR
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.svd", None)
    linalg_torch._REGISTERED_FUNCTIONS.pop("linalg.qr", None)
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        config.register_torch_linalg(
            mode="real",
            stabilized=True,
            qr_rank_policy="error",
            qr_rank_tol_factor=2.0,
        )
        registered = {args[1]: args[2] for args, _kwargs in calls}
        assert registered["linalg.svd"].__name__.startswith(
            "SVD_real_stabilized_svd"
        )
        assert registered["linalg.qr"].__name__ == "QR_real_stabilized_qr"
        assert linalg_torch._QR_RANK_POLICY == "error"
        assert linalg_torch._QR_RANK_TOL_FACTOR == 2.0
    finally:
        linalg_torch._QR_RANK_POLICY = original_policy
        linalg_torch._QR_RANK_TOL_FACTOR = original_factor
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)


def test_register_torch_linalg_leaves_quimb_drivers_untouched_without_opt_in(
    monkeypatch,
):
    """The canonical default must not mutate Quimb's process-global drivers."""
    pytest.importorskip("torch")
    from pepsy.backends import config, linalg_torch

    calls = []
    monkeypatch.setattr(
        linalg_torch,
        "reg_quimb_torch_split_drivers",
        lambda **kwargs: calls.append(kwargs),
    )

    config.register_torch_linalg(mode="real", stabilized=True)

    assert calls == []


def test_register_torch_linalg_passes_qr_policy_to_quimb(monkeypatch):
    """Raw-block Quimb drivers receive the configured QR policy and tolerance."""
    pytest.importorskip("torch")
    from pepsy.backends import config, linalg_torch

    calls = []
    monkeypatch.setattr(
        linalg_torch,
        "reg_quimb_torch_split_drivers",
        lambda **kwargs: calls.append(kwargs),
    )

    try:
        config.register_torch_linalg(
            mode="real",
            stabilized=True,
            qr_rank_policy="error",
            qr_rank_tol_factor=3.0,
            quimb_split_drivers=True,
        )

        assert calls[0]["qr_rank_policy"] == "error"
        assert calls[0]["qr_rank_tol_factor"] == 3.0
    finally:
        config.reset_linalg_registrations(backend="torch")


def test_reset_linalg_registrations_restores_native_torch(monkeypatch):
    """The public reset helper restores native Torch mappings."""
    torch = pytest.importorskip("torch")
    import autoray as ar
    from pepsy.backends import config, linalg_torch

    calls = []
    original_registered = dict(linalg_torch._REGISTERED_FUNCTIONS)
    linalg_torch._REGISTERED_FUNCTIONS.clear()
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        config.reset_linalg_registrations(backend="torch")
    finally:
        linalg_torch._REGISTERED_FUNCTIONS.clear()
        linalg_torch._REGISTERED_FUNCTIONS.update(original_registered)

    registered = {args[1]: args[2] for args, _kwargs in calls}
    assert registered["linalg.svd"].__name__ == "native_svd_gesvd_torch"
    assert linalg_torch._same_callable(registered["linalg.qr"], torch.linalg.qr)


def test_torch_real_qr_rank_policy_error_is_strict():
    """The strict QR policy rejects rank-deficient inputs before backward."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch

    original_policy = linalg_torch._QR_RANK_POLICY
    original_factor = linalg_torch._QR_RANK_TOL_FACTOR
    matrix = torch.randn(4, 3, dtype=torch.float64)
    matrix[:, 1] = matrix[:, 0]
    try:
        linalg_torch._configure_qr_rank_policy("error")
        with pytest.raises(RuntimeError, match="rank-deficient"):
            linalg_torch.QR_real.apply(matrix)
    finally:
        linalg_torch._QR_RANK_POLICY = original_policy
        linalg_torch._QR_RANK_TOL_FACTOR = original_factor


def test_torch_real_qr_rank_policy_native_is_silent():
    """The native rank policy falls back without emitting a warning."""
    torch = pytest.importorskip("torch")
    from pepsy.backends import linalg_torch

    original_policy = linalg_torch._QR_RANK_POLICY
    original_factor = linalg_torch._QR_RANK_TOL_FACTOR
    matrix = torch.randn(4, 3, dtype=torch.float64)
    matrix[:, 1] = matrix[:, 0]
    matrix.requires_grad_()
    try:
        linalg_torch._configure_qr_rank_policy("native")
        with warnings.catch_warnings(record=True) as caught:
            q, r = linalg_torch.QR_real.apply(matrix)
            gradient = torch.autograd.grad(q.sum() + r.sum(), matrix)[0]
        assert not caught
        assert torch.isfinite(gradient).all()
    finally:
        linalg_torch._QR_RANK_POLICY = original_policy
        linalg_torch._QR_RANK_TOL_FACTOR = original_factor


def test_jax_linalg_registration_aliases_are_idempotent():
    """JAX real/relative compatibility aliases share one registration."""
    pytest.importorskip("jax")
    from pepsy.backends import linalg_jax

    original_registered = linalg_jax._SVD_REGISTERED
    original_function = linalg_jax._SVD_REGISTERED_FUNCTION
    try:
        linalg_jax._SVD_REGISTERED = False
        linalg_jax._SVD_REGISTERED_FUNCTION = None
        linalg_jax.reg_rel_svd_jax()
        assert linalg_jax._SVD_REGISTERED is True
        linalg_jax.reg_real_svd_jax()
    finally:
        linalg_jax._SVD_REGISTERED = original_registered
        linalg_jax._SVD_REGISTERED_FUNCTION = original_function


def test_jax_linalg_registration_switches_native_and_stabilized(monkeypatch):
    """JAX can explicitly switch between native and truncation-safe SVD."""
    pytest.importorskip("jax")
    import autoray as ar
    from pepsy.backends import linalg_jax

    calls = []
    original_registered = linalg_jax._SVD_REGISTERED
    original_function = linalg_jax._SVD_REGISTERED_FUNCTION
    linalg_jax._SVD_REGISTERED = False
    linalg_jax._SVD_REGISTERED_FUNCTION = None
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        linalg_jax.reg_native_svd_jax()
        linalg_jax.reg_rel_svd_jax()
    finally:
        linalg_jax._SVD_REGISTERED = original_registered
        linalg_jax._SVD_REGISTERED_FUNCTION = original_function

    assert len(calls) == 2
    assert calls[0][0][0:2] == ("jax", "linalg.svd")
    assert calls[0][0][2] is linalg_jax._native_svd_jax
    assert calls[1][0][2] is linalg_jax.svd_jax


def test_register_jax_linalg_defaults_to_native(monkeypatch):
    """The JAX umbrella registration defaults to native thin SVD."""
    pytest.importorskip("jax")
    import autoray as ar
    from pepsy.backends import config, linalg_jax

    calls = []
    original_registered = linalg_jax._SVD_REGISTERED
    original_function = linalg_jax._SVD_REGISTERED_FUNCTION
    linalg_jax._SVD_REGISTERED = False
    linalg_jax._SVD_REGISTERED_FUNCTION = None
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        config.register_jax_linalg()
    finally:
        linalg_jax._SVD_REGISTERED = original_registered
        linalg_jax._SVD_REGISTERED_FUNCTION = original_function

    assert len(calls) == 1
    assert calls[0][0][2] is linalg_jax._native_svd_jax


def test_reset_linalg_registrations_restores_native_jax(monkeypatch):
    """The public reset helper restores native JAX thin SVD."""
    pytest.importorskip("jax")
    import autoray as ar
    from pepsy.backends import config, linalg_jax

    calls = []
    original_registered = linalg_jax._SVD_REGISTERED
    original_function = linalg_jax._SVD_REGISTERED_FUNCTION
    linalg_jax._SVD_REGISTERED = False
    linalg_jax._SVD_REGISTERED_FUNCTION = None
    monkeypatch.setattr(
        ar,
        "register_function",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    try:
        config.reset_linalg_registrations(backend="jax")
    finally:
        linalg_jax._SVD_REGISTERED = original_registered
        linalg_jax._SVD_REGISTERED_FUNCTION = original_function

    assert len(calls) == 1
    assert calls[0][0][2] is linalg_jax._native_svd_jax


def test_to_float_rejects_non_scalar_backend_array_before_numpy_coercion():
    class BackendVector:
        shape = (2,)

        def __array__(self, *_args, **_kwargs):
            raise AssertionError("Non-scalar backend arrays should not coerce.")

    with pytest.raises(TypeError, match="shape"):
        pepsy.to_float(BackendVector())


def test_to_float_uses_real_component_by_default():
    value = np.asarray(2.5 - 1.0j)

    assert pepsy.to_float(value) == pytest.approx(2.5)
    with pytest.raises(TypeError):
        pepsy.to_float(value, real=False)


def test_to_float_handles_torch_scalar_if_available():
    torch = pytest.importorskip("torch")

    value = torch.tensor(3.5 + 1.25j, dtype=torch.complex128, requires_grad=True)

    assert pepsy.to_float(value) == pytest.approx(3.5)


@pytest.mark.parametrize("complex_input", (False, True))
def test_jax_svd_vjp_restores_quimb_truncated_cotangents(complex_input):
    """Fixed-rank contractions must not mix full and truncated SVD axes."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from pepsy.backends.linalg_jax import jaxsvd_bwd, jaxsvd_fwd

    matrix = jnp.arange(64, dtype=jnp.float32).reshape(8, 8) + jnp.eye(8)
    if complex_input:
        matrix = matrix * (1.0 + 0.1j)

    outputs, residual = jaxsvd_fwd(matrix)
    U, S, Vh = outputs
    retained = 2
    u_tangent = 0.25 - 0.1j if complex_input else 0.25
    vh_tangent = -0.2 + 0.3j if complex_input else -0.2
    truncated_tangents = (
        jnp.full_like(U[:, :retained], u_tangent),
        jnp.full_like(S[:retained], 0.5),
        jnp.full_like(Vh[:retained, :], vh_tangent),
    )
    actual = jaxsvd_bwd(residual, truncated_tangents)[0]

    full_tangents = type(outputs)(
        jnp.zeros_like(U).at[:, :retained].set(truncated_tangents[0]),
        jnp.zeros_like(S).at[:retained].set(truncated_tangents[1]),
        jnp.zeros_like(Vh).at[:retained, :].set(truncated_tangents[2]),
    )
    _, pullback = jax.vjp(
        lambda value: jnp.linalg.svd(value, full_matrices=False),
        matrix,
    )
    expected = pullback(full_tangents)[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
