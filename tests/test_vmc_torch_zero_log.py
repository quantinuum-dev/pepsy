"""Zero-weight native fermion configurations through compiled boundary logs."""

import pytest


def test_boundary_log_zero_sector_is_clean_in_eager_and_compiled_calls():
    torch = pytest.importorskip("torch")
    pytest.importorskip("symmray")
    from pepsy.vmc.netket import fermionic_peps_rand
    from pepsy.vmc.torch import TorchPEPSAmplitude

    peps = fermionic_peps_rand("Z2", 2, 2, 2, seed=403, dtype="float64", flat=True)
    model = TorchPEPSAmplitude(peps, contraction="boundary", chi=2, cutoff=0.,
                               dtype=torch.float64, amplitude_batching="serial")
    # Native physical codes 2/3 are singly occupied, 0/1 have even parity.
    configs = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 2]])
    with torch.no_grad():
        phase, log_abs = model.forward_log(configs)
        assert phase[1] == 0
        assert torch.isneginf(log_abs[1])
        assert torch.isfinite(log_abs[0])
        model.export_and_compile(configs, backend="eager")
        compiled_phase, compiled_log = model.compiled_forward_log(configs)
        torch.testing.assert_close(compiled_phase, phase)
        torch.testing.assert_close(compiled_log, log_abs)
