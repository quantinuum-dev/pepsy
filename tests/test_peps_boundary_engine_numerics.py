"""Exact-reference checks for the three PEPS sweep boundary configurations."""

import numpy as np
import pytest
import quimb.tensor as qtn

from pepsy.boundary import peps_infidelity, peps_norm
from pepsy.optimizers.peps import PepsOptimizer


@pytest.mark.parametrize(
    "engine,fit_mode", [("quimb-mps", "eff"), ("dmrg", "dmrg"), ("dmrg", "dmrg2")]
)
def test_entangled_peps_boundary_engine_sweep_matches_exact_fidelity(engine, fit_mode):
    """Real NLopt cleanup improves a two-gate target and preserves ownership."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nlopt")
    state = qtn.PEPS.rand(3, 3, bond_dim=2, dtype="complex128", seed=311)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    state.apply_to_arrays(lambda data: torch.tensor(data, dtype=torch.complex128))
    original = state.to_dense().clone()
    zz = torch.tensor([1., -1., -1., 1.], dtype=torch.float64)
    gates = [
        (torch.diag(torch.exp(-0.37j * zz)), ((0, 0), (0, 1))),
        (torch.diag(torch.exp(-0.23j * zz)), ((1, 1), (2, 1))),
    ]
    target = state.copy()
    for gate, where in gates:
        target.gate_(gate, where, contract="split", cutoff=0.)
    target_vector = target.to_dense().reshape(-1)

    optimizer = PepsOptimizer(
        state, gates, chi=2,
        boundary_engine=engine, fit_mode=fit_mode,
        boundary_chi=(8, 10), normalize_chi=(32, 48), evaluation_chi=(32, 48),
        contraction_opt="greedy", fit_timing=True,
    )
    # Keep the real default LD_LBFGS budget (50) and four round trips per axis.
    output = optimizer.run(infidelity_tol=0., progress=False)
    records = optimizer.get_step_records()
    assert len(records) == 1
    record = records[0]
    assert record["batch_size"] == 2
    assert record["optimizer_attempted"]
    vector = output.to_dense().reshape(-1)
    norm = torch.vdot(vector, vector).real
    target_norm = torch.vdot(target_vector, target_vector).real
    exact_infidelity = float(
        1 - torch.vdot(vector, target_vector).abs() ** 2 / (norm * target_norm)
    )
    assert exact_infidelity < record["pre_infidelity"] - 1e-3
    assert record["final_infidelity"] == pytest.approx(exact_infidelity, abs=1e-10)
    assert float(norm) == pytest.approx(1., abs=1e-10)
    assert output.max_bond() <= 2
    assert torch.equal(state.to_dense(), original)
    assert all(tensor.data.dtype == torch.complex128 for tensor in output)
    diagnostics = optimizer.get_fit_diagnostics()
    if engine == "quimb-mps":
        assert not diagnostics
    else:
        assert diagnostics
        assert all(item.status == "complete" for item in diagnostics)
        assert all(item.fit_mode == ("eff" if fit_mode == "dmrg" else "dmrg2")
                   for item in diagnostics)
        if fit_mode == "dmrg2":
            assert any(item.adaptive_sweeps == 2 and item.one_site_refinement_sweeps == 8
                       for item in diagnostics)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "method,fit_mode", [("mps", "eff"), ("dmrg", "dmrg"), ("dmrg", "dmrg2")]
)
def test_boundary_metrics_match_dense_four_by_four_reference(method, fit_mode, axis):
    """Both contraction directions recover exact norm/overlap at sufficient cap."""
    state = qtn.PEPS.rand(4, 4, bond_dim=2, dtype="complex128", seed=317)
    state.multiply_(1 / np.linalg.norm(state.to_dense()))
    gate = np.diag(np.exp(-0.31j * np.array([1., -1., -1., 1.])))
    target = state.gate(gate, ((1, 1), (1, 2)), contract="split", cutoff=0.)
    vector = state.to_dense().reshape(-1)
    target_vector = target.to_dense().reshape(-1)
    exact_infidelity = 1 - abs(np.vdot(vector, target_vector)) ** 2 / (
        np.vdot(vector, vector).real * np.vdot(target_vector, target_vector).real
    )
    options = dict(
        chi=64, method=method, fit_mode=fit_mode, fit_init_strategy="guess-src",
        fit_init_seed=0, n_iter=10, contraction_opt="greedy", direction=axis,
        max_separation=0, strip_exponent=True, progress=False,
    )
    if method == "mps":
        options["sequence"] = (axis + "min",)
    mantissa, exponent = peps_norm(state, **options)
    assert complex(mantissa * 10 ** exponent) == pytest.approx(1., abs=1e-10)
    result = peps_infidelity(state, target, norm_target=1., **options)
    assert result["infidelity"] == pytest.approx(exact_infidelity, abs=1e-10)
