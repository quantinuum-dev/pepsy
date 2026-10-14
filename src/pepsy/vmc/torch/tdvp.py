"""Fixed-shape PEPS real-time TDVP, reusing the native VMC kernels."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..torch_types import _require_torch
from .amplitude import _unique_config_rows
from .local_energy import _normalized_sample_weights
from .sr import (
    _promote_sr_tensors,
    _resolve_sr_diag_shift,
    _torch_model_parameters,
    _torch_solve_linear,
    apply_torch_sr_update,
    solve_torch_sr,
    torch_log_derivative_matrix,
)

__all__ = ["TorchTDVPResult", "TorchTDVPStepResult", "TorchTDVPDriver",
           "solve_torch_tdvp", "torch_peps_gauge_vectors"]


@dataclass(frozen=True)
class TorchTDVPResult:
    """Parameter velocity and unregularized, centered TDVP residual."""

    velocity: Any
    energy_mean: Any
    energy_variance: Any
    method: str
    info: dict[str, Any]


@dataclass(frozen=True)
class TorchTDVPStepResult:
    """One completed integration step; estimates belong to its RK stages."""

    time: float
    dt: float
    stages: tuple[TorchTDVPResult, ...]
    estimates: tuple[Any, ...]


def torch_peps_gauge_vectors(model, *, parameter_mode="holomorphic"):
    """Analytic bond-gauge generators plus projective scaling for dense PEPS.

    Columns use the same packed holomorphic coordinates as the log Jacobian.
    They may be dependent (plaquette identities or a rank-deficient state).
    The direct solver removes their numerical column space. This dense helper
    is for small parameter-space solves; minSR does not construct it.
    """
    from .amplitude import TorchPEPSAmplitude

    torch = _require_torch()
    if not isinstance(model, TorchPEPSAmplitude) or model.symmray_tensor_ids:
        raise TypeError("Gauge vectors require a dense TorchPEPSAmplitude.")
    params = _torch_model_parameters(model)
    if any(not torch.is_complex(p) for p in params):
        raise ValueError("Real-time holomorphic PEPS requires complex parameters.")
    leaves = model._params_pytree()
    offsets, offset = {}, 0
    for param in params:
        offsets[id(param)] = offset
        offset += param.numel()
    tn = model.skeleton
    if any(id(a) not in offsets for a in leaves.values()):
        raise TypeError("Gauge vectors require one dense parameter per tensor.")
    columns = []
    with torch.no_grad():
        for ind in tn.inner_inds():
            tids = tuple(tn.ind_map[ind])
            if len(tids) != 2:
                raise ValueError("Gauge vectors require pairwise virtual bonds.")
            ta, tb = (tn.tensor_map[tid] for tid in tids)
            a, b = (leaves[tid] for tid in tids)
            ia, ib = ta.inds.index(ind), tb.inds.index(ind)
            d = a.shape[ia]
            for i in range(d):
                for j in range(d):
                    e = a.new_zeros((d, d))
                    e[i, j] = 1
                    da = torch.tensordot(a, e, dims=([ia], [0])).movedim(-1, ia)
                    db = -torch.tensordot(e, b, dims=([1], [ib])).movedim(0, ib)
                    v = a.new_zeros(offset)
                    for p, delta in ((a, da), (b, db)):
                        start = offsets[id(p)]
                        v[start:start + p.numel()] = delta.reshape(-1)
                    columns.append(v)
        v = params[0].new_zeros(offset)
        v[:params[0].numel()] = params[0].reshape(-1)
        columns.append(v)
    vectors = torch.stack(columns, dim=1)
    if parameter_mode == "real-imag":
        real = torch.stack((vectors.real, vectors.imag), dim=1).reshape(2 * offset, -1)
        imag = torch.stack((-vectors.imag, vectors.real), dim=1).reshape(2 * offset, -1)
        return torch.cat((real, imag), dim=1)
    if parameter_mode != "holomorphic":
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    return vectors


def solve_torch_tdvp(log_derivatives, local_energies, *, sample_weights=None,
                     method="auto", diag_shift=1.e-8, pinv_rtol=None,
                     null_vectors=None, parameter_mode="holomorphic"):
    """Solve real-time TDVP for holomorphic or explicit real/imag coordinates.

    Direct TDVP uses centered covariances and optionally projects out analytic
    null vectors. minSR uses the *uncentered* O and E of Wu and Nys, Eq. (17),
    solving (O O^H + shift I)y = -i E, theta_dot = O^H y. It is intended for
    an underdetermined sample system. Repeated configurations and additional
    null modes can still require the existing pseudoinverse fallback.
    Neither the gauge projection nor minSR adds boundary contractions.
    Real/imag coordinates use Re(S) theta_dot = Im(F), with centered direct
    or doubled-real-design minSR. They do not assume holomorphic compression.
    """
    torch = _require_torch()
    if parameter_mode not in {"holomorphic", "real-imag"}:
        raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
    real_imag = parameter_mode == "real-imag"
    o, energy = _promote_sr_tensors(log_derivatives, local_energies)
    dtype = torch.promote_types(o.dtype, (-1j * energy).dtype)
    o, energy = o.to(dtype), energy.to(dtype)
    n_samples, n_params = o.shape
    if not n_samples or not n_params:
        raise ValueError("TDVP requires at least one sample and one parameter.")
    if not bool(torch.isfinite(o).all() and torch.isfinite(energy).all()):
        raise ValueError("TDVP derivatives and local energies must be finite.")
    method = str(method).lower()
    if method == "auto":
        method = "minsr" if (2 if real_imag else 1) * n_samples < n_params else "direct"
    if method not in {"direct", "minsr"}:
        raise ValueError("method must be 'auto', 'direct', or 'minsr'.")
    shift = _resolve_sr_diag_shift(diag_shift, step=0)
    if pinv_rtol is not None:
        pinv_rtol = float(pinv_rtol)
        if not math.isfinite(pinv_rtol) or pinv_rtol <= 0:
            raise ValueError("pinv_rtol must be positive and finite.")
    weights = (torch.full((n_samples,), 1. / n_samples, dtype=o.real.dtype, device=o.device)
               if sample_weights is None else _normalized_sample_weights(
                   sample_weights, n_steps=1, n_chains=n_samples, device=o.device))
    mean = (weights * energy).sum()
    c = o - (weights[:, None] * o).sum(dim=0)
    design = weights.sqrt()[:, None] * c
    rhs = -1j * (design.mH @ (weights.sqrt() * (energy - mean)))
    if real_imag:
        rhs = rhs.real
        design = torch.cat((design.real, design.imag), dim=0)
    info = {"gauge_rank": 0}
    if null_vectors is not None:
        if method != "direct":
            raise ValueError("null_vectors are only used by direct TDVP.")
        vectors = torch.as_tensor(null_vectors, dtype=design.dtype, device=o.device)
        if vectors.ndim != 2 or vectors.shape[0] != n_params:
            raise ValueError("null_vectors must have shape (n_params, n_null).")
        if not bool(torch.isfinite(vectors).all()):
            raise ValueError("null_vectors must be finite.")
        # Rank revelation handles plaquette relations and startup degeneracy
        # without assuming every supplied gauge generator is independent.
        u, s, _ = torch.linalg.svd(vectors, full_matrices=True)
        tolerance = (pinv_rtol if pinv_rtol is not None else
                     max(vectors.shape) * torch.finfo(s.dtype).eps)
        rank = int((s > tolerance * s.max()).sum()) if s.numel() else 0
        q = u[:, rank:]
        reduced = design @ q
        if q.shape[1]:
            system = reduced.mH @ reduced + shift * torch.eye(
                q.shape[1], dtype=reduced.dtype, device=o.device)
            y, solver = _torch_solve_linear(system, q.mH @ rhs, pinv_rtol=pinv_rtol)
            velocity = q @ y
        else:
            velocity, solver = torch.zeros_like(rhs), "empty-gauge-complement"
        info.update(gauge_rank=rank, solver=solver,
                    matrix_shape=(q.shape[1], q.shape[1]))
    else:
        # Multiplying E by -i before the solve fixes the real-time velocity
        # sign. Gauge-projected direct solves never form the unreduced QGT.
        weighted = solve_torch_sr(o, -1j * energy, sample_weights=weights,
                                  method=method, center=real_imag or method == "direct",
                                  diag_shift=shift, pinv_rtol=pinv_rtol,
                                  parameter_mode=parameter_mode)
        velocity = weighted.direction
        info.update(weighted.info)
    residual = design.mH @ (design @ velocity) - rhs
    norm = rhs.norm()
    info["tdvp_relative_residual"] = float((residual.norm() / norm.clamp_min(
        torch.finfo(norm.dtype).tiny)).detach().cpu())
    info["tdvp_relative_residual_squared"] = info["tdvp_relative_residual"] ** 2
    # A solved normal equation can hide dynamics outside the tangent span.
    # Keep the complex wavefunction residual even for real coordinates: its
    # imaginary component is physical information lost by Re(S) projection.
    tangent_rhs = -1j * weights.sqrt() * (energy - mean)
    tangent_residual = weights.sqrt() * (c @ velocity.to(c.dtype)) - tangent_rhs
    tangent_norm = tangent_rhs.norm()
    tangent_residual_norm = tangent_residual.norm()
    info["tangent_relative_residual"] = float((tangent_residual_norm / tangent_norm.clamp_min(
        torch.finfo(tangent_norm.dtype).tiny)).detach().cpu())
    info["tangent_residual_norm"] = float(tangent_residual_norm.detach().cpu())
    info["tangent_rhs_norm"] = float(tangent_norm.detach().cpu())
    info["diag_shift"] = shift
    info["parameter_mode"] = parameter_mode
    return TorchTDVPResult(velocity, mean, (weights * (energy - mean).abs().square()).sum(),
                           method, info)


class TorchTDVPDriver:
    """Real-time Euler or midpoint integration around an existing VMC driver.

    Tensor shapes, tags and bond dimensions are unchanged. Each RK stage
    resamples from its own state; supplied proposal probabilities instead
    recompute importance weights at that stage. Fixed supplied weights must
    describe the current stage and are consequently supported only by Euler.
    """

    def __init__(self, vmc, *, dt, t0=0., integrator="midpoint", method="auto",
                 diag_shift=1.e-8, pinv_rtol=None, gauge_projection=True,
                 derivative_backend="auto", parameter_mode="holomorphic"):
        from .driver import TorchVMCDriver

        if not isinstance(vmc, TorchVMCDriver):
            raise TypeError("vmc must be a TorchVMCDriver.")
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("dt must be positive and finite.")
        if not math.isfinite(float(t0)):
            raise ValueError("t0 must be finite.")
        if integrator not in {"euler", "midpoint"}:
            raise ValueError("integrator must be 'euler' or 'midpoint'.")
        if method not in {"auto", "direct", "minsr"}:
            raise ValueError("method must be 'auto', 'direct', or 'minsr'.")
        if parameter_mode not in {"holomorphic", "real-imag"}:
            raise ValueError("parameter_mode must be 'holomorphic' or 'real-imag'.")
        from .amplitude import TorchPEPSAmplitude

        if (parameter_mode == "holomorphic" and isinstance(vmc.model, TorchPEPSAmplitude)
                and vmc.model.contraction != "exact"):
            # Compression can depend on conjugate tensor entries even with
            # zero cutoff; a rank cap still truncates the complex SVD.
            raise ValueError("Compressed PEPS TDVP requires parameter_mode='real-imag'.")
        if any(not _require_torch().is_complex(p)
               for p in _torch_model_parameters(vmc.model)):
            raise ValueError("Real-time holomorphic TDVP requires complex parameters.")
        self.vmc, self.dt, self.time = vmc, float(dt), float(t0)
        self.integrator, self.method = integrator, method
        self.diag_shift = _resolve_sr_diag_shift(diag_shift, step=0)
        self.pinv_rtol, self.gauge_projection = pinv_rtol, bool(gauge_projection)
        self.derivative_backend = derivative_backend
        self.parameter_mode = parameter_mode

    def _stage(self, *, sampling, samples, weights, proposal_log_probs):
        if samples is None:
            samples = self.vmc.sample(sampling=sampling)
        estimate = self.vmc.step(samples=samples, weights=weights,
                                 proposal_log_probs=proposal_log_probs)
        torch = _require_torch()
        unique, inverse = _unique_config_rows(estimate.configs)
        energies, solve_weights = estimate.local_energies, estimate.importance_weights
        if inverse is not None:
            original_weights = (torch.full_like(energies.real, 1. / energies.numel())
                                if solve_weights is None else solve_weights)
            solve_weights = original_weights.new_zeros(unique.shape[0])
            solve_weights.index_add_(0, inverse, original_weights)
            sums = energies.new_zeros(unique.shape[0])
            sums.index_add_(0, inverse, original_weights * energies)
            positive = solve_weights > 0
            unique = unique[positive]
            solve_weights = solve_weights[positive]
            energies = sums[positive] / solve_weights
        o = torch_log_derivative_matrix(self.vmc.model, unique,
                                        derivative_backend=self.derivative_backend,
                                        complex_parameter_mode=self.parameter_mode)
        method = self.method
        if method == "auto":
            factor = 2 if self.parameter_mode == "real-imag" else 1
            method = "minsr" if factor * o.shape[0] < o.shape[1] else "direct"
        vectors = None
        if method == "direct" and self.gauge_projection:
            from .amplitude import TorchPEPSAmplitude
            if isinstance(self.vmc.model, TorchPEPSAmplitude):
                vectors = torch_peps_gauge_vectors(self.vmc.model, parameter_mode=self.parameter_mode)
        result = solve_torch_tdvp(o, energies,
                                  sample_weights=solve_weights,
                                  method=method, diag_shift=self.diag_shift,
                                  pinv_rtol=self.pinv_rtol, null_vectors=vectors,
                                  parameter_mode=self.parameter_mode)
        result.info.update(n_samples=estimate.configs.shape[0], n_unique_configs=unique.shape[0])
        return result, estimate

    def step(self, *, sampling=None, samples=None, weights=None, proposal_log_probs=None):
        """Advance one dt, restoring tensor values if a stage fails."""
        if samples is not None and sampling is not None:
            raise ValueError("Pass either sampling or samples.")
        if samples is None and (weights is not None or proposal_log_probs is not None):
            raise ValueError("weights and proposal_log_probs require samples.")
        sample_weights = weights if weights is not None else getattr(samples, "weights", None)
        if self.integrator == "midpoint" and sample_weights is not None:
            raise ValueError("Midpoint requires resampling or proposal_log_probs, not fixed weights.")
        sample_proposals = (proposal_log_probs if proposal_log_probs is not None else
                            getattr(samples, "proposal_log_probs", None))
        if self.integrator == "midpoint" and samples is not None and sample_proposals is None:
            raise ValueError("Midpoint requires proposal_log_probs for supplied samples.")
        torch = _require_torch()
        params = _torch_model_parameters(self.vmc.model)
        saved = [p.detach().clone() for p in params]

        def restore():
            with torch.no_grad():
                for p, value in zip(params, saved, strict=True):
                    p.copy_(value)

        kwargs = dict(sampling=sampling, samples=samples, weights=weights,
                      proposal_log_probs=proposal_log_probs)
        try:
            first, estimate = self._stage(**kwargs)
            stages, estimates = [first], [estimate]
            if self.integrator == "midpoint":
                apply_torch_sr_update(self.vmc.model, -first.velocity,
                                      learning_rate=self.dt / 2, parameter_mode=self.parameter_mode)
                self.vmc.refresh_amplitudes()
                second, estimate = self._stage(**kwargs)
                stages.append(second)
                estimates.append(estimate)
                restore()
            apply_torch_sr_update(self.vmc.model, -stages[-1].velocity,
                                  learning_rate=self.dt, parameter_mode=self.parameter_mode)
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise RuntimeError("TDVP update produced non-finite parameters.")
            self.vmc.refresh_amplitudes()
        except Exception:
            restore()
            self.vmc.refresh_amplitudes()
            raise
        self.time += self.dt
        return TorchTDVPStepResult(self.time, self.dt, tuple(stages), tuple(estimates))
