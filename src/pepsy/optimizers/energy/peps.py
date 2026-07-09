"""Finite-PEPS energy objectives and optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import (
    backend_cupy,
    backend_jax,
    backend_numpy,
    backend_torch,
    infer_backend_converter_from_sample,
)
from ...tensors import build_optimizer, reg_rel_svd_jax, reg_rel_svd_torch
from ..global_opt import GlobalOptimizer

__all__ = ["EnergyEstimate", "MpsEnergyOptimizer", "PepsEnergyOptimizer"]


@dataclass(frozen=True)
class EnergyEstimate:
    """Energy evaluation metadata for a finite tensor-network state."""

    energy: Any
    energy_per_site: Any
    num_sites: int
    chi: int | tuple[int, int] | None
    boundary_mode: str
    normalized: bool
    metadata: dict[str, Any] | None = None

    def as_dict(self):
        """Return this estimate as a plain dictionary."""
        return {
            "energy": self.energy,
            "energy_per_site": self.energy_per_site,
            "num_sites": self.num_sites,
            "chi": self.chi,
            "boundary_mode": self.boundary_mode,
            "normalized": self.normalized,
            "metadata": {} if self.metadata is None else dict(self.metadata),
        }


class PepsEnergyOptimizer:
    """Optimize finite PEPS energy with quimb's autodiff TNOptimizer.

    The objective is the normalized local expectation value
    ``<psi|H|psi>/<psi|psi>`` computed by
    ``PEPS.compute_local_expectation(...)``. Hamiltonians can be supplied as a
    ``qtn.LocalHam2D``, its ``.terms`` mapping, or a plain local-term mapping.
    """

    _LOSS_KEYS = frozenset({
        "chi",
        "boundary_mode",
        "cutoff",
        "normalized",
        "energy_per_site",
        "real",
        "stabilize_state",
        "contraction_opt",
        "compute_kwargs",
    })

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        chi: int | tuple[int, int] = 64,
        boundary_mode: str = "mps",
        cutoff: float = 0.0,
        normalized: bool = True,
        energy_per_site: bool = True,
        real: bool = True,
        stabilize_state: bool = False,
        contraction_opt: Any = "auto-hq",
        compute_kwargs: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
    ):
        self.state = self._as_peps_state(state)
        self.hamiltonian = hamiltonian
        self.terms = self._terms_from_hamiltonian(hamiltonian)
        self.losses: list[float] = []
        self.loss_kwargs = {
            "chi": chi,
            "boundary_mode": boundary_mode,
            "cutoff": cutoff,
            "normalized": normalized,
            "energy_per_site": energy_per_site,
            "real": real,
            "stabilize_state": stabilize_state,
            "contraction_opt": contraction_opt,
            "compute_kwargs": {} if compute_kwargs is None else dict(compute_kwargs),
        }
        if loss_kwargs is not None:
            self.set_loss_kwargs(**loss_kwargs)

    @classmethod
    def loss_kwarg_names(cls):
        """Return supported loss keyword names."""
        return tuple(sorted(cls._LOSS_KEYS))

    @staticmethod
    def _merge_opts(base, extra):
        opts = dict(base or {})
        if extra:
            opts.update(dict(extra))
        return opts

    @staticmethod
    def _sample_array(value):
        if value is None:
            return None
        blocks = getattr(value, "blocks", None)
        if blocks:
            try:
                return next(iter(blocks.values()))
            except StopIteration:
                return None
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            return value
        data = getattr(value, "data", None)
        if hasattr(data, "shape") and hasattr(data, "dtype"):
            return data
        return None

    @classmethod
    def _sample_array_from_tn(cls, state):
        tensor_map = getattr(state, "tensor_map", None)
        if tensor_map:
            for tensor in tensor_map.values():
                sample = cls._sample_array(getattr(tensor, "data", None))
                if sample is not None:
                    return sample
        try:
            tensors = iter(state)
        except TypeError:
            return None
        for tensor in tensors:
            sample = cls._sample_array(getattr(tensor, "data", None))
            if sample is not None:
                return sample
        return None

    @classmethod
    def _sample_array_from_terms(cls, terms):
        for term in dict(terms).values():
            sample = cls._sample_array(term)
            if sample is not None:
                return sample
        return None

    @staticmethod
    def _dtype_name(sample):
        if sample is None:
            return None
        try:
            return ar.get_dtype_name(sample)
        except Exception:  # pragma: no cover - defensive for unusual arrays
            dtype = getattr(sample, "dtype", None)
            return None if dtype is None else np.dtype(dtype).name

    @classmethod
    def _autodiff_dtype_name(cls, state, terms):
        dtype_names = [
            name
            for name in (
                cls._dtype_name(cls._sample_array_from_tn(state)),
                cls._dtype_name(cls._sample_array_from_terms(terms)),
            )
            if name is not None
        ]
        if not dtype_names:
            return None
        try:
            return np.result_type(*dtype_names).name
        except TypeError:
            return dtype_names[0]

    @staticmethod
    def _autodiff_backend_converter(backend, dtype_name, *, device="cpu"):
        key = str(backend).strip().lower()
        if dtype_name is None:
            return None
        if key in {"auto", "autograd"}:
            return None
        if key == "numpy":
            return backend_numpy(dtype=np.dtype(dtype_name))
        if key == "torch":
            import torch  # pylint: disable=import-outside-toplevel

            dtype = getattr(torch, dtype_name)
            return backend_torch(device=device, dtype=dtype)
        if key == "jax":
            import jax.numpy as jnp  # pylint: disable=import-outside-toplevel

            dtype = getattr(jnp, dtype_name)
            return backend_jax(device=device, dtype=dtype)
        if key == "cupy":
            import cupy as cp  # pylint: disable=import-outside-toplevel

            dtype = getattr(cp, dtype_name)
            return backend_cupy(dtype=dtype)
        return None

    @staticmethod
    def _copy_array_like(value):
        copy = getattr(value, "copy", None)
        if callable(copy):
            return copy()
        return value

    @classmethod
    def _convert_term_array(cls, term, converter):
        term = cls._copy_array_like(term)
        apply_to_arrays = getattr(term, "apply_to_arrays", None)
        if callable(apply_to_arrays):
            apply_to_arrays(converter)
            return term
        return converter(term)

    @classmethod
    def _convert_state_arrays(cls, state, converter):
        if converter is None:
            return state
        apply_to_arrays = getattr(state, "apply_to_arrays", None)
        if not callable(apply_to_arrays):
            return state
        apply_to_arrays(lambda array: cls._convert_term_array(array, converter))
        return state

    @classmethod
    def _state_for_autodiff_backend(
        cls,
        state,
        reference_state,
        autodiff_backend,
        *,
        device="cpu",
    ):
        dtype_name = cls._dtype_name(cls._sample_array_from_tn(reference_state))
        converter = cls._autodiff_backend_converter(
            autodiff_backend,
            dtype_name,
            device=device,
        )
        return cls._convert_state_arrays(state, converter)

    @classmethod
    def _terms_for_autodiff_backend(cls, terms, state, autodiff_backend, *, device="cpu"):
        dtype_name = cls._autodiff_dtype_name(state, terms)
        converter = cls._autodiff_backend_converter(
            autodiff_backend,
            dtype_name,
            device=device,
        )
        if converter is None:
            return terms
        return {
            edge: cls._convert_term_array(term, converter)
            for edge, term in dict(terms).items()
        }

    @staticmethod
    def _prepare_autodiff_backend(backend):
        key = str(backend).strip().lower()
        try:
            if key == "torch":
                # Boundary-MPS contractions use SVD compression. Torch's native
                # complex SVD backward can fail on gauge/phase-sensitive losses,
                # so install Pepsy's regularized SVD rule. Do not register QR
                # here: rectangular QR appears in quimb boundary compression and
                # has a narrower custom-backward domain.
                reg_rel_svd_torch()
            elif key == "jax":
                reg_rel_svd_jax()
        except ImportError:
            # TNOptimizer will report the missing backend if it is actually
            # needed. Keeping this hook soft preserves optional-dependency tests.
            return

    @staticmethod
    def _is_finite_number(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _initial_gradient_status(cls, tnopt):
        if not (
            hasattr(tnopt, "vectorized_value_and_grad")
            and hasattr(getattr(tnopt, "vectorizer", None), "vector")
        ):
            return True, None
        x0 = tnopt.vectorizer.vector.copy()
        loss0 = None
        try:
            loss0, grad0 = tnopt.vectorized_value_and_grad(x0)
            finite_loss = cls._is_finite_number(loss0)
            finite_grad = bool(np.all(np.isfinite(grad0)))
            return finite_loss and finite_grad, loss0 if finite_loss else None
        except (FloatingPointError, RuntimeError, ValueError):
            return False, loss0 if cls._is_finite_number(loss0) else None
        finally:
            tnopt.vectorizer.vector[:] = x0
            cls._reset_tnopt_tracking(tnopt)

    @staticmethod
    def _reset_tnopt_tracking(tnopt):
        for name in ("losses", "loss_diffs"):
            values = getattr(tnopt, name, None)
            if hasattr(values, "clear"):
                values.clear()
        if hasattr(tnopt, "lgrdm"):
            try:
                tnopt.lgrdm = type(tnopt.lgrdm)()
            except TypeError:
                pass
        for name, value in (
            ("loss", float("inf")),
            ("loss_best", float("inf")),
            ("_n", 0),
        ):
            if hasattr(tnopt, name):
                setattr(tnopt, name, value)

    @classmethod
    def _pick_loss_kwargs(cls, options):
        incoming = dict(options or {})
        unknown = sorted(set(incoming) - cls._LOSS_KEYS)
        if unknown:
            allowed = ", ".join(sorted(cls._LOSS_KEYS))
            raise TypeError(
                f"Unknown PEPS energy option(s): {', '.join(unknown)}. "
                f"Allowed options: {allowed}."
            )
        return incoming

    @staticmethod
    def _as_peps_state(state):
        if hasattr(state, "compute_local_expectation"):
            return state
        peps = getattr(state, "peps", None)
        if peps is not None and hasattr(peps, "compute_local_expectation"):
            return peps
        tn = getattr(state, "tn", None)
        if tn is not None and hasattr(tn, "compute_local_expectation"):
            return tn
        raise TypeError("state must be a PEPS-like object with compute_local_expectation().")

    @classmethod
    def _terms_from_hamiltonian(cls, hamiltonian):
        if hamiltonian is None:
            raise ValueError("hamiltonian is required.")
        if isinstance(hamiltonian, Mapping):
            if "local_terms" in hamiltonian:
                return cls._terms_from_hamiltonian(hamiltonian["local_terms"])
            if "local_ham" in hamiltonian:
                return cls._terms_from_hamiltonian(hamiltonian["local_ham"])
            return hamiltonian
        terms = getattr(hamiltonian, "terms", None)
        if terms is None:
            raise TypeError("hamiltonian must be a LocalHam2D, terms mapping, or payload mapping.")
        return terms

    @staticmethod
    def _num_sites(state):
        num_sites = getattr(state, "num_sites", None)
        if num_sites is not None:
            return int(num_sites() if callable(num_sites) else num_sites)
        if hasattr(state, "Lx") and hasattr(state, "Ly"):
            return int(state.Lx) * int(state.Ly)
        if hasattr(state, "gen_site_coos"):
            return len(tuple(state.gen_site_coos()))
        sites = getattr(state, "sites", None)
        if sites is not None:
            return len(tuple(sites))
        raise ValueError("Could not infer the number of PEPS sites.")

    @staticmethod
    def _boundary_mode(mode):
        key = str(mode).strip().lower().replace("_", "-")
        if key in {"mps", "boundary-mps", "boundary"}:
            return "mps"
        if key in {"projector", "ctmrg", "ctm"}:
            return "projector"
        if key in {"exact", "full", "full-contract", "full-contraction"}:
            return "exact"
        return mode

    @staticmethod
    def _normalization_mode(mode):
        key = str(mode).strip().lower().replace("_", "-")
        if key in {"projector", "ctmrg", "ctm"}:
            return "ctmrg"
        if key in {"mps", "boundary-mps", "boundary"}:
            return "mps"
        return mode

    @staticmethod
    def _maybe_real(value):
        try:
            return ar.do("real", value)
        except Exception:  # pragma: no cover - defensive for unusual scalar types
            return value.real

    @staticmethod
    def _state_uses_symmray(state):
        tensor_map = getattr(state, "tensor_map", None)
        tensors = tensor_map.values() if tensor_map else state
        try:
            iterator = iter(tensors)
        except TypeError:
            return False
        for tensor in iterator:
            data = getattr(tensor, "data", None)
            if data is None:
                continue
            module = type(data).__module__.split(".", maxsplit=1)[0]
            if module == "symmray":
                return True
            if hasattr(data, "blocks") and hasattr(data, "apply_to_arrays"):
                return True
        return False

    @classmethod
    def _stabilize_state(cls, state):
        if hasattr(state, "balance_bonds_") and not cls._state_uses_symmray(state):
            state.balance_bonds_()
        if hasattr(state, "equalize_norms_"):
            state.equalize_norms_(1.0)
        return state

    @classmethod
    def _loss_state(
        cls,
        state,
        *,
        terms,
        chi=64,
        boundary_mode="mps",
        cutoff=0.0,
        normalized=True,
        energy_per_site=True,
        real=True,
        stabilize_state=False,
        contraction_opt="auto-hq",
        compute_kwargs=None,
    ):
        state = cls._as_peps_state(state)
        terms = cls._terms_from_hamiltonian(terms)
        if stabilize_state:
            cls._stabilize_state(state)
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)

        kwargs = dict(compute_kwargs or {})
        mode = cls._boundary_mode(boundary_mode)
        if mode == "exact":
            exact_expectation = getattr(
                state,
                "compute_local_expectation_exact",
                None,
            )
            if not callable(exact_expectation):
                raise TypeError(
                    "boundary_mode='exact' requires a PEPS-like state with "
                    "compute_local_expectation_exact()."
                )
            value = exact_expectation(
                terms,
                optimize=contraction_opt,
                normalized=bool(normalized),
                **kwargs,
            )
        else:
            value = state.compute_local_expectation(
                terms,
                max_bond=chi,
                cutoff=cutoff,
                normalized=bool(normalized),
                mode=mode,
                contract_optimize=contraction_opt,
                **kwargs,
            )
        if energy_per_site:
            value = value / cls._num_sites(state)
        if real:
            value = cls._maybe_real(value)
        return value

    @staticmethod
    def _tnopt_loss(state, *, terms, **loss_kwargs):
        """Adapter for ``qtn.TNOptimizer``."""
        return PepsEnergyOptimizer._loss_state(state, terms=terms, **loss_kwargs)

    def set_loss_kwargs(self, **kwargs):
        """Update stored defaults for energy loss evaluation."""
        self.loss_kwargs.update(self._pick_loss_kwargs(kwargs))
        return self

    def loss(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Evaluate the configured PEPS energy loss."""
        state = self.state if state is None else self._as_peps_state(state)
        terms_use = self.terms
        if hamiltonian is not None:
            terms_use = self._terms_from_hamiltonian(hamiltonian)
        if terms is not None:
            terms_use = self._terms_from_hamiltonian(terms)
        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        return self._loss_state(state, terms=terms_use, **opts)

    def energy(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Return full and per-site energy estimates for ``state``."""
        state = self.state if state is None else self._as_peps_state(state)
        terms_use = self.terms
        if hamiltonian is not None:
            terms_use = self._terms_from_hamiltonian(hamiltonian)
        if terms is not None:
            terms_use = self._terms_from_hamiltonian(terms)

        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        opts_full = dict(opts)
        opts_full["energy_per_site"] = False
        energy = self._loss_state(state, terms=terms_use, **opts_full)
        num_sites = self._num_sites(state)
        energy_per_site = energy / num_sites
        return EnergyEstimate(
            energy=energy,
            energy_per_site=energy_per_site,
            num_sites=num_sites,
            chi=opts["chi"],
            boundary_mode=str(self._boundary_mode(opts["boundary_mode"])),
            normalized=bool(opts["normalized"]),
            metadata={
                "cutoff": opts["cutoff"],
                "real": opts["real"],
                "stabilize_state": opts["stabilize_state"],
                "contraction_opt": opts["contraction_opt"],
            },
        )

    def normalize(self, state=None, **kwargs):
        """Normalize ``state`` in place using Pepsy's global normalization helper."""
        state = self.state if state is None else self._as_peps_state(state)
        opts = {
            "chi": self.loss_kwargs["chi"],
            "mode": self._normalization_mode(self.loss_kwargs["boundary_mode"]),
            "cutoff": self.loss_kwargs["cutoff"],
            "opt": self.loss_kwargs["contraction_opt"],
        }
        opts.update(kwargs)
        return GlobalOptimizer._normalize_state(state, **opts)

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a configured :class:`quimb.tensor.TNOptimizer`."""
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        optimizer = GlobalOptimizer._normalize_optimizer_name(optimizer)
        self._prepare_autodiff_backend(autodiff_backend)
        incoming_constants = dict(loss_constants or {})
        terms = incoming_constants.pop("terms", self.terms)
        constants = {
            "terms": self._terms_for_autodiff_backend(
                terms,
                self.state,
                autodiff_backend,
                device=device,
            )
        }
        constants.update(incoming_constants)
        return qtn.TNOptimizer(
            self.state,
            self._tnopt_loss,
            loss_constants=constants,
            loss_kwargs=merged_loss_kwargs,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
            **tnopt_kwargs,
        )

    def optimize(
        self,
        *,
        n=220,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        return_losses: bool = False,
        normalize: bool = False,
        normalize_kwargs: Mapping[str, Any] | None = None,
        check_finite_gradient: bool = True,
        fallback_boundary_mode: str | None = "exact",
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize`` and store the optimized PEPS."""
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        tnopt = self.make_tn_optimizer(
            loss_kwargs=merged_loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        if check_finite_gradient:
            finite_gradient, finite_loss = self._initial_gradient_status(tnopt)
            fallback_mode = (
                None
                if fallback_boundary_mode is None
                else self._boundary_mode(fallback_boundary_mode)
            )
            current_mode = self._boundary_mode(merged_loss_kwargs["boundary_mode"])
            if (
                (not finite_gradient)
                and fallback_mode is not None
                and current_mode != fallback_mode
            ):
                fallback_loss_kwargs = dict(merged_loss_kwargs)
                fallback_loss_kwargs["boundary_mode"] = fallback_mode
                tnopt = self.make_tn_optimizer(
                    loss_kwargs=fallback_loss_kwargs,
                    loss_constants=loss_constants,
                    autodiff_backend=autodiff_backend,
                    optimizer=optimizer,
                    progbar=progbar,
                    jit_fn=jit_fn,
                    device=device,
                )
                finite_gradient, fallback_loss = self._initial_gradient_status(tnopt)
                finite_loss = fallback_loss if fallback_loss is not None else finite_loss
            if not finite_gradient:
                self.losses = [] if finite_loss is None else [float(finite_loss)]
                warnings.warn(
                    "PEPS energy autodiff produced a non-finite initial gradient; "
                    "returning the unmodified state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if return_losses:
                    return self.state, tuple(self.losses)
                return self.state
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        out = self._state_for_autodiff_backend(
            out,
            self.state,
            autodiff_backend,
            device=device,
        )
        if normalize:
            out = self.normalize(out, **dict(normalize_kwargs or {}))
        self.state = out
        if return_losses:
            return out, tuple(self.losses)
        return out


class MpsEnergyOptimizer(PepsEnergyOptimizer):
    """Optimize finite MPS energy with exact 1D local expectations.

    The objective is the normalized local expectation value
    ``<psi|H|psi>/<psi|psi>``. Local-term Hamiltonians are evaluated with
    ``MPS.compute_local_expectation_exact(...)``. MPO Hamiltonians are evaluated
    directly as ``(<psi| & H_mpo & |psi>).contract(all, optimize=...)``, using
    ``contraction_opt`` for the full network contraction. Hamiltonians can be
    supplied as a ``qtn.MatrixProductOperator``, a ``qtn.LocalHam1D``-like
    object with ``.terms``, a Pepsy symmetric Hamiltonian, or a plain local-term
    mapping.
    """

    _LOSS_KEYS = frozenset({
        "normalized",
        "energy_per_site",
        "real",
        "contraction_opt",
        "compute_kwargs",
        "progbar",
    })

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        normalized: bool = True,
        energy_per_site: bool = True,
        real: bool = True,
        contraction_opt: Any = "auto-hq",
        progbar: bool = False,
        compute_kwargs: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
    ):
        self.state = self._as_mps_state(state)
        self.hamiltonian = hamiltonian
        self.terms = self._terms_from_hamiltonian(hamiltonian)
        self.losses: list[float] = []
        self.loss_kwargs = {
            "normalized": normalized,
            "energy_per_site": energy_per_site,
            "real": real,
            "contraction_opt": contraction_opt,
            "progbar": progbar,
            "compute_kwargs": {} if compute_kwargs is None else dict(compute_kwargs),
        }
        if loss_kwargs is not None:
            self.set_loss_kwargs(**loss_kwargs)

    @classmethod
    def _pick_loss_kwargs(cls, options):
        incoming = dict(options or {})
        unknown = sorted(set(incoming) - cls._LOSS_KEYS)
        if unknown:
            allowed = ", ".join(sorted(cls._LOSS_KEYS))
            raise TypeError(
                f"Unknown MPS energy option(s): {', '.join(unknown)}. "
                f"Allowed options: {allowed}."
            )
        return incoming

    @staticmethod
    def _looks_like_mps(state):
        if not hasattr(state, "compute_local_expectation_exact"):
            return False
        if hasattr(state, "Lx") and hasattr(state, "Ly"):
            return False
        return hasattr(state, "L") or hasattr(state, "phys_inds")

    @classmethod
    def _as_mps_state(cls, state):
        if cls._looks_like_mps(state):
            return state
        mps = getattr(state, "mps", None)
        if mps is not None and cls._looks_like_mps(mps):
            return mps
        tn = getattr(state, "tn", None)
        if tn is not None and cls._looks_like_mps(tn):
            return tn
        raise TypeError(
            "state must be an MPS-like object with "
            "compute_local_expectation_exact()."
        )

    @staticmethod
    def _is_mpo_hamiltonian(hamiltonian):
        return isinstance(hamiltonian, qtn.MatrixProductOperator)

    @staticmethod
    def _is_symmray_array(value):
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @staticmethod
    def _is_fermionic_array(value):
        return "fermionic" in type(value).__name__.lower()

    @classmethod
    def _mpo_uses_bosonic_symmray(cls, mpo):
        for tensor in mpo:
            data = getattr(tensor, "data", None)
            if cls._is_symmray_array(data) and not cls._is_fermionic_array(data):
                return True
        return False

    @staticmethod
    def _symmray_symmetry_name(data):
        symmetry = getattr(data, "symmetry", None)
        if symmetry is not None:
            return symmetry
        name = type(data).__name__
        for candidate in ("U1U1", "Z2Z2", "U1", "Z2"):
            if name.startswith(candidate):
                return candidate
        return None

    @staticmethod
    def _charge_zero_like(charge):
        if isinstance(charge, tuple):
            return tuple(0 for _ in charge)
        return 0

    @staticmethod
    def _charge_add(a, b):
        if a is None:
            return b
        if b is None:
            return a
        if isinstance(a, tuple) or isinstance(b, tuple):
            if not isinstance(a, tuple):
                a = (a,) * len(b)
            if not isinstance(b, tuple):
                b = (b,) * len(a)
            return tuple(x + y for x, y in zip(a, b))
        return a + b

    @staticmethod
    def _charge_neg(charge):
        if isinstance(charge, tuple):
            return tuple(-x for x in charge)
        return -charge

    @classmethod
    def _charge_sub(cls, a, b):
        return cls._charge_add(a, cls._charge_neg(b))

    @staticmethod
    def _charge_parity(charge):
        if charge is None:
            return 0
        if isinstance(charge, tuple):
            return sum(int(x) for x in charge) % 2
        return int(charge) % 2

    @classmethod
    def _bosonize_fermionic_tn(cls, tn):
        import symmray.utils as sr_utils  # pylint: disable=import-outside-toplevel

        target = tn.copy()
        if not hasattr(target, "L"):
            tensors = tuple(target)
        else:
            tensors = tuple(target[site] for site in range(int(target.L)))

        tensor_charges = [
            getattr(getattr(tensor, "data", None), "charge", None)
            for tensor in tensors
        ]
        tensor_is_fermionic = [
            cls._is_symmray_array(getattr(tensor, "data", None))
            and cls._is_fermionic_array(getattr(tensor, "data", None))
            for tensor in tensors
        ]
        prefix_charge = None
        for site, tensor in enumerate(tensors):
            data = getattr(tensor, "data", None)
            if not (cls._is_symmray_array(data) and cls._is_fermionic_array(data)):
                prefix_charge = cls._charge_add(
                    prefix_charge,
                    getattr(data, "charge", None),
                )
                continue
            symmetry = cls._symmray_symmetry_name(data)
            if symmetry is None:
                continue
            # Materialize any lazy fermionic swap phases into the block data
            # before dropping to a bosonic array; otherwise the JW image of the
            # state loses those phases and the MPO sandwich is wrong.
            phase_sync = getattr(data, "phase_sync", None)
            if callable(phase_sync):
                data = phase_sync()
            array_cls = sr_utils.get_array_cls(symmetry, fermionic=False)
            tensor_charge = getattr(data, "charge", None)
            if prefix_charge is None:
                prefix_charge = cls._charge_zero_like(tensor_charge)
            prefix_after = cls._charge_add(prefix_charge, tensor_charge)

            phys_ind = f"k{site}"
            try:
                phys_axis = tensor.inds.index(phys_ind)
            except ValueError:
                phys_axis = len(tensor.inds) - 1

            right_axis = None
            if site + 1 < len(tensors):
                right_inds = [
                    ind
                    for ind in tensor.inds
                    if ind in set(tensors[site + 1].inds)
                ]
                if right_inds:
                    right_axis = tensor.inds.index(right_inds[0])
            next_dummy_parity = (
                cls._charge_parity(tensor_charges[site + 1])
                if site + 1 < len(tensors) and tensor_is_fermionic[site + 1]
                else 0
            )
            dummy_crossing_parity = (
                cls._charge_parity(prefix_after) & next_dummy_parity
            )

            blocks = {}
            for sector, block in data.blocks.items():
                exponent = 0
                if right_axis is not None:
                    # This is the left-to-right Symmray fermionic contraction
                    # phase for the outgoing virtual mode, plus the dummy-mode
                    # phase created by the next odd tensor.
                    right_parity = cls._charge_parity(sector[right_axis])
                    phys_parity = cls._charge_parity(sector[phys_axis])
                    exponent ^= right_parity & (phys_parity ^ 1)
                    exponent ^= dummy_crossing_parity
                block = cls._copy_array_like(block)
                blocks[sector] = -block if exponent else block
            tensor.modify(
                data=array_cls(
                    indices=data.indices,
                    charge=tensor_charge,
                    blocks=blocks,
                    symmetry=symmetry,
                )
            )
            prefix_charge = prefix_after
        return target

    @classmethod
    def _debosonize_fermionic_tn(cls, tn):
        """Inverse of :meth:`_bosonize_fermionic_tn`.

        Convert a bosonic Jordan-Wigner Symmray MPS (for example a ``SymDMRG2``
        ground state, whose site tensors are plain abelian ``U1``/``U1U1``
        arrays) back into a native fermionic Symmray MPS. It undoes the
        left-to-right contraction sign flips introduced by bosonization and
        restores the fermionic dummy modes via ``from_blocks(..., label=site)``.

        The transform is a per-site gauge change: bond dimensions, tensor
        charges, block sectors, and block shapes are all preserved, so the
        fermionized MPS has exactly the same bond dimension as the input. The
        sign pattern is self-inverse, hence ``debosonize(bosonize(x)) == x``.
        """
        import symmray.utils as sr_utils  # pylint: disable=import-outside-toplevel

        target = tn.copy()
        if not hasattr(target, "L"):
            tensors = tuple(target)
        else:
            tensors = tuple(target[site] for site in range(int(target.L)))

        tensor_charges = [
            getattr(getattr(tensor, "data", None), "charge", None)
            for tensor in tensors
        ]
        # Sites whose bosonic array we convert back to a fermionic array.
        tensor_is_target = [
            cls._is_symmray_array(getattr(tensor, "data", None))
            and not cls._is_fermionic_array(getattr(tensor, "data", None))
            for tensor in tensors
        ]
        prefix_charge = None
        for site, tensor in enumerate(tensors):
            data = getattr(tensor, "data", None)
            if not tensor_is_target[site]:
                if cls._is_symmray_array(data):
                    prefix_charge = cls._charge_add(
                        prefix_charge,
                        getattr(data, "charge", None),
                    )
                continue
            symmetry = cls._symmray_symmetry_name(data)
            if symmetry is None:
                continue
            # Normalize the leg order so the physical index is last, matching
            # the convention assumed by the fermionic sign reconstruction (and
            # produced by ``_bosonize_fermionic_tn`` / ``SymMPS.for_model``).
            # Some producers (notably quimb's ``DMRG2``) leave the physical leg
            # first on the boundary tensor; reconstructing the fermionic array
            # from that order corrupts the fermionic swap phases and yields a
            # state that is right for the bosonic sandwich but wrong for native
            # fermionic gates/off-diagonal correlators. A bosonic transpose is
            # phase-free, so this reorder is safe and gauge-preserving.
            phys_ind = f"k{site}"
            if phys_ind in tensor.inds and tensor.inds[-1] != phys_ind:
                new_order = [ind for ind in tensor.inds if ind != phys_ind]
                new_order.append(phys_ind)
                tensor.transpose(*new_order, inplace=True)
                data = tensor.data
            array_cls = sr_utils.get_array_cls(symmetry, fermionic=True)
            tensor_charge = getattr(data, "charge", None)
            if prefix_charge is None:
                prefix_charge = cls._charge_zero_like(tensor_charge)
            prefix_after = cls._charge_add(prefix_charge, tensor_charge)

            try:
                phys_axis = tensor.inds.index(phys_ind)
            except ValueError:
                phys_axis = len(tensor.inds) - 1

            right_axis = None
            if site + 1 < len(tensors):
                right_inds = [
                    ind
                    for ind in tensor.inds
                    if ind in set(tensors[site + 1].inds)
                ]
                if right_inds:
                    right_axis = tensor.inds.index(right_inds[0])
            next_dummy_parity = (
                cls._charge_parity(tensor_charges[site + 1])
                if site + 1 < len(tensors) and tensor_is_target[site + 1]
                else 0
            )
            dummy_crossing_parity = (
                cls._charge_parity(prefix_after) & next_dummy_parity
            )

            blocks = {}
            for sector, block in data.blocks.items():
                exponent = 0
                if right_axis is not None:
                    # Same left-to-right fermionic contraction phase used by
                    # bosonization; applying it a second time inverts it.
                    right_parity = cls._charge_parity(sector[right_axis])
                    phys_parity = cls._charge_parity(sector[phys_axis])
                    exponent ^= right_parity & (phys_parity ^ 1)
                    exponent ^= dummy_crossing_parity
                block = cls._copy_array_like(block)
                blocks[sector] = -block if exponent else block
            tensor.modify(
                data=array_cls.from_blocks(
                    blocks,
                    duals=data.duals,
                    charge=tensor_charge,
                    symmetry=symmetry,
                    label=site,
                )
            )
            prefix_charge = prefix_after
        return target

    @classmethod
    def _terms_from_hamiltonian(cls, hamiltonian):
        if cls._is_mpo_hamiltonian(hamiltonian):
            return hamiltonian
        return super()._terms_from_hamiltonian(hamiltonian)

    @staticmethod
    def _num_sites(state):
        num_sites = getattr(state, "num_sites", None)
        if num_sites is not None:
            return int(num_sites() if callable(num_sites) else num_sites)
        length = getattr(state, "L", None)
        if length is not None:
            return int(length)
        sites = getattr(state, "sites", None)
        if sites is not None:
            return len(tuple(sites))
        phys_inds = getattr(state, "phys_inds", None)
        if phys_inds is not None:
            return len(tuple(phys_inds() if callable(phys_inds) else phys_inds))
        raise ValueError("Could not infer the number of MPS sites.")

    @staticmethod
    def _max_bond(state):
        max_bond = getattr(state, "max_bond", None)
        if callable(max_bond):
            return int(max_bond())
        return None

    @classmethod
    def _terms_for_state_backend(cls, terms, state):
        if cls._is_mpo_hamiltonian(terms):
            return terms
        sample = cls._sample_array_from_tn(state)
        try:
            converter = infer_backend_converter_from_sample(sample)
        except (ImportError, TypeError, ValueError):
            converter = None
        if converter is None:
            return terms
        return {
            edge: cls._convert_term_array(term, converter)
            for edge, term in dict(terms).items()
        }

    @classmethod
    def _iter_tn_data(cls, state):
        tensor_map = getattr(state, "tensor_map", None)
        if tensor_map:
            tensors = tensor_map.values()
        else:
            try:
                tensors = iter(state)
            except TypeError:
                return
        for tensor in tensors:
            yield getattr(tensor, "data", None)

    @classmethod
    def _state_uses_bosonic_symmray(cls, state):
        found_symmray = False
        for data in cls._iter_tn_data(state):
            if not cls._is_symmray_array(data):
                continue
            found_symmray = True
            if cls._is_fermionic_array(data):
                return False
        return found_symmray

    @classmethod
    def _terms_use_fermionic_symmray(cls, terms):
        return any(
            cls._is_symmray_array(term) and cls._is_fermionic_array(term)
            for term in dict(terms).values()
        )

    @classmethod
    def _check_local_terms_state_compatible(cls, state, terms):
        if (
            cls._terms_use_fermionic_symmray(terms)
            and cls._state_uses_bosonic_symmray(state)
        ):
            raise ValueError(
                "Fermionic Symmray local-term dictionaries cannot be evaluated "
                "on a bosonic/Jordan-Wigner MPS state. Use the corresponding "
                "bosonic/Jordan-Wigner MPO for energy measurements."
            )

    @classmethod
    def _mpo_expectation(
        cls,
        state,
        mpo,
        *,
        normalized=True,
        contraction_opt="auto-hq",
    ):
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)
        state = cls._as_mps_state(state)
        if cls._mpo_uses_bosonic_symmray(mpo):
            ket = cls._bosonize_fermionic_tn(state)
        else:
            ket = state.copy()
        bra = ket.H
        bra.reindex_({f"k{i}": f"b{i}" for i in range(ket.L)})
        network = bra | mpo | ket
        for site in range(ket.L):
            network ^= f"I{site}"
        value = network.contract(all, optimize=contraction_opt)
        if normalized:
            norm = (ket.H & ket).contract(all, optimize=contraction_opt)
            if norm == 0.0:
                raise ValueError("Cannot compute normalized energy for a zero-norm state.")
            value = value / norm
        return value

    @classmethod
    def _loss_state(
        cls,
        state,
        *,
        terms,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
        compute_kwargs=None,
        progbar=False,
    ):
        state = cls._as_mps_state(state)
        terms = cls._terms_from_hamiltonian(terms)
        if cls._is_mpo_hamiltonian(terms):
            value = cls._mpo_expectation(
                state,
                terms,
                normalized=normalized,
                contraction_opt=contraction_opt,
            )
            if energy_per_site:
                value = value / cls._num_sites(state)
            if real:
                value = cls._maybe_real(value)
            return value

        cls._check_local_terms_state_compatible(state, terms)
        terms = cls._terms_for_state_backend(terms, state)
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)

        kwargs = dict(compute_kwargs or {})
        kwargs.setdefault("progbar", bool(progbar))
        value = state.compute_local_expectation_exact(
            terms,
            optimize=contraction_opt,
            normalized=bool(normalized),
            **kwargs,
        )
        if energy_per_site:
            value = value / cls._num_sites(state)
        if real:
            value = cls._maybe_real(value)
        return value

    @staticmethod
    def _tnopt_loss(state, *, terms, **loss_kwargs):
        """Adapter for ``qtn.TNOptimizer``."""
        return MpsEnergyOptimizer._loss_state(state, terms=terms, **loss_kwargs)

    def loss(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Evaluate the configured MPS energy loss."""
        state = self.state if state is None else self._as_mps_state(state)
        terms_use = self.terms
        if hamiltonian is not None:
            terms_use = self._terms_from_hamiltonian(hamiltonian)
        if terms is not None:
            terms_use = self._terms_from_hamiltonian(terms)
        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        return self._loss_state(state, terms=terms_use, **opts)

    def energy(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Return full and per-site MPS energy estimates for ``state``."""
        state = self.state if state is None else self._as_mps_state(state)
        terms_use = self.terms
        if hamiltonian is not None:
            terms_use = self._terms_from_hamiltonian(hamiltonian)
        if terms is not None:
            terms_use = self._terms_from_hamiltonian(terms)

        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        opts_full = dict(opts)
        opts_full["energy_per_site"] = False
        energy = self._loss_state(state, terms=terms_use, **opts_full)
        num_sites = self._num_sites(state)
        energy_per_site = energy / num_sites
        return EnergyEstimate(
            energy=energy,
            energy_per_site=energy_per_site,
            num_sites=num_sites,
            chi=self._max_bond(state),
            boundary_mode="exact",
            normalized=bool(opts["normalized"]),
            metadata={
                "real": opts["real"],
                "contraction_opt": opts["contraction_opt"],
                "progbar": opts["progbar"],
                "compute_kwargs": dict(opts["compute_kwargs"]),
            },
        )

    def normalize(self, state=None, **kwargs):
        """Normalize ``state`` in place using the MPS' native normalization."""
        state = self.state if state is None else self._as_mps_state(state)
        normalize = getattr(state, "normalize", None)
        if not callable(normalize):
            raise TypeError("state must provide normalize() for MPS normalization.")
        normalize(**kwargs)
        return state

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a configured :class:`quimb.tensor.TNOptimizer`."""
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        optimizer = GlobalOptimizer._normalize_optimizer_name(optimizer)
        self._prepare_autodiff_backend(autodiff_backend)
        incoming_constants = dict(loss_constants or {})
        terms = incoming_constants.pop("terms", self.terms)
        if self._is_mpo_hamiltonian(terms):
            constants = {"terms": terms}
            constants.update(incoming_constants)
            return qtn.TNOptimizer(
                self.state,
                self._tnopt_loss,
                loss_constants=constants,
                loss_kwargs=merged_loss_kwargs,
                autodiff_backend=autodiff_backend,
                optimizer=optimizer,
                progbar=progbar,
                jit_fn=jit_fn,
                device=device,
                **tnopt_kwargs,
            )
        constants = {
            "terms": self._terms_for_autodiff_backend(
                terms,
                self.state,
                autodiff_backend,
                device=device,
            )
        }
        constants.update(incoming_constants)
        return qtn.TNOptimizer(
            self.state,
            self._tnopt_loss,
            loss_constants=constants,
            loss_kwargs=merged_loss_kwargs,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
            **tnopt_kwargs,
        )

    def optimize(
        self,
        *,
        n=220,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        jit_fn: bool = False,
        device: str = "cpu",
        return_losses: bool = False,
        normalize: bool = False,
        normalize_kwargs: Mapping[str, Any] | None = None,
        check_finite_gradient: bool = True,
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize`` and store the optimized MPS."""
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        tnopt = self.make_tn_optimizer(
            loss_kwargs=merged_loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        if check_finite_gradient:
            finite_gradient, finite_loss = self._initial_gradient_status(tnopt)
            if not finite_gradient:
                self.losses = [] if finite_loss is None else [float(finite_loss)]
                warnings.warn(
                    "MPS energy autodiff produced a non-finite initial gradient; "
                    "returning the unmodified state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                if return_losses:
                    return self.state, tuple(self.losses)
                return self.state
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        out = self._state_for_autodiff_backend(
            out,
            self.state,
            autodiff_backend,
            device=device,
        )
        if normalize:
            out = self.normalize(out, **dict(normalize_kwargs or {}))
        self.state = out
        if return_losses:
            return out, tuple(self.losses)
        return out
