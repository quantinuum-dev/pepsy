"""MERA energy objective and optimization shell."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import (
    get_default_array_backend,
    infer_backend_converter_from_sample,
    resolve_backend_sample_data_from_tn,
)
from ...tensors import build_optimizer, reg_rel_svd_jax, reg_rel_svd_torch
from ..energy import EnergyEstimate
from ..global_opt import GlobalOptimizer
from .lightcones import (
    build_lightcone_chunks,
    build_qmera_lightcone_chunks,
    local_lightcone_expectation,
)
from .terms import convert_local_terms, normalize_local_terms

__all__ = ["MeraEnergyOptimizer"]


class MeraEnergyOptimizer:
    """Evaluate and optimize local energies of MERA-like tensor networks.

    This first implementation assumes a fixed MERA-like tensor network with
    physical site tags. It contracts each Hamiltonian term over its selected
    reverse lightcone rather than over the full state.
    """

    _LOSS_KEYS = frozenset({
        "normalized",
        "energy_per_site",
        "real",
        "contraction_opt",
        "array_backend",
        "convert_terms",
        "precompute_tags",
        "simplify",
        "gate_contract",
        "contract_opts",
    })

    def __init__(
        self,
        state,
        hamiltonian,
        *,
        normalized: bool = True,
        energy_per_site: bool = True,
        real: bool = True,
        isometrize_method: str | None = "exp",
        contraction_opt: Any = "auto-hq",
        schedule=None,
        array_backend=None,
        convert_terms: bool = True,
        precompute_tags: bool = True,
        simplify: bool | str = False,
        gate_contract: bool = True,
        contract_opts: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
    ):
        self.schedule = self._resolve_schedule(state, schedule)
        self.state = self._as_mera_state(state)
        self.hamiltonian = hamiltonian
        self.terms = normalize_local_terms(hamiltonian)
        self.losses: list[float] = []
        self.isometrize_method = isometrize_method
        self.loss_kwargs = {
            "normalized": normalized,
            "energy_per_site": energy_per_site,
            "real": real,
            "contraction_opt": contraction_opt,
            "array_backend": array_backend,
            "convert_terms": convert_terms,
            "precompute_tags": precompute_tags,
            "simplify": simplify,
            "gate_contract": gate_contract,
            "contract_opts": {} if contract_opts is None else dict(contract_opts),
        }
        if loss_kwargs is not None:
            self.set_loss_kwargs(**loss_kwargs)
        self.lightcones = self._build_chunks_if_requested(
            self.state,
            self.terms,
            schedule=self.schedule,
        )

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

    @classmethod
    def _pick_loss_kwargs(cls, options):
        incoming = dict(options or {})
        unknown = sorted(set(incoming) - cls._LOSS_KEYS)
        if unknown:
            allowed = ", ".join(sorted(cls._LOSS_KEYS))
            raise TypeError(
                f"Unknown MERA energy option(s): {', '.join(unknown)}. "
                f"Allowed options: {allowed}."
            )
        return incoming

    @staticmethod
    def _as_mera_state(state):
        if hasattr(state, "select") and hasattr(state, "gate"):
            return state
        ansatz_state = getattr(state, "state", None)
        if (
            ansatz_state is not None
            and hasattr(ansatz_state, "select")
            and hasattr(ansatz_state, "gate")
        ):
            return ansatz_state
        tn = getattr(state, "tn", None)
        if tn is not None and hasattr(tn, "select") and hasattr(tn, "gate"):
            return tn
        raise TypeError(
            "state must be a MERA-like TensorNetwork with select() and gate()."
        )

    @staticmethod
    def _resolve_schedule(state, schedule=None):
        if schedule is not None:
            return schedule
        return getattr(state, "schedule", None)

    @staticmethod
    def _num_sites(state):
        num_sites = getattr(state, "num_sites", None)
        if num_sites is not None:
            return int(num_sites() if callable(num_sites) else num_sites)
        sites = getattr(state, "sites", None)
        if sites is not None:
            return len(tuple(sites))
        length = getattr(state, "L", None)
        if length is not None:
            return int(length)
        raise ValueError("Could not infer the number of MERA sites.")

    @staticmethod
    def _max_bond(state):
        max_bond = getattr(state, "max_bond", None)
        if callable(max_bond):
            return max_bond()
        return max_bond

    @staticmethod
    def _maybe_real(value):
        try:
            return ar.do("real", value)
        except Exception:  # pragma: no cover - defensive for unusual scalar types
            return value.real

    @staticmethod
    def _is_finite_number(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _prepare_autodiff_backend(backend):
        key = str(backend).strip().lower()
        try:
            if key == "torch":
                reg_rel_svd_torch()
            elif key == "jax":
                reg_rel_svd_jax()
        except ImportError:
            return

    @classmethod
    def _array_backend_for_state(cls, state, explicit_backend=None):
        if explicit_backend is not None:
            return explicit_backend
        default_backend = get_default_array_backend()
        if default_backend is not None:
            return default_backend
        sample = resolve_backend_sample_data_from_tn(state)
        return infer_backend_converter_from_sample(sample)

    @classmethod
    def _prepare_terms(cls, state, terms, *, array_backend=None, convert_terms=True):
        terms = normalize_local_terms(terms)
        if not convert_terms:
            return terms
        backend = cls._array_backend_for_state(state, array_backend)
        return convert_local_terms(terms, backend)

    def _build_chunks_if_requested(self, state, terms, *, schedule=None, opts=None):
        opts = self.loss_kwargs if opts is None else opts
        if not opts.get("precompute_tags", True):
            return None
        prepared_terms = self._prepare_terms(
            state,
            terms,
            array_backend=opts.get("array_backend"),
            convert_terms=opts.get("convert_terms", True),
        )
        if schedule is not None:
            return build_qmera_lightcone_chunks(state, schedule, prepared_terms)
        return build_lightcone_chunks(state, prepared_terms)

    @classmethod
    def _chunks_for_loss(cls, state, terms, *, chunks=None, schedule=None, **opts):
        if chunks is not None:
            return chunks
        prepared_terms = cls._prepare_terms(
            state,
            terms,
            array_backend=opts.get("array_backend"),
            convert_terms=opts.get("convert_terms", True),
        )
        if schedule is not None:
            return build_qmera_lightcone_chunks(state, schedule, prepared_terms)
        return build_lightcone_chunks(state, prepared_terms)

    @classmethod
    def _loss_state(
        cls,
        state,
        *,
        terms,
        chunks=None,
        schedule=None,
        normalized=True,
        energy_per_site=True,
        real=True,
        contraction_opt="auto-hq",
        array_backend=None,
        convert_terms=True,
        precompute_tags=True,
        simplify=False,
        gate_contract=True,
        contract_opts=None,
    ):
        del precompute_tags
        state = cls._as_mera_state(state)
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)
        chunks = cls._chunks_for_loss(
            state,
            terms,
            chunks=chunks,
            schedule=schedule,
            array_backend=array_backend,
            convert_terms=convert_terms,
        )
        value = None
        for chunk in chunks:
            term_value = local_lightcone_expectation(
                state,
                chunk,
                optimize=contraction_opt,
                normalized=normalized,
                real=False,
                simplify=simplify,
                gate_contract=gate_contract,
                contract_opts=contract_opts,
            )
            value = term_value if value is None else value + term_value
        if value is None:
            raise ValueError("hamiltonian contains no local terms.")
        if energy_per_site:
            value = value / cls._num_sites(state)
        if real:
            value = cls._maybe_real(value)
        return value

    @staticmethod
    def _tnopt_loss(state, *, terms, chunks=None, schedule=None, **loss_kwargs):
        """Adapter for :class:`quimb.tensor.TNOptimizer`."""
        return MeraEnergyOptimizer._loss_state(
            state,
            terms=terms,
            chunks=chunks,
            schedule=schedule,
            **loss_kwargs,
        )

    def set_loss_kwargs(self, **kwargs):
        """Update stored defaults for energy loss evaluation."""
        self.loss_kwargs.update(self._pick_loss_kwargs(kwargs))
        self.lightcones = self._build_chunks_if_requested(
            self.state,
            self.terms,
            schedule=self.schedule,
        )
        return self

    def loss(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Evaluate the configured MERA local energy loss."""
        state = self.state if state is None else self._as_mera_state(state)
        terms_use = self.terms
        chunks = self.lightcones if state is self.state else None
        if hamiltonian is not None:
            terms_use = normalize_local_terms(hamiltonian)
            chunks = None
        if terms is not None:
            terms_use = normalize_local_terms(terms)
            chunks = None
        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        if chunks is not None and (
            opts.get("array_backend") is not self.loss_kwargs.get("array_backend")
            or opts.get("convert_terms") != self.loss_kwargs.get("convert_terms")
        ):
            chunks = None
        return self._loss_state(
            state,
            terms=terms_use,
            chunks=chunks,
            schedule=self.schedule,
            **opts,
        )

    def energy(self, state=None, *, hamiltonian=None, terms=None, **kwargs):
        """Return full and per-site MERA energy estimates."""
        state = self.state if state is None else self._as_mera_state(state)
        opts = self._merge_opts(self.loss_kwargs, self._pick_loss_kwargs(kwargs))
        opts_full = dict(opts)
        opts_full["energy_per_site"] = False
        energy = self.loss(
            state,
            hamiltonian=hamiltonian,
            terms=terms,
            **opts_full,
        )
        num_sites = self._num_sites(state)
        energy_per_site = energy / num_sites
        chunks = self.lightcones if state is self.state else None
        metadata = {
            "real": opts["real"],
            "contraction_opt": opts["contraction_opt"],
            "simplify": opts["simplify"],
            "gate_contract": opts["gate_contract"],
            "num_terms": len(self.terms),
        }
        if chunks is not None:
            metadata.update(self.lightcone_diagnostics(chunks=chunks))
        return EnergyEstimate(
            energy=energy,
            energy_per_site=energy_per_site,
            num_sites=num_sites,
            chi=self._max_bond(state),
            boundary_mode="lightcone-exact",
            normalized=bool(opts["normalized"]),
            metadata=metadata,
        )

    def lightcone_diagnostics(self, *, chunks=None):
        """Return compact diagnostics for cached lightcone chunks."""
        chunks = self.lightcones if chunks is None else chunks
        if chunks is None:
            chunks = self._build_chunks_if_requested(self.state, self.terms)
        if not chunks:
            return {
                "max_lightcone_tensors": 0,
                "max_lightcone_indices": 0,
                "max_physical_width": 0,
            }
        return {
            "max_lightcone_tensors": max(chunk.num_tensors for chunk in chunks),
            "max_lightcone_indices": max(chunk.num_indices for chunk in chunks),
            "max_physical_width": max(chunk.physical_width for chunk in chunks),
            "max_schedule_width": max(chunk.schedule_width for chunk in chunks),
            "lightcone_sources": tuple(chunk.source for chunk in chunks),
            "num_tensors_by_term": tuple(chunk.num_tensors for chunk in chunks),
            "num_indices_by_term": tuple(chunk.num_indices for chunk in chunks),
            "physical_width_by_term": tuple(chunk.physical_width for chunk in chunks),
            "schedule_width_by_term": tuple(chunk.schedule_width for chunk in chunks),
        }

    def _norm_fn(self, state):
        method = self.isometrize_method
        if method is None:
            return state
        isometrize = getattr(state, "isometrize", None)
        if not callable(isometrize):
            return state
        return isometrize(method=method, inplace=False)

    def make_tn_optimizer(
        self,
        *,
        loss_kwargs: Mapping[str, Any] | None = None,
        loss_constants: Mapping[str, Any] | None = None,
        autodiff_backend: str = "torch",
        optimizer: str = "adam",
        progbar: bool = True,
        device: str = "cpu",
        **tnopt_kwargs,
    ):
        """Construct a configured :class:`quimb.tensor.TNOptimizer`."""
        del device
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        optimizer = GlobalOptimizer._normalize_optimizer_name(optimizer)
        self._prepare_autodiff_backend(autodiff_backend)
        incoming_constants = dict(loss_constants or {})
        terms = incoming_constants.pop("terms", self.terms)
        chunks = incoming_constants.pop(
            "chunks",
            self.lightcones if merged_loss_kwargs.get("precompute_tags", True) else None,
        )
        constants = {"terms": terms, "chunks": chunks, "schedule": self.schedule}
        constants.update(incoming_constants)
        return qtn.TNOptimizer(
            self.state,
            self._tnopt_loss,
            norm_fn=self._norm_fn if self.isometrize_method is not None else None,
            loss_constants=constants,
            loss_kwargs=merged_loss_kwargs,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
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
        return_losses: bool = False,
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize`` and store the optimized MERA state."""
        tnopt = self.make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
        )
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        self.state = out
        self.lightcones = self._build_chunks_if_requested(
            self.state,
            self.terms,
            schedule=self.schedule,
        )
        if return_losses:
            return out, tuple(self.losses)
        return out
