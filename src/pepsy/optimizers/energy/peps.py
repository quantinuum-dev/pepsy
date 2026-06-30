"""Finite-PEPS energy objectives and optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import autoray as ar
import numpy as np
import quimb.tensor as qtn

from ...backends import backend_cupy, backend_jax, backend_numpy, backend_torch
from ...tensors import build_optimizer
from ..global_opt import GlobalOptimizer

__all__ = ["EnergyEstimate", "PepsEnergyOptimizer"]


@dataclass(frozen=True)
class EnergyEstimate:
    """Energy evaluation metadata for a finite PEPS."""

    energy: Any
    energy_per_site: Any
    num_sites: int
    chi: int | tuple[int, int]
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
        value = state.compute_local_expectation(
            terms,
            max_bond=chi,
            cutoff=cutoff,
            normalized=bool(normalized),
            mode=cls._boundary_mode(boundary_mode),
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
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize`` and store the optimized PEPS."""
        tnopt = self.make_tn_optimizer(
            loss_kwargs=loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=progbar,
            jit_fn=jit_fn,
            device=device,
        )
        out = tnopt.optimize(n=n, **optimize_kwargs)
        self.losses = list(getattr(tnopt, "losses", ()))
        if normalize:
            out = self.normalize(out, **dict(normalize_kwargs or {}))
        self.state = out
        if return_losses:
            return out, tuple(self.losses)
        return out
