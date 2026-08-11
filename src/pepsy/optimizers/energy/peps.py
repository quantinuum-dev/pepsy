"""Finite-PEPS energy objectives and optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
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
    TorchLinalgConfig,
)
from ...tensors import build_optimizer, reg_rel_svd_jax
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
    _VALIDATION_TOL = 1.0e-8
    _VALIDATION_MAX_CHECKS = 5

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
            if key == "jax":
                reg_rel_svd_jax()
        except ImportError:
            # TNOptimizer will report the missing backend if it is actually
            # needed. Keeping this hook soft preserves optional-dependency tests.
            return

    @classmethod
    def _configure_torch_linalg(cls, state, terms, *, quimb_split_drivers):
        """Configure one consistent Torch linalg stack for an optimizer run."""
        dtype_name = cls._autodiff_dtype_name(state, terms)
        mode = "complex" if "complex" in str(dtype_name).lower() else "real"
        try:
            # Energy optimization differentiates through both SVD and QR.
            # Keep those registrations together in the public policy class so
            # the optimizer cannot accidentally regularize one decomposition
            # while leaving the other on an incompatible backend.
            TorchLinalgConfig(
                mode=mode,
                stabilized=True,
                quimb_split_drivers=quimb_split_drivers,
            ).register()
        except ImportError:
            # TNOptimizer will report a missing requested backend if it is
            # actually needed; preserve the optional-dependency behavior.
            return

    @staticmethod
    def _is_finite_number(value):
        try:
            return bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            return False

    @classmethod
    def _track_tnopt_best_checkpoint(cls, tnopt):
        """Track the parameter vector for the best finite loss seen.

        Quimb records ``TNOptimizer.loss_best`` as a scalar only.  That is
        insufficient when an optimizer backend aborts after evaluating a
        worse trial, because the current vector can then be worse than the
        best vector.  Wrapping the backend handler covers both SciPy's
        ``vectorized_value_and_grad`` route and Quimb's direct NLopt callback.
        """
        vectorizer = getattr(tnopt, "vectorizer", None)
        vector = getattr(vectorizer, "vector", None)
        tracker = {
            "loss": float("inf"),
            "vector": None if vector is None else vector.copy(),
        }
        if vector is None:
            return tracker

        def record(result):
            if not cls._is_finite_number(result):
                return
            loss = float(result)
            if loss < tracker["loss"]:
                tracker["loss"] = loss
                tracker["vector"] = vector.copy()

        handler = getattr(tnopt, "handler", None)
        if handler is None:
            return tracker

        value_and_grad = getattr(handler, "value_and_grad", None)
        if callable(value_and_grad):
            def tracked_value_and_grad(arrays):
                result, gradients = value_and_grad(arrays)
                record(result)
                return result, gradients

            handler.value_and_grad = tracked_value_and_grad

        value = getattr(handler, "value", None)
        if callable(value):
            def tracked_value(arrays):
                result = value(arrays)
                record(result)
                return result

            handler.value = tracked_value

        return tracker

    @staticmethod
    def _is_recoverable_optimizer_error(exc, optlib):
        """Return whether an optimizer exception should yield its best state."""
        if str(optlib).strip().lower() == "nlopt":
            # nlopt.runtime_error and nlopt.roundoff_limited are not Python's
            # built-in RuntimeError, but all are exposed from the ``nlopt``
            # module and represent an optimizer stop rather than a loss bug.
            return "nlopt" in type(exc).__module__.lower()
        return isinstance(exc, RuntimeError)

    @classmethod
    def _best_tnopt_state(cls, tnopt, tracker):
        """Restore and extract the best tracked TNOptimizer checkpoint."""
        vector = tracker.get("vector")
        if vector is not None:
            tnopt.vectorizer.vector[:] = vector
        return tnopt.get_tn_opt()

    @staticmethod
    def _validation_chi(chi):
        """Choose the automatic higher-chi validation bond dimension."""
        if isinstance(chi, Integral) and not isinstance(chi, bool):
            return max(1, 2 * int(chi))
        if isinstance(chi, (tuple, list)):
            return tuple(max(1, 2 * int(value)) for value in chi)
        return chi

    @classmethod
    def _validation_chunk_size(cls, n, validation_interval):
        """Resolve the optimizer evaluations between validation checks."""
        n_total = max(0, int(n))
        if validation_interval is None:
            return max(
                10,
                int(np.ceil(n_total / cls._VALIDATION_MAX_CHECKS)),
            )
        if isinstance(validation_interval, bool) or not isinstance(
            validation_interval,
            Integral,
        ):
            raise TypeError("validation_interval must be a positive integer or None.")
        validation_interval = int(validation_interval)
        if validation_interval <= 0:
            raise ValueError("validation_interval must be a positive integer or None.")
        return validation_interval

    @classmethod
    def _validation_loss(cls, state, *, terms, loss_kwargs):
        """Evaluate a candidate with the automatic validation bond dimension."""
        kwargs = dict(loss_kwargs)
        kwargs["chi"] = cls._validation_chi(kwargs["chi"])
        return cls._loss_state(state, terms=terms, **kwargs)

    def _optimize_with_validation(
        self,
        tnopt,
        *,
        n,
        optimize_kwargs,
        optlib,
        best_tracker,
        terms,
        loss_kwargs,
        autodiff_backend,
        device,
        progbar,
        validation_interval,
    ):
        """Run short optimizer chunks with sparse higher-chi rollback checks."""

        def validation_state():
            # Quimb's ``TNOptimizer.get_tn_opt`` deliberately converts its
            # injected variables back to NumPy. That is fine for ordinary
            # dense PEPS, but native Symmray PEPS can then contain NumPy
            # blocks next to Torch blocks. Convert the complete candidate
            # before contracting the validation loss.
            candidate = tnopt.get_tn_opt()
            return self._state_for_autodiff_backend(
                candidate,
                self.state,
                autodiff_backend,
                device=device,
            )

        n_total = max(0, int(n))
        chunk_size = self._validation_chunk_size(n_total, validation_interval)
        train_chi = loss_kwargs["chi"]
        validation_chi = self._validation_chi(train_chi)
        pbar = None
        if progbar:
            from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel

            pbar = tqdm(
                total=n_total,
                desc="PEPS validate",
                unit="eval",
                leave=True,
                position=0,
                ascii=True,
                dynamic_ncols=True,
                mininterval=0.2,
            )

        def update_progress(completed, local_energy, check_energy, status):
            if pbar is None:
                return
            pbar.update(completed - pbar.n)

            def format_energy(value):
                if self._is_finite_number(value):
                    return f"{float(value):+.3e}"
                return "n/a"

            pbar.set_postfix(
                train_chi=train_chi,
                check_chi=validation_chi,
                step=chunk_size,
                local_E=format_energy(local_energy),
                check_E=format_energy(check_energy),
                status=status,
            )

        initial_vector = tnopt.vectorizer.vector.copy()
        validated_vector = initial_vector.copy()
        initial_state = validation_state()
        initial_loss = self._validation_loss(
            initial_state,
            terms=terms,
            loss_kwargs=loss_kwargs,
        )
        update_progress(0, best_tracker.get("loss"), initial_loss, "initial")
        if not self._is_finite_number(initial_loss):
            warnings.warn(
                "PEPS validation energy is non-finite for the initial state; "
                "returning the unmodified state.",
                RuntimeWarning,
                stacklevel=2,
            )
            tnopt.vectorizer.vector[:] = initial_vector
            self.validation_history = [(0, None)]
            if pbar is not None:
                pbar.close()
            return validation_state()

        validated_loss = float(initial_loss)
        self.validation_history = [(0, validated_loss)]
        if n_total == 0:
            tnopt.vectorizer.vector[:] = validated_vector
            if pbar is not None:
                pbar.close()
            return validation_state()

        completed = 0
        while completed < n_total:
            chunk = min(chunk_size, n_total - completed)
            try:
                tnopt.optimize(n=chunk, **optimize_kwargs)
            except Exception as exc:
                if not self._is_recoverable_optimizer_error(exc, optlib):
                    if pbar is not None:
                        pbar.close()
                    raise
                tnopt.vectorizer.vector[:] = validated_vector
                warnings.warn(
                    f"{optlib} optimization stopped early ({exc}); returning "
                    "the last validated PEPS state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                update_progress(
                    completed,
                    best_tracker.get("loss"),
                    validated_loss,
                    "stopped",
                )
                break

            completed += chunk
            candidate_vector = best_tracker.get("vector")
            if candidate_vector is None:
                candidate_vector = tnopt.vectorizer.vector.copy()
            tnopt.vectorizer.vector[:] = candidate_vector
            candidate_state = validation_state()
            candidate_loss = self._validation_loss(
                candidate_state,
                terms=terms,
                loss_kwargs=loss_kwargs,
            )
            candidate_finite = self._is_finite_number(candidate_loss)
            if (
                not candidate_finite
                or float(candidate_loss) > validated_loss + self._VALIDATION_TOL
            ):
                tnopt.vectorizer.vector[:] = validated_vector
                self.validation_history.append(
                    (
                        completed,
                        None if not candidate_finite else float(candidate_loss),
                    )
                )
                warnings.warn(
                    "PEPS validation energy worsened or became non-finite; "
                    "rolling back to the last validated state.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                update_progress(
                    completed,
                    best_tracker.get("loss"),
                    candidate_loss,
                    "rollback",
                )
                break

            validated_vector = candidate_vector.copy()
            validated_loss = float(candidate_loss)
            self.validation_history.append((completed, validated_loss))
            update_progress(
                completed,
                best_tracker.get("loss"),
                validated_loss,
                "accepted",
            )

        if pbar is not None:
            pbar.close()
        tnopt.vectorizer.vector[:] = validated_vector
        return validation_state()

    @staticmethod
    def _coerce_parameter_bounds(bounds, dimension):
        """Validate and normalize flat-parameter bounds for TNOptimizer."""
        array = np.asarray(bounds, dtype=float)
        if array.shape == (2,):
            array = np.broadcast_to(array, (dimension, 2)).copy()
        if array.shape != (dimension, 2) and array.shape == (2, dimension):
            array = array.T
        if array.shape != (dimension, 2):
            raise ValueError(
                "parameter bounds must have shape (dimension, 2) or "
                "be a (lower, upper) pair of length dimension."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("parameter bounds must contain only finite values.")
        if np.any(array[:, 0] > array[:, 1]):
            raise ValueError("each parameter lower bound must not exceed its upper bound.")
        return array

    @classmethod
    def _apply_parameter_bounds(
        cls,
        tnopt,
        *,
        bounds=None,
        max_parameter_step=None,
    ):
        """Apply optional cheap trust-box bounds to a TNOptimizer."""
        if bounds is not None and max_parameter_step is not None:
            raise ValueError(
                "pass either bounds or max_parameter_step, not both."
            )
        if bounds is None and max_parameter_step is None:
            return

        vector = np.asarray(tnopt.vectorizer.vector, dtype=float)
        dimension = vector.size
        if max_parameter_step is not None:
            try:
                step = float(max_parameter_step)
            except (TypeError, ValueError) as exc:
                raise TypeError("max_parameter_step must be a positive number.") from exc
            if not np.isfinite(step) or step <= 0.0:
                raise ValueError("max_parameter_step must be a positive finite number.")
            bounds = np.column_stack((vector - step, vector + step))
        bounds = cls._coerce_parameter_bounds(bounds, dimension)
        # Quimb's public ``bounds`` setter accepts only one scalar pair and
        # broadcasts it across all variables. Preserve per-variable bounds by
        # writing its backing field directly; the optimizer backends read the
        # public property afterward. Small test doubles without that property
        # retain the ordinary assignment path.
        if isinstance(getattr(type(tnopt), "bounds", None), property):
            tnopt._bounds = bounds
        else:
            tnopt.bounds = bounds

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

    @staticmethod
    def _is_symmray_array(value):
        """Return whether ``value`` is a native Symmray block array."""
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @classmethod
    def _terms_use_symmray(cls, terms):
        return all(cls._is_symmray_array(term) for term in terms.values())

    @staticmethod
    def _term_sites(state, where):
        """Normalize a local-term key to the corresponding PEPS sites."""
        has_site = getattr(state, "has_site", None)
        if callable(has_site) and has_site(where):
            return (where,)
        if not isinstance(where, (tuple, list)):
            return (where,)
        sites = tuple(where)
        if len(sites) not in {1, 2}:
            raise ValueError("PEPS exact energy terms must act on one or two sites.")
        return sites

    @classmethod
    def _symmray_exact_local_expectation(
        cls,
        state,
        terms,
        *,
        optimize,
        normalized,
        contract_opts,
    ):
        """Contract native Symmray local terms without forming a dense RDM.

        Quimb's generic ``compute_local_expectation_exact`` forms a reduced
        density matrix and fuses its physical legs. Individual Symmray blocks
        do not carry the full physical rank, so that dense-only fusion fails.
        Directly contracting each operator-inserted ket with the bra preserves
        the native block structure and fermionic metadata.
        """
        bra = state.H
        total = 0
        for where, term in terms.items():
            sites = cls._term_sites(state, where)
            inds = [state.site_ind(site) for site in sites]
            gated = qtn.tensor_network_gate_inds(
                state,
                term,
                inds,
                contract="split",
                tags=[],
                info=None,
                inplace=False,
            )
            total = total + (bra | gated).contract(
                all,
                optimize=optimize,
                **contract_opts,
            )

        if normalized:
            norm = (state.H & state).contract(
                all,
                optimize=optimize,
                **contract_opts,
            )
            total = total / norm
        return total

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
            if cls._state_uses_symmray(state) and cls._terms_use_symmray(terms):
                value = cls._symmray_exact_local_expectation(
                    state,
                    terms,
                    optimize=contraction_opt,
                    normalized=bool(normalized),
                    contract_opts=kwargs,
                )
            else:
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
        if str(autodiff_backend).strip().lower() == "torch":
            # Symmray sends raw Torch blocks through Quimb's composed split
            # paths. Configure both Autoray and those raw-block paths through
            # the one public Torch-linalg entry point.
            self._configure_torch_linalg(
                self.state,
                terms,
                quimb_split_drivers=True,
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
        fallback_boundary_mode: str | None = "exact",
        bounds=None,
        max_parameter_step: float | None = None,
        validate: bool = False,
        validation_interval: int | None = None,
        **optimize_kwargs,
    ):
        """Run ``TNOptimizer.optimize`` and store the optimized PEPS.

        ``max_parameter_step`` creates a cheap hard trust box around the
        initial flattened PEPS parameters. It is useful for noisy or
        truncated PEPS objectives where an L-BFGS line search can probe a
        very high-energy trial point. It does not perform an additional
        energy evaluation. Pass explicit ``bounds`` instead when different
        lower and upper limits are required.

        ``validate=True`` runs the optimizer in a few chunks and checks each
        candidate with an automatically doubled boundary bond dimension. A
        candidate that worsens this validation energy is rejected and the
        last validated state is returned. When ``progbar=True``, validation
        uses one outer progress bar showing training/validation chi, checked
        energy per site, validation step, and the accepted/rollback status of
        each chunk. ``validation_interval`` controls the number of optimizer
        evaluations between checks; ``None`` chooses it automatically from
        ``n``.
        """
        merged_loss_kwargs = self._merge_opts(
            self.loss_kwargs,
            self._pick_loss_kwargs(loss_kwargs),
        )
        tnopt_progbar = False if validate else progbar
        tnopt = self.make_tn_optimizer(
            loss_kwargs=merged_loss_kwargs,
            loss_constants=loss_constants,
            autodiff_backend=autodiff_backend,
            optimizer=optimizer,
            progbar=tnopt_progbar,
            jit_fn=jit_fn,
            device=device,
        )
        self._apply_parameter_bounds(
            tnopt,
            bounds=bounds,
            max_parameter_step=max_parameter_step,
        )
        optlib = optimize_kwargs.get("optlib", "scipy")
        best_tracker = self._track_tnopt_best_checkpoint(tnopt)
        active_loss_kwargs = merged_loss_kwargs
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
                active_loss_kwargs = fallback_loss_kwargs
                tnopt = self.make_tn_optimizer(
                    loss_kwargs=fallback_loss_kwargs,
                    loss_constants=loss_constants,
                    autodiff_backend=autodiff_backend,
                    optimizer=optimizer,
                    progbar=tnopt_progbar,
                    jit_fn=jit_fn,
                    device=device,
                )
                self._apply_parameter_bounds(
                    tnopt,
                    bounds=bounds,
                    max_parameter_step=max_parameter_step,
                )
                best_tracker = self._track_tnopt_best_checkpoint(tnopt)
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
        if validate:
            validation_constants = dict(loss_constants or {})
            validation_terms = validation_constants.pop("terms", self.terms)
            validation_terms = self._terms_for_autodiff_backend(
                validation_terms,
                self.state,
                autodiff_backend,
                device=device,
            )
            out = self._optimize_with_validation(
                tnopt,
                n=n,
                optimize_kwargs=optimize_kwargs,
                optlib=optlib,
                best_tracker=best_tracker,
                terms=validation_terms,
                loss_kwargs=active_loss_kwargs,
                autodiff_backend=autodiff_backend,
                device=device,
                progbar=progbar,
                validation_interval=validation_interval,
            )
        else:
            try:
                out = tnopt.optimize(n=n, **optimize_kwargs)
            except Exception as exc:
                if not self._is_recoverable_optimizer_error(exc, optlib):
                    raise
                out = self._best_tnopt_state(tnopt, best_tracker)
                best_loss = best_tracker["loss"]
                if self._is_finite_number(best_loss):
                    message = (
                        f"{optlib} optimization stopped early ({exc}); returning "
                        f"the best finite checkpoint with loss {best_loss:.12g}."
                    )
                else:
                    message = (
                        f"{optlib} optimization stopped before a finite loss was "
                        f"recorded ({exc}); returning the initial state."
                    )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
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
    ``MPS.compute_local_expectation_exact(...)``. Bosonic MPO Hamiltonians are
    evaluated directly as ``(<psi| & H_mpo & |psi>).contract(all,
    optimize=...)``, using ``contraction_opt`` for the full network
    contraction. Native fermionic MPOs are applied sitewise as a factorized
    MPO-MPS network, which preserves Symmray's graded contraction ordering
    without materializing a global operator. Repeated native-MPO evaluations
    reuse a per-optimizer contraction path cache. Optional compression can be
    requested with ``native_mpo_compression={"max_bond": ..., "cutoff": ...}``;
    it is disabled by default so the energy remains exact.
    Native fermionic Symmray states use native local terms by default when a
    mapped ``SymHamiltonian`` is supplied. A bosonic/Jordan-Wigner Symmray MPO
    cannot be silently contracted with a native fermionic state, since the
    required re-encoding can create very large block contractions; pass
    ``allow_encoding_conversion=True`` to explicitly request that conversion.
    Hamiltonians can be supplied as a ``qtn.MatrixProductOperator``, a
    ``qtn.LocalHam1D``-like object with ``.terms``, a Pepsy symmetric
    Hamiltonian, or a plain local-term mapping. Use the explicit ``terms=``
    constructor keyword when passing a local-term mapping or a Pepsy
    ``SymHamiltonian``; ``hamiltonian=`` remains supported as a compatibility
    alias.
    """

    _LOSS_KEYS = frozenset({
        "normalized",
        "energy_per_site",
        "real",
        "contraction_opt",
        "compute_kwargs",
        "progbar",
        "allow_encoding_conversion",
        "native_mpo_compression",
    })

    def __init__(
        self,
        state,
        hamiltonian=None,
        *,
        terms=None,
        normalized: bool = True,
        energy_per_site: bool = True,
        real: bool = True,
        contraction_opt: Any = "auto-hq",
        progbar: bool = False,
        compute_kwargs: Mapping[str, Any] | None = None,
        loss_kwargs: Mapping[str, Any] | None = None,
        allow_encoding_conversion: bool = False,
        native_mpo_compression: Mapping[str, Any] | None = None,
    ):
        if hamiltonian is not None and terms is not None:
            raise TypeError("pass either hamiltonian or terms, not both")
        hamiltonian = terms if terms is not None else hamiltonian
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
            "allow_encoding_conversion": bool(allow_encoding_conversion),
            "native_mpo_compression": (
                None
                if native_mpo_compression is None
                else dict(native_mpo_compression)
            ),
        }
        self._native_mpo_path_optimizer = None
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

    def _prepare_native_mpo_options(self, state, terms, options):
        """Reuse one cotengra optimizer for repeated native MPO losses."""
        options = dict(options)
        if not self._is_mpo_hamiltonian(terms):
            return options
        if self._symmray_encoding(state) != "native_fermionic":
            return options
        if self._symmray_encoding(terms) != "native_fermionic":
            return options

        contraction_opt = options.get("contraction_opt", "auto-hq")
        if contraction_opt is None or contraction_opt == "auto-hq":
            if self._native_mpo_path_optimizer is None:
                self._native_mpo_path_optimizer = build_optimizer(
                    progbar=bool(options.get("progbar", False)),
                )
            options["contraction_opt"] = self._native_mpo_path_optimizer
        return options

    @staticmethod
    def _is_mpo_hamiltonian(hamiltonian):
        return isinstance(hamiltonian, qtn.MatrixProductOperator)

    @staticmethod
    def _is_symmray_array(value):
        return hasattr(value, "blocks") and hasattr(value, "indices")

    @staticmethod
    def _is_fermionic_array(value):
        if bool(getattr(value, "fermionic", False)):
            return True
        return "fermionic" in type(value).__name__.lower()

    @classmethod
    def _symmray_encoding(cls, value):
        """Infer the representation encoding of a Symmray TN-like object."""
        encoding = getattr(value, "encoding", None)
        if encoding in {"native_fermionic", "bosonic_symmray", "mixed_symmray"}:
            return encoding

        fermionic = getattr(value, "fermionic", None)
        if fermionic is not None:
            nested = getattr(value, "tn", None)
            if nested is not None and nested is not value:
                nested_encoding = cls._symmray_encoding(nested)
                if nested_encoding is not None:
                    return nested_encoding
            if cls._is_symmray_array(value) or hasattr(value, "fermionic"):
                return "native_fermionic" if bool(fermionic) else "bosonic_symmray"

        flags = []
        for data in cls._iter_tn_data(value):
            if cls._is_symmray_array(data):
                flags.append(cls._is_fermionic_array(data))
        if not flags:
            return None
        if all(flags):
            return "native_fermionic"
        if not any(flags):
            return "bosonic_symmray"
        return "mixed_symmray"

    @classmethod
    def _mpo_uses_bosonic_symmray(cls, mpo):
        return cls._symmray_encoding(mpo) == "bosonic_symmray"

    @classmethod
    def _native_mpo_expectation(
        cls,
        state,
        mpo,
        *,
        normalized=True,
        contraction_opt="auto-hq",
        native_mpo_compression=None,
    ):
        """Evaluate a native graded MPO through a factorized MPO-MPS network.

        Symmray fermionic contractions are order-sensitive. A conventional
        MPO sandwich allows Quimb's path optimizer to interleave local MPO
        factors with bra and ket tensors, which can change the graded phase.
        Applying the MPO sitewise first keeps each local graded contraction
        together and leaves the operator bond factorized, so the contraction
        scales with the MPS and MPO bond dimensions rather than the global
        Hilbert-space dimension.
        """
        gated = qtn.tensor_network_apply_op_vec(
            mpo,
            state,
            which_A="lower",
            contract=True,
            fuse_multibonds=True,
            compress=False,
            inplace=False,
            inplace_A=False,
        )
        if native_mpo_compression is not None:
            compression_opts = dict(native_mpo_compression)
            max_bond = compression_opts.get("max_bond")
            if max_bond is None:
                raise ValueError(
                    "native_mpo_compression requires an explicit max_bond."
                )
            max_bond = int(max_bond)
            if max_bond < 1:
                raise ValueError(
                    "native_mpo_compression max_bond must be positive."
                )
            cutoff = compression_opts.get("cutoff", 1e-12)
            if cutoff < 0.0:
                raise ValueError(
                    "native_mpo_compression cutoff must be non-negative."
                )
            compression_opts["max_bond"] = max_bond
            compression_opts["cutoff"] = cutoff
            compression_opts.setdefault("method", "svd")
            gated.compress(**compression_opts)
        value = (state.H | gated).contract(all, optimize=contraction_opt)
        if normalized:
            norm = (state.H & state).contract(all, optimize=contraction_opt)
            if norm == 0.0:
                raise ValueError("Cannot compute normalized energy for a zero-norm state.")
            value = value / norm
        return value

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
        if cls._is_fermionic_sym_hamiltonian(hamiltonian):
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
    def _is_fermionic_sym_hamiltonian(cls, hamiltonian):
        terms = getattr(hamiltonian, "terms", None)
        to_mpo = getattr(hamiltonian, "to_mpo", None)
        return (
            terms is not None
            and callable(to_mpo)
            and cls._terms_use_fermionic_symmray(terms)
        )

    @classmethod
    def _fermionic_hamiltonian_mpo_for_state(cls, hamiltonian, state):
        try:
            return hamiltonian.to_mpo(
                L=cls._num_sites(state),
                compress=True,
                cutoff=1e-12,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Fermionic SymHamiltonian energy on an MPS is evaluated through "
                "the string-aware MPO path. If the Hamiltonian edges use lattice "
                "coordinates, build the MPO explicitly with "
                "`hamiltonian.to_mpo(mapper=...)` and pass that MPO to "
                "MpsEnergyOptimizer."
            ) from exc

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
        return cls._symmray_encoding(state) == "bosonic_symmray"

    @classmethod
    def _state_uses_fermionic_symmray(cls, state):
        return cls._symmray_encoding(state) == "native_fermionic"

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
    def _can_use_native_local_terms(cls, hamiltonian, state):
        """Whether a symmetric Hamiltonian is already indexed by MPS sites."""
        if not cls._state_uses_fermionic_symmray(state):
            return False
        if not cls._terms_use_fermionic_symmray(hamiltonian.terms):
            return False
        return all(
            len(where) in {1, 2}
            and all(isinstance(site, Integral) for site in where)
            for where in hamiltonian.edges
        )

    @classmethod
    def _mpo_expectation(
        cls,
        state,
        mpo,
        *,
        normalized=True,
        contraction_opt="auto-hq",
        allow_encoding_conversion=False,
        native_mpo_compression=None,
    ):
        if contraction_opt is None:
            contraction_opt = build_optimizer(progbar=False)
        state = cls._as_mps_state(state)
        state_encoding = cls._symmray_encoding(state)
        mpo_encoding = cls._symmray_encoding(mpo)
        if (
            state_encoding == "native_fermionic"
            and mpo_encoding == "bosonic_symmray"
            and not allow_encoding_conversion
        ):
            raise ValueError(
                "Symmray encoding mismatch in MPS energy evaluation: the state "
                "is native fermionic but the MPO is bosonic/Jordan-Wigner. "
                "Use a mapped native local-term Hamiltonian (for example "
                "`hamiltonian=ham.terms`) for large-chi energy evaluation, or "
                "pass `allow_encoding_conversion=True` to explicitly request "
                "the potentially memory-intensive re-encoding."
            )
        if mpo_encoding == "native_fermionic":
            if state_encoding != "native_fermionic":
                raise ValueError(
                    "Native fermionic MPO energy evaluation requires a native "
                    "fermionic Symmray MPS state."
                )
            return cls._native_mpo_expectation(
                state,
                mpo,
                normalized=normalized,
                contraction_opt=contraction_opt,
                native_mpo_compression=native_mpo_compression,
            )
        if native_mpo_compression is not None:
            raise ValueError(
                "native_mpo_compression is only supported for native "
                "fermionic Symmray MPOs."
            )
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
        allow_encoding_conversion=False,
        native_mpo_compression=None,
    ):
        state = cls._as_mps_state(state)
        terms = cls._terms_from_hamiltonian(terms)
        if cls._is_fermionic_sym_hamiltonian(terms):
            if cls._can_use_native_local_terms(terms, state):
                terms = terms.terms
            else:
                terms = cls._fermionic_hamiltonian_mpo_for_state(terms, state)
        if cls._is_mpo_hamiltonian(terms):
            value = cls._mpo_expectation(
                state,
                terms,
                normalized=normalized,
                contraction_opt=contraction_opt,
                allow_encoding_conversion=allow_encoding_conversion,
                native_mpo_compression=native_mpo_compression,
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
        opts = self._prepare_native_mpo_options(state, terms_use, opts)
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
        opts = self._prepare_native_mpo_options(state, terms_use, opts)
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
                "native_mpo_compression": (
                    None
                    if opts["native_mpo_compression"] is None
                    else dict(opts["native_mpo_compression"])
                ),
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
        if str(autodiff_backend).strip().lower() == "torch":
            # MPS optimization uses the same SVD/QR autodiff policy, but has
            # no Quimb PEPS boundary raw-block path to configure.
            self._configure_torch_linalg(
                self.state,
                {},
                quimb_split_drivers=False,
            )
        if self._is_fermionic_sym_hamiltonian(terms):
            if self._can_use_native_local_terms(terms, self.state):
                terms = terms.terms
            else:
                terms = self._fermionic_hamiltonian_mpo_for_state(terms, self.state)
        merged_loss_kwargs = self._prepare_native_mpo_options(
            self.state,
            terms,
            merged_loss_kwargs,
        )
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
