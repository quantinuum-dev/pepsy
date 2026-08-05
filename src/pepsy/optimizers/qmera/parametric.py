"""Parameter-dict qMERA energy optimizer shell."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ...backends import backend_jax, backend_torch
from ...solvers import GradientOptimizer, GradSolverResult

__all__ = ["QMeraEnergyOptimizer", "QMeraParametricEnergyOptimizer"]


def _solver_backend(solver):
    key = str(solver).strip().lower()
    if key.startswith("jax-"):
        return "jax"
    if key.startswith("torch-") or key in {"adam", "adamw", "radam", "nadam", "lbfgs"}:
        return "torch"
    return "torch"


def _array_backend_for_train_backend(backend, *, dtype=None, device="cpu"):
    if callable(backend) or backend is None:
        return backend
    key = str(backend).strip().lower().replace("_", "-")
    if key in {"torch", "pytorch"}:
        if dtype is None:
            try:
                import torch  # pylint: disable=import-outside-toplevel

                dtype = torch.complex128
            except ImportError:  # pragma: no cover - backend_torch raises next
                pass
        return backend_torch(device=device, dtype=dtype, requires_grad=False)
    if key == "jax":
        return backend_jax(device=device, dtype=dtype)
    return None


@dataclass
class QMeraEnergyOptimizer:
    """Optimize parameterized qMERA gates using schedule-only lightcones.

    The built-in qMERA gate families are unitary (and the fermion families
    are symmetry/parity preserving) by construction.  Consequently the
    optimizer does not normalize the state by default.  Pass
    ``normalized=True`` in ``loss_kwargs`` or to ``loss``/``run`` when using
    a custom gate family that is not norm preserving.
    """

    builder: Any
    schedule: Any
    hamiltonian: Any
    chunks: Any = None
    compiled_chunks: Any = None
    parameters: Mapping[str, Any] | None = None
    loss_kwargs: Mapping[str, Any] | None = None
    result: GradSolverResult | None = None
    losses: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.loss_kwargs = {} if self.loss_kwargs is None else dict(self.loss_kwargs)
        self.loss_kwargs.setdefault("normalized", False)
        if self.parameters is None:
            self.parameters = self.builder.initialize_parameters(self.schedule)
        else:
            self.parameters = dict(self.parameters)

    @staticmethod
    def _merge_opts(base, extra):
        opts = dict(base or {})
        if extra:
            opts.update(dict(extra))
        return opts

    def cast_params(
        self,
        values,
        *,
        trainable: bool = True,
        backend=None,
        dtype=None,
        device="cpu",
        stop_grad_fn=None,
    ):
        """Cast a parameter dictionary through the owning qMERA builder."""
        return self.builder.cast_params(
            values,
            trainable=trainable,
            backend=backend,
            dtype=dtype,
            device=device,
            stop_grad_fn=stop_grad_fn,
        )

    def _chunks_for_backend(self, *, chunks=None, array_backend=None, convert_terms=True):
        if chunks is not None:
            return chunks
        if self.hamiltonian is None:
            return self.chunks
        return self.builder.parametric_lightcone_chunks(
            self.hamiltonian,
            self.schedule,
            array_backend=array_backend,
            convert_terms=convert_terms,
        )

    def loss(self, parameters=None, **kwargs):
        """Evaluate ``loss(params)`` for the configured schedule and terms."""
        params = self.parameters if parameters is None else parameters
        opts = self._merge_opts(self.loss_kwargs, kwargs)
        chunks = self._chunks_for_backend(
            chunks=opts.pop("chunks", self.chunks),
            array_backend=opts.get("array_backend"),
            convert_terms=opts.get("convert_terms", True),
        )
        return self.builder.parametric_loss(
            params,
            self.hamiltonian,
            schedule=self.schedule,
            chunks=chunks,
            **opts,
        )

    def loss_fn(self, **kwargs):
        """Return a pure callable with signature ``loss(params) -> scalar``."""

        def _loss(parameters):
            return self.loss(parameters, **kwargs)

        return _loss

    def compile(
        self,
        *,
        chunks=None,
        array_backend=None,
        convert_terms=True,
        contraction_opt="auto-hq",
        expression_opts=None,
        path_cache=None,
    ):
        """Compile static contraction expressions for configured local cones."""
        chunks = self._chunks_for_backend(
            chunks=chunks if chunks is not None else self.chunks,
            array_backend=array_backend,
            convert_terms=convert_terms,
        )
        self.chunks = chunks
        self.compiled_chunks = self.builder.compile_parametric_lightcones(
            schedule=self.schedule,
            chunks=chunks,
            array_backend=array_backend,
            convert_terms=convert_terms,
            contraction_opt=contraction_opt,
            expression_opts=expression_opts,
            path_cache=path_cache,
        )
        return self.compiled_chunks

    def compiled_loss(self, parameters=None, **kwargs):
        """Evaluate loss with precompiled local-cone contraction expressions."""
        params = self.parameters if parameters is None else parameters
        opts = self._merge_opts(self.loss_kwargs, kwargs)
        compiled_chunks = opts.pop("compiled_chunks", self.compiled_chunks)
        if compiled_chunks is None:
            compiled_chunks = self.compile(
                chunks=opts.get("chunks", self.chunks),
                array_backend=opts.get("array_backend"),
                convert_terms=opts.get("convert_terms", True),
                contraction_opt=opts.get("contraction_opt", "auto-hq"),
                expression_opts=opts.get("expression_opts"),
                path_cache=opts.get("path_cache"),
            )
        return self.builder.compiled_parametric_loss(
            params,
            self.hamiltonian,
            schedule=self.schedule,
            chunks=opts.pop("chunks", self.chunks),
            compiled_chunks=compiled_chunks,
            **opts,
        )

    def compiled_loss_fn(self, **kwargs):
        """Return a pure compiled ``loss(params) -> scalar`` callable."""

        def _loss(parameters):
            return self.compiled_loss(parameters, **kwargs)

        return _loss

    def run(
        self,
        params_init=None,
        *,
        solver="torch-adam",
        backend=None,
        array_backend=None,
        dtype=None,
        array_dtype=None,
        device="cpu",
        options: Mapping[str, Any] | None = None,
        n_steps: int = 100,
        log_every: int = 20,
        progress: bool = False,
        desc: str | None = None,
        progress_callback=None,
        compiled: bool = False,
        expression_opts=None,
        **loss_kwargs,
    ):
        """Optimize parameters with :class:`pepsy.solvers.GradientOptimizer`."""
        backend = _solver_backend(solver) if backend is None else backend
        array_backend = (
            _array_backend_for_train_backend(backend, dtype=array_dtype, device=device)
            if array_backend is None
            else array_backend
        )
        params = self.parameters if params_init is None else params_init
        params = self.cast_params(
            params,
            trainable=True,
            backend=backend,
            dtype=dtype,
            device=device,
        )
        opts = self._merge_opts(self.loss_kwargs, loss_kwargs)
        opts["array_backend"] = array_backend
        chunks = self._chunks_for_backend(
            array_backend=array_backend,
            convert_terms=opts.get("convert_terms", True),
        )
        opts["chunks"] = chunks
        if compiled:
            self.compiled_chunks = self.builder.compile_parametric_lightcones(
                schedule=self.schedule,
                chunks=chunks,
                array_backend=array_backend,
                convert_terms=opts.get("convert_terms", True),
                contraction_opt=opts.get("contraction_opt", "auto-hq"),
                expression_opts=expression_opts,
                path_cache=opts.get("path_cache"),
            )
            opts["compiled_chunks"] = self.compiled_chunks

        runner = GradientOptimizer(
            solver=solver,
            options=options,
            n_steps=n_steps,
            log_every=log_every,
            progress=progress,
            desc=desc,
        )
        result = runner.run(
            params_init=params,
            loss_fn=self.compiled_loss_fn(**opts) if compiled else self.loss_fn(**opts),
            progress_callback=progress_callback,
        )
        self.result = result
        self.losses = list(result.history)
        self.parameters = dict(result.params)
        self.chunks = chunks
        return result

    optimize = run


# Compatibility name retained for callers of the original parameter-dict
# qMERA API.  The canonical public name is now QMeraEnergyOptimizer.
QMeraParametricEnergyOptimizer = QMeraEnergyOptimizer
