"""PEPS amplitude models and connected-configuration batching.

The classes in this module own PEPS parameter packing, exact/approximate
amplitude evaluation, boundary reuse, and the small batching helpers shared by
local estimators and samplers.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

from ..torch_types import _check_positive_int, _require_torch
from ..api import ContractionFallbackWarning
from ...boundary.metrics import (
    _ctmrg_stabilization_kwargs,
    quimb_ctmrg_projector_compat,
)
from ._common import (
    _as_contraction_options,
    _as_long_matrix,
    _check_nonnegative_int,
    _run_cheap_torch_kernel,
    _torch_finfo_tiny,
    _validate_contraction,
)
from .connections import TorchConnections
from ._graded import _GradedTorchProjector, _find_symmray_tensors


_PROPOSAL_BATCHING_MODES = {"auto", "cache", "vmap"}
_AMPLITUDE_BATCHING_MODES = {"auto", "serial", "vmap"}
_BOUNDARY_VMAP_CONNECTION_THRESHOLD = 64

__all__ = [
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "make_torch_peps_amplitude_model",
    "_call_amplitude_fn",
    "_default_connected_amplitudes",
    "_normalize_amplitude_batching",
    "_normalize_chunk_size",
    "_resolve_log_amplitude_fn",
]


def _as_torch_scalar(value, reference):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value
    if reference is None:
        return torch.as_tensor(value)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _normalize_chunk_size(chunk_size, *, name="chunk_size"):
    if chunk_size is None:
        return None
    return _check_positive_int(name, chunk_size)


def _normalize_proposal_batching(proposal_batching):
    """Normalize the boundary-proposal batching policy."""
    mode = str(proposal_batching).replace("_", "-").lower()
    if mode not in _PROPOSAL_BATCHING_MODES:
        choices = ", ".join(
            repr(choice) for choice in sorted(_PROPOSAL_BATCHING_MODES)
        )
        raise ValueError(f"proposal_batching must be one of {choices}.")
    return mode


def _normalize_amplitude_batching(amplitude_batching):
    """Normalize the independent-amplitude batching policy."""
    mode = str(amplitude_batching).replace("_", "-").lower()
    aliases = {"loop": "serial"}
    mode = aliases.get(mode, mode)
    if mode not in _AMPLITUDE_BATCHING_MODES:
        choices = ", ".join(
            repr(choice) for choice in sorted(_AMPLITUDE_BATCHING_MODES)
        )
        raise ValueError(f"amplitude_batching must be one of {choices}.")
    return mode


def _call_amplitude_fn(amplitude_fn, configs, *, chunk_size=None):
    """Evaluate ``amplitude_fn`` on ``configs``, optionally in chunks."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    chunk_size = _normalize_chunk_size(chunk_size)
    if chunk_size is None or configs.shape[0] <= chunk_size:
        return torch.as_tensor(amplitude_fn(configs), device=configs.device)

    pieces = []
    for start in range(0, configs.shape[0], chunk_size):
        stop = min(start + chunk_size, configs.shape[0])
        pieces.append(torch.as_tensor(
            amplitude_fn(configs[start:stop]),
            device=configs.device,
        ))
    return torch.cat(pieces, dim=0)


def _resolve_log_amplitude_fn(amplitude_fn, log_amplitude_fn=None):
    """Resolve an optional ``(phase, log_abs)`` amplitude interface."""
    if log_amplitude_fn is False:
        return None
    if log_amplitude_fn is not None:
        if not callable(log_amplitude_fn):
            raise TypeError("log_amplitude_fn must be callable or False.")
        return log_amplitude_fn
    for name in ("forward_log", "log_amplitude"):
        candidate = getattr(amplitude_fn, name, None)
        if callable(candidate):
            return candidate
    return None


def _call_log_amplitude_fn(log_amplitude_fn, configs, *, chunk_size=None):
    """Evaluate a log-amplitude function in optional fixed-size chunks."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    chunk_size = _normalize_chunk_size(chunk_size)
    if chunk_size is None or configs.shape[0] <= chunk_size:
        chunks = (configs,)
    else:
        chunks = (
            configs[start:min(start + chunk_size, configs.shape[0])]
            for start in range(0, configs.shape[0], chunk_size)
        )

    phases = []
    log_abs = []
    for chunk in chunks:
        result = log_amplitude_fn(chunk)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError(
                "log_amplitude_fn must return a (phase, log_abs) pair."
            )
        phase, chunk_log_abs = result
        phase = torch.as_tensor(phase, device=chunk.device)
        chunk_log_abs = torch.as_tensor(chunk_log_abs, device=chunk.device)
        if phase.ndim != 1 or chunk_log_abs.ndim != 1:
            raise ValueError(
                "log_amplitude_fn must return one phase and one log magnitude "
                "per configuration."
            )
        if phase.shape[0] != chunk.shape[0] or (
            chunk_log_abs.shape[0] != chunk.shape[0]
        ):
            raise ValueError(
                "log_amplitude_fn outputs must have one entry per configuration."
            )
        phases.append(phase)
        log_abs.append(chunk_log_abs.real)
    return torch.cat(phases, dim=0), torch.cat(log_abs, dim=0)


def _diagonal_connection_mask(configs, connections):
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=configs.device)
    parents = configs[connections.batch_ids]
    return torch.all(connections.configs == parents, dim=1)


def _connection_key_rows(batch_ids, configs):
    """Pack fixed-width connection keys before eager unique grouping."""
    torch = _require_torch()
    return torch.cat((batch_ids.reshape(-1, 1), configs), dim=1)


def _coalesce_connections(connections, *, device=None, compile_kernels=False):
    """Merge duplicate ``(batch_id, connected_config)`` rows.

    Local Hamiltonians assembled from several terms can emit the same target
    configuration more than once. Summing those coefficients before amplitude
    evaluation avoids redundant contractions while preserving the exact local
    energy.
    """
    torch = _require_torch()
    if connections.configs.numel() == 0:
        return connections

    target_device = connections.configs.device if device is None else device
    configs = connections.configs.to(device=target_device, dtype=torch.long)
    batch_ids = connections.batch_ids.to(device=target_device, dtype=torch.long)
    coeffs = connections.coeffs.to(device=target_device)
    keys = _run_cheap_torch_kernel(
        "connection-key-rows",
        _connection_key_rows,
        batch_ids,
        configs,
        compile_kernels=compile_kernels,
    )
    unique_keys, inverse = torch.unique(
        keys,
        dim=0,
        return_inverse=True,
        sorted=False,
    )
    unique_coeffs = torch.zeros(
        unique_keys.shape[0],
        dtype=coeffs.dtype,
        device=target_device,
    )
    unique_coeffs.index_add_(0, inverse, coeffs)
    nonzero = unique_coeffs != 0
    return TorchConnections(
        configs=unique_keys[nonzero, 1:],
        coeffs=unique_coeffs[nonzero],
        batch_ids=unique_keys[nonzero, 0],
    )


def _unique_config_rows(configs):
    """Return unique configuration rows and an inverse scatter index."""
    torch = _require_torch()
    if configs.shape[0] <= 1:
        return configs, None
    return torch.unique(
        configs,
        dim=0,
        return_inverse=True,
        sorted=False,
    )


def _default_connected_amplitudes(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    """Evaluate connected amplitudes, copying diagonal terms when possible."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if connections.configs.numel() == 0:
        return torch.empty(0, dtype=amplitudes.dtype, device=configs.device)

    if not reuse_diagonal:
        return _call_amplitude_fn(
            amplitude_fn,
            connections.configs,
            chunk_size=chunk_size,
        )

    diag = _diagonal_connection_mask(configs, connections)
    if not bool(torch.any(diag)):
        return _call_amplitude_fn(
            amplitude_fn,
            connections.configs,
            chunk_size=chunk_size,
        )

    out = torch.empty(
        connections.configs.shape[0],
        dtype=amplitudes.dtype,
        device=configs.device,
    )
    out[diag] = amplitudes[connections.batch_ids[diag]]
    offdiag = ~diag
    if bool(torch.any(offdiag)):
        unique_configs, inverse = _unique_config_rows(
            connections.configs[offdiag]
        )
        unique_amplitudes = _call_amplitude_fn(
            amplitude_fn,
            unique_configs,
            chunk_size=chunk_size,
        )
        if inverse is None:
            out[offdiag] = unique_amplitudes
        else:
            out[offdiag] = unique_amplitudes[inverse].to(
                dtype=out.dtype,
                device=out.device,
            )
    return out


class TorchPEPSAmplitude:
    """Torch-optimizable amplitude wrapper for a quimb PEPS-like network.

    The input configuration rows are physical indices in the PEPS site order by
    default. For spin PEPS this usually means binary rows ``0/1``. For spinful
    Hubbard PEPS use a four-state row encoding that matches the PEPS physical
    basis, for example :class:`FermionSiteEncoding.symmray`.

    This class deliberately stays pure PEPS/TNS: it registers the packed PEPS
    tensor leaves as torch parameters and evaluates amplitudes by selecting
    physical indices then contracting the resulting quimb tensor network.
    Dense quimb tensors and Symmray block-sparse tensors are both handled
    through ``quimb.tensor.pack`` / ``unpack``. For Symmray, this preserves the
    array's own pytree metadata, including fermionic phases and charge sectors,
    while replacing numeric block leaves with torch trainable parameters.

    Set ``graded_torch=True`` for native sparse U1U1 PEPS to use the exact
    fixed-shape Torch projector. That opt-in path compiles Symmray's graded
    charge and phase rules once, then performs the per-configuration dense
    contractions under ``torch.vmap``.

    ``amplitude_batching`` controls independent configuration batches. The
    default ``"auto"`` probes ``torch.vmap`` once and permanently falls back
    to the serial contraction path if the selected PEPS backend is not
    vmappable. Use ``"vmap"`` for flat Z2 Symmray PEPS when the fast path is
    known to be supported, or ``"serial"`` for native U1/U1U1 PEPS and other
    dynamic block-sparse contractions. A failed explicit ``"vmap"`` request
    still falls back safely rather than changing numerical semantics.

    For a long native CTMRG run, ``contraction_opts`` may include
    ``fallback_contraction="boundary_mps"``. The first CTMRG exception then
    switches the model to boundary-MPS for the remainder of the run and
    records the fallback cause; omit it (or use ``"none"``) for strict
    failure behavior.
    """

    def __init__(
        self,
        peps,
        *,
        contraction="exact",
        chi=None,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        site_order=None,
        graded_torch=False,
        amplitude_batching="auto",
    ):
        torch = _require_torch()
        import quimb as qu
        import quimb.tensor as qtn

        from ..api import _resolve_contraction_config
        contraction, chi, cutoff, contraction_opts = _resolve_contraction_config(
            contraction,
            chi,
            cutoff,
            contraction_opts,
        )

        self.contraction = _validate_contraction(contraction, chi)
        self.chi = None if chi is None else int(chi)
        self.cutoff = (
            0.0
            if cutoff is None and self.contraction == "exact"
            else 1.0e-10 if cutoff is None else float(cutoff)
        )
        contraction_opts = _as_contraction_options(contraction_opts)
        fallback_contraction = contraction_opts.pop(
            "fallback_contraction", None
        )
        if fallback_contraction is not None:
            fallback_key = str(fallback_contraction).replace("_", "-").lower()
            if fallback_key in {"none", "off", "false"}:
                fallback_contraction = None
            else:
                fallback_contraction = _validate_contraction(
                    fallback_contraction,
                    self.chi,
                )
                if fallback_contraction != "boundary":
                    raise ValueError(
                        "fallback_contraction currently supports only "
                        "'boundary_mps'."
                    )
        self.contraction_fallback = fallback_contraction
        self.requested_contraction = self.contraction
        self.contraction_opts = contraction_opts
        if self.contraction == "boundary":
            self.contraction_opts.setdefault("mode", "mps")

        tn = getattr(peps, "tn", peps)
        if not hasattr(tn, "sites"):
            raise TypeError("peps must be a quimb PEPS-like object with sites.")
        self.symmray_tensor_ids = tuple(_find_symmray_tensors(tn))
        self.sites = tuple(tn.sites if site_order is None else site_order)
        missing = [site for site in self.sites if site not in tn.sites]
        if missing:
            raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")
        self.site_inds = tuple(tn.site_ind(site) for site in self.sites)
        self.cutoff_fallbacks = 0
        # Count outer CTMRG contractions, not the internal CTMRG iterations
        # performed by Quimb. This is useful for estimating the cost of a
        # stored-sample replay.
        self.ctmrg_calls = 0
        self.ctmrg_failures = 0
        self.contraction_fallbacks = 0
        self.contraction_fallback_error = None
        self.final_optimizer_fallbacks = 0
        self._warned_final_optimizer_fallback = False
        self.graded_torch = bool(graded_torch)
        self.amplitude_batching = _normalize_amplitude_batching(
            amplitude_batching
        )
        self.last_amplitude_batching = None
        if self.graded_torch:
            if self.contraction != "exact":
                raise ValueError(
                    "graded_torch currently supports contraction='exact' only."
                )
            if not self.symmray_tensor_ids:
                raise TypeError(
                    "graded_torch requires a native Symmray fermionic PEPS."
                )
            self._graded_torch_projector = _GradedTorchProjector(
                tn,
                self.sites,
                contraction_opts=self.contraction_opts,
            )
        else:
            self._graded_torch_projector = None

        params, skeleton = qtn.pack(tn)
        flat_params, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        leaves = []
        for leaf in flat_params:
            tensor = torch.as_tensor(leaf, dtype=dtype, device=device)
            leaves.append(torch.nn.Parameter(tensor.clone()))
        self.params = torch.nn.ParameterList(leaves)
        self.params_pytree = params_pytree
        self.skeleton = skeleton
        if (
            self.contraction == "ctmrg"
            and self.symmray_tensor_ids
            and self.contraction_opts.get("mode") is None
        ):
            # The scoped native-fermionic CTMRG compatibility layer repairs
            # projector insertion and zero sectors. Keep the projector route
            # as the default for Torch-backed Symmray: the direct boundary
            # compressor can mix NumPy intermediates with Symmray blocks on
            # larger native PEPS. A caller-provided mode remains an explicit
            # override.
            self.contraction_opts["mode"] = "projector"
        # ``torch.vmap`` can batch the pure tensor contractions for dense and
        # compatible Symmray PEPS. Keep a per-model fallback for contraction
        # paths or optional backends that cannot be vmapped.
        has_vmap = callable(getattr(torch, "vmap", None))
        self._vmap_forward_enabled = has_vmap
        self._vmap_log_enabled = has_vmap
        # Connected estimators and proposal batches are independent fast
        # paths. A failed ordinary amplitude trace must not poison either one.
        self._connection_vmap_enabled = has_vmap
        # Populated only by the explicit export/compile opt-in. Keeping these
        # separate from the ordinary vmap flags preserves the established
        # eager fallback for unsupported contraction paths.
        self._exported_amplitude_program = None
        self._compiled_amplitude_forward = None
        self._exported_log_amplitude_program = None
        self._compiled_log_amplitude_forward = None
        self._compiled_batch_size = None
        self._compiled_mode = None
        self._compiled_backend = None

    @property
    def is_symmray(self):
        """Whether the wrapped PEPS contains Symmray tensor data."""
        return bool(self.symmray_tensor_ids)

    @property
    def n_sites(self):
        """Number of physical sites expected in each config row."""
        return len(self.sites)

    @property
    def n_params(self):
        """Number of scalar PEPS tensor parameters."""
        return int(sum(p.numel() for p in self.params))

    def parameters(self):
        """Return trainable PEPS tensor parameters for ``torch.optim``."""
        return self.params.parameters()

    def named_parameters(self):
        """Return named trainable PEPS tensor parameters."""
        return self.params.named_parameters()

    def zero_grad(self, *, set_to_none=True):
        """Clear parameter gradients."""
        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    def to(self, *args, **kwargs):
        """Move/cast PEPS tensor parameters, mirroring ``torch.nn.Module.to``."""
        self.clear_compiled()
        self.params.to(*args, **kwargs)
        return self

    def clear_compiled(self):
        """Drop exported and compiled amplitude graphs.

        Exported graphs contain device- and shape-specific constants. Parameter
        value updates do not require clearing because packed PEPS leaves are
        explicit graph inputs, but moving or casting the model does.
        """
        self._exported_amplitude_program = None
        self._compiled_amplitude_forward = None
        self._exported_log_amplitude_program = None
        self._compiled_log_amplitude_forward = None
        self._compiled_batch_size = None
        self._compiled_mode = None
        self._compiled_backend = None
        return self

    def export_and_compile(
        self,
        example_configs,
        *,
        mode="default",
        backend="inductor",
        dynamic=False,
        strict=False,
        compile_log=True,
    ):
        """Export, batch, and compile the exact PEPS amplitude closure.

        This follows the GPU VMC pipeline used by the reference
        ``vmc_torch`` implementation: export one configuration with
        ``torch.export``, add the walker axis with ``torch.vmap``, and compile
        the fixed-shape batch with ``torch.compile``. The feature is opt-in
        because export support depends on the tensor-network backend.

        ``example_configs`` may be one configuration or a batch. Its batch
        size becomes the compiled size. Calls through :meth:`forward` use the
        compiled route when they provide that size; :meth:`compiled_forward`
        makes the fixed-size requirement explicit. ``dynamic`` is forwarded to
        ``torch.compile``, but the default is static to match compiled GPU
        sampling.

        Only ``contraction="exact"`` is supported. Approximate boundary,
        CTMRG, and HOTRG paths retain their existing eager/vmap policies. The
        native ``graded_torch`` projector likewise keeps its own fixed-shape
        route.

        ``strict=False`` is the default because quimb's packed pytree contains
        static Symmray metadata tensors that PyTorch 2.x can lift in non-strict
        export, while strict export rejects them on some releases.

        Parameters
        ----------
        example_configs : array-like
            Shape ``(n_sites,)`` or ``(batch, n_sites)``.
        mode, backend, dynamic
            Forwarded to ``torch.compile``.
        strict : bool
            Forwarded to ``torch.export.export``.
        compile_log : bool
            Also compile the stable ``(phase, log_abs)`` closure used by
            Metropolis acceptance.

        Returns
        -------
        TorchPEPSAmplitude
            This model, for setup chaining.
        """
        torch = _require_torch()
        if self.contraction != "exact":
            raise ValueError(
                "export_and_compile currently supports contraction='exact' "
                "only; keep approximate PEPS paths on eager/vmap evaluation."
            )
        if self.graded_torch:
            raise ValueError(
                "export_and_compile is for dense or flat PEPS; "
                "graded_torch already provides its own fixed-shape projector."
            )

        exporter = getattr(getattr(torch, "export", None), "export", None)
        compiler = getattr(torch, "compile", None)
        if not callable(exporter) or not callable(compiler):
            raise RuntimeError(
                "export_and_compile requires a PyTorch build exposing both "
                "torch.export.export and torch.compile."
            )

        example_configs = _as_long_matrix(example_configs)
        if example_configs.shape[0] == 0:
            raise ValueError("example_configs must contain at least one row.")
        if example_configs.shape[1] != self.n_sites:
            raise ValueError(
                "example_configs must have one physical index per site; "
                f"expected {self.n_sites}, got {example_configs.shape[1]}."
            )

        # The public model intentionally remains outside nn.Module. Passing
        # each packed leaf explicitly keeps the existing optimizer interface
        # while making graph parameters visible to torch.export.
        model = self

        class _SingleAmplitudeModule(torch.nn.Module):
            def __init__(self, *, log=False):
                super().__init__()
                self.log = bool(log)

            def forward(self, config, *params):
                if self.log:
                    phase, log_abs = model.forward_log(
                        config.reshape(1, -1),
                        params=params,
                    )
                    return phase[0], log_abs[0]
                return model.amplitude(config, params=params)

        params = list(self.params)
        in_dims = (0,) + (None,) * len(params)

        def build(log=False):
            wrapper = _SingleAmplitudeModule(log=log)
            try:
                # PyTorch 2.6 emits harmless FX warnings while un lifting
                # static quimb/Symmray metadata. Keep this opt-in API quiet.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Attempted to insert a get_attr Node.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Node lifted_tensor_.*",
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message="Additional .* warnings suppressed about "
                        "get_attr references",
                        category=UserWarning,
                    )
                    exported = exporter(
                        wrapper,
                        (example_configs[0], *params),
                        strict=strict,
                    )
                    vmapped = torch.vmap(
                        exported.module(),
                        in_dims=in_dims,
                    )
                    compile_kwargs = {
                        "backend": backend,
                        "dynamic": dynamic,
                    }
                    if mode is not None:
                        compile_kwargs["mode"] = mode
                    compiled = compiler(vmapped, **compile_kwargs)
            except Exception as error:
                name = "log-amplitude" if log else "amplitude"
                raise RuntimeError(
                    f"Could not export and compile the exact PEPS {name} "
                    "closure. Try strict=False, backend='eager' for a "
                    "diagnostic, or retain the ordinary vmap/serial path."
                ) from error
            return exported, compiled

        # Rebuilding is atomic: an unsuccessful rebuild cannot leave a graph
        # tied to the previous device or batch shape.
        self.clear_compiled()
        exported_amplitude, compiled_amplitude = build(log=False)
        exported_log = compiled_log_fn = None
        if compile_log:
            try:
                exported_log, compiled_log_fn = build(log=True)
            except RuntimeError:
                self.clear_compiled()
                raise

        self._exported_amplitude_program = exported_amplitude
        self._compiled_amplitude_forward = compiled_amplitude
        self._exported_log_amplitude_program = exported_log
        self._compiled_log_amplitude_forward = compiled_log_fn
        self._compiled_batch_size = int(example_configs.shape[0])
        self._compiled_mode = mode
        self._compiled_backend = backend
        return self

    def compiled_forward(self, configs):
        """Evaluate a fixed-size batch through the compiled graph."""
        if self._compiled_amplitude_forward is None:
            raise RuntimeError(
                "No compiled amplitude graph is available; call "
                "export_and_compile(...) first."
            )
        configs = _as_long_matrix(configs)
        if configs.shape[0] != self._compiled_batch_size:
            raise ValueError(
                "compiled_forward requires exactly "
                f"{self._compiled_batch_size} configurations, got "
                f"{configs.shape[0]}."
            )
        if configs.shape[1] != self.n_sites:
            raise ValueError(
                f"configs must have {self.n_sites} sites per row."
            )
        self.last_amplitude_batching = "export-vmap-compile"
        return self._compiled_amplitude_forward(
            configs,
            *list(self.params),
        )

    def compiled_forward_log(self, configs):
        """Evaluate stable ``(phase, log_abs)`` through the compiled graph."""
        if self._compiled_log_amplitude_forward is None:
            raise RuntimeError(
                "No compiled log-amplitude graph is available; call "
                "export_and_compile(..., compile_log=True) first."
            )
        configs = _as_long_matrix(configs)
        if configs.shape[0] != self._compiled_batch_size:
            raise ValueError(
                "compiled_forward_log requires exactly "
                f"{self._compiled_batch_size} configurations, got "
                f"{configs.shape[0]}."
            )
        if configs.shape[1] != self.n_sites:
            raise ValueError(
                f"configs must have {self.n_sites} sites per row."
            )
        self.last_amplitude_batching = "export-vmap-compile-log"
        return self._compiled_log_amplitude_forward(
            configs,
            *list(self.params),
        )

    def _params_pytree(self, params=None):
        import quimb as qu

        if params is None:
            params = list(self.params)
        elif isinstance(params, _require_torch().nn.ParameterList):
            params = list(params)
        return qu.utils.tree_unflatten(params, self.params_pytree)

    def to_peps(self, *, detach=True, device="cpu"):
        """Return a quimb PEPS-like object with the current tensor parameters."""
        import quimb.tensor as qtn

        leaves = []
        for param in self.params:
            leaf = param.detach() if detach else param
            if device is not None:
                leaf = leaf.to(device)
            leaves.append(leaf)
        return qtn.unpack(self._params_pytree(leaves), self.skeleton)

    def _unpack_tn(self, params=None):
        import quimb.tensor as qtn

        return qtn.unpack(self._params_pytree(params), self.skeleton)

    def _reference_tensor(self, params=None):
        torch = _require_torch()
        if params is None:
            params = self.params
        if isinstance(params, torch.nn.ParameterList):
            params = list(params)
        try:
            return next(iter(params))
        except StopIteration:
            return None

    def _select_config(self, tn, config):
        if config.shape[0] != self.n_sites:
            raise ValueError(
                f"config row has length {config.shape[0]}, expected {self.n_sites}."
            )
        return tn.isel({ind: config[i] for i, ind in enumerate(self.site_inds)})

    def _final_contraction_options(self, *, strip_exponent=None):
        """Return the caller's path options for the exact scalar closure."""
        options = dict(self.contraction_opts.get("final_contract_opts") or {})
        if strip_exponent is not None:
            options.setdefault("strip_exponent", strip_exponent)
        return options

    def _ctmrg_options(self, tnx):
        """Return CTMRG options with the native Symmray safety defaults."""
        options = dict(self.contraction_opts)
        defaults = _ctmrg_stabilization_kwargs(
            tnx,
            reduce_opts=options.get("reduce_opts"),
            gauge_smudge=options.get("gauge_smudge"),
        )
        for key, value in defaults.items():
            if key != "canonize_opts":
                if options.get(key) is None:
                    options[key] = value
                continue
            canonize_opts = dict(value)
            canonize_opts.update(options.get(key) or {})
            options[key] = canonize_opts
        return options

    def _boundary_fallback_options(self):
        """Adapt CTMRG options for a boundary-MPS retry."""
        options = dict(self.contraction_opts)
        # These options control CTMRG's projector and gauge construction and
        # are not accepted by the boundary-MPS route. ``mode='mps'`` is
        # deliberate: this fallback is intended to avoid the failing CTMRG
        # projector path, not to re-enter it through a different alias.
        for key in ("reduce_opts", "gauge_smudge", "canonize_opts", "mode"):
            options.pop(key, None)
        options["mode"] = "mps"
        return options

    def _activate_contraction_fallback(self, error):
        """Switch a failed CTMRG model to its configured fallback route."""
        if self.contraction != "ctmrg":
            return False

        self.ctmrg_failures += 1
        if self.contraction_fallback is None:
            return False

        self.contraction = self.contraction_fallback
        self.contraction_opts = self._boundary_fallback_options()
        self.contraction_fallbacks += 1
        self.contraction_fallback_error = (
            f"{type(error).__name__}: {error}"
        )[:1000]
        warnings.warn(
            "CTMRG contraction failed; switching this Torch PEPS amplitude "
            "model to boundary-MPS for the remainder of the run. "
            f"Cause: {self.contraction_fallback_error}",
            ContractionFallbackWarning,
            stacklevel=3,
        )
        return True

    def _contract_remaining(self, tn, *args, final_opts=None):
        """Close an approximate PEPS contraction with the requested path."""
        if final_opts is None:
            final_opts = self._final_contraction_options()
        try:
            return tn.contract(*args, **final_opts)
        except KeyError as exc:
            # cotengra's ReusableHyperOptimizer can raise this after every
            # trial fails, leaving no ``best['tree']``. Preserve a long VMC
            # run by falling back only for this known optimizer failure.
            if exc.args != ("tree",) or final_opts.get("optimize") in (
                None,
                "auto-hq",
            ):
                raise
            self.final_optimizer_fallbacks += 1
            if not self._warned_final_optimizer_fallback:
                warnings.warn(
                    "The supplied final contraction optimizer produced no "
                    "cotengra tree; retrying affected VMC scalar closures "
                    "with optimize='auto-hq'.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned_final_optimizer_fallback = True
            fallback_opts = dict(final_opts)
            fallback_opts["optimize"] = "auto-hq"
            return tn.contract(*args, **fallback_opts)

    def _contract_approximate(self, fn, *args, close_final=False, **kwargs):
        """Contract with the requested cutoff, retrying empty sparse sectors."""
        kwargs = dict(kwargs)
        kwargs["cutoff"] = self.cutoff
        if close_final:
            # Close the final small tensor network here rather than inside
            # Quimb. This makes ``final_contract_opts`` apply identically to
            # full and cached boundary paths, and lets us recover from a
            # failed reusable cotengra search at the scalar-closure boundary.
            # Environment builders do not accept this option, hence the
            # explicit opt-in at amplitude call sites below.
            kwargs["final_contract"] = False

        def finish(value):
            if not close_final:
                return value
            # Boundary, CTMRG, and HOTRG now all return the partially
            # contracted flat TN here. Amplitude evaluation still needs a
            # scalar, so complete that last contraction with the requested
            # optimizer and strip-exponent options.
            contract = getattr(value, "contract", None)
            if not callable(contract):
                return value
            return self._contract_remaining(
                value,
                final_opts=self._final_contraction_options(
                    strip_exponent=kwargs.get("strip_exponent"),
                ),
            )

        try:
            return finish(fn(*args, **kwargs))
        except Exception:  # pragma: no cover - exact upstream exception varies
            if not self.symmray_tensor_ids or self.cutoff <= 0.0:
                raise
            retry_kwargs = dict(kwargs)
            retry_kwargs["cutoff"] = 0.0
            compress_opts = retry_kwargs.get("compress_opts")
            if isinstance(compress_opts, dict) and compress_opts.get("method") == "cholesky":
                retry_kwargs["compress_opts"] = {
                    **compress_opts,
                    "method": "svd",
                }
            self.cutoff_fallbacks += 1
            return finish(fn(*args, **retry_kwargs))

    def _counted_ctmrg(self, tnx):
        """Return a CTMRG closure that counts outer contractions."""
        def contract(*args, **kwargs):
            self.ctmrg_calls += 1
            return tnx.contract_ctmrg(*args, **kwargs)

        return contract

    def _contract_value(self, tnx, reference=None):
        try:
            if self.contraction == "hotrg":
                value = self._contract_approximate(
                    tnx.contract_hotrg,
                    max_bond=self.chi,
                    close_final=True,
                    **self.contraction_opts,
                )
            elif self.contraction == "ctmrg":
                with quimb_ctmrg_projector_compat():
                    value = self._contract_approximate(
                        self._counted_ctmrg(tnx),
                        max_bond=self.chi,
                        close_final=True,
                        **self._ctmrg_options(tnx),
                    )
            elif self.contraction == "boundary":
                value = self._contract_approximate(
                    tnx.contract_boundary,
                    max_bond=self.chi,
                    close_final=True,
                    **self.contraction_opts,
                )
            else:
                value = tnx.contract(all)
        except Exception as error:  # pragma: no cover - upstream varies
            if not self._activate_contraction_fallback(error):
                raise
            value = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                close_final=True,
                **self.contraction_opts,
            )
        return _as_torch_scalar(value, reference)

    def _contract_log_parts(self, tnx, reference=None):
        torch = _require_torch()
        try:
            if self.contraction == "hotrg":
                mantissa, exponent_10 = self._contract_approximate(
                    tnx.contract_hotrg,
                    max_bond=self.chi,
                    strip_exponent=True,
                    close_final=True,
                    **self.contraction_opts,
                )
            elif self.contraction == "ctmrg":
                with quimb_ctmrg_projector_compat():
                    mantissa, exponent_10 = self._contract_approximate(
                        self._counted_ctmrg(tnx),
                        max_bond=self.chi,
                        strip_exponent=True,
                        close_final=True,
                        **self._ctmrg_options(tnx),
                    )
            elif self.contraction == "boundary":
                mantissa, exponent_10 = self._contract_approximate(
                    tnx.contract_boundary,
                    max_bond=self.chi,
                    strip_exponent=True,
                    close_final=True,
                    **self.contraction_opts,
                )
            else:
                amp = tnx.contract(all)
                amp = _as_torch_scalar(amp, reference)
                abs_amp = amp.abs()
                tiny = _torch_finfo_tiny(abs_amp.dtype)
                phase = torch.where(
                    abs_amp > 0,
                    amp / abs_amp.to(dtype=amp.dtype),
                    torch.zeros_like(amp),
                )
                return phase, torch.log(abs_amp.clamp_min(tiny))
        except Exception as error:  # pragma: no cover - upstream varies
            if not self._activate_contraction_fallback(error):
                raise
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                strip_exponent=True,
                close_final=True,
                **self.contraction_opts,
            )

        mantissa = _as_torch_scalar(mantissa, reference)
        if isinstance(exponent_10, torch.Tensor):
            exponent_10 = exponent_10.to(device=mantissa.device)
        else:
            exponent_dtype = (
                mantissa.real.dtype if mantissa.is_complex() else mantissa.dtype
            )
            exponent_10 = torch.as_tensor(
                exponent_10,
                dtype=exponent_dtype,
                device=mantissa.device,
            )
        abs_mantissa = mantissa.abs()
        tiny = _torch_finfo_tiny(abs_mantissa.dtype)
        phase = torch.where(
            abs_mantissa > 0,
            mantissa / abs_mantissa.to(dtype=mantissa.dtype),
            torch.zeros_like(mantissa),
        )
        log_abs = torch.log(abs_mantissa.clamp_min(tiny)) + exponent_10 * torch.log(
            torch.as_tensor(10.0, dtype=exponent_10.dtype, device=mantissa.device)
        )
        return phase, log_abs

    def _graded_torch_forward(self, configs, *, params=None):
        """Evaluate configs through the fixed graded Torch projector."""
        if self._graded_torch_projector is None:  # pragma: no cover - guard
            raise RuntimeError("graded_torch projector is not initialized.")
        tn = self._unpack_tn(params)
        dense_leaves = [tn[site].data.to_dense() for site in self.sites]
        return self._graded_torch_projector(
            dense_leaves,
            configs,
            use_vmap=self.amplitude_batching != "serial",
        )

    def amplitude(self, config, params=None):
        """Evaluate a single configuration amplitude."""
        config = _as_long_matrix(config).reshape(-1)
        if self.graded_torch:
            return self._graded_torch_forward(config.reshape(1, -1), params=params)[0]
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return self._contract_value(self._select_config(tn, config), reference)

    def _try_vmapped_forward(self, configs, *, params=None, force=False):
        """Attempt a native batched amplitude contraction.

        Symmray fermionic arrays can support ``torch.vmap`` directly when the
        selected contraction route is pure and all required block operations
        have batching rules. A failed trace disables only this optional fast
        path; the established serial route remains numerically authoritative.
        """
        torch = _require_torch()
        if self.graded_torch:
            return None
        if self.amplitude_batching == "serial":
            return None
        if force:
            if not self._proposal_vmap_enabled:
                return None
        elif not self._vmap_forward_enabled:
            return None
        try:
            return torch.vmap(
                lambda config: self.amplitude(config, params=params)
            )(configs)
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            if force:
                self._proposal_vmap_enabled = False
            else:
                self._vmap_forward_enabled = False
            return None

    def _try_vmapped_forward_log(self, configs, *, params=None):
        """Attempt a vmapped phase/log-magnitude contraction.

        Stable-log sampling must not silently turn the flat-Z2 fast path back
        into a scalar loop. Exact contractions use the same pure selected-TN
        closure as :meth:`amplitude`, while approximate boundary/CTMRG paths
        simply decline this optional route and retain their serial fallback.
        """
        torch = _require_torch()
        if self.graded_torch or self.amplitude_batching == "serial":
            return None
        if not self._vmap_log_enabled:
            return None
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)

        def evaluate(config):
            return self._contract_log_parts(
                self._select_config(tn, config),
                reference,
            )

        try:
            phases, log_abs = torch.vmap(evaluate)(configs)
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self._vmap_log_enabled = False
            return None
        return phases, log_abs

    def forward(self, configs, params=None, *, chunk_size=None):
        """Evaluate a batch of configuration amplitudes."""
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        chunk_size = _normalize_chunk_size(chunk_size)
        if chunk_size is not None and configs.shape[0] > chunk_size:
            return torch.cat([
                self.forward(
                    configs[start:start + chunk_size],
                    params=params,
                    chunk_size=None,
                )
                for start in range(0, configs.shape[0], chunk_size)
            ])

        if (
            params is None
            and chunk_size is None
            and self._compiled_amplitude_forward is not None
            and configs.shape[0] == self._compiled_batch_size
        ):
            return self.compiled_forward(configs)

        if self.graded_torch:
            self.last_amplitude_batching = (
                "graded-vmap"
                if self.amplitude_batching != "serial"
                else "graded-serial"
            )
            return self._graded_torch_forward(configs, params=params)

        vmapped = self._try_vmapped_forward(configs, params=params)
        if vmapped is not None:
            self.last_amplitude_batching = "vmap"
            return vmapped

        self.last_amplitude_batching = "serial"
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return torch.stack([
            self._contract_value(self._select_config(tn, row), reference)
            for row in configs
        ])

    def forward_log(self, configs, params=None):
        """Return ``(phase, log_abs)`` for a batch of configurations."""
        configs = _as_long_matrix(configs)
        if (
            params is None
            and self._compiled_log_amplitude_forward is not None
            and configs.shape[0] == self._compiled_batch_size
        ):
            return self.compiled_forward_log(configs)
        if self.graded_torch:
            torch = _require_torch()
            amp = self.forward(configs, params=params)
            abs_amp = amp.abs()
            tiny = _torch_finfo_tiny(abs_amp.dtype)
            phase = torch.where(
                abs_amp > 0,
                amp / abs_amp.to(dtype=amp.dtype),
                torch.zeros_like(amp),
            )
            return phase, torch.log(abs_amp.clamp_min(tiny))
        vmapped = self._try_vmapped_forward_log(configs, params=params)
        if vmapped is not None:
            self.last_amplitude_batching = "log-vmap"
            return vmapped
        self.last_amplitude_batching = "serial"
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        phases = []
        log_abs = []
        for row in configs:
            phase, log_scale = self._contract_log_parts(
                self._select_config(tn, row),
                reference,
            )
            phases.append(phase)
            log_abs.append(log_scale)
        torch = _require_torch()
        return torch.stack(phases), torch.stack(log_abs)

    def proposal_amplitudes(
        self,
        parent_configs,
        target_configs,
        current_amplitudes,
        *,
        chunk_size=None,
    ):
        """Evaluate proposals, padding to the compiled batch when available.

        The compiled GPU graph has one fixed walker dimension. For a sweep
        where only ``n`` walkers changed, repeat the first target row to fill
        the remaining slots and discard those auxiliary outputs. This mirrors
        the upstream compiled sampler while preserving the exact result for
        the requested proposals. The path is used only when no explicit
        ``chunk_size`` is active; otherwise the ordinary proposal batching
        policy remains authoritative.
        """
        del parent_configs, current_amplitudes
        torch = _require_torch()
        target_configs = _as_long_matrix(target_configs)
        n_targets = int(target_configs.shape[0])
        batch_size = self._compiled_batch_size
        if (
            self._compiled_amplitude_forward is not None
            and chunk_size is None
            and n_targets > 0
            and batch_size is not None
            and n_targets <= batch_size
        ):
            if n_targets < batch_size:
                padding = target_configs[:1].expand(
                    batch_size - n_targets,
                    -1,
                )
                evaluation_configs = torch.cat((target_configs, padding), dim=0)
            else:
                evaluation_configs = target_configs
            return self.compiled_forward(evaluation_configs)[:n_targets]
        return _call_amplitude_fn(
            self,
            target_configs,
            chunk_size=chunk_size,
        )

    def proposal_log_amplitudes(
        self,
        parent_configs,
        target_configs,
        *,
        chunk_size=None,
    ):
        """Evaluate proposal log amplitudes with the fixed compiled batch."""
        del parent_configs
        target_configs = _as_long_matrix(target_configs)
        n_targets = int(target_configs.shape[0])
        batch_size = self._compiled_batch_size
        if (
            self._compiled_log_amplitude_forward is not None
            and chunk_size is None
            and n_targets > 0
            and batch_size is not None
            and n_targets <= batch_size
        ):
            torch = _require_torch()
            if n_targets < batch_size:
                padding = target_configs[:1].expand(
                    batch_size - n_targets,
                    -1,
                )
                evaluation_configs = torch.cat((target_configs, padding), dim=0)
            else:
                evaluation_configs = target_configs
            phase, log_abs = self.compiled_forward_log(evaluation_configs)
            return phase[:n_targets], log_abs[:n_targets]
        return _call_log_amplitude_fn(
            self.forward_log,
            target_configs,
            chunk_size=chunk_size,
        )

    def connected_amplitudes(
        self,
        configs,
        amplitudes,
        connections,
        *,
        chunk_size=None,
        reuse_diagonal=True,
    ):
        """Evaluate amplitudes for Hamiltonian-connected configurations.

        Diagonal connections reuse the already available parent amplitudes.
        Future boundary-environment reuse can specialize this method without
        changing :func:`local_energy_from_connections` or the VMC driver API.
        """
        return _default_connected_amplitudes(
            configs,
            amplitudes,
            connections,
            self,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )

    def __call__(self, configs, params=None, *, chunk_size=None):
        """Alias for :meth:`forward`."""
        return self.forward(configs, params=params, chunk_size=chunk_size)


class TorchPEPSBoundaryAmplitude(TorchPEPSAmplitude):
    """PEPS amplitude wrapper with boundary-environment connected reuse.

    The base :class:`TorchPEPSAmplitude` evaluates every off-diagonal connected
    configuration with a fresh contraction. This subclass keeps the same public
    call interface but specializes ``connected_amplitudes(...)`` for finite
    quimb PEPS using boundary-MPS environments around each parent walker. For a
    local update, only the touched row or column window is recontracted.

    ``boundary_workers`` optionally evaluates independent cached-window
    closures concurrently during no-grad CPU measurements. It defaults to one;
    use a small value such as two or four only after checking for BLAS and
    contraction-optimizer oversubscription.

    Unsupported PEPS geometries or non-boundary contractions fall back to the
    base implementation.
    """

    def __init__(
        self,
        peps,
        *,
        contraction="boundary",
        chi=None,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        site_order=None,
        graded_torch=False,
        amplitude_batching="auto",
        environment_radius=0,
        boundary_cache_size=128,
        proposal_batching="auto",
        proposal_vmap_min_batch=8,
        boundary_workers=1,
    ):
        super().__init__(
            peps,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            dtype=dtype,
            device=device,
            site_order=site_order,
            graded_torch=graded_torch,
            amplitude_batching=amplitude_batching,
        )
        self.environment_radius = _check_nonnegative_int(
            "environment_radius",
            environment_radius,
        )
        self.boundary_cache_size = _check_positive_int(
            "boundary_cache_size",
            boundary_cache_size,
        )
        self.proposal_batching = _normalize_proposal_batching(proposal_batching)
        self.proposal_vmap_min_batch = _check_positive_int(
            "proposal_vmap_min_batch",
            proposal_vmap_min_batch,
        )
        # Native U1/U1U1 boundary contractions are not reliably vmappable.
        # Independent cached-window closures can nevertheless be evaluated in
        # parallel during no-grad CPU measurements. Keep this opt-in so the
        # default remains deterministic and avoids BLAS/thread oversubscription.
        self.boundary_workers = _check_positive_int(
            "boundary_workers",
            boundary_workers,
        )
        self._proposal_vmap_enabled = callable(
            getattr(_require_torch(), "vmap", None)
        )
        self._boundary_geometry = self._infer_boundary_geometry(self._unpack_tn())
        self.last_connected_reuse_stats = None
        self.last_proposal_cache_stats = None
        self.last_amplitude_cache_stats = None
        self._boundary_cache_token = None
        self._boundary_environment_cache = {}
        self._boundary_transition_cache = {}
        self._boundary_amplitude_cache = {}
        # Parent-selected strip templates are much smaller than a full PEPS
        # contraction and let connected local estimators replace only their
        # changed physical projectors.
        self._boundary_strip_cache = {}
        # Compiled boundary reuse is keyed by a static geometry class, for
        # example ``('x', (1, 2))``.  The boundary environments are inputs to
        # these graphs rather than captured constants, so parameter updates
        # only require rebuilding the environments, not re-exporting the
        # contraction graph.
        self._compiled_boundary_reuse = {}
        self._compiled_boundary_environments = {}
        self._compiled_boundary_batch_size = None
        self._compiled_boundary_mode = None
        self._compiled_boundary_backend = None
        self._compiled_boundary_environment_mode = None
        self.last_boundary_compile_report = None

    def _parameter_cache_token(self):
        """Return a cheap token that changes when torch leaves are updated."""
        return tuple(int(getattr(param, "_version", 0)) for param in self.params)

    def clear_boundary_cache(self):
        """Clear cached boundary environments and proposal transitions."""
        self._boundary_environment_cache.clear()
        self._boundary_transition_cache.clear()
        self._boundary_strip_cache.clear()
        self._boundary_amplitude_cache.clear()
        self._boundary_cache_token = self._parameter_cache_token()
        self.last_connected_reuse_stats = None
        self.last_proposal_cache_stats = None
        self.last_amplitude_cache_stats = None
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.clear_boundary_cache()
        return self

    def clear_compiled(self):
        """Clear full-amplitude and boundary-reuse compiled graphs."""
        super().clear_compiled()
        self._compiled_boundary_reuse = {}
        self._compiled_boundary_environments = {}
        self._compiled_boundary_batch_size = None
        self._compiled_boundary_mode = None
        self._compiled_boundary_backend = None
        self._compiled_boundary_environment_mode = None
        self.last_boundary_compile_report = None
        return self

    def _ensure_boundary_cache_current(self):
        token = self._parameter_cache_token()
        if token != self._boundary_cache_token:
            self._boundary_environment_cache.clear()
            self._boundary_transition_cache.clear()
            self._boundary_strip_cache.clear()
            self._boundary_amplitude_cache.clear()
            self._boundary_cache_token = token

    def _boundary_reuse_patterns(self, directions=("x", "y"), widths=(1, 2)):
        """Return static row/column geometry classes for boundary reuse."""
        if self._boundary_geometry is None:
            return ()
        patterns = []
        for direction in directions:
            direction = str(direction).lower()
            if direction not in {"x", "y"}:
                raise ValueError("directions must contain only 'x' and 'y'.")
            length = self._boundary_geometry[
                "Lx" if direction == "x" else "Ly"
            ]
            for width in widths:
                width = _check_positive_int("width", width)
                for start in range(length - width + 1):
                    patterns.append(
                        (direction, tuple(range(start, start + width)))
                    )
        return tuple(patterns)

    @staticmethod
    def _pack_boundary_network(boundary_tn):
        """Pack one cached boundary network for graph inputs."""
        import quimb as qu
        import quimb.tensor as qtn

        params, skeleton = qtn.pack(boundary_tn)
        flat, pytree = qu.utils.tree_flatten(params, get_ref=True)
        return tuple(flat), pytree, skeleton

    @staticmethod
    @contextmanager
    def _boundary_export_warnings():
        """Return the harmless export warnings emitted by packed Quimb data."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Attempted to insert a get_attr Node.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Node lifted_tensor_.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Additional .* warnings suppressed about get_attr references",
                category=UserWarning,
            )
            yield

    def _boundary_environment_manifest(self, envs):
        """Describe packed environments with stable reconstruction metadata."""
        manifest = []
        for key in sorted(envs):
            flat, pytree, skeleton = self._pack_boundary_network(envs[key])
            manifest.append({
                "key": key,
                "n_leaves": len(flat),
                "pytree": pytree,
                "skeleton": skeleton,
                "shapes": tuple(tuple(leaf.shape) for leaf in flat),
            })
        return tuple(manifest)

    def _boundary_environment_options_for_mode(self, mode):
        options = self._boundary_environment_options()
        options["mode"] = mode
        return options

    def _build_boundary_environment_compiler(
        self,
        axis,
        example_config,
        *,
        environment_mode,
        mode,
        backend,
        strict,
    ):
        """Export one batched boundary-environment builder."""
        torch = _require_torch()
        import quimb as qu
        import quimb.tensor as qtn

        params = list(self.params)
        manifest_options = self._boundary_environment_options_for_mode(
            environment_mode
        )
        with torch.no_grad():
            tn = self._unpack_tn()
            selected = self._select_config(tn, example_config)
            if axis == "x":
                example_envs = selected.compute_x_environments(
                    max_bond=self.chi,
                    cutoff=self.cutoff,
                    **manifest_options,
                )
            else:
                example_envs = selected.compute_y_environments(
                    max_bond=self.chi,
                    cutoff=self.cutoff,
                    **manifest_options,
                )
        manifest = self._boundary_environment_manifest(example_envs)
        params_pytree = self.params_pytree
        skeleton = self.skeleton
        site_inds = self.site_inds
        chi = self.chi
        cutoff = self.cutoff

        def wrapper(config, *flat_params):
            tn_params = qu.utils.tree_unflatten(list(flat_params), params_pytree)
            tns = qtn.unpack(tn_params, skeleton)
            selected = tns.isel({
                ind: config[index]
                for index, ind in enumerate(site_inds)
            })
            if axis == "x":
                envs = selected.compute_x_environments(
                    max_bond=chi,
                    cutoff=cutoff,
                    **manifest_options,
                )
            else:
                envs = selected.compute_y_environments(
                    max_bond=chi,
                    cutoff=cutoff,
                    **manifest_options,
                )
            outputs = []
            for item in manifest:
                env_params, _ = qtn.pack(envs[item["key"]])
                flat, _ = qu.utils.tree_flatten(env_params, get_ref=True)
                outputs.extend(flat)
            return tuple(outputs)

        class _EnvironmentModule(torch.nn.Module):
            def forward(self, config, *flat_params):
                return wrapper(config, *flat_params)

        with self._boundary_export_warnings():
            exported = torch.export.export(
                _EnvironmentModule(),
                (example_config, *params),
                strict=strict,
            )
        exported_module = exported.module()
        vmapped = torch.vmap(
            exported_module,
            in_dims=(0,) + (None,) * len(params),
        )
        compile_kwargs = {
            "backend": backend,
            "dynamic": False,
        }
        if mode is not None:
            compile_kwargs["mode"] = mode
        compiled = torch.compile(vmapped, **compile_kwargs)
        return {
            "fn": compiled,
            "exported": exported,
            "manifest": manifest,
            "axis": axis,
            "environment_mode": environment_mode,
        }

    def _boundary_environment_batch(self, configs, axis):
        """Evaluate a fixed-size compiled environment batch, if available."""
        torch = _require_torch()
        entry = self._compiled_boundary_environments.get(axis)
        if entry is None:
            return None
        configs = _as_long_matrix(configs)
        n_configs = int(configs.shape[0])
        batch_size = int(self._compiled_boundary_batch_size)
        if n_configs == 0 or n_configs > batch_size:
            return None
        if n_configs < batch_size:
            configs = torch.cat((
                configs,
                configs[:1].expand(batch_size - n_configs, -1),
            ), dim=0)
        outputs = tuple(entry["fn"](configs, *list(self.params)))
        grouped = {}
        offset = 0
        for item in entry["manifest"]:
            n_leaves = item["n_leaves"]
            grouped[item["key"]] = tuple(
                output[:n_configs]
                for output in outputs[offset:offset + n_leaves]
            )
            offset += n_leaves
        return grouped

    def _materialize_boundary_environment_batch(
        self,
        axis,
        grouped,
        *,
        n_configs=None,
    ):
        """Turn batched packed environment leaves back into Quimb networks."""
        import quimb as qu
        import quimb.tensor as qtn

        entry = self._compiled_boundary_environments[axis]
        result = []
        if n_configs is None:
            n_configs = next(
                (
                    int(leaf.shape[0])
                    for leaves in grouped.values()
                    for leaf in leaves
                ),
                0,
            )
        for index in range(n_configs):
            envs = {}
            for item in entry["manifest"]:
                leaves = [leaf[index] for leaf in grouped[item["key"]]]
                params = qu.utils.tree_unflatten(leaves, item["pytree"])
                envs[item["key"]] = qtn.unpack(params, item["skeleton"])
            result.append(envs)
        return result

    def _populate_compiled_boundary_environments(self, configs, axis):
        """Populate the ordinary environment cache from compiled batches."""
        torch = _require_torch()
        if axis not in self._compiled_boundary_environments:
            return 0
        configs = _as_long_matrix(configs)
        batch_size = int(self._compiled_boundary_batch_size)
        built = 0
        missing = []
        seen = set()
        for config in configs:
            key = (axis, self._configuration_key(config))
            if key not in self._boundary_environment_cache and key not in seen:
                missing.append(config)
                seen.add(key)
        for start in range(0, len(missing), batch_size):
            batch = torch.stack(missing[start:start + batch_size], dim=0)
            grouped = self._boundary_environment_batch(batch, axis)
            if grouped is None:
                continue
            # ``_materialize_boundary_environment_batch`` returns the padded
            # graph size. Only retain rows belonging to this request.
            materialized = self._materialize_boundary_environment_batch(
                axis,
                {
                    key: tuple(leaf[:batch.shape[0]] for leaf in leaves)
                    for key, leaves in grouped.items()
                },
                n_configs=int(batch.shape[0]),
            )
            for index, config in enumerate(batch):
                self._cache_put(
                    self._boundary_environment_cache,
                    (axis, self._configuration_key(config)),
                    materialized[index],
                )
                built += 1
        return built

    def _compiled_boundary_environment_inputs(
        self,
        parent_configs,
        axis,
        indices,
    ):
        """Pack cached endpoint environments for one compiled geometry class."""
        torch = _require_torch()
        import quimb as qu
        import quimb.tensor as qtn

        entry = self._compiled_boundary_reuse.get((axis, tuple(indices)))
        if entry is None:
            return None
        tn = self._unpack_tn()
        endpoint_keys = entry["endpoint_keys"]
        packed = []
        for endpoint in endpoint_keys:
            rows = []
            for parent_config in parent_configs:
                cache_key = (axis, self._configuration_key(parent_config))
                envs = self._boundary_environment_cache.get(cache_key)
                if envs is None:
                    if axis in self._compiled_boundary_environments:
                        return None
                    envs, _ = self._cached_boundary_environments(
                        tn,
                        parent_config,
                        axis,
                    )
                params, _ = qtn.pack(envs[endpoint])
                flat, _ = qu.utils.tree_flatten(params, get_ref=True)
                expected = entry["environment_manifests"][endpoint]
                if len(flat) != expected["n_leaves"]:
                    return None
                if any(
                    tuple(leaf.shape) != shape
                    for leaf, shape in zip(flat, expected["shapes"])
                ):
                    return None
                rows.append(tuple(flat))
            if rows:
                n_leaves = len(rows[0])
                packed.append(tuple(
                    torch.stack([
                        row[leaf_index].to(device=parent_configs.device)
                        for row in rows
                    ], dim=0)
                    for leaf_index in range(n_leaves)
                ))
            else:
                packed.append(())
        return tuple(packed)

    def _compiled_boundary_reuse_batch(
        self,
        parent_configs,
        target_configs,
        axis,
        indices,
        *,
        log=False,
    ):
        """Evaluate one fixed-shape compiled boundary-reuse group."""
        torch = _require_torch()
        key = (axis, tuple(indices))
        entry = self._compiled_boundary_reuse.get(key)
        if entry is None or torch.is_grad_enabled():
            return None
        parent_configs = _as_long_matrix(parent_configs)
        target_configs = _as_long_matrix(target_configs)
        if parent_configs.shape != target_configs.shape:
            raise ValueError("Boundary reuse parent and target shapes differ.")
        n_configs = int(target_configs.shape[0])
        batch_size = int(self._compiled_boundary_batch_size)
        if n_configs == 0 or n_configs > batch_size:
            return None
        env_inputs = self._compiled_boundary_environment_inputs(
            parent_configs,
            axis,
            indices,
        )
        if env_inputs is None:
            return None
        if n_configs < batch_size:
            target_configs = torch.cat((
                target_configs,
                target_configs[:1].expand(batch_size - n_configs, -1),
            ), dim=0)

        padded_env_inputs = []
        for endpoint_inputs in env_inputs:
            for leaf in endpoint_inputs:
                if n_configs < batch_size:
                    leaf = torch.cat((
                        leaf,
                        leaf[:1].expand(
                            (batch_size - n_configs,) + tuple(leaf.shape[1:])
                        ),
                    ), dim=0)
                padded_env_inputs.append(leaf)
        outputs = entry["log_fn" if log else "fn"](
            target_configs,
            *list(self.params),
            *padded_env_inputs,
        )
        if log:
            return outputs[0][:n_configs], outputs[1][:n_configs]
        return outputs[:n_configs]

    def _compiled_boundary_proposals(
        self,
        parent_configs,
        target_configs,
        *,
        log=False,
    ):
        """Evaluate proposal rows through compiled geometry classes."""
        torch = _require_torch()
        compiled_reuse = getattr(self, "_compiled_boundary_reuse", {})
        if torch.is_grad_enabled() or not compiled_reuse:
            return None
        parent_configs = _as_long_matrix(parent_configs)
        target_configs = _as_long_matrix(target_configs)
        if parent_configs.shape != target_configs.shape:
            raise ValueError("Boundary proposal parent and target shapes differ.")

        candidates = {}
        for index in range(int(parent_configs.shape[0])):
            windows = self._changed_axis_windows(
                parent_configs[index],
                target_configs[index],
            )
            if not windows:
                continue
            axis, indices = windows[0]
            key = (axis, indices)
            entry = compiled_reuse.get(key)
            if entry is None or (log and entry.get("log_fn") is None):
                continue
            candidates.setdefault(key, []).append(index)
        if not candidates:
            return None

        stats = {
            "num_compiled_groups": 0,
            "num_compiled_connections": 0,
            "num_environment_compiled": 0,
        }
        for axis, _indices in candidates:
            stats["num_environment_compiled"] += (
                self._populate_compiled_boundary_environments(
                    parent_configs,
                    axis,
                )
            )

        n_configs = int(target_configs.shape[0])
        handled = torch.zeros(
            n_configs,
            dtype=torch.bool,
            device=target_configs.device,
        )
        values = [None] * n_configs
        batch_size = int(self._compiled_boundary_batch_size)
        for (axis, indices), indices_for_group in candidates.items():
            for start in range(0, len(indices_for_group), batch_size):
                group_indices = indices_for_group[start:start + batch_size]
                index_tensor = torch.as_tensor(
                    group_indices,
                    dtype=torch.long,
                    device=target_configs.device,
                )
                try:
                    result = self._compiled_boundary_reuse_batch(
                        parent_configs[index_tensor],
                        target_configs[index_tensor],
                        axis,
                        indices,
                        log=log,
                    )
                except Exception:  # pragma: no cover - backend-specific fallback
                    result = None
                if result is None:
                    continue
                if log:
                    result = (
                        result[0].to(device=target_configs.device),
                        result[1].to(device=target_configs.device),
                    )
                    for offset, index in enumerate(group_indices):
                        values[index] = (result[0][offset], result[1][offset])
                else:
                    result = result.to(device=target_configs.device)
                    for offset, index in enumerate(group_indices):
                        values[index] = result[offset]
                handled[index_tensor] = True
                stats["num_compiled_groups"] += 1
                stats["num_compiled_connections"] += len(group_indices)

        if not bool(torch.any(handled)):
            return None
        if log:
            phases = torch.stack([
                value[0] for value in values if value is not None
            ])
            log_abs = torch.stack([
                value[1] for value in values if value is not None
            ])
            # The compact arrays above are only used for handled rows; the
            # caller scatters them with the same boolean mask.
            result = (phases, log_abs)
        else:
            result = torch.stack([
                value for value in values if value is not None
            ])
        return result, handled, stats

    def export_and_compile_boundary_reuse(
        self,
        example_configs,
        *,
        mode="default",
        backend="inductor",
        strict=False,
        directions=("x", "y"),
        widths=(1, 2),
        compile_log=True,
        compile_environments=True,
        environment_mode="full-bond",
        reuse_mode="direct",
    ):
        """Compile static row/column boundary-environment reuse patterns.

        This is the boundary analogue of :meth:`export_and_compile`. One
        graph is exported for each fixed row/column window, with the PEPS
        parameters and the two endpoint boundary environments as explicit
        inputs. ``torch.vmap`` adds the walker dimension and ``torch.compile``
        fuses the resulting fixed-shape batch. Connected local-energy calls
        dispatch compatible targets to these geometry-class graphs.

        ``compile_environments=True`` also compiles the environment builder.
        The default ``environment_mode='full-bond'`` is intentional: Quimb's
        MPS compression path contains data-dependent SVD shape guards that
        PyTorch export cannot represent in current releases. The resulting
        static path is opt-in and retains the ordinary eager fallback when a
        PEPS or Symmray backend is not exportable. ``reuse_mode='direct'``
        contracts the assembled boundary environments and selected strip
        directly, matching the static reuse strategy in the reference
        implementation. ``reuse_mode='boundary'`` requests an additional
        compression sweep and may not be exportable for all Quimb versions.
        """
        torch = _require_torch()
        if self.contraction != "boundary":
            raise ValueError(
                "export_and_compile_boundary_reuse requires "
                "contraction='boundary'."
            )
        if self._boundary_geometry is None:
            raise ValueError("The PEPS geometry is not a finite rectangular grid.")
        reuse_mode = str(reuse_mode).lower().replace("_", "-")
        if reuse_mode not in {"direct", "boundary", "auto"}:
            raise ValueError("reuse_mode must be 'direct', 'boundary', or 'auto'.")
        environment_mode = str(environment_mode).lower().replace("_", "-")
        directions = tuple(str(direction).lower() for direction in directions)
        widths = tuple(widths)
        self._boundary_reuse_patterns(directions=directions, widths=widths)
        example_configs = _as_long_matrix(example_configs)
        if example_configs.shape[0] == 0:
            raise ValueError("example_configs must contain at least one row.")
        if example_configs.shape[1] != self.n_sites:
            raise ValueError(
                "example_configs must have one physical index per site; "
                f"expected {self.n_sites}, got {example_configs.shape[1]}."
            )
        if environment_mode not in {"mps", "full-bond"}:
            raise ValueError("environment_mode must be 'mps' or 'full-bond'.")

        self.clear_compiled()
        self._compiled_boundary_batch_size = int(example_configs.shape[0])
        self._compiled_boundary_mode = mode
        self._compiled_boundary_backend = backend
        self._compiled_boundary_environment_mode = environment_mode
        report = {
            "batch_size": self._compiled_boundary_batch_size,
            "environment_mode": environment_mode,
            "reuse_mode": reuse_mode,
            "environment_compiled": {},
            "reuse_compiled": [],
            "reuse_failures": {},
        }

        first_config = example_configs[0]
        if compile_environments:
            for axis in directions:
                try:
                    self._compiled_boundary_environments[axis] = (
                        self._build_boundary_environment_compiler(
                            axis,
                            first_config,
                            environment_mode=environment_mode,
                            mode=mode,
                            backend=backend,
                            strict=strict,
                        )
                    )
                except Exception as error:  # pragma: no cover - backend-specific
                    report["environment_compiled"][axis] = False
                    report["reuse_failures"][f"environment:{axis}"] = (
                        f"{type(error).__name__}: {error}"
                    )[:1000]
                else:
                    report["environment_compiled"][axis] = True

        # Materialize one environment set per direction to establish the
        # fixed packed skeletons used by every reuse graph in that direction.
        with torch.no_grad():
            for axis in directions:
                if axis in self._compiled_boundary_environments:
                    grouped = self._boundary_environment_batch(
                        first_config.reshape(1, -1),
                        axis,
                    )
                    if grouped is not None:
                        envs = self._materialize_boundary_environment_batch(
                            axis,
                            {
                                key: tuple(leaf[:1] for leaf in leaves)
                                for key, leaves in grouped.items()
                            },
                            n_configs=1,
                        )[0]
                        self._cache_put(
                            self._boundary_environment_cache,
                            (axis, self._configuration_key(first_config)),
                            envs,
                        )
                else:
                    envs, _ = self._cached_boundary_environments(
                        self._unpack_tn(),
                        first_config,
                        axis,
                    )

                for direction, indices in self._boundary_reuse_patterns(
                    directions=(axis,), widths=widths
                ):
                    endpoint_keys = (
                        ("xmin", indices[0]), ("xmax", indices[-1])
                    ) if direction == "x" else (
                        ("ymin", indices[0]), ("ymax", indices[-1])
                    )
                    environment_manifests = {}
                    environment_flats = []
                    for endpoint in endpoint_keys:
                        flat, pytree, skeleton = self._pack_boundary_network(
                            envs[endpoint]
                        )
                        environment_manifests[endpoint] = {
                            "n_leaves": len(flat),
                            "pytree": pytree,
                            "skeleton": skeleton,
                            "shapes": tuple(tuple(leaf.shape) for leaf in flat),
                        }
                        environment_flats.append(flat)

                    def build_reuse(log, selected_direction=direction,
                                    selected_indices=indices,
                                    selected_endpoints=endpoint_keys,
                                    selected_manifests=environment_manifests,
                                    selected_flats=tuple(environment_flats),
                                    selected_mode=None):
                        import quimb as qu
                        import quimb.tensor as qtn

                        if selected_mode is None:
                            selected_mode = reuse_mode

                        tn_params_pytree = self.params_pytree
                        tn_skeleton = self.skeleton
                        site_inds = self.site_inds
                        view_kwargs = self._boundary_geometry["view_kwargs"]
                        chi = self.chi
                        cutoff = self.cutoff
                        boundary_options = self._boundary_environment_options()
                        final_options = self._final_contraction_options()
                        log_options = dict(final_options)
                        log_options["strip_exponent"] = True
                        n_tn = len(self.params)
                        n_min = selected_manifests[selected_endpoints[0]]["n_leaves"]

                        def wrapper(config, *flat_args):
                            tn_params = qu.utils.tree_unflatten(
                                list(flat_args[:n_tn]), tn_params_pytree
                            )
                            min_params = qu.utils.tree_unflatten(
                                list(flat_args[n_tn:n_tn + n_min]),
                                selected_manifests[selected_endpoints[0]]["pytree"],
                            )
                            max_params = qu.utils.tree_unflatten(
                                list(flat_args[n_tn + n_min:]),
                                selected_manifests[selected_endpoints[1]]["pytree"],
                            )
                            tns = qtn.unpack(tn_params, tn_skeleton)
                            boundary_min = qtn.unpack(
                                min_params,
                                selected_manifests[selected_endpoints[0]]["skeleton"],
                            )
                            boundary_max = qtn.unpack(
                                max_params,
                                selected_manifests[selected_endpoints[1]]["skeleton"],
                            )
                            amp = tns.isel({
                                ind: config[index]
                                for index, ind in enumerate(site_inds)
                            })
                            tags = [
                                tns.x_tag(index) if selected_direction == "x"
                                else tns.y_tag(index)
                                for index in selected_indices
                            ]
                            reuse_tn = boundary_min | amp.select(tags, which="any") | boundary_max
                            reuse_tn.view_as_(qtn.PEPS, **view_kwargs)
                            if selected_mode in {"boundary", "auto"}:
                                if selected_direction == "x":
                                    reuse_tn.contract_boundary_from_xmin_(
                                        xrange=[selected_indices[0], selected_indices[-1] + 1],
                                        max_bond=chi,
                                        cutoff=cutoff,
                                        **boundary_options,
                                    )
                                else:
                                    reuse_tn.contract_boundary_from_ymin_(
                                        yrange=[selected_indices[0], selected_indices[-1] + 1],
                                        max_bond=chi,
                                        cutoff=cutoff,
                                        **boundary_options,
                                    )
                            if log:
                                return reuse_tn.contract(all, **log_options)
                            return reuse_tn.contract(all, **final_options)

                        class _ReuseModule(torch.nn.Module):
                            def forward(self, config, *flat_args):
                                return wrapper(config, *flat_args)

                        args = (
                            first_config,
                            *list(self.params),
                            *selected_flats[0],
                            *selected_flats[1],
                        )
                        with self._boundary_export_warnings():
                            exported = torch.export.export(
                                _ReuseModule(),
                                args,
                                strict=strict,
                            )
                        exported_module = exported.module()
                        in_dims = (
                            (0,)
                            + (None,) * n_tn
                            + (0,) * len(selected_flats[0])
                            + (0,) * len(selected_flats[1])
                        )
                        vmapped = torch.vmap(exported_module, in_dims=in_dims)
                        compile_kwargs = {"backend": backend, "dynamic": False}
                        if mode is not None:
                            compile_kwargs["mode"] = mode
                        return exported, torch.compile(vmapped, **compile_kwargs)

                    try:
                        exported, compiled = build_reuse(False)
                        log_export = log_compiled = None
                        if compile_log:
                            log_export, log_compiled = build_reuse(True)
                    except Exception as error:  # pragma: no cover - backend-specific
                        if reuse_mode == "auto":
                            try:
                                reuse_mode = "direct"
                                exported, compiled = build_reuse(False)
                                log_export = log_compiled = None
                                if compile_log:
                                    log_export, log_compiled = build_reuse(True)
                            except Exception as retry_error:
                                report["reuse_failures"][str((direction, indices))] = (
                                    f"{type(retry_error).__name__}: {retry_error}"
                                )[:1000]
                                continue
                        else:
                            report["reuse_failures"][str((direction, indices))] = (
                                f"{type(error).__name__}: {error}"
                            )[:1000]
                            continue

                    self._compiled_boundary_reuse[(direction, indices)] = {
                        "fn": compiled,
                        "log_fn": log_compiled,
                        "exported": exported,
                        "log_exported": log_export,
                        "endpoint_keys": endpoint_keys,
                        "environment_manifests": environment_manifests,
                    }
                    report["reuse_compiled"].append((direction, indices))

        report["reuse_mode"] = reuse_mode
        self.last_boundary_compile_report = report
        if not report["reuse_compiled"]:
            raise RuntimeError(
                "Could not export any boundary-reuse geometry class. "
                "Try environment_mode='full-bond', reuse_mode='direct', "
                "backend='eager', or retain the eager boundary path."
            )
        return self

    @staticmethod
    def _configuration_key(config):
        return tuple(int(value) for value in config.detach().cpu().tolist())

    def _cache_put(self, cache, key, value, *, max_size=None):
        cache[key] = value
        if max_size is None:
            max_size = self.boundary_cache_size
        while len(cache) > max_size:
            cache.pop(next(iter(cache)))

    def forward(self, configs, params=None, *, chunk_size=None):
        """Evaluate amplitudes, caching serial boundary contractions safely.

        Boundary amplitudes are cached only for detached/no-grad calls using
        the model's own parameters. Gradient-enabled contractions and custom
        parameter pytrees always use the base implementation, so the cache
        never retains an autograd graph or returns stale derivatives.
        """
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        vmap_preferred = (
            self.amplitude_batching != "serial"
            and self._vmap_forward_enabled
        )
        if (
            params is not None
            or torch.is_grad_enabled()
            or self.contraction != "boundary"
            or self.graded_torch
            or vmap_preferred
        ):
            return super().forward(configs, params=params, chunk_size=chunk_size)
        if configs.shape[0] == 0:
            return super().forward(configs, params=params, chunk_size=chunk_size)

        self._ensure_boundary_cache_current()
        unique_configs, inverse = _unique_config_rows(configs)
        if inverse is None:
            inverse = torch.zeros(
                1,
                dtype=torch.long,
                device=configs.device,
            )
        cached_values = [None] * int(unique_configs.shape[0])
        missing_indices = []
        num_hits = 0
        for index, config in enumerate(unique_configs):
            key = self._configuration_key(config)
            try:
                cached_values[index] = self._boundary_amplitude_cache[key]
            except KeyError:
                missing_indices.append(index)
            else:
                num_hits += 1

        if missing_indices:
            missing = torch.as_tensor(
                missing_indices,
                dtype=torch.long,
                device=configs.device,
            )
            computed = TorchPEPSAmplitude.forward(
                self,
                unique_configs[missing],
                params=params,
                chunk_size=chunk_size,
            )
            for offset, index in enumerate(missing_indices):
                value = computed[offset].detach()
                cached_values[index] = value
                self._cache_put(
                    self._boundary_amplitude_cache,
                    self._configuration_key(unique_configs[index]),
                    value,
                )

        unique_amplitudes = torch.stack([
            value.to(device=configs.device)
            for value in cached_values
        ])
        self.last_amplitude_batching = "serial"
        self.last_amplitude_cache_stats = {
            "num_requests": int(configs.shape[0]),
            "num_unique_requests": int(unique_configs.shape[0]),
            "num_hits": num_hits,
            "num_misses": len(missing_indices),
        }
        return unique_amplitudes[inverse]

    def _cached_boundary_environments(
        self,
        tn,
        parent_config,
        axis,
        *,
        parent_tn=None,
    ):
        """Get one walker's MPS environments, retaining them across sweeps."""
        self._ensure_boundary_cache_current()
        key = (axis, self._configuration_key(parent_config))
        try:
            return self._boundary_environment_cache[key], True
        except KeyError:
            if parent_tn is None:
                parent_tn = self._select_config(tn, parent_config)
            envs = self._compute_boundary_environments(parent_tn, axis)
            self._cache_put(self._boundary_environment_cache, key, envs)
            return envs, False

    def _cached_boundary_strip(
        self,
        tn,
        parent_config,
        axis,
        indices,
        *,
        parent_tn=None,
    ):
        """Get a parent-selected strip template for local impurity updates."""
        self._ensure_boundary_cache_current()
        parent_key = self._configuration_key(parent_config)
        key = (axis, tuple(indices), parent_key)
        try:
            return self._boundary_strip_cache[key], True
        except KeyError:
            if parent_tn is None:
                parent_tn = self._select_config(tn, parent_config)
            tags = (
                [tn.x_tag(index) for index in indices]
                if axis == "x"
                else [tn.y_tag(index) for index in indices]
            )
            strip_tn = parent_tn.select(tags, which="any")
            # Keep this LRU smaller than the environment cache: a long-range
            # observable can otherwise retain many selected PEPS strips.
            self._cache_put(
                self._boundary_strip_cache,
                key,
                strip_tn,
                max_size=min(self.boundary_cache_size, 32),
            )
            return strip_tn, False

    def _boundary_transition_amplitude(
        self,
        tn,
        parent_config,
        target_config,
        reference,
    ):
        """Evaluate one local proposal using the parent's cached boundaries."""
        if self._boundary_geometry is None:
            return self._contract_value(
                self._select_config(tn, target_config),
                reference,
            ), 0, 0, False

        windows = self._changed_axis_windows(parent_config, target_config)
        if not windows:
            return self._contract_value(
                self._select_config(tn, target_config),
                reference,
            ), 0, 0, False

        # A periodic Hamiltonian edge is local in the transverse boundary
        # sweep direction even when its endpoints are far apart in the
        # longitudinal coordinate. Try that short strip first. Some upstream
        # tensor backends reject a particular sweep direction, so use the
        # other cached boundary direction before a full contraction fallback.
        num_environment_hits = 0
        num_environment_builds = 0
        for window_index, (axis, indices) in enumerate(windows):
            try:
                envs, reused = self._cached_boundary_environments(
                    tn,
                    parent_config,
                    axis,
                )
                value = self._contract_axis_window(
                    tn,
                    target_config,
                    axis,
                    indices,
                    envs,
                    reference,
                )
            except Exception:  # pragma: no cover - upstream exceptions vary
                continue
            if reused:
                num_environment_hits += 1
            else:
                num_environment_builds += 1
            return (
                value,
                num_environment_hits,
                num_environment_builds,
                window_index > 0,
            )

        return self._contract_value(
            self._select_config(tn, target_config),
            reference,
        ), num_environment_hits, num_environment_builds, False

    def _should_vmap_proposals(self, *, n_changed, device):
        """Whether this proposal batch should prefer full vmapped amplitudes."""
        if not self._proposal_vmap_enabled:
            return False
        if self.proposal_batching == "cache":
            return False
        if self.proposal_batching == "vmap":
            return True
        return (
            device.type == "cuda"
            and n_changed >= self.proposal_vmap_min_batch
        )

    def proposal_amplitudes(
        self,
        parent_configs,
        target_configs,
        current_amplitudes,
        *,
        chunk_size=None,
    ):
        """Evaluate local Metropolis proposals with cached MPS environments.

        The cache is deliberately attached to the amplitude model rather than
        the sampler. It therefore survives burn-in/thinning sweeps, while
        :meth:`clear_boundary_cache` invalidates it when VMC parameters change.
        Unsupported geometries fall back to ordinary batched amplitudes.
        """
        torch = _require_torch()
        parent_configs = _as_long_matrix(parent_configs)
        target_configs = _as_long_matrix(target_configs)
        if parent_configs.shape != target_configs.shape:
            raise ValueError(
                "parent_configs and target_configs must have matching shapes."
            )
        current_amplitudes = torch.as_tensor(
            current_amplitudes,
            device=parent_configs.device,
        )
        if current_amplitudes.shape != (parent_configs.shape[0],):
            raise ValueError(
                "current_amplitudes must have one value per proposal."
            )
        if self.contraction != "boundary" or self._boundary_geometry is None:
            return _call_amplitude_fn(
                self,
                target_configs,
                chunk_size=chunk_size,
            )

        self._ensure_boundary_cache_current()
        out = current_amplitudes.clone()
        stats = {
            "num_requests": int(parent_configs.shape[0]),
            "num_vmapped": 0,
            "num_vmap_fallback": 0,
            "num_compiled_groups": 0,
            "num_compiled_connections": 0,
            "num_environment_compiled": 0,
            "num_transition_cache_hits": 0,
            "num_environment_cache_hits": 0,
            "num_environment_builds": 0,
            "num_alternative_axis_reused": 0,
            "num_fallback": 0,
        }
        changed = torch.any(parent_configs != target_configs, dim=1)
        unresolved = changed.clone()
        compiled_result = self._compiled_boundary_proposals(
            parent_configs[changed],
            target_configs[changed],
        )
        if compiled_result is not None:
            compiled_values, compiled_handled, compiled_stats = compiled_result
            changed_indices = changed.nonzero(as_tuple=True)[0]
            handled_indices = changed_indices[compiled_handled]
            out[handled_indices] = compiled_values.to(
                dtype=out.dtype,
                device=out.device,
            )
            unresolved[handled_indices] = False
            for name in compiled_stats:
                stats[name] += compiled_stats[name]

        n_unresolved = int(unresolved.sum().item())
        if n_unresolved and self._should_vmap_proposals(
            n_changed=n_unresolved,
            device=parent_configs.device,
        ):
            vmapped = self._try_vmapped_forward(
                target_configs[unresolved],
                force=True,
            )
            if vmapped is None and self.proposal_batching == "vmap":
                # Explicit batching is a stable API promise even when a
                # particular upstream contraction has no native vmap rule.
                # Evaluate the whole proposal set through the normal batch
                # entry point rather than rebuilding one boundary per move.
                vmapped = _call_amplitude_fn(
                    self,
                    target_configs[unresolved],
                    chunk_size=chunk_size,
                )
                stats["num_vmap_fallback"] = n_unresolved
            if vmapped is not None:
                out[unresolved] = vmapped.to(dtype=out.dtype, device=out.device)
                stats["num_vmapped"] = n_unresolved
                stats["num_fallback"] = int(
                    (~torch.isfinite(vmapped)).sum().item()
                )
                self.last_proposal_cache_stats = stats
                return out

        tn = self._unpack_tn()
        reference = self._reference_tensor()
        for index_tensor in unresolved.nonzero(as_tuple=True)[0]:
            index = int(index_tensor)
            parent_config = parent_configs[index]
            target_config = target_configs[index]
            if torch.equal(parent_config, target_config):
                continue
            parent_key = self._configuration_key(parent_config)
            target_key = self._configuration_key(target_config)
            cache_key = (parent_key, target_key)
            try:
                value = self._boundary_transition_cache[cache_key]
                stats["num_transition_cache_hits"] += 1
            except KeyError:
                (
                    value,
                    num_environment_hits,
                    num_environment_builds,
                    alternative_axis,
                ) = self._boundary_transition_amplitude(
                    tn,
                    parent_config,
                    target_config,
                    reference,
                )
                stats["num_environment_cache_hits"] += num_environment_hits
                stats["num_environment_builds"] += num_environment_builds
                if alternative_axis:
                    stats["num_alternative_axis_reused"] += 1
                self._cache_put(self._boundary_transition_cache, cache_key, value)
            if not torch.isfinite(torch.as_tensor(value)).all():
                stats["num_fallback"] += 1
            out[index] = value
        self.last_proposal_cache_stats = stats
        return out

    def proposal_log_amplitudes(
        self,
        parent_configs,
        target_configs,
        *,
        chunk_size=None,
    ):
        """Evaluate proposal log amplitudes through compiled reuse classes."""
        torch = _require_torch()
        parent_configs = _as_long_matrix(parent_configs)
        target_configs = _as_long_matrix(target_configs)
        compiled_result = self._compiled_boundary_proposals(
            parent_configs,
            target_configs,
            log=True,
        )
        if compiled_result is None:
            return super().proposal_log_amplitudes(
                parent_configs,
                target_configs,
                chunk_size=chunk_size,
            )

        (compiled_phase, compiled_log_abs), handled, _stats = compiled_result
        phases = torch.empty(
            target_configs.shape[0],
            dtype=compiled_phase.dtype,
            device=target_configs.device,
        )
        log_abs = torch.empty(
            target_configs.shape[0],
            dtype=compiled_log_abs.dtype,
            device=target_configs.device,
        )
        phases[handled] = compiled_phase
        log_abs[handled] = compiled_log_abs
        unresolved = ~handled
        if bool(torch.any(unresolved)):
            fallback_phase, fallback_log_abs = _call_log_amplitude_fn(
                self.forward_log,
                target_configs[unresolved],
                chunk_size=chunk_size,
            )
            phases[unresolved] = fallback_phase
            log_abs[unresolved] = fallback_log_abs
        return phases, log_abs

    def _infer_boundary_geometry(self, tn):
        try:
            Lx = getattr(tn, "Lx", None)
            Ly = getattr(tn, "Ly", None)
            if Lx is None:
                Lx = getattr(tn, "_Lx")
            if Ly is None:
                Ly = getattr(tn, "_Ly")
            Lx = int(Lx)
            Ly = int(Ly)
            view_kwargs = {
                "site_tag_id": getattr(tn, "_site_tag_id"),
                "x_tag_id": getattr(tn, "_x_tag_id"),
                "y_tag_id": getattr(tn, "_y_tag_id"),
                "site_ind_id": getattr(tn, "_site_ind_id"),
                "Lx": Lx,
                "Ly": Ly,
            }
        except AttributeError:
            return None

        coords = []
        for site in self.sites:
            if not isinstance(site, tuple) or len(site) != 2:
                return None
            try:
                x, y = int(site[0]), int(site[1])
            except (TypeError, ValueError):
                return None
            if not (0 <= x < Lx and 0 <= y < Ly):
                return None
            coords.append((x, y))

        return {
            "Lx": Lx,
            "Ly": Ly,
            "coords": tuple(coords),
            "view_kwargs": view_kwargs,
        }

    def _changed_axis_windows(self, parent_config, target_config):
        """Return cached boundary windows ordered by estimated work.

        Both directions are available as a safe fallback. In particular, a
        wrap-around Hamiltonian bond has two distant endpoints in one lattice
        direction but only one changed plane in the transverse direction.
        """
        torch = _require_torch()
        changed = torch.nonzero(parent_config != target_config, as_tuple=True)[0]
        if changed.numel() == 0:
            return ()

        geometry = self._boundary_geometry
        coords = [geometry["coords"][int(i)] for i in changed.detach().cpu()]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        radius = self.environment_radius
        x0 = max(0, min(xs) - radius)
        x1 = min(geometry["Lx"], max(xs) + radius + 1)
        y0 = max(0, min(ys) - radius)
        y1 = min(geometry["Ly"], max(ys) + radius + 1)

        windows = (
            ("x", tuple(range(x0, x1))),
            ("y", tuple(range(y0, y1))),
        )
        # A complete plane has the other lattice extent. This is equivalent to
        # comparing window widths on a fixed geometry, but makes the choice
        # explicit and deterministic for separated/PBC updates.
        return tuple(sorted(
            windows,
            key=lambda item: (
                len(item[1]) * (
                    geometry["Ly"] if item[0] == "x" else geometry["Lx"]
                ),
                item[0],
            ),
        ))

    def _changed_axis_window(self, parent_config, target_config):
        """Return the preferred cached boundary window for compatibility."""
        windows = self._changed_axis_windows(parent_config, target_config)
        return None if not windows else windows[0]

    def _boundary_environment_options(self):
        """Adapt full-boundary options to Quimb environment builders."""
        options = dict(self.contraction_opts)
        # These belong to ``contract_boundary``'s final scalar contraction,
        # not to ``compute_*_environments`` or ``contract_boundary_from_*``.
        for key in (
            "final_contract",
            "final_contract_opts",
            "sequence",
            "inplace",
            "progbar",
            "optimize",
            "max_separation",
        ):
            options.pop(key, None)
        return options

    def _compute_boundary_environments(self, parent_tn, axis):
        options = self._boundary_environment_options()
        if axis == "x":
            return self._contract_approximate(
                parent_tn.compute_x_environments,
                max_bond=self.chi,
                **options,
            )
        return self._contract_approximate(
            parent_tn.compute_y_environments,
            max_bond=self.chi,
            **options,
        )

    def _replace_strip_projectors(
        self,
        tn,
        strip_tn,
        parent_config,
        target_config,
    ):
        """Copy a parent strip and replace only its changed physical tensors."""
        torch = _require_torch()
        changed = torch.nonzero(parent_config != target_config, as_tuple=True)[0]
        target_strip = strip_tn.copy()
        for config_index in changed.detach().cpu().tolist():
            site = self.sites[int(config_index)]
            site_tag = tn.site_tag(site)
            physical_index = tn.site_ind(site)
            selected_tensor = tn[site_tag].isel({
                physical_index: int(target_config[config_index].item()),
            })
            # ``strip_tn.copy()`` clones tensor objects while sharing immutable
            # parent data, so modifying this tensor leaves the cached template
            # untouched for the next connected configuration.
            target_strip[site_tag].modify(data=selected_tensor.data)
        return target_strip

    def _contract_axis_strip(self, tn, strip_tn, axis, indices, envs, reference):
        import quimb.tensor as qtn

        options = self._boundary_environment_options()
        first = indices[0]
        last = indices[-1]
        if axis == "x":
            reuse_tn = envs[("xmin", first)] | strip_tn | envs[("xmax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_xmin_(
                xrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **options,
            )
        else:
            reuse_tn = envs[("ymin", first)] | strip_tn | envs[("ymax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_ymin_(
                yrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **options,
            )
        return _as_torch_scalar(
            self._contract_remaining(reuse_tn, all),
            reference,
        )

    def _contract_cached_axis_window(
        self,
        tn,
        parent_config,
        target_config,
        axis,
        indices,
        envs,
        strip_tn,
        reference,
    ):
        """Contract a cached parent strip with its local target impurities."""
        target_strip = self._replace_strip_projectors(
            tn,
            strip_tn,
            parent_config,
            target_config,
        )
        return self._contract_axis_strip(
            tn,
            target_strip,
            axis,
            indices,
            envs,
            reference,
        )

    def _contract_axis_window(self, tn, target_config, axis, indices, envs, reference):
        """Compatibility contraction path that rebuilds the selected strip."""
        target_tn = self._select_config(tn, target_config)
        tags = (
            [tn.x_tag(index) for index in indices]
            if axis == "x"
            else [tn.y_tag(index) for index in indices]
        )
        strip_tn = target_tn.select(tags, which="any")
        return self._contract_axis_strip(
            tn,
            strip_tn,
            axis,
            indices,
            envs,
            reference,
        )

    def connected_amplitudes(
        self,
        configs,
        amplitudes,
        connections,
        *,
        chunk_size=None,
        reuse_diagonal=True,
    ):
        """Evaluate connected amplitudes with parent boundary environments."""
        torch = _require_torch()
        configs = _as_long_matrix(configs)
        amplitudes = torch.as_tensor(amplitudes, device=configs.device)
        # A previous parent/fallback amplitude call must not be mistaken for
        # the cache statistics of this connected-target measurement.
        self.last_amplitude_cache_stats = None
        if connections.configs.numel() == 0:
            self.last_connected_reuse_stats = {
                "num_requests": 0,
                "num_diagonal": 0,
                "num_reused": 0,
                "num_batched": 0,
                "num_parallel": 0,
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_compiled_groups": 0,
                "num_compiled_connections": 0,
                "num_environment_compiled": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": 0,
            }
            return torch.empty(0, dtype=amplitudes.dtype, device=configs.device)

        if self.contraction != "boundary" or self._boundary_geometry is None:
            num_diagonal = (
                int(_diagonal_connection_mask(configs, connections).sum().item())
                if reuse_diagonal
                else 0
            )
            result = super().connected_amplitudes(
                configs,
                amplitudes,
                connections,
                chunk_size=chunk_size,
                reuse_diagonal=reuse_diagonal,
            )
            self.last_connected_reuse_stats = {
                "num_requests": int(connections.configs.shape[0]),
                "num_diagonal": num_diagonal,
                "num_reused": 0,
                "num_batched": 0,
                "num_parallel": 0,
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_compiled_groups": 0,
                "num_compiled_connections": 0,
                "num_environment_compiled": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": int(connections.configs.shape[0]) - num_diagonal,
            }
            return result

        diag = (
            _diagonal_connection_mask(configs, connections)
            if reuse_diagonal
            else torch.zeros(
                connections.configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        )
        offdiag = (~diag).nonzero(as_tuple=True)[0]
        if (
            self.amplitude_batching != "serial"
            and self._connection_vmap_enabled
            and offdiag.numel() >= _BOUNDARY_VMAP_CONNECTION_THRESHOLD
        ):
            previous_vmap_state = self._vmap_forward_enabled
            self._vmap_forward_enabled = True
            try:
                result = super().connected_amplitudes(
                    configs,
                    amplitudes,
                    connections,
                    chunk_size=chunk_size,
                    reuse_diagonal=reuse_diagonal,
                )
            finally:
                self._vmap_forward_enabled = previous_vmap_state
            self.last_connected_reuse_stats = {
                "num_requests": int(connections.configs.shape[0]),
                "num_diagonal": int(diag.sum().item()),
                "num_reused": 0,
                "num_batched": int(offdiag.numel()),
                "num_parallel": 0,
                "num_groups": 0,
                "num_grouped_connections": 0,
                "num_strip_cache_hits": 0,
                "num_strip_builds": 0,
                "num_alternative_axis_reused": 0,
                "num_fallback": 0,
            }
            return result

        out = torch.empty(
            connections.configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
        if bool(torch.any(diag)):
            out[diag] = amplitudes[connections.batch_ids[diag]]

        self._ensure_boundary_cache_current()
        tn = self._unpack_tn()
        reference = self._reference_tensor()
        stats = {
            "num_requests": int(connections.configs.shape[0]),
            "num_diagonal": int(diag.sum().item()),
            "num_reused": 0,
            "num_batched": 0,
            "num_parallel": 0,
            "num_groups": 0,
            "num_grouped_connections": 0,
            "num_compiled_groups": 0,
            "num_compiled_connections": 0,
            "num_environment_compiled": 0,
            "num_environment_cache_hits": 0,
            "num_environment_builds": 0,
            "num_strip_cache_hits": 0,
            "num_strip_builds": 0,
            "num_alternative_axis_reused": 0,
            "num_fallback": 0,
        }

        # Group by the first/cheapest boundary strip. Compiled geometry
        # classes are shared across parent walkers, matching the paper's
        # batched reuse path. Unsupported or over-sized groups retain the
        # parent-local eager reuse implementation below.
        compiled_reuse = getattr(self, "_compiled_boundary_reuse", {})
        groups = {}
        compiled_groups = {}
        fallback_indices = []
        for conn_idx_tensor in offdiag:
            conn_idx = int(conn_idx_tensor)
            parent_idx = int(connections.batch_ids[conn_idx].item())
            parent_config = configs[parent_idx]
            target_config = connections.configs[conn_idx]
            windows = self._changed_axis_windows(parent_config, target_config)
            if not windows:
                fallback_indices.append(conn_idx)
                stats["num_fallback"] += 1
                continue
            axis, indices = windows[0]
            reuse_key = (axis, indices)
            if (
                not torch.is_grad_enabled()
                and reuse_key in compiled_reuse
            ):
                compiled_groups.setdefault(reuse_key, []).append(
                    (conn_idx, parent_idx, windows)
                )
            else:
                groups.setdefault((parent_idx, axis, indices), []).append(
                    (conn_idx, windows)
                )

        # A compiled environment builder is also fixed-shape. Populate the
        # ordinary cache first so both compiled reuse and eager fallback see
        # one consistent environment snapshot for this local-energy batch.
        for axis, _indices in compiled_groups:
            try:
                stats["num_environment_compiled"] += (
                    self._populate_compiled_boundary_environments(configs, axis)
                )
            except Exception:  # pragma: no cover - backend-specific fallback
                continue

        # Dispatch each geometry class as one full-batch compiled call. A
        # local Hamiltonian normally has one or two changed sites, so the
        # groups below correspond to the paper's one-row/two-row or
        # one-column/two-column reuse kernels.
        for (axis, indices), entries in compiled_groups.items():
            batch_size = int(self._compiled_boundary_batch_size)
            for start in range(0, len(entries), batch_size):
                entries_chunk = entries[start:start + batch_size]
                conn_indices = [entry[0] for entry in entries_chunk]
                parent_indices = torch.as_tensor(
                    [entry[1] for entry in entries_chunk],
                    dtype=torch.long,
                    device=configs.device,
                )
                conn_index_tensor = torch.as_tensor(
                    conn_indices,
                    dtype=torch.long,
                    device=configs.device,
                )
                try:
                    compiled_values = self._compiled_boundary_reuse_batch(
                        configs[parent_indices],
                        connections.configs[conn_index_tensor],
                        axis,
                        indices,
                    )
                except Exception:  # pragma: no cover - backend-specific fallback
                    compiled_values = None
                if compiled_values is None:
                    for conn_idx, parent_idx, windows in entries_chunk:
                        groups.setdefault(
                            (parent_idx, axis, indices), []
                        ).append((conn_idx, windows))
                    continue
                out[conn_index_tensor] = compiled_values.to(
                    dtype=out.dtype,
                    device=out.device,
                )
                stats["num_compiled_groups"] += 1
                stats["num_compiled_connections"] += len(entries_chunk)
                stats["num_batched"] += len(entries_chunk)
                stats["num_reused"] += len(entries_chunk)

        stats["num_groups"] = len(groups)
        stats["num_grouped_connections"] = sum(
            len(entries) for entries in groups.values()
        )
        parent_tns = {}
        contexts = {}

        def get_context(parent_idx, parent_config, axis, indices):
            cache_key = (parent_idx, axis, indices)
            if cache_key in contexts:
                return contexts[cache_key]
            try:
                parent_key = self._configuration_key(parent_config)
                environment_key = (axis, parent_key)
                strip_key = (axis, tuple(indices), parent_key)
                envs = self._boundary_environment_cache.get(environment_key)
                strip_tn = self._boundary_strip_cache.get(strip_key)
                environment_reused = envs is not None
                strip_reused = strip_tn is not None
                if not (environment_reused and strip_reused):
                    parent_tn = parent_tns.get(parent_idx)
                    if parent_tn is None:
                        parent_tn = self._select_config(tn, parent_config)
                        parent_tns[parent_idx] = parent_tn
                    if not environment_reused:
                        envs, environment_reused = (
                            self._cached_boundary_environments(
                                tn,
                                parent_config,
                                axis,
                                parent_tn=parent_tn,
                            )
                        )
                    if not strip_reused:
                        strip_tn, strip_reused = self._cached_boundary_strip(
                            tn,
                            parent_config,
                            axis,
                            indices,
                            parent_tn=parent_tn,
                        )
            except Exception:  # pragma: no cover - upstream exceptions vary
                contexts[cache_key] = None
                return None
            stats[
                "num_environment_cache_hits" if environment_reused
                else "num_environment_builds"
            ] += 1
            stats["num_strip_cache_hits" if strip_reused else "num_strip_builds"] += 1
            contexts[cache_key] = (envs, strip_tn)
            return contexts[cache_key]

        # Build all primary contexts before dispatching work. This keeps cache
        # mutation and boundary-environment construction on the caller thread;
        # only independent scalar closures are allowed to run concurrently.
        primary_jobs = []
        alternative_jobs = []
        for (parent_idx, axis, indices), entries in groups.items():
            parent_config = configs[parent_idx]
            primary_context = get_context(
                parent_idx,
                parent_config,
                axis,
                indices,
            )
            for conn_idx, windows in entries:
                job = (
                    conn_idx,
                    parent_idx,
                    axis,
                    indices,
                    windows,
                    primary_context,
                )
                if primary_context is None:
                    alternative_jobs.append(job)
                else:
                    primary_jobs.append(job)

        def contract_primary(job):
            conn_idx, parent_idx, axis, indices, windows, context = job
            parent_config = configs[parent_idx]
            target_config = connections.configs[conn_idx]
            envs, strip_tn = context
            try:
                value = self._contract_cached_axis_window(
                    tn,
                    parent_config,
                    target_config,
                    axis,
                    indices,
                    envs,
                    strip_tn,
                    reference,
                )
            except Exception:  # pragma: no cover - upstream exceptions vary
                return job, None, False
            return job, value, True

        def contract_primary_no_grad(job):
            # Torch's grad mode is thread-local; explicitly carry the
            # measurement mode into worker threads.
            with torch.no_grad():
                return contract_primary(job)

        reference_device = getattr(
            getattr(reference, "device", None),
            "type",
            None,
        )
        boundary_workers = _check_positive_int(
            "boundary_workers",
            getattr(self, "boundary_workers", 1),
        )
        use_parallel = (
            boundary_workers > 1
            and len(primary_jobs) > 1
            and not torch.is_grad_enabled()
            and reference_device in (None, "cpu")
        )
        if use_parallel:
            # Threading is deliberately restricted to no-grad CPU inference.
            # The PEPS parameters and cached environments remain shared, while
            # every worker contracts into its own temporary TensorNetwork.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=boundary_workers) as executor:
                primary_results = executor.map(
                    contract_primary_no_grad,
                    primary_jobs,
                )
                for job, value, value_found in primary_results:
                    if value_found:
                        conn_idx = job[0]
                        out[conn_idx] = value
                        stats["num_reused"] += 1
                    else:
                        alternative_jobs.append(job)
            stats["num_parallel"] = len(primary_jobs)
        else:
            for job in primary_jobs:
                job, value, value_found = contract_primary(job)
                if value_found:
                    conn_idx = job[0]
                    out[conn_idx] = value
                    stats["num_reused"] += 1
                else:
                    alternative_jobs.append(job)

        # Alternative-axis retries stay serial because they may create new
        # cached contexts. They are uncommon for ordinary nearest-neighbor
        # updates and retain the existing robustness path.
        for conn_idx, parent_idx, _axis, _indices, windows, _context in alternative_jobs:
            parent_config = configs[parent_idx]
            target_config = connections.configs[conn_idx]
            value_found = False
            for alt_axis, alt_indices in windows[1:]:
                context = get_context(
                    parent_idx,
                    parent_config,
                    alt_axis,
                    alt_indices,
                )
                if context is None:
                    continue
                envs, strip_tn = context
                try:
                    out[conn_idx] = self._contract_cached_axis_window(
                        tn,
                        parent_config,
                        target_config,
                        alt_axis,
                        alt_indices,
                        envs,
                        strip_tn,
                        reference,
                    )
                except Exception:  # pragma: no cover - upstream exceptions vary
                    continue
                value_found = True
                stats["num_reused"] += 1
                stats["num_alternative_axis_reused"] += 1
                break

            if not value_found:
                fallback_indices.append(conn_idx)
                stats["num_fallback"] += 1

        if fallback_indices:
            fallback_indices = torch.as_tensor(
                fallback_indices,
                dtype=torch.long,
                device=configs.device,
            )
            # Re-enter ``forward`` rather than directly contracting each
            # target. In inference mode this deduplicates and consults the
            # persistent boundary-amplitude cache; when vmap is available it
            # can also evaluate unresolved targets as one fixed batch.
            # Gradient-enabled calls deliberately retain ``forward``'s normal
            # uncached differentiable path.
            out[fallback_indices] = self.forward(
                connections.configs[fallback_indices],
                chunk_size=chunk_size,
            ).to(dtype=out.dtype, device=out.device)

        self.last_connected_reuse_stats = stats
        return out


def make_torch_peps_amplitude_model(peps, **kwargs):
    """Build the appropriate torch amplitude model for ``contraction``."""
    from ..api import _resolve_contraction_config
    contraction, chi, cutoff, contraction_opts = _resolve_contraction_config(
        kwargs.get("contraction", "exact"),
        kwargs.get("chi"),
        kwargs.get("cutoff"),
        kwargs.get("contraction_opts"),
    )
    kwargs = dict(kwargs)
    kwargs.update(
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
    )
    contraction = _validate_contraction(
        contraction,
        chi,
    )
    if contraction == "boundary":
        return TorchPEPSBoundaryAmplitude(peps, **kwargs)
    return TorchPEPSAmplitude(peps, **kwargs)
