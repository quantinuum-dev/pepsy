"""PyTorch kernels for lightweight VMC loops.

The routines here are intentionally small and optional-dependency friendly.
They cover the sampler and local-energy pieces that are useful around PEPS
amplitude models without vendoring a full VMC framework.
"""

from __future__ import annotations

from itertools import product
import time
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

__all__ = [
    "FermionSiteEncoding",
    "TorchFermionVMCMetadata",
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchMCMCSamples",
    "TorchChainDiagnostics",
    "TorchMetropolisSampler",
    "TorchBPMetropolisSampler",
    "TorchVMCDriver",
    "TorchFermionVMC",
    "TorchVMCEnergyEstimate",
    "TorchVMCImportanceEstimate",
    "TorchVMCStepResult",
    "TorchSRResult",
    "TorchSquareLattice",
    "apply_torch_sr_update",
    "count_spinful_particles",
    "heisenberg_connections",
    "local_energy_from_connections",
    "torch_chain_diagnostics",
    "metropolis_local_sampler",
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "propose_spinful_u1_exchange_or_hopping",
    "propose_spinful_z2_exchange_or_hopping",
    "propose_spinful_z2z2_exchange_or_hopping",
    "random_spin_configs",
    "random_spinful_configs",
    "make_torch_peps_amplitude_model",
    "solve_torch_sr",
    "spinful_fermi_hubbard_connections",
    "torch_log_derivative_matrix",
    "transverse_ising_connections",
    "torch_hamiltonian_connections",
]


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.torch requires optional dependency 'torch'. "
            "Install it with `pip install pepsy[torch]` or `pip install torch`."
        ) from exc
    return torch


def _check_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _as_long_matrix(configs, *, name="configs"):
    torch = _require_torch()
    configs = torch.as_tensor(configs, dtype=torch.long)
    if configs.ndim == 1:
        configs = configs.reshape(1, -1)
    if configs.ndim != 2:
        raise ValueError(f"{name} must have shape (n_batch, n_sites).")
    return configs


def _edge_value(value, edge):
    if isinstance(value, dict):
        i, j = edge
        if edge in value:
            return value[edge]
        if (j, i) in value:
            return value[(j, i)]
        return 0.0
    return value


def _site_value(value, site):
    if isinstance(value, dict):
        return value.get(site, 0.0)
    return value


_CONTRACTION_ALIASES = {
    "exact": "exact",
    "hotrg": "hotrg",
    "ctmrg": "ctmrg",
    "boundary": "boundary",
    "contract_boundary": "boundary",
    "mps": "boundary",
    "boundary_mps": "boundary",
    "contract-boundary": "boundary",
    "boundary-mps": "boundary",
}


def _validate_contraction(contraction, chi):
    key = str(contraction).replace("_", "-").lower()
    try:
        contraction = _CONTRACTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            "contraction must be 'exact', 'hotrg', 'ctmrg', or 'boundary'."
        ) from exc
    if contraction in {"hotrg", "ctmrg", "boundary"} and chi is None:
        raise ValueError(f"contraction={contraction!r} requires chi.")
    return contraction


def _as_contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


def _torch_finfo_tiny(dtype):
    torch = _require_torch()
    if dtype.is_complex:
        dtype = torch.empty((), dtype=dtype).real.dtype
    return torch.finfo(dtype).tiny


def _is_symmray_data(data):
    cls = type(data)
    return cls.__module__.split(".", 1)[0] == "symmray"


def _find_symmray_tensors(tn):
    tensor_map = getattr(tn, "tensor_map", {})
    return [
        tensor_id
        for tensor_id, tensor in tensor_map.items()
        if _is_symmray_data(getattr(tensor, "data", None))
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


def _check_nonnegative_int(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


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
        out[offdiag] = _call_amplitude_fn(
            amplitude_fn,
            connections.configs[offdiag],
            chunk_size=chunk_size,
        ).to(dtype=out.dtype, device=out.device)
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
    ):
        torch = _require_torch()
        import quimb as qu
        import quimb.tensor as qtn

        self.contraction = _validate_contraction(contraction, chi)
        self.chi = None if chi is None else int(chi)
        self.cutoff = (
            0.0
            if cutoff is None and self.contraction == "exact"
            else 1.0e-10 if cutoff is None else float(cutoff)
        )
        self.contraction_opts = _as_contraction_options(contraction_opts)
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

        params, skeleton = qtn.pack(tn)
        flat_params, params_pytree = qu.utils.tree_flatten(params, get_ref=True)
        leaves = []
        for leaf in flat_params:
            tensor = torch.as_tensor(leaf, dtype=dtype, device=device)
            leaves.append(torch.nn.Parameter(tensor.clone()))
        self.params = torch.nn.ParameterList(leaves)
        self.params_pytree = params_pytree
        self.skeleton = skeleton

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
        self.params.to(*args, **kwargs)
        return self

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

    def _contract_approximate(self, fn, *args, **kwargs):
        """Contract with the requested cutoff, retrying empty sparse sectors."""
        kwargs = dict(kwargs)
        kwargs["cutoff"] = self.cutoff
        try:
            return fn(*args, **kwargs)
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
            return fn(*args, **retry_kwargs)

    def _contract_value(self, tnx, reference=None):
        if self.contraction == "hotrg":
            value = self._contract_approximate(
                tnx.contract_hotrg,
                max_bond=self.chi,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            value = self._contract_approximate(
                tnx.contract_ctmrg,
                max_bond=self.chi,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            value = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                **self.contraction_opts,
            )
        else:
            value = tnx.contract(all)
        return _as_torch_scalar(value, reference)

    def _contract_log_parts(self, tnx, reference=None):
        torch = _require_torch()
        if self.contraction == "hotrg":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_hotrg,
                max_bond=self.chi,
                strip_exponent=True,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_ctmrg,
                max_bond=self.chi,
                strip_exponent=True,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            mantissa, exponent_10 = self._contract_approximate(
                tnx.contract_boundary,
                max_bond=self.chi,
                strip_exponent=True,
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

    def amplitude(self, config, params=None):
        """Evaluate a single configuration amplitude."""
        config = _as_long_matrix(config).reshape(-1)
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return self._contract_value(self._select_config(tn, config), reference)

    def forward(self, configs, params=None, *, chunk_size=None):
        """Evaluate a batch of configuration amplitudes."""
        configs = _as_long_matrix(configs)
        chunk_size = _normalize_chunk_size(chunk_size)
        if chunk_size is not None and configs.shape[0] > chunk_size:
            return _require_torch().cat([
                self.forward(
                    configs[start:start + chunk_size],
                    params=params,
                    chunk_size=None,
                )
                for start in range(0, configs.shape[0], chunk_size)
            ])
        tn = self._unpack_tn(params)
        reference = self._reference_tensor(params)
        return _require_torch().stack([
            self._contract_value(self._select_config(tn, row), reference)
            for row in configs
        ])

    def forward_log(self, configs, params=None):
        """Return ``(phase, log_abs)`` for a batch of configurations."""
        configs = _as_long_matrix(configs)
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
        environment_radius=0,
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
        )
        self.environment_radius = _check_nonnegative_int(
            "environment_radius",
            environment_radius,
        )
        self._boundary_geometry = self._infer_boundary_geometry(self._unpack_tn())
        self.last_connected_reuse_stats = None

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

    def _changed_axis_window(self, parent_config, target_config):
        torch = _require_torch()
        changed = torch.nonzero(parent_config != target_config, as_tuple=True)[0]
        if changed.numel() == 0:
            return None

        geometry = self._boundary_geometry
        coords = [geometry["coords"][int(i)] for i in changed.detach().cpu()]
        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        radius = self.environment_radius
        x0 = max(0, min(xs) - radius)
        x1 = min(geometry["Lx"], max(xs) + radius + 1)
        y0 = max(0, min(ys) - radius)
        y1 = min(geometry["Ly"], max(ys) + radius + 1)

        if (x1 - x0) <= (y1 - y0):
            return "x", tuple(range(x0, x1))
        return "y", tuple(range(y0, y1))

    def _compute_boundary_environments(self, parent_tn, axis):
        if axis == "x":
            return self._contract_approximate(
                parent_tn.compute_x_environments,
                max_bond=self.chi,
                **self.contraction_opts,
            )
        return self._contract_approximate(
            parent_tn.compute_y_environments,
            max_bond=self.chi,
            **self.contraction_opts,
        )

    def _contract_axis_window(self, tn, target_config, axis, indices, envs, reference):
        import quimb.tensor as qtn

        target_tn = self._select_config(tn, target_config)
        first = indices[0]
        last = indices[-1]
        if axis == "x":
            tags = [tn.x_tag(i) for i in indices]
            window_tn = target_tn.select(tags, which="any")
            reuse_tn = envs[("xmin", first)] | window_tn | envs[("xmax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_xmin_(
                xrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        else:
            tags = [tn.y_tag(i) for i in indices]
            window_tn = target_tn.select(tags, which="any")
            reuse_tn = envs[("ymin", first)] | window_tn | envs[("ymax", last)]
            reuse_tn.view_as_(
                qtn.PEPS,
                **self._boundary_geometry["view_kwargs"],
            )
            reuse_tn.contract_boundary_from_ymin_(
                yrange=[first, last + 1],
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        return _as_torch_scalar(reuse_tn.contract(all), reference)

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
        if connections.configs.numel() == 0:
            self.last_connected_reuse_stats = {
                "num_diagonal": 0,
                "num_reused": 0,
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
                "num_diagonal": num_diagonal,
                "num_reused": 0,
                "num_fallback": int(connections.configs.shape[0]) - num_diagonal,
            }
            return result

        out = torch.empty(
            connections.configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
        diag = (
            _diagonal_connection_mask(configs, connections)
            if reuse_diagonal
            else torch.zeros(
                connections.configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        )
        if bool(torch.any(diag)):
            out[diag] = amplitudes[connections.batch_ids[diag]]

        tn = self._unpack_tn()
        reference = self._reference_tensor()
        env_cache = {}
        stats = {
            "num_diagonal": int(diag.sum().item()),
            "num_reused": 0,
            "num_fallback": 0,
        }

        offdiag = (~diag).nonzero(as_tuple=True)[0]
        for conn_idx_tensor in offdiag:
            conn_idx = int(conn_idx_tensor)
            parent_idx = int(connections.batch_ids[conn_idx].item())
            parent_config = configs[parent_idx]
            target_config = connections.configs[conn_idx]
            window = self._changed_axis_window(parent_config, target_config)
            if window is None:
                out[conn_idx] = self._contract_value(
                    self._select_config(tn, target_config),
                    reference,
                )
                stats["num_fallback"] += 1
                continue

            axis, indices = window
            cache_key = (parent_idx, axis)
            try:
                envs = env_cache[cache_key]
            except KeyError:
                parent_tn = self._select_config(tn, parent_config)
                envs = self._compute_boundary_environments(parent_tn, axis)
                env_cache[cache_key] = envs

            try:
                out[conn_idx] = self._contract_axis_window(
                    tn,
                    target_config,
                    axis,
                    indices,
                    envs,
                    reference,
                )
                stats["num_reused"] += 1
            except Exception:  # pragma: no cover - exact upstream exception varies
                out[conn_idx] = self._contract_value(
                    self._select_config(tn, target_config),
                    reference,
                )
                stats["num_fallback"] += 1

        self.last_connected_reuse_stats = stats
        return out


def make_torch_peps_amplitude_model(peps, **kwargs):
    """Build a :class:`TorchPEPSAmplitude` from a quimb PEPS-like object."""
    return TorchPEPSAmplitude(peps, **kwargs)


@dataclass(frozen=True)
class TorchSRResult:
    """Result of a torch stochastic-reconfiguration linear solve."""

    direction: Any
    energy_mean: Any
    energy_variance: Any
    force: Any
    centered_log_derivatives: Any
    method: str
    diag_shift: float
    info: dict[str, Any]


def _torch_model_parameters(model):
    try:
        params = list(model.parameters())
    except AttributeError as exc:
        raise TypeError("model must expose a parameters() method.") from exc
    if not params:
        raise ValueError("model must expose at least one trainable parameter.")
    return params


def _flatten_torch_tensors(tensors, refs):
    torch = _require_torch()
    pieces = []
    for tensor, ref in zip(tensors, refs, strict=True):
        if tensor is None:
            tensor = torch.zeros_like(ref)
        pieces.append(tensor.reshape(-1))
    return torch.cat(pieces) if pieces else torch.empty(0)


def torch_log_derivative_matrix(
    model,
    configs,
    *,
    amplitude_floor=None,
    create_graph=False,
):
    """Return per-sample log-amplitude derivatives for a torch model.

    The returned matrix has shape ``(n_samples, n_params)`` and entries
    ``d psi(config) / d theta / psi(config)``. It is intended for real-valued
    PEPS amplitudes; complex-amplitude SR needs an explicit real/imaginary
    parameter convention and is not silently guessed here.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    params = _torch_model_parameters(model)
    rows = []

    for config in configs:
        amp = model(config.reshape(1, -1))
        amp = torch.as_tensor(amp).reshape(-1)
        if amp.numel() != 1:
            raise ValueError("model(config) must return one amplitude per row.")
        amp = amp[0]
        if torch.is_complex(amp):
            raise NotImplementedError(
                "torch_log_derivative_matrix currently supports real scalar "
                "amplitudes. Split complex parameters/amplitudes explicitly "
                "before using torch SR."
            )
        if not amp.requires_grad:
            raise RuntimeError("model amplitude does not require gradients.")

        amp_abs = amp.detach().abs()
        if amplitude_floor is None:
            if amp_abs.item() == 0:
                raise ZeroDivisionError(
                    "Encountered a zero amplitude while forming log derivatives."
                )
            denom = amp
        else:
            floor = torch.as_tensor(
                amplitude_floor,
                dtype=amp.dtype,
                device=amp.device,
            )
            sign = torch.where(
                amp.detach() >= 0,
                torch.ones_like(amp),
                -torch.ones_like(amp),
            )
            denom = torch.where(amp_abs < floor, sign * floor, amp)

        grads = torch.autograd.grad(
            amp,
            params,
            retain_graph=create_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        row = _flatten_torch_tensors(grads, params) / denom
        if not create_graph:
            row = row.detach()
        rows.append(row)

    return torch.stack(rows, dim=0)


def _promote_sr_tensors(log_derivatives, local_energies):
    torch = _require_torch()
    log_derivatives = torch.as_tensor(log_derivatives)
    if log_derivatives.ndim != 2:
        raise ValueError("log_derivatives must have shape (n_samples, n_params).")
    if not torch.is_floating_point(log_derivatives) and not torch.is_complex(
        log_derivatives
    ):
        log_derivatives = log_derivatives.to(torch.float64)

    local_energies = torch.as_tensor(local_energies, device=log_derivatives.device)
    if local_energies.ndim != 1:
        local_energies = local_energies.reshape(-1)
    if local_energies.shape[0] != log_derivatives.shape[0]:
        raise ValueError("local_energies must have one entry per sample.")
    if not torch.is_floating_point(local_energies) and not torch.is_complex(
        local_energies
    ):
        local_energies = local_energies.to(log_derivatives.dtype)

    dtype = torch.promote_types(log_derivatives.dtype, local_energies.dtype)
    return log_derivatives.to(dtype), local_energies.to(dtype)


def _torch_solve_linear(matrix, rhs):
    torch = _require_torch()
    try:
        return torch.linalg.solve(matrix, rhs), "solve"
    except RuntimeError:
        return torch.linalg.lstsq(matrix, rhs).solution, "lstsq"


def solve_torch_sr(
    log_derivatives,
    local_energies,
    *,
    diag_shift=1.0e-4,
    method="auto",
    center=True,
):
    """Solve direct SR or sample-space minSR for a torch VMC batch.

    ``method="direct"`` forms the parameter-space covariance matrix.
    ``method="minsr"`` solves the equivalent sample-space system, which is
    preferable when the number of PEPS parameters is much larger than the
    number of Monte Carlo samples. ``method="auto"`` picks minSR when
    ``n_samples < n_params``.
    """
    torch = _require_torch()
    if diag_shift < 0:
        raise ValueError("diag_shift must be non-negative.")
    log_derivatives, local_energies = _promote_sr_tensors(
        log_derivatives,
        local_energies,
    )
    n_samples, n_params = log_derivatives.shape
    if n_samples == 0 or n_params == 0:
        raise ValueError("SR requires at least one sample and one parameter.")

    method_key = str(method).replace("_", "").replace("-", "").lower()
    if method_key == "auto":
        method_key = "minsr" if n_samples < n_params else "direct"
    if method_key not in {"direct", "sr", "minsr"}:
        raise ValueError("method must be 'auto', 'direct', or 'minsr'.")
    if method_key == "sr":
        method_key = "direct"

    energy_mean = local_energies.mean()
    energy_residual = local_energies - energy_mean if center else local_energies
    centered = (
        log_derivatives - log_derivatives.mean(dim=0, keepdim=True)
        if center
        else log_derivatives
    )
    force = centered.conj().transpose(0, 1) @ energy_residual / n_samples
    shift = torch.as_tensor(
        diag_shift,
        dtype=log_derivatives.dtype,
        device=log_derivatives.device,
    )

    if method_key == "direct":
        eye = torch.eye(
            n_params,
            dtype=log_derivatives.dtype,
            device=log_derivatives.device,
        )
        sr_matrix = centered.conj().transpose(0, 1) @ centered / n_samples
        direction, solver = _torch_solve_linear(sr_matrix + shift * eye, force)
        matrix_shape = tuple(sr_matrix.shape)
    else:
        eye = torch.eye(
            n_samples,
            dtype=log_derivatives.dtype,
            device=log_derivatives.device,
        )
        gram = centered @ centered.conj().transpose(0, 1)
        alpha, solver = _torch_solve_linear(
            gram + n_samples * shift * eye,
            energy_residual,
        )
        direction = centered.conj().transpose(0, 1) @ alpha
        matrix_shape = tuple(gram.shape)

    energy_variance = energy_residual.abs().square().mean()
    return TorchSRResult(
        direction=direction,
        energy_mean=energy_mean,
        energy_variance=energy_variance.real,
        force=force,
        centered_log_derivatives=centered,
        method=method_key,
        diag_shift=float(diag_shift),
        info={"solver": solver, "matrix_shape": matrix_shape},
    )


def apply_torch_sr_update(model, direction, *, learning_rate=1.0):
    """Apply ``theta <- theta - learning_rate * direction`` in place."""
    torch = _require_torch()
    params = _torch_model_parameters(model)
    n_params = sum(param.numel() for param in params)
    direction = torch.as_tensor(direction)
    if direction.numel() != n_params:
        raise ValueError(
            f"direction has {direction.numel()} entries, expected {n_params}."
        )

    offset = 0
    with torch.no_grad():
        for param in params:
            size = param.numel()
            update = direction[offset:offset + size].reshape_as(param)
            if torch.is_complex(update) and not torch.is_complex(param):
                if update.imag.abs().max().item() > 1.0e-12:
                    raise ValueError(
                        "Cannot apply a complex SR direction to real parameters."
                    )
                update = update.real
            update = update.to(dtype=param.dtype, device=param.device)
            param.sub_(learning_rate * update)
            offset += size
    return model


@dataclass(frozen=True)
class FermionSiteEncoding:
    """Four-state spinful-fermion on-site encoding.

    Native torch VMC and Pepsy's four-sector fermionic PEPS use
    ``0=empty, 1=down, 2=up, 3=double``. The ``symmray`` constructor is kept
    for callers that explicitly use the alternate legacy labels. Use the
    class constructors to make the choice explicit at interop boundaries.
    """

    empty: int = 0
    double: int = 1
    up: int = 2
    down: int = 3

    def __post_init__(self):
        values = (self.empty, self.double, self.up, self.down)
        if len(set(values)) != 4 or any(v < 0 for v in values):
            raise ValueError("Fermion site codes must be unique non-negative ints.")

    @classmethod
    def symmray(cls):
        """Return Pepsy/Symmray's spinful physical-index convention."""
        return cls(empty=0, double=1, up=2, down=3)

    @classmethod
    def vmc_torch(cls):
        """Return the convention used by ``sjdu10/vmc_torch``."""
        return cls(empty=0, double=3, up=2, down=1)

    @classmethod
    def from_fermion(cls, fermion, *, physical_charges=None):
        """Return the PEPS physical-index encoding for a native ``Fermion``.

        When the PEPS exposes four resolved ``U1U1`` physical charges, their
        ordered charge map is authoritative. For charge-collapsed spinful
        ``U1`` data, the conventional four-state PEPS order is used. This
        keeps the physical-index contract distinct from the dense local basis
        used internally while constructing native Fermion operators.
        """
        if not bool(getattr(fermion, "spinful", False)):
            raise ValueError("FermionSiteEncoding.from_fermion requires spinful=True.")
        if physical_charges:
            try:
                return cls.from_physical_charges(physical_charges)
            except ValueError:
                pass
        return cls.vmc_torch()

    @classmethod
    def from_physical_charges(cls, physical_charges):
        """Return an encoding from four resolved ``(n_up, n_down)`` charges."""
        lookup = {}
        for code, charge in enumerate(tuple(physical_charges)):
            if not isinstance(charge, tuple) or len(charge) != 2:
                raise ValueError("Physical charges must be two-component tuples.")
            charge = tuple(int(value) for value in charge)
            if charge in lookup:
                raise ValueError("PEPS physical charges must be unique.")
            lookup[charge] = code
        required = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if set(lookup) != required:
            raise ValueError(
                "Physical charges must contain exactly the four spinful states."
            )
        return cls(
            empty=lookup[(0, 0)],
            down=lookup[(0, 1)],
            up=lookup[(1, 0)],
            double=lookup[(1, 1)],
        )

    @property
    def max_code(self):
        return max(self.empty, self.double, self.up, self.down)

    def validate(self, configs):
        """Raise if ``configs`` contains a code outside this encoding."""
        torch = _require_torch()
        valid = (
            (configs == self.empty)
            | (configs == self.double)
            | (configs == self.up)
            | (configs == self.down)
        )
        if not torch.all(valid):
            bad = torch.unique(configs[~valid]).detach().cpu().tolist()
            raise ValueError(f"Unknown fermion site code(s): {bad!r}.")

    def decode(self, configs):
        """Return ``(n_up, n_down)`` tensors for encoded site configs."""
        torch = _require_torch()
        configs = torch.as_tensor(configs, dtype=torch.long)
        self.validate(configs)
        lookup = torch.zeros(
            (self.max_code + 1, 2),
            dtype=torch.long,
            device=configs.device,
        )
        lookup[self.up, 0] = 1
        lookup[self.down, 1] = 1
        lookup[self.double, 0] = 1
        lookup[self.double, 1] = 1
        occ = lookup[configs]
        return occ[..., 0], occ[..., 1]

    def encode(self, n_up, n_down):
        """Encode ``(n_up, n_down)`` occupation tensors as site states."""
        torch = _require_torch()
        n_up = torch.as_tensor(n_up)
        n_down = torch.as_tensor(n_down, device=n_up.device)
        code = torch.full_like(n_up.long(), self.empty)
        code = torch.where((n_up == 1) & (n_down == 0), self.up, code)
        code = torch.where((n_up == 0) & (n_down == 1), self.down, code)
        code = torch.where((n_up == 1) & (n_down == 1), self.double, code)
        return code


@dataclass(frozen=True)
class TorchSquareLattice:
    """Square-lattice nearest-neighbor graph with grouped row/column edges."""

    Lx: int
    Ly: int
    pbc: bool | tuple[bool, bool] = False

    def __post_init__(self):
        Lx = _check_positive_int("Lx", self.Lx)
        Ly = _check_positive_int("Ly", self.Ly)
        if isinstance(self.pbc, bool):
            pbc_x = pbc_y = self.pbc
        else:
            pbc_x, pbc_y = self.pbc

        row_edges = {i: [] for i in range(Lx)}
        for i in range(Lx):
            for j in range(Ly - 1):
                row_edges[i].append((i * Ly + j, i * Ly + j + 1))
            if pbc_y and Ly > 2:
                row_edges[i].append((i * Ly + Ly - 1, i * Ly))

        col_edges = {j: [] for j in range(Ly)}
        for j in range(Ly):
            for i in range(Lx - 1):
                col_edges[j].append((i * Ly + j, (i + 1) * Ly + j))
            if pbc_x and Lx > 2:
                col_edges[j].append(((Lx - 1) * Ly + j, j))

        object.__setattr__(self, "Lx", Lx)
        object.__setattr__(self, "Ly", Ly)
        object.__setattr__(self, "row_edges", {k: tuple(v) for k, v in row_edges.items()})
        object.__setattr__(self, "col_edges", {k: tuple(v) for k, v in col_edges.items()})

    @property
    def n_sites(self):
        return self.Lx * self.Ly

    @property
    def edges(self):
        edges = []
        for group in self.row_edges.values():
            edges.extend(group)
        for group in self.col_edges.values():
            edges.extend(group)
        return tuple(edges)


def _peps_physical_axis(tn, site):
    """Return the physical tensor axis and dimension for ``site``."""
    tensor = tn[site]
    try:
        axis = tuple(tensor.inds).index(tn.site_ind(site))
    except (AttributeError, ValueError) as exc:
        raise ValueError(
            f"Could not locate the physical index for PEPS site {site!r}."
        ) from exc
    shape = getattr(tensor, "shape", None)
    if shape is None:
        shape = getattr(getattr(tensor, "data", None), "shape", None)
    if shape is None:
        raise ValueError(f"Could not determine the physical dimension at site {site!r}.")
    return axis, int(shape[axis])


def _peps_physical_charges(tn, site):
    """Return ordered Symmray physical charges, when available."""
    tensor = tn[site]
    data = getattr(tensor, "data", None)
    if not _is_symmray_data(data):
        return ()
    try:
        axis, _ = _peps_physical_axis(tn, site)
        index = data.indices[axis]
        chargemap = getattr(index, "chargemap", None)
    except (AttributeError, IndexError, TypeError, ValueError):
        return ()
    if chargemap is None:
        return ()
    return tuple(chargemap.keys())


def _peps_symmetry(tn, site_order):
    """Return the named Symmray symmetry carried by the PEPS, if present."""
    for site in site_order:
        symmetry = getattr(getattr(tn[site], "data", None), "symmetry", None)
        if symmetry is not None:
            return str(symmetry).upper()
    return None


def _peps_lattice_edges(site_order, Lx, Ly, *, pbc=False):
    """Infer coordinate-labelled nearest-neighbor edges from PEPS metadata."""
    site_order = tuple(site_order)
    if not all(
        isinstance(site, tuple)
        and len(site) == 2
        and all(isinstance(value, Integral) for value in site)
        for site in site_order
    ):
        raise ValueError(
            "PEPS sites must be coordinate labels to infer lattice edges; "
            "pass edges explicitly for non-coordinate site labels."
        )
    site_order = tuple((int(site[0]), int(site[1])) for site in site_order)
    by_coord = {site: site for site in site_order}
    expected = {(x, y) for x in range(Lx) for y in range(Ly)}
    if set(by_coord) != expected:
        raise ValueError(
            "PEPS coordinate sites do not form the inferred rectangular grid."
        )

    if isinstance(pbc, bool):
        pbc_x = pbc_y = pbc
    else:
        try:
            pbc_x, pbc_y = pbc
        except (TypeError, ValueError) as exc:
            raise ValueError("pbc must be a bool or a two-entry tuple.") from exc

    edges = []
    for x in range(Lx):
        for y in range(Ly - 1):
            edges.append(((x, y), (x, y + 1)))
        if pbc_y and Ly > 2:
            edges.append(((x, Ly - 1), (x, 0)))
    for y in range(Ly):
        for x in range(Lx - 1):
            edges.append(((x, y), (x + 1, y)))
        if pbc_x and Lx > 2:
            edges.append(((Lx - 1, y), (0, y)))
    return tuple(edges)


def _coerce_labelled_edges(edges, site_order):
    """Normalize explicit edges to labels in ``site_order``."""
    site_order = tuple(site_order)
    positions = {site: i for i, site in enumerate(site_order)}
    normalized = []
    for edge in tuple(edges):
        try:
            left, right = tuple(edge)
        except (TypeError, ValueError) as exc:
            raise ValueError("Each edge must contain exactly two site labels.") from exc
        if left in positions and right in positions:
            normalized.append((left, right))
            continue
        if (
            isinstance(left, Integral)
            and not isinstance(left, bool)
            and isinstance(right, Integral)
            and not isinstance(right, bool)
            and 0 <= int(left) < len(site_order)
            and 0 <= int(right) < len(site_order)
        ):
            normalized.append((site_order[int(left)], site_order[int(right)]))
            continue
        raise ValueError(
            f"Edge {(left, right)!r} contains a site not present in the PEPS."
        )
    return tuple(normalized)


def _sum_site_charges(tn, site_order):
    """Infer a fixed global charge from Symmray tensor charge metadata."""
    charges = []
    for site in site_order:
        charge = getattr(getattr(tn[site], "data", None), "charge", None)
        if charge is None:
            return None
        if isinstance(charge, tuple):
            charge = tuple(int(value) for value in charge)
        else:
            charge = int(charge)
        charges.append(charge)
    if not charges:
        return None
    first = charges[0]
    if isinstance(first, tuple):
        if not all(
            isinstance(charge, tuple) and len(charge) == len(first)
            for charge in charges
        ):
            return None
        return tuple(
            sum(charge[axis] for charge in charges)
            for axis in range(len(first))
        )
    if any(isinstance(charge, tuple) for charge in charges):
        return None
    return sum(charges)


def _coerce_fermion_sector(sector, symmetry):
    """Normalize a requested physical sector for the supported spinful modes."""
    symmetry = str(symmetry).upper()
    if symmetry == "Z2":
        if isinstance(sector, bool) or not isinstance(sector, Integral):
            raise ValueError("A spinful Z2 sector must be parity 0 or 1.")
        return int(sector) % 2
    if symmetry == "U1":
        if isinstance(sector, bool) or not isinstance(sector, Integral):
            raise ValueError(
                "A spinful U1 sector must be an integer total particle number."
            )
        return int(sector)
    if symmetry == "U1U1":
        try:
            sector = tuple(sector)
        except TypeError as exc:
            raise ValueError(
                "A spinful U1U1 sector must be (N_up, N_down)."
            ) from exc
        if len(sector) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in sector
        ):
            raise ValueError("A spinful U1U1 sector must be (N_up, N_down).")
        return tuple(int(value) for value in sector)
    if symmetry == "Z2Z2":
        try:
            sector = tuple(sector)
        except TypeError as exc:
            raise ValueError(
                "A spinful Z2Z2 sector must be (parity_up, parity_down)."
            ) from exc
        if len(sector) != 2 or any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in sector
        ):
            raise ValueError("A spinful Z2Z2 sector must be (parity_up, parity_down).")
        return tuple(int(value) % 2 for value in sector)
    raise NotImplementedError(
        f"Automatic Torch Fermion VMC does not support {symmetry!r}."
    )


def _validate_fermion_sector(sector, symmetry, n_sites):
    sector = _coerce_fermion_sector(sector, symmetry)
    if symmetry in {"Z2", "Z2Z2"}:
        return sector
    if symmetry == "U1":
        if not 0 <= sector <= 2 * n_sites:
            raise ValueError(
                f"U1 total particle sector must be between 0 and {2 * n_sites}."
            )
    elif any(value < 0 or value > n_sites for value in sector):
        raise ValueError(
            f"U1U1 sector entries must each be between 0 and {n_sites}."
        )
    return sector


@dataclass(frozen=True)
class TorchFermionVMCMetadata:
    """Validated PEPS/Fermion metadata used by :class:`TorchFermionVMC`."""

    site_order: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    graph_edges: tuple[tuple[int, int], ...]
    Lx: int
    Ly: int
    physical_dim: int
    symmetry: str
    spinful: bool
    encoding: FermionSiteEncoding
    sector: int | tuple[int, int] | None
    physical_charges: tuple[Any, ...] = ()

    @property
    def n_sites(self):
        return len(self.site_order)

    @property
    def graph(self):
        """Return the integer graph consumed by the Torch sampler."""
        return self.graph_edges


def _infer_torch_fermion_metadata(
    peps,
    fermion,
    *,
    sector=None,
    edges=None,
    pbc=False,
    site_order=None,
):
    """Infer and validate all static metadata for native spinful PEPS VMC."""
    tn = getattr(peps, "tn", peps)
    if not hasattr(tn, "sites"):
        raise TypeError("peps must be a quimb PEPS-like object with sites.")
    site_order = tuple(tn.sites if site_order is None else site_order)
    if not site_order:
        raise ValueError("The PEPS must contain at least one physical site.")
    if len(set(site_order)) != len(site_order):
        raise ValueError("PEPS site_order must contain unique site labels.")
    missing = [site for site in site_order if site not in tn.sites]
    if missing:
        raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")

    Lx = getattr(tn, "Lx", None)
    Ly = getattr(tn, "Ly", None)
    if Lx is None:
        Lx = getattr(tn, "_Lx", None)
    if Ly is None:
        Ly = getattr(tn, "_Ly", None)
    if Lx is None or Ly is None:
        if all(
            isinstance(site, tuple)
            and len(site) == 2
            and all(isinstance(value, Integral) for value in site)
            for site in site_order
        ):
            Lx = max(int(site[0]) for site in site_order) + 1
            Ly = max(int(site[1]) for site in site_order) + 1
        else:
            raise ValueError(
                "Could not infer PEPS Lx/Ly; use coordinate PEPS sites or "
                "pass explicit site_order and edges."
            )
    Lx = _check_positive_int("Lx", Lx)
    Ly = _check_positive_int("Ly", Ly)
    if len(site_order) != Lx * Ly:
        raise ValueError(
            f"PEPS has {len(site_order)} sites but inferred geometry is {Lx}x{Ly}."
        )

    if edges is None:
        edges = _peps_lattice_edges(site_order, Lx, Ly, pbc=pbc)
    else:
        edges = _coerce_labelled_edges(edges, site_order)
    positions = {site: i for i, site in enumerate(site_order)}
    graph_edges = tuple((positions[left], positions[right]) for left, right in edges)

    dimensions = []
    physical_charges = []
    for site in site_order:
        _, dimension = _peps_physical_axis(tn, site)
        dimensions.append(dimension)
        charges = _peps_physical_charges(tn, site)
        if charges:
            physical_charges.append(charges)
    if len(set(dimensions)) != 1:
        raise ValueError(f"PEPS physical dimensions are inconsistent: {dimensions!r}.")
    physical_dim = dimensions[0]

    peps_symmetry = _peps_symmetry(tn, site_order)
    spinful = True
    if fermion is None:
        symmetry = peps_symmetry
        if symmetry is None:
            raise ValueError(
                "Cannot infer Fermion symmetry from this PEPS. Pass fermion=... "
                "or use a Symmray PEPS with symmetry metadata."
            )
    else:
        spinful = bool(getattr(fermion, "spinful", False))
        if not spinful:
            raise NotImplementedError(
                "TorchFermionVMC currently supports spinful Fermion objects only."
            )
        symmetry = str(getattr(fermion, "symmetry", "")).upper()
        if peps_symmetry is not None and peps_symmetry != symmetry:
            raise ValueError(
                f"PEPS symmetry {peps_symmetry!r} does not match Fermion "
                f"symmetry {symmetry!r}."
            )
    if symmetry not in {"U1", "U1U1", "Z2", "Z2Z2"}:
        raise NotImplementedError(
            "TorchFermionVMC currently supports U1, U1U1, Z2, and Z2Z2, "
            f"not {symmetry!r}."
        )
    if physical_dim != 4:
        raise ValueError(
            f"Spinful Fermion VMC requires PEPS physical dimension 4, got {physical_dim}."
        )
    if fermion is None:
        sectors = {
            "U1": {0: 1, 1: 2, 2: 1},
            "U1U1": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
            "Z2": {0: 2, 1: 2},
            "Z2Z2": {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
        }[symmetry]
    else:
        sectors = getattr(fermion, "physical_sectors", None)
    if sectors is None or sum(int(size) for size in sectors.values()) != physical_dim:
        raise ValueError("Fermion and PEPS physical dimensions/sectors are incompatible.")

    if physical_charges:
        first_charges = physical_charges[0]
        if any(charges != first_charges for charges in physical_charges[1:]):
            raise ValueError("PEPS physical charge ordering differs between sites.")
        expected_charges = tuple(sectors)
        if symmetry == "U1U1" and first_charges != expected_charges:
            raise ValueError(
                "PEPS and Fermion U1U1 physical charge orders differ; refusing "
                "to apply an implicit local basis permutation."
            )
        if symmetry in {"Z2", "Z2Z2"} and first_charges != expected_charges:
            raise ValueError(
                "PEPS and Fermion parity physical charge sectors differ; refusing "
                "to apply an implicit local basis permutation."
            )
        physical_charges = first_charges
    else:
        physical_charges = ()

    if symmetry == "Z2":
        encoding = FermionSiteEncoding.symmray()
    elif symmetry == "Z2Z2" and physical_charges:
        encoding = FermionSiteEncoding.from_physical_charges(physical_charges)
    elif symmetry == "Z2Z2":
        encoding = FermionSiteEncoding.vmc_torch()
    elif fermion is None:
        if symmetry == "U1U1" and physical_charges:
            encoding = FermionSiteEncoding.from_physical_charges(physical_charges)
        else:
            encoding = FermionSiteEncoding.vmc_torch()
    else:
        encoding = FermionSiteEncoding.from_fermion(
            fermion,
            physical_charges=physical_charges,
        )
    if sector is None:
        sector = _sum_site_charges(tn, site_order)
    if sector is not None:
        sector = _validate_fermion_sector(sector, symmetry, len(site_order))
    return TorchFermionVMCMetadata(
        site_order=site_order,
        edges=tuple(edges),
        graph_edges=graph_edges,
        Lx=Lx,
        Ly=Ly,
        physical_dim=physical_dim,
        symmetry=symmetry,
        spinful=spinful,
        encoding=encoding,
        sector=sector,
        physical_charges=tuple(physical_charges),
    )


@dataclass(frozen=True)
class TorchConnections:
    """Batched Hamiltonian connections.

    ``configs[k]`` is connected to source sample ``batch_ids[k]`` with
    coefficient ``coeffs[k]``.
    """

    configs: Any
    coeffs: Any
    batch_ids: Any


@dataclass(frozen=True)
class TorchMetropolisResult:
    """Result of one Metropolis sweep."""

    configs: Any
    amplitudes: Any
    n_proposed: int
    n_accepted: int
    log_abs_amplitudes: Any = None
    nonzero_amplitudes: Any = None

    @property
    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


@dataclass(frozen=True)
class TorchMCMCSamples:
    """Chain-preserving samples and diagnostics from a torch sampler.

    ``configs`` and ``amplitudes`` have shape
    ``(n_samples_per_chain, n_chains, ...)``. ``n_samples`` is the actual
    number of returned samples, so it can be larger than the requested total
    when that total is not divisible by ``n_chains``.
    """

    configs: Any
    amplitudes: Any
    n_samples: int
    n_samples_per_chain: int
    n_chains: int
    n_discard_per_chain: int
    sweep_size: int
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    elapsed_seconds: float
    samples_per_second: float
    log_abs_amplitudes: Any = None

    def diagnostics(self, values=None, *, max_lag=None):
        """Compute chain diagnostics for a scalar observable.

        If ``values`` is omitted, the sampled ``|psi|**2`` values are used as
        a generic mixing diagnostic. For VMC convergence, pass local
        observable values with shape ``(n_samples_per_chain, n_chains)``.
        """
        if values is None:
            values = self.amplitudes.abs().square()
        return torch_chain_diagnostics(values, max_lag=max_lag)


@dataclass(frozen=True)
class TorchChainDiagnostics:
    """MCMC convergence diagnostics for chain-shaped scalar values."""

    r_hat: Any
    integrated_autocorrelation_time: Any
    effective_sample_size: Any
    n_samples_per_chain: int
    n_chains: int

    @property
    def rhat(self):
        """Alias for :attr:`r_hat`."""
        return self.r_hat

    @property
    def tau(self):
        """Alias for :attr:`integrated_autocorrelation_time`."""
        return self.integrated_autocorrelation_time


def _make_torch_generator(seed, *, device=None):
    """Construct a reproducible torch generator for a target device."""
    torch = _require_torch()
    if seed is None:
        return None
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _iter_edges(graph):
    if hasattr(graph, "edges"):
        edges = graph.edges
        if callable(edges):
            edges = edges()
    else:
        edges = graph
    return tuple((int(i), int(j)) for i, j in edges)


def _term_items(terms):
    """Return ``(where, operator)`` pairs from common Hamiltonian containers."""
    if hasattr(terms, "terms"):
        terms = terms.terms
    if hasattr(terms, "items"):
        return tuple(terms.items())
    try:
        return tuple(terms)
    except TypeError as exc:
        raise TypeError(
            "terms must be a mapping, a SymHamiltonian-like object with "
            "`.terms`, or an iterable of (where, operator) pairs."
        ) from exc


def _term_dense_array(operator):
    """Convert a local operator to a dense array without changing its backend."""
    if hasattr(operator, "to_dense"):
        operator = operator.to_dense()
    elif hasattr(operator, "data") and not hasattr(operator, "shape"):
        operator = operator.data
    return operator


def _term_site_indices(where, rank, *, site_order, n_sites):
    """Resolve a one- or two-site term location to config-column indices."""
    n_local_sites = rank // 2
    if n_local_sites not in (1, 2):
        raise ValueError(
            "Hamiltonian operators must have rank 2 or 4, with output axes "
            "followed by input axes."
        )
    if n_local_sites == 1:
        where = (where,)
    elif isinstance(where, (str, bytes)):
        raise ValueError("A two-site operator location must contain two sites.")
    else:
        try:
            where = tuple(where)
        except TypeError as exc:
            raise ValueError(
                "A two-site operator location must contain two sites."
            ) from exc
        if len(where) != 2:
            raise ValueError("A two-site operator location must contain two sites.")

    if site_order is None:
        site_order = tuple(range(n_sites))
    position = {site: i for i, site in enumerate(site_order)}
    missing = [site for site in where if site not in position]
    if missing:
        raise ValueError(
            f"Hamiltonian term site(s) {missing!r} are not in site_order. "
            "Pass site_order matching the PEPS physical-site order."
        )
    return tuple(position[site] for site in where)


def torch_hamiltonian_connections(
    configs,
    terms,
    *,
    site_order=None,
    coefficient_cutoff=0.0,
):
    """Build connected configurations from explicit local Hamiltonian terms.

    ``terms`` can be a :class:`SymHamiltonian`, its ``.terms`` mapping, or an
    iterable of ``(where, operator)`` pairs. Operators must be dense or expose
    ``to_dense()`` and use output axes followed by input axes, as do Pepsy's
    native one- and two-site fermionic operators. This lets torch VMC measure
    the exact terms supplied by the caller without guessing ``t``, ``U``, or
    a model-specific connection function.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    all_etas = []
    all_coeffs = []
    all_bids = []

    for where, operator in _term_items(terms):
        dense = torch.as_tensor(_term_dense_array(operator), device=device)
        if dense.ndim not in (2, 4):
            raise ValueError(
                f"Hamiltonian term at {where!r} has rank {dense.ndim}; "
                "only one- and two-site terms are supported."
            )
        if dense.shape[: dense.ndim // 2] != dense.shape[dense.ndim // 2 :]:
            raise ValueError(
                f"Hamiltonian term at {where!r} must have square input/output "
                f"dimensions, got shape {tuple(dense.shape)}."
            )
        local_sites = _term_site_indices(
            where,
            dense.ndim,
            site_order=site_order,
            n_sites=n_sites,
        )
        local_dim = int(dense.shape[0])
        if any(int(configs[:, site].max()) >= local_dim for site in local_sites):
            raise ValueError(
                f"Hamiltonian term at {where!r} has local dimension {local_dim}, "
                "which is too small for the supplied configurations."
            )

        nonzero = torch.nonzero(
            dense.abs() > float(coefficient_cutoff),
            as_tuple=False,
        )
        n_local_sites = len(local_sites)
        for entry in nonzero:
            entry = tuple(int(x) for x in entry.tolist())
            outputs = entry[:n_local_sites]
            inputs = entry[n_local_sites:]
            mask = torch.ones(batch, dtype=torch.bool, device=device)
            for site, value in zip(local_sites, inputs):
                mask &= configs[:, site] == value
            batch_ids = mask.nonzero(as_tuple=True)[0]
            if batch_ids.numel() == 0:
                continue
            eta = configs[batch_ids].clone()
            for site, value in zip(local_sites, outputs):
                eta[:, site] = value
            all_etas.append(eta)
            all_coeffs.append(
                dense[entry].expand(batch_ids.numel()).to(device=device)
            )
            all_bids.append(batch_ids)

    if not all_etas:
        return _empty_connections(configs)
    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def _driver_terms_connections(configs, graph, *, terms, site_order=None, **kwargs):
    """Adapt explicit-term connections to the driver's connection signature."""
    del graph, kwargs
    return torch_hamiltonian_connections(
        configs,
        terms,
        site_order=site_order,
    )


def _normalize_terms_site_labels(terms, site_order):
    """Map positional integer term labels onto PEPS site labels when needed."""
    site_order = tuple(site_order)
    positions = {site: i for i, site in enumerate(site_order)}

    def map_site(site):
        if site in positions:
            return site
        if (
            isinstance(site, Integral)
            and not isinstance(site, bool)
            and 0 <= int(site) < len(site_order)
        ):
            return site_order[int(site)]
        return site

    normalized = {}
    for where, operator in _term_items(terms):
        dense = _term_dense_array(operator)
        rank = getattr(dense, "ndim", None)
        if rank is None:
            rank = len(getattr(dense, "shape", ()))
        n_local_sites = int(rank) // 2
        if n_local_sites == 1:
            normalized[map_site(where)] = operator
        elif n_local_sites == 2:
            try:
                left, right = tuple(where)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "A two-site observable location must contain two sites."
                ) from exc
            normalized[(map_site(left), map_site(right))] = operator
        else:
            normalized[where] = operator
    return normalized


def _empty_connections(configs):
    torch = _require_torch()
    return TorchConnections(
        configs=configs.new_empty((0, configs.shape[1])),
        coeffs=torch.empty(0, dtype=torch.float64, device=configs.device),
        batch_ids=torch.empty(0, dtype=torch.long, device=configs.device),
    )


def count_spinful_particles(configs, *, encoding=None):
    """Return per-sample ``(n_up, n_down)`` counts."""
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    n_up, n_down = encoding.decode(configs)
    return n_up.sum(dim=-1), n_down.sum(dim=-1)


def propose_spin_exchange(i, j, configs):
    """Propose spin exchange on one edge for binary spin configs."""
    configs = _as_long_matrix(configs)
    proposed = configs.clone()
    si = configs[:, i]
    sj = configs[:, j]
    changed = si != sj
    proposed[changed, i] = sj[changed]
    proposed[changed, j] = si[changed]
    return proposed, changed


def propose_spinful_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    encoding=None,
    generator=None,
):
    """Propose spinful Hubbard exchange/hopping moves on one edge.

    The proposal preserves ``N_up`` and ``N_down``. With probability
    ``1 - hopping_rate`` it swaps the two local site states. Otherwise it uses
    local hopping-style moves over ``empty/up/down/double`` states, following
    the sampling options in ``sjdu10/vmc_torch``.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposed = configs.clone()
    device = configs.device
    batch = configs.shape[0]

    ci = configs[:, i]
    cj = configs[:, j]
    changed = ci != cj
    if not torch.any(changed):
        return proposed, changed

    n_up, n_down = encoding.decode(configs)
    ni = n_up[:, i] + n_down[:, i]
    nj = n_up[:, j] + n_down[:, j]
    delta_n = (ni - nj).abs()

    rand = torch.rand(batch, device=device, generator=generator)
    is_exchange = (rand < (1.0 - hopping_rate)) & changed
    is_hopping = (~is_exchange) & changed

    swap_mask = is_exchange | (is_hopping & (delta_n == 1))
    proposed[swap_mask, i] = cj[swap_mask]
    proposed[swap_mask, j] = ci[swap_mask]

    mask_d0 = is_hopping & (delta_n == 0)
    if torch.any(mask_d0):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d0, i] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.double, device=device),
            torch.as_tensor(encoding.empty, device=device),
        )
        proposed[mask_d0, j] = torch.where(
            bits[mask_d0],
            torch.as_tensor(encoding.empty, device=device),
            torch.as_tensor(encoding.double, device=device),
        )

    mask_d2 = is_hopping & (delta_n == 2)
    if torch.any(mask_d2):
        bits = torch.randint(
            0,
            2,
            (batch,),
            device=device,
            dtype=torch.long,
            generator=generator,
        ).bool()
        proposed[mask_d2, i] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.down, device=device),
            torch.as_tensor(encoding.up, device=device),
        )
        proposed[mask_d2, j] = torch.where(
            bits[mask_d2],
            torch.as_tensor(encoding.up, device=device),
            torch.as_tensor(encoding.down, device=device),
        )

    return proposed, changed


def propose_spinful_u1_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    encoding=None,
    generator=None,
):
    """Propose moves that preserve total spinful particle number only.

    In addition to the ``U1U1``-safe exchange and hopping moves, this rule
    includes single-site ``up <-> down`` flips. Those flips allow a ``U1``
    walker to move between different spin-resolved sectors while preserving
    ``N_up + N_down``. The selected edge and endpoint are fixed before the
    local state is inspected, so no proposal-probability correction is needed
    for the spin-flip branch.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposed, changed = propose_spinful_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        encoding=encoding,
        generator=generator,
    )

    spin_flip_rate = float(spin_flip_rate)
    if not 0.0 <= spin_flip_rate <= 1.0:
        raise ValueError("spin_flip_rate must be between 0 and 1.")
    if spin_flip_rate == 0.0:
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    flip_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < spin_flip_rate
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.up) | (target == encoding.down)
    flip = flip_branch & valid
    proposed[flip_branch] = configs[flip_branch]
    if not torch.any(flip):
        return proposed, changed & ~flip_branch

    flipped = torch.where(
        target == encoding.up,
        torch.as_tensor(encoding.down, device=device),
        torch.as_tensor(encoding.up, device=device),
    )
    proposed[flip & choose_i, i] = flipped[flip & choose_i]
    proposed[flip & ~choose_i, j] = flipped[flip & ~choose_i]
    changed = torch.where(flip_branch, flip, changed)
    return proposed, changed


def propose_spinful_z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
):
    """Propose moves that preserve spinful total fermion parity.

    The U1-preserving exchange, hopping, and spin-flip moves are augmented by
    an ``empty <-> double`` toggle on a randomly selected endpoint. The latter
    changes particle number by two, allowing the chain to explore the full
    fixed-parity sector rather than remaining in one fixed-number sector.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposed, changed = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        encoding=encoding,
        generator=generator,
    )

    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        return proposed, changed

    device = configs.device
    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < pair_toggle_rate
    choose_i = torch.rand(
        configs.shape[0],
        device=device,
        generator=generator,
    ) < 0.5
    target = torch.where(choose_i, ci, cj)
    valid = (target == encoding.empty) | (target == encoding.double)
    pair_toggle = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    if not torch.any(pair_toggle):
        return proposed, changed & ~pair_branch

    toggled = torch.where(
        target == encoding.empty,
        torch.as_tensor(encoding.double, device=device),
        torch.as_tensor(encoding.empty, device=device),
    )
    proposed[pair_toggle & choose_i, i] = toggled[pair_toggle & choose_i]
    proposed[pair_toggle & ~choose_i, j] = toggled[pair_toggle & ~choose_i]
    changed = torch.where(pair_branch, pair_toggle, changed)
    return proposed, changed


def propose_spinful_z2z2_exchange_or_hopping(
    i,
    j,
    configs,
    *,
    hopping_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
):
    """Propose moves preserving spin-resolved parity ``Z2 x Z2``.

    Spin flips are deliberately disabled because they change both resolved
    parities. Exchange, species-preserving hopping, and empty/double toggles
    preserve each parity independently.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    proposed, changed = propose_spinful_u1_exchange_or_hopping(
        i,
        j,
        configs,
        hopping_rate=hopping_rate,
        spin_flip_rate=0.0,
        encoding=encoding,
        generator=generator,
    )
    pair_toggle_rate = float(pair_toggle_rate)
    if not 0.0 <= pair_toggle_rate <= 1.0:
        raise ValueError("pair_toggle_rate must be between 0 and 1.")
    if pair_toggle_rate == 0.0:
        return proposed, changed

    ci = configs[:, i]
    cj = configs[:, j]
    pair_branch = torch.rand(
        configs.shape[0],
        device=configs.device,
        generator=generator,
    ) < pair_toggle_rate
    valid_empty_double = (ci == encoding.empty) & (cj == encoding.empty)
    valid_double_empty = (ci == encoding.double) & (cj == encoding.double)
    valid_up_up = (ci == encoding.up) & (cj == encoding.up)
    valid_down_down = (ci == encoding.down) & (cj == encoding.down)
    valid = (
        valid_empty_double
        | valid_double_empty
        | valid_up_up
        | valid_down_down
    )
    pair_move = pair_branch & valid
    proposed[pair_branch] = configs[pair_branch]
    proposed[pair_move & valid_empty_double, i] = encoding.double
    proposed[pair_move & valid_empty_double, j] = encoding.double
    proposed[pair_move & valid_double_empty, i] = encoding.empty
    proposed[pair_move & valid_double_empty, j] = encoding.empty
    proposed[pair_move & valid_up_up, i] = encoding.down
    proposed[pair_move & valid_up_up, j] = encoding.down
    proposed[pair_move & valid_down_down, i] = encoding.up
    proposed[pair_move & valid_down_down, j] = encoding.up
    changed = torch.where(pair_branch, pair_move, changed)
    return proposed, changed


def _safe_metropolis_ratio(proposed_amps, current_amps):
    torch = _require_torch()
    numerator = proposed_amps.abs().square()
    denominator = current_amps.abs().square()
    zero = torch.zeros_like(numerator)
    inf = torch.full_like(numerator, float("inf"))
    return torch.where(
        denominator > 0,
        numerator / denominator,
        torch.where(numerator > 0, inf, zero),
    )


def _safe_metropolis_log_ratio(
    proposed_log_abs,
    current_log_abs,
    *,
    proposed_nonzero=None,
    current_nonzero=None,
):
    """Return clipped Metropolis ratios from log magnitudes."""
    torch = _require_torch()
    log_ratio = 2.0 * (proposed_log_abs - current_log_abs)
    # ``inf - inf`` can occur for user-provided log amplitudes. The explicit
    # support masks below decide those zero-amplitude cases, so make the
    # finite-ratio branch harmless instead of propagating NaNs into RNG tests.
    log_ratio = torch.where(
        torch.isnan(log_ratio),
        torch.zeros_like(log_ratio),
        log_ratio,
    )
    ratio = torch.exp(torch.minimum(log_ratio, torch.zeros_like(log_ratio)))
    if proposed_nonzero is None or current_nonzero is None:
        return ratio
    zero = torch.zeros_like(ratio)
    one = torch.ones_like(ratio)
    return torch.where(
        current_nonzero,
        torch.where(proposed_nonzero, ratio, zero),
        torch.where(proposed_nonzero, one, zero),
    )


def metropolis_exchange_sweep(
    configs,
    amplitude_fn,
    graph,
    *,
    current_amplitudes=None,
    current_log_abs=None,
    current_nonzero=None,
    log_amplitude_fn=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    generator=None,
    chunk_size=None,
):
    """Run one nearest-neighbor Metropolis sweep.

    ``amplitude_fn`` should accept a ``(batch, n_sites)`` torch integer tensor
    and return a batch of amplitudes. The sampler evaluates only changed
    proposals when possible. ``chunk_size`` caps proposal-amplitude batch size
    without changing the Markov chain.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs).clone()
    current = (
        _call_amplitude_fn(amplitude_fn, configs, chunk_size=chunk_size)
        if current_amplitudes is None
        else current_amplitudes
    )
    current = torch.as_tensor(current, device=configs.device)
    log_amplitude_fn = _resolve_log_amplitude_fn(
        amplitude_fn,
        log_amplitude_fn,
    )
    if log_amplitude_fn is not None:
        try:
            if current_log_abs is None or current_nonzero is None:
                current_phase, computed_log_abs = _call_log_amplitude_fn(
                    log_amplitude_fn,
                    configs,
                    chunk_size=chunk_size,
                )
                if current_log_abs is None:
                    current_log_abs = computed_log_abs
                if current_nonzero is None:
                    current_nonzero = current_phase.abs() > 0
            current_log_abs = torch.as_tensor(
                current_log_abs,
                dtype=torch.float64,
                device=configs.device,
            ).clone()
            current_nonzero = torch.as_tensor(
                current_nonzero,
                dtype=torch.bool,
                device=configs.device,
            ).clone()
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            # Some approximate sparse contractions expose ``forward_log`` but
            # cannot represent every intermediate charge sector. Keep the
            # raw-amplitude sampler path usable in that case.
            log_amplitude_fn = None
            current_log_abs = None
            current_nonzero = None
    n_proposed = 0
    n_accepted = 0

    for i, j in _iter_edges(graph):
        if proposal in {"spin", "spin_exchange", "heisenberg"}:
            proposed, flags = propose_spin_exchange(i, j, configs)
        elif proposal in {"spinful", "hubbard", "spinful_exchange_hopping"}:
            proposed, flags = propose_spinful_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                encoding=encoding,
                generator=generator,
            )
        elif proposal in {
            "spinful_u1",
            "u1_spinful",
            "spinful_total",
            "spinful_total_exchange_hopping",
        }:
            proposed, flags = propose_spinful_u1_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                encoding=encoding,
                generator=generator,
            )
        elif proposal in {
            "spinful_z2",
            "z2_spinful",
            "spinful_parity",
            "spinful_parity_exchange_hopping",
        }:
            proposed, flags = propose_spinful_z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                spin_flip_rate=spin_flip_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
            )
        elif proposal in {
            "spinful_z2z2",
            "z2z2_spinful",
            "spinful_resolved_parity",
        }:
            proposed, flags = propose_spinful_z2z2_exchange_or_hopping(
                i,
                j,
                configs,
                hopping_rate=hopping_rate,
                pair_toggle_rate=pair_toggle_rate,
                encoding=encoding,
                generator=generator,
            )
        else:
            raise ValueError(
                "proposal must be 'spin', 'spinful_exchange_hopping', or "
                "'spinful_u1', 'spinful_z2', or 'spinful_z2z2'."
            )

        if not torch.any(flags):
            continue

        n_changed = int(flags.sum().item())
        n_proposed += n_changed
        proposed_amps = current.clone()
        proposed_amps[flags] = _call_amplitude_fn(
            amplitude_fn,
            proposed[flags],
            chunk_size=chunk_size,
        )
        if log_amplitude_fn is None:
            ratio = _safe_metropolis_ratio(proposed_amps, current)
        else:
            try:
                proposed_phase, proposed_log_abs_values = (
                    _call_log_amplitude_fn(
                        log_amplitude_fn,
                        proposed[flags],
                        chunk_size=chunk_size,
                    )
                )
                proposed_log_abs = current_log_abs.clone()
                proposed_log_abs[flags] = proposed_log_abs_values
                proposed_nonzero = current_nonzero.clone()
                proposed_nonzero[flags] = proposed_phase.abs() > 0
                ratio = _safe_metropolis_log_ratio(
                    proposed_log_abs,
                    current_log_abs,
                    proposed_nonzero=proposed_nonzero,
                    current_nonzero=current_nonzero,
                )
            except (AttributeError, IndexError, KeyError, NotImplementedError,
                    RuntimeError, TypeError, ValueError):
                log_amplitude_fn = None
                current_log_abs = None
                current_nonzero = None
                ratio = _safe_metropolis_ratio(proposed_amps, current)
        accept = flags & (
            torch.rand(configs.shape[0], device=configs.device, generator=generator)
            < ratio
        )

        if torch.any(accept):
            n_accept = int(accept.sum().item())
            n_accepted += n_accept
            configs[accept] = proposed[accept]
            current[accept] = proposed_amps[accept]
            if log_amplitude_fn is not None:
                current_log_abs[accept] = proposed_log_abs[accept]
                current_nonzero[accept] = proposed_nonzero[accept]

    return TorchMetropolisResult(
        configs=configs,
        amplitudes=current,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
        log_abs_amplitudes=(
            current_log_abs if log_amplitude_fn is not None else None
        ),
        nonzero_amplitudes=(
            current_nonzero if log_amplitude_fn is not None else None
        ),
    )


class TorchMetropolisSampler:
    """Stateful batched Metropolis sampler for torch amplitude models.

    The first configuration axis represents independent chains. Sampling
    retains that axis and returns arrays shaped as
    ``(n_samples_per_chain, n_chains, n_sites)``. ``sweep_size`` is measured
    in the graph sweeps performed by :func:`metropolis_exchange_sweep`; the
    ``n_thin`` spelling is accepted as a convenience alias.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        configs,
        *,
        amplitudes=None,
        n_chains=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
        seed=None,
        n_sites=None,
        log_amplitude_fn=None,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        configs = _as_long_matrix(configs).clone()
        if n_sites is not None:
            n_sites = _check_positive_int("n_sites", n_sites)
            if configs.shape[1] != n_sites:
                raise ValueError(
                    f"n_sites={n_sites} does not match configs with "
                    f"{configs.shape[1]} sites."
                )
        if n_chains is None:
            n_chains = int(configs.shape[0])
        n_chains = _check_positive_int("n_chains", n_chains)
        if configs.shape[0] == 1 and n_chains > 1:
            configs = configs.expand(n_chains, -1).clone()
        elif configs.shape[0] != n_chains:
            raise ValueError(
                "configs must contain exactly one initial configuration per "
                f"chain: expected {n_chains}, got {configs.shape[0]}."
            )

        self.amplitude_fn = amplitude_fn
        self.graph = graph
        self.configs = configs
        self.n_chains = n_chains
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.log_amplitude_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        self.generator = (
            _make_torch_generator(seed, device=configs.device)
            if seed is not None
            else generator
        )
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.numel() == 1 and n_chains > 1:
                amplitudes = amplitudes.reshape(1).expand(n_chains).clone()
            if amplitudes.shape != (n_chains,):
                raise ValueError(
                    "amplitudes must have one value per chain, got "
                    f"shape {tuple(amplitudes.shape)}."
                )
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes(
            log_abs_amplitudes=log_abs_amplitudes,
            nonzero_amplitudes=nonzero_amplitudes,
        )

    @property
    def n_sites(self):
        """Number of physical sites in each chain configuration."""
        return int(self.configs.shape[1])

    def refresh_amplitudes(self):
        """Recompute the amplitudes at the current chain positions."""
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(
        self,
        *,
        log_abs_amplitudes=None,
        nonzero_amplitudes=None,
    ):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        if log_abs_amplitudes is not None and nonzero_amplitudes is not None:
            torch = _require_torch()
            self.log_abs_amplitudes = torch.as_tensor(
                log_abs_amplitudes,
                dtype=torch.float64,
                device=self.configs.device,
            )
            self.nonzero_amplitudes = torch.as_tensor(
                nonzero_amplitudes,
                dtype=torch.bool,
                device=self.configs.device,
            )
            if self.log_abs_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "log_abs_amplitudes must have one value per chain."
                )
            if self.nonzero_amplitudes.shape != (self.n_chains,):
                raise ValueError(
                    "nonzero_amplitudes must have one value per chain."
                )
            return
        try:
            phase, self.log_abs_amplitudes = _call_log_amplitude_fn(
                self.log_amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
            self.nonzero_amplitudes = phase.abs() > 0
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self.log_amplitude_fn = None
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None

    def reset(self, configs=None, *, amplitudes=None):
        """Reset chain positions, optionally supplying their amplitudes."""
        if configs is None:
            raise ValueError("reset requires explicit configs.")
        configs = _as_long_matrix(configs).clone()
        if tuple(configs.shape) != tuple(self.configs.shape):
            raise ValueError(
                "reset configs must have shape "
                f"{tuple(self.configs.shape)}, got {tuple(configs.shape)}."
            )
        self.configs = configs
        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=configs.device,
            )
            if amplitudes.shape != (self.n_chains,):
                raise ValueError("reset amplitudes must have one value per chain.")
            self.amplitudes = amplitudes
        self._refresh_log_amplitudes()
        return self

    def sample_sweep(self, *, n_sweeps=1):
        """Advance all chains by one or more graph sweeps."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.amplitude_fn,
                    self.graph,
                    current_amplitudes=self.amplitudes,
                    current_log_abs=self.log_abs_amplitudes,
                    current_nonzero=self.nonzero_amplitudes,
                    log_amplitude_fn=(
                        self.log_amplitude_fn
                        if self.log_amplitude_fn is not None
                        else False
                    ),
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    spin_flip_rate=self.spin_flip_rate,
                    pair_toggle_rate=self.pair_toggle_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
                self.log_abs_amplitudes = result.log_abs_amplitudes
                self.nonzero_amplitudes = result.nonzero_amplitudes
                if result.log_abs_amplitudes is None:
                    self.log_amplitude_fn = None
        return result

    def sample(
        self,
        *,
        n_samples=1024,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
    ):
        """Discard and collect chain-preserving Metropolis samples.

        ``n_samples`` is the requested total across all chains. As in
        NetKet, the chain length is rounded up so every chain contributes the
        same number of samples. ``n_discard`` and ``n_thin`` are aliases for
        ``n_discard_per_chain`` and ``sweep_size`` respectively.
        """
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if n_discard_per_chain is not None and n_discard is not None:
            raise ValueError(
                "Pass either n_discard_per_chain=... or n_discard=..., not both."
            )
        if sweep_size is not None and n_thin is not None:
            raise ValueError("Pass either sweep_size=... or n_thin=..., not both.")
        if n_discard_per_chain is None:
            n_discard_per_chain = 32 if n_discard is None else n_discard
        if sweep_size is None:
            sweep_size = self.n_sites if n_thin is None else n_thin
        n_discard_per_chain = _check_nonnegative_int(
            "n_discard_per_chain",
            n_discard_per_chain,
        )
        sweep_size = _check_positive_int("sweep_size", sweep_size)
        n_samples_per_chain = (
            n_samples + self.n_chains - 1
        ) // self.n_chains
        total_sweeps = (
            n_discard_per_chain + n_samples_per_chain
        ) * sweep_size
        bar = _make_progress(
            progress,
            total=total_sweeps,
            desc="Torch Metropolis",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0

        def advance_one_sweep():
            nonlocal n_proposed, n_accepted
            result = self.sample_sweep(n_sweeps=1)
            n_proposed += result.n_proposed
            n_accepted += result.n_accepted
            if bar is not None:
                bar.update(1)

        for _ in range(n_discard_per_chain * sweep_size):
            advance_one_sweep()

        configs = []
        amplitudes = []
        log_abs_amplitudes = [] if self.log_abs_amplitudes is not None else None
        for _ in range(n_samples_per_chain):
            for _ in range(sweep_size):
                advance_one_sweep()
            configs.append(self.configs.clone())
            amplitudes.append(self.amplitudes.clone())
            if log_abs_amplitudes is not None:
                log_abs_amplitudes.append(self.log_abs_amplitudes.clone())
        if bar is not None:
            bar.close()

        configs = torch.stack(configs, dim=0)
        amplitudes = torch.stack(amplitudes, dim=0)
        if log_abs_amplitudes is not None:
            log_abs_amplitudes = torch.stack(log_abs_amplitudes, dim=0)
        actual_samples = int(configs.shape[0] * configs.shape[1])
        elapsed = time.perf_counter() - start
        return TorchMCMCSamples(
            configs=configs,
            amplitudes=amplitudes,
            n_samples=actual_samples,
            n_samples_per_chain=int(configs.shape[0]),
            n_chains=self.n_chains,
            n_discard_per_chain=n_discard_per_chain,
            sweep_size=sweep_size,
            acceptance_rate=(
                n_accepted / n_proposed if n_proposed else 0.0
            ),
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            elapsed_seconds=elapsed,
            samples_per_second=(
                actual_samples / elapsed if elapsed > 0 else float("inf")
            ),
            log_abs_amplitudes=log_abs_amplitudes,
        )


class TorchBPMetropolisSampler(TorchMetropolisSampler):
    """Independence Metropolis sampler driven by a BP proposal.

    ``proposal_sampler`` should return ``configs`` and BP proposal
    probabilities in ``omegas``, as :class:`pepsy.sampling.PepsBpSampler`
    does. Initial chains are drawn from that proposal, so the proposal
    probability of every current chain is known. Later proposals are accepted
    with the exact independence Metropolis-Hastings ratio.

    ``symmetry`` and ``sector`` optionally filter spinful fermion proposals.
    This is important for approximate BP distributions, which can assign
    probability to configurations outside a globally fixed charge sector.
    """

    def __init__(
        self,
        amplitude_fn,
        graph,
        proposal_sampler,
        configs=None,
        *,
        amplitudes=None,
        initial_log_q=None,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        valid_config_fn=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        chunk_size=None,
        generator=None,
        seed=None,
        device=None,
        log_amplitude_fn=None,
    ):
        torch = _require_torch()
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        max_init_attempts = _check_positive_int(
            "max_init_attempts",
            max_init_attempts,
        )
        if configs is None:
            if n_chains is None:
                raise ValueError(
                    "n_chains is required when BP initializes the chains."
                )
            n_chains = _check_positive_int("n_chains", n_chains)
        else:
            configs = _as_long_matrix(configs)
            if n_chains is None:
                n_chains = int(configs.shape[0])
            n_chains = _check_positive_int("n_chains", n_chains)

        self.amplitude_fn = amplitude_fn
        self.proposal_sampler = proposal_sampler
        self.proposal_sample_kwargs = dict(sample_kwargs or {})
        self.symmetry = None if symmetry is None else str(symmetry).upper()
        self.sector = sector
        self.fermion_encoding = encoding
        self.valid_config_fn = valid_config_fn
        self.amplitude_floor = float(amplitude_floor)
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self._initial_log_fn = _resolve_log_amplitude_fn(
            amplitude_fn,
            log_amplitude_fn,
        )
        if device is None:
            try:
                device = next(amplitude_fn.parameters()).device
            except (AttributeError, StopIteration, TypeError):
                device = None
        self._proposal_device = (
            torch.device(device) if device is not None else None
        )

        if configs is None:
            configs, amplitudes, initial_log_q = self._draw_initial_chains(
                n_chains,
                max_attempts=max_init_attempts,
            )
        else:
            configs = configs.clone()
            if self._proposal_device is not None:
                configs = configs.to(device=self._proposal_device)
            if configs.shape[0] == 1 and n_chains > 1:
                configs = configs.expand(n_chains, -1).clone()
            elif configs.shape[0] != n_chains:
                raise ValueError(
                    "configs must contain exactly one initial configuration "
                    f"per chain: expected {n_chains}, got {configs.shape[0]}."
                )
            if initial_log_q is None:
                raise ValueError(
                    "initial_log_q is required for explicit initial configs; "
                    "omit configs to initialize chains from BP."
                )
            initial_log_q = torch.as_tensor(
                initial_log_q,
                dtype=torch.float64,
                device=configs.device,
            )
            if initial_log_q.shape != (n_chains,):
                raise ValueError(
                    "initial_log_q must have one value per initial chain."
                )
            if amplitudes is None:
                with torch.no_grad():
                    amplitudes = _call_amplitude_fn(
                        amplitude_fn,
                        configs,
                        chunk_size=self.chunk_size,
                    )

        super().__init__(
            amplitude_fn,
            graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            # The parent stores amplitude/log-amplitude state. Its local
            # proposal is not used because this class overrides the sweep.
            proposal="spin",
            encoding=encoding,
            chunk_size=self.chunk_size,
            generator=generator,
            seed=seed,
            log_amplitude_fn=log_amplitude_fn,
        )
        self.log_proposal_probabilities = torch.as_tensor(
            initial_log_q,
            dtype=torch.float64,
            device=self.configs.device,
        ).clone()
        if self.log_proposal_probabilities.shape != (self.n_chains,):
            raise ValueError(
                "initial_log_q must have one value per initial chain."
            )
        if not bool(torch.isfinite(self.log_proposal_probabilities).all()):
            raise ValueError("Initial BP proposal probabilities must be positive and finite.")
        self._validate_current_support()

    def _proposal_sample(self, n_samples):
        """Draw configurations and decode their BP log probabilities."""
        kwargs = dict(self.proposal_sample_kwargs)
        kwargs["samples"] = int(n_samples)
        kwargs.setdefault("progbar", False)
        try:
            proposed = self.proposal_sampler.sample(**kwargs)
        except TypeError:
            kwargs.pop("progbar", None)
            proposed = self.proposal_sampler.sample(**kwargs)
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        if self._proposal_device is not None:
            configs = configs.to(device=self._proposal_device)
        log_q = _proposal_log_probabilities(
            proposed.omegas,
            device=configs.device,
            allow_zero=True,
        )
        n_samples = int(n_samples)
        if configs.shape[0] != n_samples or log_q.shape != (n_samples,):
            raise ValueError(
                "The BP proposal must return exactly one config and omega per "
                f"requested sample ({n_samples})."
            )
        return configs, log_q

    def _sector_mask(self, configs):
        """Return the requested symmetry-sector mask for configurations."""
        torch = _require_torch()
        if self.valid_config_fn is not None:
            mask = torch.as_tensor(
                self.valid_config_fn(configs),
                dtype=torch.bool,
                device=configs.device,
            )
            if mask.shape != (configs.shape[0],):
                raise ValueError(
                    "valid_config_fn must return one boolean per configuration."
                )
            return mask
        if self.symmetry is None or self.sector is None:
            return torch.ones(
                configs.shape[0],
                dtype=torch.bool,
                device=configs.device,
            )
        n_up, n_down = count_spinful_particles(
            configs,
            encoding=self.fermion_encoding,
        )
        if self.symmetry == "U1":
            return n_up + n_down == int(self.sector)
        if self.symmetry == "Z2":
            return (n_up + n_down) % 2 == int(self.sector) % 2
        try:
            sector = tuple(int(value) for value in self.sector)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{self.symmetry} sectors must contain two integer charges."
            ) from exc
        if len(sector) != 2:
            raise ValueError(f"{self.symmetry} sectors must contain two charges.")
        if self.symmetry == "U1U1":
            return (n_up == sector[0]) & (n_down == sector[1])
        if self.symmetry == "Z2Z2":
            return ((n_up % 2) == sector[0] % 2) & (
                (n_down % 2) == sector[1] % 2
            )
        raise ValueError(
            "symmetry must be one of U1, U1U1, Z2, or Z2Z2 when sector "
            "filtering is enabled."
        )

    def _draw_initial_chains(self, n_chains, *, max_attempts):
        """Draw nonzero, sector-valid initial chains from BP."""
        torch = _require_torch()
        configs_out = []
        amplitudes_out = []
        log_q_out = []
        n_kept = 0
        for _ in range(max_attempts):
            configs, log_q = self._proposal_sample(n_chains)
            keep = self._sector_mask(configs) & torch.isfinite(log_q)
            if not bool(torch.any(keep)):
                continue
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(
                    self.amplitude_fn,
                    configs[keep],
                    chunk_size=self.chunk_size,
                )
            if self._initial_log_fn is not None:
                try:
                    phase, log_abs = _call_log_amplitude_fn(
                        self._initial_log_fn,
                        configs[keep],
                        chunk_size=self.chunk_size,
                    )
                    support = (
                        torch.isfinite(log_abs)
                        & (phase.abs() > 0)
                        & (
                            log_abs
                            > (
                                -torch.inf
                                if self.amplitude_floor == 0.0
                                else float(torch.log(torch.tensor(self.amplitude_floor)))
                            )
                        )
                    )
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    NotImplementedError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ):
                    self._initial_log_fn = None
                    support = (
                        torch.isfinite(amplitudes.abs())
                        & (amplitudes.abs() > self.amplitude_floor)
                    )
            else:
                support = (
                    torch.isfinite(amplitudes.abs())
                    & (amplitudes.abs() > self.amplitude_floor)
                )
            if not bool(torch.any(support)):
                continue
            configs_out.append(configs[keep][support])
            amplitudes_out.append(amplitudes[support])
            log_q_out.append(log_q[keep][support])
            n_kept += int(support.sum().item())
            if n_kept >= n_chains:
                break
        if n_kept < n_chains:
            raise RuntimeError(
                "Could not initialize enough nonzero BP configurations in the "
                "requested Fermion sector. Check the BP encoding/sector or "
                "increase max_init_attempts."
            )
        return (
            torch.cat(configs_out, dim=0)[:n_chains],
            torch.cat(amplitudes_out, dim=0)[:n_chains],
            torch.cat(log_q_out, dim=0)[:n_chains],
        )

    def _validate_current_support(self):
        """Reject undefined initial states rather than creating 0/0 ratios."""
        torch = _require_torch()
        valid = self._sector_mask(self.configs)
        if self.log_amplitude_fn is not None:
            valid &= self.nonzero_amplitudes
            valid &= torch.isfinite(self.log_abs_amplitudes)
            if self.amplitude_floor:
                valid &= self.log_abs_amplitudes > float(
                    torch.log(torch.tensor(self.amplitude_floor))
                )
        else:
            valid &= torch.isfinite(self.amplitudes.abs())
            valid &= self.amplitudes.abs() > self.amplitude_floor
        if not bool(torch.all(valid)):
            raise ValueError(
                "Initial BP Metropolis walkers must be finite, nonzero, and "
                "inside the requested symmetry sector."
            )

    def reset(self, configs=None, *, amplitudes=None, log_proposal_probabilities=None):
        """Reset chains and proposal probabilities together."""
        if log_proposal_probabilities is None:
            raise ValueError(
                "log_proposal_probabilities is required when resetting a BP "
                "Metropolis sampler."
            )
        super().reset(configs, amplitudes=amplitudes)
        torch = _require_torch()
        log_q = torch.as_tensor(
            log_proposal_probabilities,
            dtype=torch.float64,
            device=self.configs.device,
        )
        if log_q.shape != (self.n_chains,) or not bool(torch.isfinite(log_q).all()):
            raise ValueError(
                "log_proposal_probabilities must be finite with one value per chain."
            )
        self.log_proposal_probabilities = log_q.clone()
        self._validate_current_support()
        return self

    def sample_sweep(self, *, n_sweeps=1):
        """Advance all chains with BP independence proposals."""
        torch = _require_torch()
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        with torch.no_grad():
            for _ in range(n_sweeps):
                proposed, proposed_log_q = self._proposal_sample(self.n_chains)
                proposed = proposed.to(device=self.configs.device)
                proposed_log_q = proposed_log_q.to(device=self.configs.device)
                proposal_valid = self._sector_mask(proposed) & torch.isfinite(
                    proposed_log_q
                )
                proposed_amplitudes = self.amplitudes.clone()
                if bool(torch.any(proposal_valid)):
                    proposed_amplitudes[proposal_valid] = _call_amplitude_fn(
                        self.amplitude_fn,
                        proposed[proposal_valid],
                        chunk_size=self.chunk_size,
                    )

                if self.log_amplitude_fn is not None and bool(torch.any(proposal_valid)):
                    try:
                        phase, proposed_log_valid = _call_log_amplitude_fn(
                            self.log_amplitude_fn,
                            proposed[proposal_valid],
                            chunk_size=self.chunk_size,
                        )
                        proposed_log_abs = self.log_abs_amplitudes.clone()
                        proposed_nonzero = self.nonzero_amplitudes.clone()
                        proposed_log_abs[proposal_valid] = proposed_log_valid
                        proposed_nonzero[proposal_valid] = (
                            (phase.abs() > 0)
                            & torch.isfinite(proposed_log_valid)
                            & (
                                proposed_log_valid
                                > (
                                    -torch.inf
                                    if self.amplitude_floor == 0.0
                                    else float(torch.log(torch.tensor(self.amplitude_floor)))
                                )
                            )
                        )
                    except (
                        AttributeError,
                        IndexError,
                        KeyError,
                        NotImplementedError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        self.log_amplitude_fn = None
                        self.log_abs_amplitudes = None
                        self.nonzero_amplitudes = None
                elif self.log_amplitude_fn is not None:
                    proposed_log_abs = self.log_abs_amplitudes.clone()
                    proposed_nonzero = torch.zeros_like(self.nonzero_amplitudes)

                if self.log_amplitude_fn is None:
                    current_abs = self.amplitudes.abs()
                    proposed_abs = proposed_amplitudes.abs()
                    current_log_abs = torch.where(
                        current_abs > 0,
                        current_abs.to(dtype=torch.float64).log(),
                        torch.full_like(current_abs, -torch.inf, dtype=torch.float64),
                    )
                    proposed_log_abs = torch.where(
                        proposed_abs > 0,
                        proposed_abs.to(dtype=torch.float64).log(),
                        torch.full_like(proposed_abs, -torch.inf, dtype=torch.float64),
                    )
                    current_nonzero = (
                        torch.isfinite(current_abs)
                        & (current_abs > self.amplitude_floor)
                    )
                    proposed_nonzero = (
                        proposal_valid
                        & torch.isfinite(proposed_abs)
                        & (proposed_abs > self.amplitude_floor)
                    )
                else:
                    current_log_abs = self.log_abs_amplitudes
                    current_nonzero = self.nonzero_amplitudes
                    proposed_nonzero &= proposal_valid

                log_ratio = (
                    2.0 * (proposed_log_abs - current_log_abs)
                    + self.log_proposal_probabilities
                    - proposed_log_q
                )
                log_ratio = torch.where(
                    torch.isnan(log_ratio),
                    torch.zeros_like(log_ratio),
                    log_ratio,
                )
                log_ratio = torch.minimum(log_ratio, torch.zeros_like(log_ratio))
                uniform = torch.rand(
                    self.n_chains,
                    device=self.configs.device,
                    generator=self.generator,
                )
                accept = (
                    proposal_valid
                    & current_nonzero
                    & proposed_nonzero
                    & (
                        torch.log(
                            uniform.clamp_min(torch.finfo(torch.float64).tiny)
                        )
                        < log_ratio
                    )
                )
                n_accepted = int(accept.sum().item())
                self.configs[accept] = proposed[accept]
                self.amplitudes[accept] = proposed_amplitudes[accept]
                self.log_proposal_probabilities[accept] = proposed_log_q[accept]
                if self.log_amplitude_fn is not None:
                    self.log_abs_amplitudes[accept] = proposed_log_abs[accept]
                    self.nonzero_amplitudes[accept] = proposed_nonzero[accept]
                result = TorchMetropolisResult(
                    configs=self.configs,
                    amplitudes=self.amplitudes,
                    n_proposed=self.n_chains,
                    n_accepted=n_accepted,
                    log_abs_amplitudes=(
                        self.log_abs_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                    nonzero_amplitudes=(
                        self.nonzero_amplitudes
                        if self.log_amplitude_fn is not None
                        else None
                    ),
                )
        return result


def metropolis_local_sampler(
    configs,
    amplitude_fn,
    graph,
    *,
    n_sites=None,
    n_samples=1024,
    n_chains=None,
    n_discard_per_chain=None,
    n_discard=None,
    sweep_size=None,
    n_thin=None,
    proposal="spinful",
    hopping_rate=0.25,
    spin_flip_rate=0.25,
    pair_toggle_rate=0.25,
    encoding=None,
    chunk_size=None,
    amplitudes=None,
    generator=None,
    seed=None,
    progress=False,
    log_amplitude_fn=None,
):
    """Run a convenience batched Metropolis sampling call.

    ``n_sites`` is optional validation only; it is inferred from ``configs``.
    Use :class:`TorchMetropolisSampler` when the chain state must be retained
    for multiple sampling calls.
    """
    sampler = TorchMetropolisSampler(
        amplitude_fn,
        graph,
        configs,
        amplitudes=amplitudes,
        n_chains=n_chains,
        proposal=proposal,
        hopping_rate=hopping_rate,
        spin_flip_rate=spin_flip_rate,
        pair_toggle_rate=pair_toggle_rate,
        encoding=encoding,
        chunk_size=chunk_size,
        generator=generator,
        seed=seed,
        n_sites=n_sites,
        log_amplitude_fn=log_amplitude_fn,
    )
    return sampler.sample(
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        n_discard=n_discard,
        sweep_size=sweep_size,
        n_thin=n_thin,
        progress=progress,
    )


def _mode_occupations(n_up, n_down, *, order):
    torch = _require_torch()
    if order in {"down-up", "du", "symmray"}:
        return torch.stack((n_down, n_up), dim=-1).reshape(n_up.shape[0], -1)
    if order in {"up-down", "ud", "netket"}:
        return torch.stack((n_up, n_down), dim=-1).reshape(n_up.shape[0], -1)
    raise ValueError("mode_order must be 'down-up' or 'up-down'.")


def _mode_index(site, spin, *, order):
    if order in {"down-up", "du", "symmray"}:
        offset = 1 if spin == "up" else 0
    elif order in {"up-down", "ud", "netket"}:
        offset = 0 if spin == "up" else 1
    else:
        raise ValueError("mode_order must be 'down-up' or 'up-down'.")
    return 2 * site + offset


def spinful_fermi_hubbard_connections(
    configs,
    graph,
    *,
    t=1.0,
    U=8.0,
    encoding=None,
    mode_order="down-up",
):
    """Return batched spinful Fermi-Hubbard connected configurations.

    The local state encoding is configurable. Fermion signs use a site-major
    mode order; ``mode_order='down-up'`` matches Symmray/vmc_torch convention.
    """
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    n_up, n_down = encoding.decode(configs)
    modes = _mode_occupations(n_up, n_down, order=mode_order)

    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(t, edge))
        if coeff == 0.0:
            continue
        for spin, occ in (("up", n_up), ("down", n_down)):
            valid = occ[:, i] != occ[:, j]
            if not torch.any(valid):
                continue
            idx = valid.nonzero(as_tuple=True)[0]
            new_up = n_up[idx].clone()
            new_down = n_down[idx].clone()
            target = new_up if spin == "up" else new_down
            tmp = target[:, i].clone()
            target[:, i] = target[:, j]
            target[:, j] = tmp

            p = _mode_index(i, spin, order=mode_order)
            q = _mode_index(j, spin, order=mode_order)
            if p > q:
                p, q = q, p
            between = modes[idx, p + 1:q].sum(dim=-1) % 2
            phase = 1.0 - 2.0 * between.to(torch.float64)

            all_etas.append(encoding.encode(new_up, new_down))
            all_coeffs.append(-coeff * phase)
            all_bids.append(idx)

    for site in range(n_sites):
        coeff = float(_site_value(U, site))
        if coeff == 0.0:
            continue
        valid = (n_up[:, site] == 1) & (n_down[:, site] == 1)
        if not torch.any(valid):
            continue
        idx = valid.nonzero(as_tuple=True)[0]
        all_etas.append(configs[idx].clone())
        all_coeffs.append(torch.full(
            (idx.numel(),),
            coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(idx)

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def heisenberg_connections(configs, graph, *, J=1.0):
    """Return batched spin-1/2 Heisenberg connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch = configs.shape[0]
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diff = configs[:, i] != configs[:, j]
        if torch.any(diff):
            idx = diff.nonzero(as_tuple=True)[0]
            eta = configs[idx].clone()
            tmp = eta[:, i].clone()
            eta[:, i] = eta[:, j]
            eta[:, j] = tmp
            all_etas.append(eta)
            all_coeffs.append(torch.full(
                (idx.numel(),),
                0.5 * coeff,
                dtype=torch.float64,
                device=device,
            ))
            all_bids.append(idx)

        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def transverse_ising_connections(configs, graph, *, J=1.0, h=1.0):
    """Return batched transverse-field Ising connected configurations."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    batch, n_sites = configs.shape
    device = configs.device
    batch_ids = torch.arange(batch, dtype=torch.long, device=device)
    all_etas = []
    all_coeffs = []
    all_bids = []

    for edge in _iter_edges(graph):
        i, j = edge
        coeff = float(_edge_value(J, edge))
        if coeff == 0.0:
            continue
        diag_sign = 1.0 - 2.0 * ((configs[:, i] - configs[:, j]).abs() % 2).to(
            torch.float64
        )
        all_etas.append(configs.clone())
        all_coeffs.append(0.25 * coeff * diag_sign)
        all_bids.append(batch_ids.clone())

    for site in range(n_sites):
        coeff = float(_site_value(h, site))
        if coeff == 0.0:
            continue
        eta = configs.clone()
        eta[:, site] = 1 - eta[:, site]
        all_etas.append(eta)
        all_coeffs.append(torch.full(
            (batch,),
            0.5 * coeff,
            dtype=torch.float64,
            device=device,
        ))
        all_bids.append(batch_ids.clone())

    if not all_etas:
        return _empty_connections(configs)

    return TorchConnections(
        configs=torch.cat(all_etas, dim=0),
        coeffs=torch.cat(all_coeffs, dim=0),
        batch_ids=torch.cat(all_bids, dim=0),
    )


def local_energy_from_connections(
    configs,
    amplitudes,
    connections,
    amplitude_fn,
    *,
    chunk_size=None,
    reuse_diagonal=True,
):
    """Accumulate local energies from connected configs and amplitudes.

    If ``amplitude_fn`` exposes ``connected_amplitudes(...)`` that method is
    used. Otherwise diagonal connections can reuse the supplied parent
    amplitudes and off-diagonal amplitudes are evaluated in optional chunks.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )

    connected_amplitudes = getattr(amplitude_fn, "connected_amplitudes", None)
    if callable(connected_amplitudes):
        conn_amps = connected_amplitudes(
            configs,
            amplitudes,
            connections,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )
    else:
        conn_amps = _default_connected_amplitudes(
            configs,
            amplitudes,
            connections,
            amplitude_fn,
            chunk_size=chunk_size,
            reuse_diagonal=reuse_diagonal,
        )
    conn_amps = torch.as_tensor(conn_amps, device=configs.device)
    ratios = conn_amps / amplitudes[connections.batch_ids]
    contrib = connections.coeffs.to(dtype=ratios.dtype) * ratios
    energy = torch.zeros(
        configs.shape[0],
        dtype=contrib.dtype,
        device=configs.device,
    )
    energy.index_add_(0, connections.batch_ids, contrib)
    return energy


def _energy_mean_and_variance(local_energies):
    energy_mean = local_energies.mean()
    centered = local_energies - energy_mean
    variance = centered.abs().square().mean()
    return energy_mean, variance.real


def torch_chain_diagnostics(values, *, max_lag=None):
    """Return ``R-hat``, integrated autocorrelation time, and ESS.

    ``values`` must have shape ``(n_samples_per_chain, n_chains)``. The
    implementation uses split-chain-independent Gelman--Rubin statistics and
    an FFT autocorrelation estimate with an initial-positive-sequence cutoff.
    Complex values are reduced to their real parts, as appropriate for a
    Hermitian local observable.
    """
    torch = _require_torch()
    values = torch.as_tensor(values)
    if values.ndim != 2:
        raise ValueError(
            "values must have shape (n_samples_per_chain, n_chains)."
        )
    n_steps, n_chains = (int(value) for value in values.shape)
    if n_steps < 2:
        raise ValueError("At least two samples per chain are required.")
    if n_chains < 2:
        raise ValueError("At least two chains are required.")
    if not torch.is_floating_point(values) and not torch.is_complex(values):
        values = values.to(torch.float64)
    values = values.real if values.is_complex() else values
    if values.dtype != torch.float64:
        values = values.to(torch.float64)

    chain_means = values.mean(dim=0)
    within = values.var(dim=0, unbiased=True).mean()
    between = n_steps * chain_means.var(unbiased=True)
    variance_hat = (
        (n_steps - 1) * within + between
    ) / n_steps
    if bool(within == 0):
        r_hat = torch.where(
            between == 0,
            torch.ones_like(variance_hat),
            torch.full_like(variance_hat, float("inf")),
        )
    else:
        r_hat = torch.sqrt(torch.clamp(variance_hat / within, min=1.0))

    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = _check_nonnegative_int("max_lag", max_lag)
        max_lag = min(max_lag, n_steps - 1)

    centered = values - chain_means
    fft_size = 1 << (2 * n_steps - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=0)
    autocovariance = torch.fft.irfft(
        spectrum.conj() * spectrum,
        n=fft_size,
        dim=0,
    )[:n_steps]
    variances = autocovariance[0].real
    normalized = torch.where(
        variances > 0,
        autocovariance.real / variances,
        torch.zeros_like(autocovariance.real),
    )
    rho = normalized.mean(dim=1)
    tau = torch.ones((), dtype=values.dtype, device=values.device)
    for lag in range(1, max_lag + 1):
        if bool(rho[lag] <= 0):
            break
        tau = tau + 2 * rho[lag]
    tau = torch.clamp(tau, min=1.0)
    total_samples = n_steps * n_chains
    effective_sample_size = torch.as_tensor(
        total_samples,
        dtype=values.dtype,
        device=values.device,
    ) / tau
    return TorchChainDiagnostics(
        r_hat=r_hat,
        integrated_autocorrelation_time=tau,
        effective_sample_size=effective_sample_size,
        n_samples_per_chain=n_steps,
        n_chains=n_chains,
    )


def _resolve_connection_fn(connection_fn):
    if callable(connection_fn):
        return None, connection_fn
    key = str(connection_fn).replace("-", "_").lower()
    aliases = {
        "fermi_hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "fh": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "hubbard": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful": ("spinful_fermi_hubbard", spinful_fermi_hubbard_connections),
        "spinful_fermi_hubbard": (
            "spinful_fermi_hubbard",
            spinful_fermi_hubbard_connections,
        ),
        "heisenberg": ("heisenberg", heisenberg_connections),
        "heis": ("heisenberg", heisenberg_connections),
        "transverse_ising": ("transverse_ising", transverse_ising_connections),
        "tfim": ("transverse_ising", transverse_ising_connections),
        "ising": ("transverse_ising", transverse_ising_connections),
    }
    try:
        return aliases[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unknown torch VMC connection_fn {connection_fn!r}. "
            f"Expected a callable or one of: {allowed}."
        ) from exc


@dataclass(frozen=True)
class TorchVMCStepResult:
    """Result of one :class:`TorchVMCDriver` step."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    sr: Any = None


@dataclass(frozen=True)
class TorchVMCEnergyEstimate:
    """Observable estimate and sampling diagnostics from a torch VMC run.

    ``chain_diagnostics`` is populated when the estimate retained at least
    two samples from each of at least two chains.
    """

    configs: Any
    amplitudes: Any
    local_energies: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    acceptance_rate: float
    n_proposed: int
    n_accepted: int
    n_samples: int
    n_measurements: int
    elapsed_seconds: float
    samples_per_second: float
    chain_diagnostics: Any = None


@dataclass(frozen=True)
class TorchVMCImportanceEstimate:
    """Energy estimate from an external proposal distribution."""

    configs: Any
    amplitudes: Any
    local_energies: Any
    weights: Any
    energy_mean: Any
    energy_variance: Any
    energy_stderr: Any
    effective_sample_size: Any
    n_samples: int
    n_valid: int
    elapsed_seconds: float
    samples_per_second: float


def _make_progress(progress, *, total, desc):
    """Create an optional tqdm progress iterator without making tqdm required."""
    if not progress:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "progress=True requires optional dependency 'tqdm'."
        ) from exc
    return tqdm(total=total, desc=desc)


def _proposal_log_probabilities(omegas, *, device, allow_zero=False):
    """Decode ``PepsBpSampler`` mantissa/exponent proposal probabilities."""
    torch = _require_torch()
    if not isinstance(omegas, (tuple, list)) or len(omegas) != 2:
        raise ValueError(
            "proposal samples must expose omegas as (mantissas, exponents)."
        )
    mantissas = torch.as_tensor(omegas[0], dtype=torch.float64, device=device)
    exponents = torch.as_tensor(omegas[1], dtype=torch.float64, device=device)
    if mantissas.ndim != 1 or exponents.shape != mantissas.shape:
        raise ValueError("proposal mantissas and exponents must be one-dimensional.")
    if torch.any(mantissas < 0):
        raise ValueError("proposal probabilities cannot have negative mantissas.")
    if not allow_zero and torch.any(mantissas <= 0):
        raise ValueError("proposal probabilities must have positive mantissas.")
    positive = mantissas > 0
    log_prob = torch.where(
        positive,
        mantissas.log() + exponents * torch.log(
            torch.as_tensor(10.0, dtype=torch.float64, device=device)
        ),
        torch.full_like(mantissas, -torch.inf),
    )
    return log_prob


class TorchVMCDriver:
    """Small PyTorch-native VMC loop around Pepsy's torch kernels.

    The driver keeps walker configurations and amplitudes in sync, runs
    Metropolis exchange/hopping sweeps, evaluates local energies with optional
    chunking/diagonal reuse, and can apply one SR/minSR update per step.
    """

    def __init__(
        self,
        model,
        graph,
        configs,
        connection_fn=None,
        *,
        terms=None,
        site_order=None,
        connection_kwargs=None,
        amplitudes=None,
        proposal="spinful",
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
    ):
        self.model = model
        self.graph = graph
        self.configs = _as_long_matrix(configs)
        if terms is not None and connection_fn is not None:
            raise ValueError(
                "Pass either terms=... or connection_fn=..., not both."
            )
        self.terms = terms
        if terms is not None:
            self.connection_name = "terms"
            if site_order is None and hasattr(graph, "Lx") and hasattr(graph, "Ly"):
                site_order = tuple(
                    (x, y)
                    for x in range(int(graph.Lx))
                    for y in range(int(graph.Ly))
                )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_fn = _driver_terms_connections
            self.connection_kwargs = {
                "terms": terms,
                "site_order": self.site_order,
            }
        else:
            if connection_fn is None:
                connection_fn = "spinful_fermi_hubbard"
            self.connection_name, self.connection_fn = _resolve_connection_fn(
                connection_fn
            )
            self.site_order = None if site_order is None else tuple(site_order)
            self.connection_kwargs = (
                {} if connection_kwargs is None else dict(connection_kwargs)
            )
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.spin_flip_rate = float(spin_flip_rate)
        self.pair_toggle_rate = float(pair_toggle_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.generator = generator
        self.log_amplitude_fn = _resolve_log_amplitude_fn(self.model)
        self.log_abs_amplitudes = None
        self.nonzero_amplitudes = None

        if (
            self.connection_name == "spinful_fermi_hubbard"
            and encoding is not None
            and "encoding" not in self.connection_kwargs
        ):
            self.connection_kwargs["encoding"] = encoding

        if amplitudes is None:
            self.refresh_amplitudes()
        else:
            self.amplitudes = _require_torch().as_tensor(
                amplitudes,
                device=self.configs.device,
            )
            self._refresh_log_amplitudes()

    @property
    def n_walkers(self):
        """Number of active walkers."""
        return int(self.configs.shape[0])

    @property
    def n_sites(self):
        """Number of sites in each walker configuration."""
        return int(self.configs.shape[1])

    def refresh_amplitudes(self):
        """Recompute current walker amplitudes from the current model."""
        with _require_torch().no_grad():
            self.amplitudes = _call_amplitude_fn(
                self.model,
                self.configs,
                chunk_size=self.chunk_size,
            )
        self._refresh_log_amplitudes()
        return self.amplitudes

    def _refresh_log_amplitudes(self):
        if self.log_amplitude_fn is None:
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None
            return
        try:
            phase, self.log_abs_amplitudes = _call_log_amplitude_fn(
                self.log_amplitude_fn,
                self.configs,
                chunk_size=self.chunk_size,
            )
            self.nonzero_amplitudes = phase.abs() > 0
        except (AttributeError, IndexError, KeyError, NotImplementedError,
                RuntimeError, TypeError, ValueError):
            self.log_amplitude_fn = None
            self.log_abs_amplitudes = None
            self.nonzero_amplitudes = None

    def make_sampler(
        self,
        *,
        configs=None,
        amplitudes=None,
        n_chains=None,
        seed=None,
        sampler_seed=None,
    ):
        """Create a stateful sampler initialized from the current driver."""
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        if configs is None:
            configs = self.configs
            if amplitudes is None:
                amplitudes = self.amplitudes
        return TorchMetropolisSampler(
            self.model,
            self.graph,
            configs,
            amplitudes=amplitudes,
            n_chains=n_chains,
            proposal=self.proposal,
            hopping_rate=self.hopping_rate,
            spin_flip_rate=self.spin_flip_rate,
            pair_toggle_rate=self.pair_toggle_rate,
            encoding=self.encoding,
            chunk_size=self.chunk_size,
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
            log_abs_amplitudes=(
                self.log_abs_amplitudes
                if configs is self.configs
                else None
            ),
            nonzero_amplitudes=(
                self.nonzero_amplitudes
                if configs is self.configs
                else None
            ),
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
        )

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Create a BP independence sampler for this amplitude model.

        The base driver requires an explicit ``proposal_sampler``. The
        :class:`TorchFermionVMC` specialization creates a compatible
        :class:`pepsy.sampling.PepsBpSampler` automatically from its PEPS.
        """
        if proposal_sampler is None:
            raise ValueError(
                "proposal_sampler is required for TorchVMCDriver; use "
                "TorchFermionVMC to infer PepsBpSampler from a PEPS."
            )
        if seed is not None and sampler_seed is not None:
            raise ValueError("Pass either seed=... or sampler_seed=..., not both.")
        n_chains = self.n_walkers if n_chains is None else n_chains
        return TorchBPMetropolisSampler(
            self.model,
            self.graph,
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            chunk_size=self.chunk_size,
            generator=(
                self.generator
                if seed is None and sampler_seed is None
                else None
            ),
            seed=seed if seed is not None else sampler_seed,
            device=_model_device(self.model),
            log_amplitude_fn=(
                self.log_amplitude_fn
                if self.log_amplitude_fn is not None
                else False
            ),
        )

    def sample_bp(
        self,
        proposal_sampler=None,
        *,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        sample_kwargs=None,
        symmetry=None,
        sector=None,
        encoding=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Collect chain-preserving samples with BP independence proposals."""
        sampler = self.make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=symmetry,
            sector=sector,
            encoding=encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.log_abs_amplitudes = sampler.log_abs_amplitudes
        self.nonzero_amplitudes = sampler.nonzero_amplitudes
        self.generator = sampler.generator
        self._bp_sampler = sampler
        return result

    def sample(
        self,
        *,
        n_samples=1024,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        progress=False,
        seed=None,
        sampler_seed=None,
    ):
        """Collect chain-preserving samples and update the driver state."""
        sampler = self.make_sampler(
            n_chains=n_chains,
            seed=seed,
            sampler_seed=sampler_seed,
        )
        result = sampler.sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain,
            n_discard=n_discard,
            sweep_size=sweep_size,
            n_thin=n_thin,
            progress=progress,
        )
        self.configs = sampler.configs
        self.amplitudes = sampler.amplitudes
        self.generator = sampler.generator
        return result

    def make_connections(self, configs=None):
        """Build Hamiltonian-connected configurations for ``configs``."""
        configs = self.configs if configs is None else _as_long_matrix(configs)
        return self.connection_fn(configs, self.graph, **self.connection_kwargs)

    def sample_sweep(self, *, n_sweeps=1):
        """Run one or more Metropolis sweeps and update driver state."""
        n_sweeps = _check_positive_int("n_sweeps", n_sweeps)
        result = None
        with _require_torch().no_grad():
            for _ in range(n_sweeps):
                result = metropolis_exchange_sweep(
                    self.configs,
                    self.model,
                    self.graph,
                    current_amplitudes=self.amplitudes,
                    current_log_abs=self.log_abs_amplitudes,
                    current_nonzero=self.nonzero_amplitudes,
                    log_amplitude_fn=(
                        self.log_amplitude_fn
                        if self.log_amplitude_fn is not None
                        else False
                    ),
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    spin_flip_rate=self.spin_flip_rate,
                    pair_toggle_rate=self.pair_toggle_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
                self.log_abs_amplitudes = result.log_abs_amplitudes
                self.nonzero_amplitudes = result.nonzero_amplitudes
                if result.log_abs_amplitudes is None:
                    self.log_amplitude_fn = None
        return result

    def local_energies(self, *, connections=None):
        """Evaluate local energies for the current walkers."""
        connections = self.make_connections() if connections is None else connections
        with _require_torch().no_grad():
            return local_energy_from_connections(
                self.configs,
                self.amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
            )

    def energy_estimate(self):
        """Return ``(mean, variance, local_energies)`` for current walkers."""
        local_energies = self.local_energies()
        energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
        return energy_mean, energy_variance, local_energies

    def estimate_observable(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
        n_samples=None,
        n_chains=None,
        n_discard_per_chain=None,
        n_discard=None,
        sweep_size=None,
        n_thin=None,
        sampler=None,
        seed=None,
        sampler_seed=None,
    ):
        """Run burn-in and estimate the configured local observable.

        The driver keeps the configured walkers and collects all walker local
        observable values after each measurement. ``n_samples`` therefore equals
        ``n_walkers * n_measurements``. The returned ``samples_per_second``
        measures completed observable samples, including sampling and
        contraction time.

        The result retains the historical ``TorchVMCEnergyEstimate`` type and
        ``energy_*`` field names for compatibility. They describe the
        observable encoded by this driver's configured ``terms`` or connection
        function and are not restricted to a Hamiltonian.
        """
        modern_sampling = (
            sampler is not None
            or n_samples is not None
            or n_chains is not None
            or n_discard_per_chain is not None
            or n_discard is not None
            or sweep_size is not None
            or n_thin is not None
            or seed is not None
            or sampler_seed is not None
        )
        if modern_sampling:
            if sampler is None:
                samples = self.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_chains=n_chains,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                    seed=seed,
                    sampler_seed=sampler_seed,
                )
            else:
                if any(
                    value is not None
                    for value in (
                        n_chains,
                        seed,
                        sampler_seed,
                    )
                ):
                    raise ValueError(
                        "n_chains and sampler seeds must be configured on an "
                        "explicit sampler."
                    )
                samples = sampler.sample(
                    n_samples=1024 if n_samples is None else n_samples,
                    n_discard_per_chain=n_discard_per_chain,
                    n_discard=n_discard,
                    sweep_size=sweep_size,
                    n_thin=n_thin,
                    progress=progress,
                )
            start = time.perf_counter()
            sample_configs = samples.configs
            sample_amplitudes = samples.amplitudes
            flat_configs = sample_configs.reshape(-1, self.n_sites)
            flat_amplitudes = sample_amplitudes.reshape(-1)
            connections = self.make_connections(flat_configs)
            with _require_torch().no_grad():
                local_values = local_energy_from_connections(
                    flat_configs,
                    flat_amplitudes,
                    connections,
                    self.model,
                    chunk_size=self.chunk_size,
                    reuse_diagonal=True,
                )
            observable_mean, observable_variance = _energy_mean_and_variance(
                local_values
            )
            n_actual = int(local_values.numel())
            chain_values = local_values.reshape(sample_configs.shape[:-1])
            chain_diagnostics = None
            if chain_values.shape[0] >= 2 and chain_values.shape[1] >= 2:
                chain_diagnostics = torch_chain_diagnostics(chain_values)
            elapsed = time.perf_counter() - start + samples.elapsed_seconds
            return TorchVMCEnergyEstimate(
                configs=sample_configs,
                amplitudes=sample_amplitudes,
                local_energies=chain_values,
                energy_mean=observable_mean,
                energy_variance=observable_variance,
                energy_stderr=_require_torch().sqrt(
                    observable_variance / max(n_actual, 1)
                ),
                acceptance_rate=samples.acceptance_rate,
                n_proposed=samples.n_proposed,
                n_accepted=samples.n_accepted,
                n_samples=n_actual,
                n_measurements=samples.n_samples_per_chain,
                elapsed_seconds=elapsed,
                samples_per_second=(
                    n_actual / elapsed if elapsed > 0 else float("inf")
                ),
                chain_diagnostics=chain_diagnostics,
            )

        burn_in = int(burn_in)
        n_measurements = _check_positive_int("n_measurements", n_measurements)
        sweeps_between = _check_positive_int("sweeps_between", sweeps_between)
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

        total_sweeps = burn_in + n_measurements * sweeps_between
        bar = _make_progress(
            progress,
            total=total_sweeps,
            desc="Torch VMC",
        )
        start = time.perf_counter()
        n_proposed = 0
        n_accepted = 0

        def run_sweeps(count):
            nonlocal n_proposed, n_accepted
            for _ in range(count):
                sample = self.sample_sweep(n_sweeps=1)
                n_proposed += sample.n_proposed
                n_accepted += sample.n_accepted
                if bar is not None:
                    bar.update(1)

        run_sweeps(burn_in)
        measurements = []
        for _ in range(n_measurements):
            run_sweeps(sweeps_between)
            measurements.append(self.local_energies().detach())
        if bar is not None:
            bar.close()

        local_energies = _require_torch().cat(measurements, dim=0)
        energy_mean, energy_variance = _energy_mean_and_variance(local_energies)
        chain_diagnostics = None
        if n_measurements >= 2 and self.n_walkers >= 2:
            chain_diagnostics = torch_chain_diagnostics(
                _require_torch().stack(measurements, dim=0)
            )
        n_samples = int(local_energies.numel())
        elapsed = time.perf_counter() - start
        acceptance = n_accepted / n_proposed if n_proposed else 0.0
        return TorchVMCEnergyEstimate(
            configs=self.configs,
            amplitudes=self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=_require_torch().sqrt(
                energy_variance / max(n_samples, 1)
            ),
            acceptance_rate=acceptance,
            n_proposed=n_proposed,
            n_accepted=n_accepted,
            n_samples=n_samples,
            n_measurements=n_measurements,
            elapsed_seconds=elapsed,
            samples_per_second=n_samples / elapsed if elapsed > 0 else float("inf"),
            chain_diagnostics=chain_diagnostics,
        )

    def estimate_energy(
        self,
        *,
        burn_in=0,
        n_measurements=1,
        sweeps_between=1,
        progress=False,
    ):
        """Compatibility wrapper for :meth:`estimate_observable`."""
        return self.estimate_observable(
            burn_in=burn_in,
            n_measurements=n_measurements,
            sweeps_between=sweeps_between,
            progress=progress,
        )

    def importance_energy_estimate(
        self,
        proposal_sampler,
        *,
        n_samples=128,
        sample_kwargs=None,
        amplitude_floor=0.0,
        progress=False,
    ):
        """Measure the driver Hamiltonian using an external proposal sampler.

        ``proposal_sampler`` should expose ``sample(samples=..., progbar=...)``
        and return a PEPS-BP-style result with ``configs`` and ``omegas``.
        The sampler proposes configurations; torch evaluates their PEPS
        amplitudes and local energies. The returned self-normalized weights are
        ``|psi(x)|**2 / q(x)`` and include an effective sample-size diagnostic.
        """
        torch = _require_torch()
        n_samples = _check_positive_int("n_samples", n_samples)
        if amplitude_floor < 0:
            raise ValueError("amplitude_floor must be non-negative.")
        sample_kwargs = dict(sample_kwargs or {})
        sample_kwargs.setdefault("samples", n_samples)
        sample_kwargs.setdefault("progbar", bool(progress))
        start = time.perf_counter()
        try:
            proposed = proposal_sampler.sample(**sample_kwargs)
        except TypeError:
            # Small custom proposal samplers often don't expose ``progbar``.
            sample_kwargs.pop("progbar", None)
            proposed = proposal_sampler.sample(**sample_kwargs)

        device = self.configs.device
        configs = _as_long_matrix(proposed.configs, name="proposal configs")
        configs = configs.to(device=device)
        if configs.shape[0] != n_samples:
            n_samples = int(configs.shape[0])
        log_q = _proposal_log_probabilities(proposed.omegas, device=device)
        if log_q.shape[0] != configs.shape[0]:
            raise ValueError("proposal probabilities must match proposal configs.")

        with torch.no_grad():
            amplitudes = _call_amplitude_fn(
                self.model,
                configs,
                chunk_size=self.chunk_size,
            )
            amp_abs = amplitudes.abs()
            valid = torch.isfinite(amp_abs) & (amp_abs > float(amplitude_floor))
            if not torch.any(valid):
                raise ValueError(
                    "The proposal produced no configurations with non-zero "
                    "torch PEPS amplitude."
                )
            valid_configs = configs[valid]
            valid_amplitudes = amplitudes[valid]
            connections = self.make_connections(valid_configs)
            local_energies = local_energy_from_connections(
                valid_configs,
                valid_amplitudes,
                connections,
                self.model,
                chunk_size=self.chunk_size,
                reuse_diagonal=True,
            )
            log_weights = 2.0 * valid_amplitudes.abs().log() - log_q[valid]
            log_weights = log_weights - log_weights.max()
            weights = torch.exp(log_weights)
            weights = weights / weights.sum()
            energy_mean = (weights.to(local_energies.dtype) * local_energies).sum()
            energy_variance = (
                weights * (local_energies - energy_mean).abs().square()
            ).sum().real
            effective_sample_size = 1.0 / weights.square().sum()

        n_valid = int(valid.sum().item())
        elapsed = time.perf_counter() - start
        return TorchVMCImportanceEstimate(
            configs=valid_configs,
            amplitudes=valid_amplitudes,
            local_energies=local_energies,
            weights=weights,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            energy_stderr=torch.sqrt(energy_variance / effective_sample_size),
            effective_sample_size=effective_sample_size,
            n_samples=n_samples,
            n_valid=n_valid,
            elapsed_seconds=elapsed,
            samples_per_second=n_valid / elapsed if elapsed > 0 else float("inf"),
        )

    def step(
        self,
        *,
        sample_sweeps=1,
        sr=False,
        learning_rate=1.0,
        sr_diag_shift=1.0e-4,
        sr_method="auto",
        amplitude_floor=None,
    ):
        """Run sampling, estimate energy, and optionally apply an SR update."""
        sample = self.sample_sweep(n_sweeps=sample_sweeps)
        local_energies = self.local_energies()
        energy_mean, energy_variance = _energy_mean_and_variance(local_energies)

        sr_result = None
        if sr:
            log_derivatives = torch_log_derivative_matrix(
                self.model,
                self.configs,
                amplitude_floor=amplitude_floor,
            )
            sr_result = solve_torch_sr(
                log_derivatives,
                local_energies,
                method=sr_method,
                diag_shift=sr_diag_shift,
            )
            apply_torch_sr_update(
                self.model,
                sr_result.direction,
                learning_rate=learning_rate,
            )
            self.refresh_amplitudes()

        return TorchVMCStepResult(
            configs=self.configs,
            amplitudes=self.amplitudes,
            local_energies=local_energies,
            energy_mean=energy_mean,
            energy_variance=energy_variance,
            acceptance_rate=0.0 if sample is None else sample.acceptance_rate,
            n_proposed=0 if sample is None else sample.n_proposed,
            n_accepted=0 if sample is None else sample.n_accepted,
            sr=sr_result,
        )

    def run(self, n_steps, **step_kwargs):
        """Run ``n_steps`` VMC steps and return their result records."""
        n_steps = _check_positive_int("n_steps", n_steps)
        return [self.step(**step_kwargs) for _ in range(n_steps)]


def _fermion_sector_from_configs(configs, metadata):
    """Return the unique conserved sector represented by ``configs``."""
    n_up, n_down = count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        values = n_up + n_down
        sector = int(values[0].item())
    elif metadata.symmetry == "Z2":
        values = n_up + n_down
        sector = int((values[0] % 2).item())
    elif metadata.symmetry == "Z2Z2":
        values = list(
            zip(
                (n_up % 2).detach().cpu().tolist(),
                (n_down % 2).detach().cpu().tolist(),
            )
        )
        sector = tuple(int(value) for value in values[0])
    else:
        values = list(
            zip(n_up.detach().cpu().tolist(), n_down.detach().cpu().tolist())
        )
        sector = tuple(int(value) for value in values[0])
    if metadata.symmetry == "U1":
        if not bool((values == sector).all()):
            raise ValueError("All initial walkers must have the same U1 sector.")
    elif metadata.symmetry == "Z2":
        if not bool(((values % 2) == sector).all()):
            raise ValueError("All initial walkers must have the same Z2 sector.")
    elif metadata.symmetry == "Z2Z2":
        if any(tuple(value) != sector for value in values):
            raise ValueError("All initial walkers must have the same Z2Z2 sector.")
    elif any(tuple(value) != sector for value in values):
        raise ValueError("All initial walkers must have the same U1U1 sector.")
    return sector


def _fermion_sector_counts(sector, symmetry, n_sites):
    """Choose spin-resolved counts for a spinful initial configuration batch."""
    if symmetry == "U1U1":
        return tuple(sector)
    if symmetry == "Z2":
        total = n_sites if n_sites % 2 == sector else n_sites - 1
        n_up = total // 2
        return n_up, total - n_up
    if symmetry == "Z2Z2":
        n_up = n_sites if n_sites % 2 == int(sector[0]) else n_sites - 1
        n_down = n_sites if n_sites % 2 == int(sector[1]) else n_sites - 1
        return n_up, n_down
    total = int(sector)
    n_up = total // 2
    return n_up, total - n_up


def _fermion_sector_mask(configs, metadata):
    """Return a boolean mask selecting walkers in ``metadata.sector``."""
    n_up, n_down = count_spinful_particles(
        configs,
        encoding=metadata.encoding,
    )
    if metadata.symmetry == "U1":
        return n_up + n_down == metadata.sector
    if metadata.symmetry == "Z2":
        return (n_up + n_down) % 2 == metadata.sector
    if metadata.symmetry == "Z2Z2":
        return (n_up % 2 == metadata.sector[0]) & (
            n_down % 2 == metadata.sector[1]
        )
    return (n_up == metadata.sector[0]) & (n_down == metadata.sector[1])


def _model_device(model, device=None):
    torch = _require_torch()
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _initial_fermion_walkers(
    model,
    metadata,
    n_walkers,
    *,
    device,
    generator=None,
    amplitude_floor=0.0,
    max_attempts=32,
    max_states=100_000,
):
    """Find nonzero PEPS amplitudes inside the requested conserved sector."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    max_attempts = _check_positive_int("init_max_attempts", max_attempts)
    max_states = _check_positive_int("init_max_states", max_states)
    if amplitude_floor < 0:
        raise ValueError("amplitude_floor must be non-negative.")

    n_up, n_down = _fermion_sector_counts(
        metadata.sector,
        metadata.symmetry,
        metadata.n_sites,
    )
    kept_configs = []
    kept_amplitudes = []

    def keep(candidate):
        with torch.no_grad():
            candidate_amplitudes = _call_amplitude_fn(model, candidate)
        valid = (
            torch.isfinite(candidate_amplitudes.abs())
            & (candidate_amplitudes.abs() > float(amplitude_floor))
        )
        if bool(torch.any(valid)):
            kept_configs.append(candidate[valid])
            kept_amplitudes.append(candidate_amplitudes[valid])
        return int(valid.sum().item())

    n_kept = 0
    for _ in range(max_attempts):
        candidate = random_spinful_configs(
            n_walkers,
            metadata.n_sites,
            n_up,
            n_down,
            encoding=metadata.encoding,
            device=device,
            generator=generator,
        )
        n_kept += keep(candidate)
        if n_kept >= n_walkers:
            break

    if n_kept == 0 and 4 ** metadata.n_sites <= max_states:
        candidate = torch.as_tensor(
            tuple(product(range(4), repeat=metadata.n_sites)),
            dtype=torch.long,
            device=device,
        )
        candidate = candidate[_fermion_sector_mask(candidate, metadata)]
        if candidate.numel():
            n_kept += keep(candidate)

    if n_kept == 0:
        raise RuntimeError(
            "Could not find a nonzero PEPS amplitude in the requested Fermion "
            "sector. Pass valid configs or increase init_max_attempts."
        )

    configs = torch.cat(kept_configs, dim=0)
    amplitudes = torch.cat(kept_amplitudes, dim=0)
    if configs.shape[0] < n_walkers:
        choice = torch.randint(
            configs.shape[0],
            (n_walkers,),
            device=device,
            generator=generator,
        )
    else:
        choice = torch.arange(n_walkers, device=device)
    return configs[choice], amplitudes[choice]


class TorchFermionVMC(TorchVMCDriver):
    """Automatic native spinful Fermion VMC around a Quimb PEPS.

    The constructor derives the PEPS lattice, physical dimension, local basis,
    charge sector, and default native Hamiltonian. ``fermion`` can be omitted
    when explicit ``terms`` are supplied and the PEPS exposes Symmray symmetry
    metadata. The lower-level
    :class:`TorchVMCDriver` remains available when callers need full manual
    control over configurations or connection functions.
    """

    def __init__(
        self,
        peps,
        fermion=None,
        terms=None,
        *,
        observables=None,
        edges=None,
        pbc=False,
        site_order=None,
        sector=None,
        configs=None,
        n_walkers=128,
        contraction="exact",
        chi=None,
        cutoff=None,
        contraction_opts=None,
        dtype=None,
        device=None,
        proposal=None,
        hopping_rate=0.25,
        spin_flip_rate=0.25,
        pair_toggle_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
        seed=None,
        amplitude_floor=0.0,
        init_max_attempts=32,
        init_max_states=100_000,
    ):
        torch = _require_torch()
        if terms is not None and observables is not None:
            raise ValueError("Pass either terms=... or observables=..., not both.")
        if observables is not None:
            terms = observables

        metadata = _infer_torch_fermion_metadata(
            peps,
            fermion,
            sector=sector,
            edges=edges,
            pbc=pbc,
            site_order=site_order,
        )
        if encoding is not None and encoding != metadata.encoding:
            raise ValueError(
                "The supplied encoding does not match the native Fermion local "
                "basis. Omit encoding=... to infer it safely."
            )

        model = make_torch_peps_amplitude_model(
            peps,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            dtype=dtype,
            device=device,
            site_order=metadata.site_order,
        )
        model_device = _model_device(model, device=device)
        if generator is not None and seed is not None:
            raise ValueError("Pass either generator=... or seed=..., not both.")
        if seed is not None:
            try:
                generator = torch.Generator(device=model_device)
            except (RuntimeError, TypeError, ValueError):
                generator = torch.Generator()
            generator.manual_seed(int(seed))

        if terms is None:
            if fermion is None:
                raise ValueError(
                    "Pass fermion=... when terms are omitted so the default "
                    "Hamiltonian can be constructed."
                )
            hamiltonian = fermion.hamiltonian(metadata.edges)
            terms = hamiltonian.terms
        else:
            hamiltonian = terms if hasattr(terms, "terms") else None
        terms = _normalize_terms_site_labels(terms, metadata.site_order)

        if configs is None:
            if metadata.sector is None:
                raise ValueError(
                    "Could not infer the PEPS charge sector. Pass sector=... or "
                    "provide initial configs in the target sector."
                )
            configs, amplitudes = _initial_fermion_walkers(
                model,
                metadata,
                n_walkers,
                device=model_device,
                generator=generator,
                amplitude_floor=amplitude_floor,
                max_attempts=init_max_attempts,
                max_states=init_max_states,
            )
        else:
            configs = _as_long_matrix(configs).to(device=model_device)
            if configs.shape[1] != metadata.n_sites:
                raise ValueError(
                    f"configs must have {metadata.n_sites} sites, got {configs.shape[1]}."
                )
            metadata.encoding.validate(configs)
            actual_sector = _fermion_sector_from_configs(configs, metadata)
            if metadata.sector is not None and actual_sector != metadata.sector:
                raise ValueError(
                    f"configs are in sector {actual_sector}, expected {metadata.sector}."
                )
            if metadata.sector is None:
                metadata = replace(metadata, sector=actual_sector)
            with torch.no_grad():
                amplitudes = _call_amplitude_fn(model, configs)
            valid = (
                torch.isfinite(amplitudes.abs())
                & (amplitudes.abs() > float(amplitude_floor))
            )
            if not bool(torch.all(valid)):
                raise ValueError(
                    "configs contain zero, non-finite, or below-floor PEPS amplitudes."
                )

        self.peps = peps
        self.fermion = fermion
        self.metadata = metadata
        self.hamiltonian = hamiltonian
        self.physical_charges = metadata.physical_charges
        if proposal is None:
            proposal = {
                "U1": "spinful_u1",
                "U1U1": "spinful",
                "Z2": "spinful_z2",
                "Z2Z2": "spinful_z2z2",
            }[metadata.symmetry]

        super().__init__(
            model,
            metadata.graph,
            configs,
            terms=terms,
            site_order=metadata.site_order,
            amplitudes=amplitudes,
            proposal=proposal,
            hopping_rate=hopping_rate,
            spin_flip_rate=spin_flip_rate,
            pair_toggle_rate=pair_toggle_rate,
            encoding=metadata.encoding,
            chunk_size=chunk_size,
            generator=generator,
        )

    @property
    def Lx(self):
        return self.metadata.Lx

    @property
    def Ly(self):
        return self.metadata.Ly

    def make_bp_sampler(
        self,
        proposal_sampler=None,
        *,
        n_chains=None,
        sample_kwargs=None,
        bp_sampler_kwargs=None,
        amplitude_floor=0.0,
        max_init_attempts=32,
        seed=None,
        sampler_seed=None,
    ):
        """Create a symmetry-aware BP independence sampler from this PEPS."""
        if proposal_sampler is None:
            from ..sampling import PepsBpSampler  # pylint: disable=import-outside-toplevel

            proposal_sampler = PepsBpSampler(
                self.peps,
                encoding=self.metadata.encoding,
                site_order=self.metadata.site_order,
                sample_kwargs=bp_sampler_kwargs,
            )
        return super().make_bp_sampler(
            proposal_sampler,
            n_chains=n_chains,
            sample_kwargs=sample_kwargs,
            symmetry=self.metadata.symmetry,
            sector=self.metadata.sector,
            encoding=self.metadata.encoding,
            amplitude_floor=amplitude_floor,
            max_init_attempts=max_init_attempts,
            seed=seed,
            sampler_seed=sampler_seed,
        )

    @property
    def sector(self):
        return self.metadata.sector


def random_spin_configs(n_walkers, n_sites, n_up, *, device=None, generator=None):
    """Generate binary spin configs with fixed number of up spins."""
    torch = _require_torch()
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    configs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm = torch.randperm(n_sites, device=device, generator=generator)
        configs[row, perm[:n_up]] = 1
    return configs


def random_spinful_configs(
    n_walkers,
    n_sites,
    n_up,
    n_down,
    *,
    encoding=None,
    device=None,
    generator=None,
):
    """Generate spinful fermion configs with fixed ``N_up`` and ``N_down``."""
    torch = _require_torch()
    encoding = FermionSiteEncoding.vmc_torch() if encoding is None else encoding
    n_walkers = _check_positive_int("n_walkers", n_walkers)
    n_sites = _check_positive_int("n_sites", n_sites)
    if n_up < 0 or n_up > n_sites:
        raise ValueError("n_up must be between 0 and n_sites.")
    if n_down < 0 or n_down > n_sites:
        raise ValueError("n_down must be between 0 and n_sites.")
    ups = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    downs = torch.zeros((n_walkers, n_sites), dtype=torch.long, device=device)
    for row in range(n_walkers):
        perm_up = torch.randperm(n_sites, device=device, generator=generator)
        perm_down = torch.randperm(n_sites, device=device, generator=generator)
        ups[row, perm_up[:n_up]] = 1
        downs[row, perm_down[:n_down]] = 1
    return encoding.encode(ups, downs)
