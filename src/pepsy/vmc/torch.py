"""PyTorch kernels for lightweight VMC loops.

The routines here are intentionally small and optional-dependency friendly.
They cover the sampler and local-energy pieces that are useful around PEPS
amplitude models without vendoring a full VMC framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

__all__ = [
    "FermionSiteEncoding",
    "TorchPEPSAmplitude",
    "TorchPEPSBoundaryAmplitude",
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchVMCDriver",
    "TorchVMCStepResult",
    "TorchSRResult",
    "TorchSquareLattice",
    "apply_torch_sr_update",
    "count_spinful_particles",
    "heisenberg_connections",
    "local_energy_from_connections",
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "random_spin_configs",
    "random_spinful_configs",
    "make_torch_peps_amplitude_model",
    "solve_torch_sr",
    "spinful_fermi_hubbard_connections",
    "torch_log_derivative_matrix",
    "transverse_ising_connections",
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
        cutoff=0.0,
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
        self.cutoff = float(cutoff)
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

    def _contract_value(self, tnx, reference=None):
        if self.contraction == "hotrg":
            value = tnx.contract_hotrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            value = tnx.contract_ctmrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            value = tnx.contract_boundary(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        else:
            value = tnx.contract(all)
        return _as_torch_scalar(value, reference)

    def _contract_log_parts(self, tnx, reference=None):
        torch = _require_torch()
        if self.contraction == "hotrg":
            mantissa, exponent_10 = tnx.contract_hotrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                strip_exponent=True,
                **self.contraction_opts,
            )
        elif self.contraction == "ctmrg":
            mantissa, exponent_10 = tnx.contract_ctmrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                strip_exponent=True,
                **self.contraction_opts,
            )
        elif self.contraction == "boundary":
            mantissa, exponent_10 = tnx.contract_boundary(
                max_bond=self.chi,
                cutoff=self.cutoff,
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
        cutoff=0.0,
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
            return parent_tn.compute_x_environments(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        return parent_tn.compute_y_environments(
            max_bond=self.chi,
            cutoff=self.cutoff,
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
            except (
                AttributeError,
                KeyError,
                NotImplementedError,
                TypeError,
                ValueError,
            ):
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

    Pepsy's Symmray physical-index convention is ``0=empty, 1=double,
    2=up, 3=down``. The reference ``vmc_torch`` code often uses
    ``0=empty, 1=down, 2=up, 3=double``. Use the class constructors to make
    that choice explicit.
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

    @property
    def acceptance_rate(self):
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed


def _iter_edges(graph):
    if hasattr(graph, "edges"):
        edges = graph.edges
        if callable(edges):
            edges = edges()
    else:
        edges = graph
    return tuple((int(i), int(j)) for i, j in edges)


def _empty_connections(configs):
    torch = _require_torch()
    return TorchConnections(
        configs=configs.new_empty((0, configs.shape[1])),
        coeffs=torch.empty(0, dtype=torch.float64, device=configs.device),
        batch_ids=torch.empty(0, dtype=torch.long, device=configs.device),
    )


def count_spinful_particles(configs, *, encoding=None):
    """Return per-sample ``(n_up, n_down)`` counts."""
    encoding = FermionSiteEncoding.symmray() if encoding is None else encoding
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
    encoding = FermionSiteEncoding.symmray() if encoding is None else encoding
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


def metropolis_exchange_sweep(
    configs,
    amplitude_fn,
    graph,
    *,
    current_amplitudes=None,
    proposal="spinful",
    hopping_rate=0.25,
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
        else:
            raise ValueError(
                "proposal must be 'spin' or 'spinful_exchange_hopping'."
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

    return TorchMetropolisResult(
        configs=configs,
        amplitudes=current,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
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
    encoding = FermionSiteEncoding.symmray() if encoding is None else encoding
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
        connection_fn="spinful_fermi_hubbard",
        *,
        connection_kwargs=None,
        amplitudes=None,
        proposal="spinful",
        hopping_rate=0.25,
        encoding=None,
        chunk_size=None,
        generator=None,
    ):
        self.model = model
        self.graph = graph
        self.configs = _as_long_matrix(configs)
        self.connection_name, self.connection_fn = _resolve_connection_fn(
            connection_fn
        )
        self.connection_kwargs = (
            {} if connection_kwargs is None else dict(connection_kwargs)
        )
        self.proposal = proposal
        self.hopping_rate = float(hopping_rate)
        self.encoding = encoding
        self.chunk_size = _normalize_chunk_size(chunk_size)
        self.generator = generator

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
        return self.amplitudes

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
                    proposal=self.proposal,
                    hopping_rate=self.hopping_rate,
                    encoding=self.encoding,
                    generator=self.generator,
                    chunk_size=self.chunk_size,
                )
                self.configs = result.configs
                self.amplitudes = result.amplitudes
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
    encoding = FermionSiteEncoding.symmray() if encoding is None else encoding
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
