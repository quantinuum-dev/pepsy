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
    "TorchConnections",
    "TorchMetropolisResult",
    "TorchSquareLattice",
    "count_spinful_particles",
    "heisenberg_connections",
    "local_energy_from_connections",
    "metropolis_exchange_sweep",
    "propose_spin_exchange",
    "propose_spinful_exchange_or_hopping",
    "random_spin_configs",
    "random_spinful_configs",
    "make_torch_peps_amplitude_model",
    "spinful_fermi_hubbard_connections",
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


class TorchPEPSAmplitude:
    """Torch-optimizable amplitude wrapper for a quimb PEPS-like network.

    The input configuration rows are physical indices in the PEPS site order by
    default. For spin PEPS this usually means binary rows ``0/1``. For spinful
    Hubbard PEPS use a four-state row encoding that matches the PEPS physical
    basis, for example :class:`FermionSiteEncoding.symmray`.

    This class deliberately stays pure PEPS/TNS: it registers the packed PEPS
    tensor leaves as torch parameters and evaluates amplitudes by selecting
    physical indices then contracting the resulting quimb tensor network.
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

    def _select_config(self, tn, config):
        if config.shape[0] != self.n_sites:
            raise ValueError(
                f"config row has length {config.shape[0]}, expected {self.n_sites}."
            )
        return tn.isel({ind: config[i] for i, ind in enumerate(self.site_inds)})

    def _contract_value(self, tnx):
        if self.contraction == "hotrg":
            return tnx.contract_hotrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        if self.contraction == "ctmrg":
            return tnx.contract_ctmrg(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        if self.contraction == "boundary":
            return tnx.contract_boundary(
                max_bond=self.chi,
                cutoff=self.cutoff,
                **self.contraction_opts,
            )
        return tnx.contract(all)

    def _contract_log_parts(self, tnx):
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
            abs_amp = amp.abs()
            tiny = _torch_finfo_tiny(abs_amp.dtype)
            phase = torch.where(
                abs_amp > 0,
                amp / abs_amp.to(dtype=amp.dtype),
                torch.zeros_like(amp),
            )
            return phase, torch.log(abs_amp.clamp_min(tiny))

        mantissa = torch.as_tensor(mantissa)
        exponent_10 = torch.as_tensor(exponent_10, device=mantissa.device)
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
        return self._contract_value(self._select_config(tn, config))

    def forward(self, configs, params=None):
        """Evaluate a batch of configuration amplitudes."""
        configs = _as_long_matrix(configs)
        tn = self._unpack_tn(params)
        return _require_torch().stack([
            self._contract_value(self._select_config(tn, row))
            for row in configs
        ])

    def forward_log(self, configs, params=None):
        """Return ``(phase, log_abs)`` for a batch of configurations."""
        configs = _as_long_matrix(configs)
        tn = self._unpack_tn(params)
        phases = []
        log_abs = []
        for row in configs:
            phase, log_scale = self._contract_log_parts(self._select_config(tn, row))
            phases.append(phase)
            log_abs.append(log_scale)
        torch = _require_torch()
        return torch.stack(phases), torch.stack(log_abs)

    def __call__(self, configs, params=None):
        """Alias for :meth:`forward`."""
        return self.forward(configs, params=params)


def make_torch_peps_amplitude_model(peps, **kwargs):
    """Build a :class:`TorchPEPSAmplitude` from a quimb PEPS-like object."""
    return TorchPEPSAmplitude(peps, **kwargs)


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
):
    """Run one nearest-neighbor Metropolis sweep.

    ``amplitude_fn`` should accept a ``(batch, n_sites)`` torch integer tensor
    and return a batch of amplitudes. The sampler evaluates only changed
    proposals when possible.
    """
    torch = _require_torch()
    configs = _as_long_matrix(configs).clone()
    current = amplitude_fn(configs) if current_amplitudes is None else current_amplitudes
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
        proposed_amps[flags] = amplitude_fn(proposed[flags])
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
):
    """Accumulate local energies from connected configs and amplitudes."""
    torch = _require_torch()
    configs = _as_long_matrix(configs)
    amplitudes = torch.as_tensor(amplitudes, device=configs.device)
    if connections.configs.numel() == 0:
        return torch.zeros(
            configs.shape[0],
            dtype=amplitudes.dtype,
            device=configs.device,
        )
    conn_amps = amplitude_fn(connections.configs)
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
