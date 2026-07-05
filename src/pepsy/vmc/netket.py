"""NetKet bridge for PEPS VMC.

This module keeps NetKet/JAX/Flax/Symmray optional. Importing it requires those
packages only when the concrete helpers are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os
from typing import Any
import warnings

import numpy as np
import quimb.tensor as qtn

__all__ = [
    "NetKetLocalConfigMap",
    "NetKetChunkSettings",
    "NetKetPEPSVMC",
    "NetKetFermiHubbardVMC",
    "NetKetVMCSettings",
    "PackedPEPS",
    "PackedFermionicPEPS",
    "SpinOrbitalColumns",
    "build_heisenberg_vmc",
    "build_ising_vmc",
    "build_fermi_hubbard_vmc",
    "fermionic_peps_rand",
    "choose_netket_chunk_size",
    "configure_jax_for_vmc",
    "config_to_phys_indices",
    "make_peps_log_amplitude_model",
    "make_peps_batched_amplitude_function",
    "make_fermionic_peps_log_amplitude_model",
    "make_fermionic_peps_batched_amplitude_function",
    "make_netket_autochunk_callback",
    "make_netket_sr_preconditioner",
    "make_netket_vmc_driver",
    "netket_spin_orbital_columns",
    "occupation_to_phys_indices",
    "pack_peps_ansatz",
    "pack_fermionic_peps_ansatz",
    "recommend_netket_vmc_settings",
    "square_lattice_edges",
    "verify_netket_spin_columns",
]


@dataclass(frozen=True)
class NetKetLocalConfigMap:
    """Map local NetKet configuration values to PEPS physical indices."""

    values: tuple[Any, ...] = (1, -1)
    phys_indices: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        values = tuple(self.values)
        phys_indices = tuple(int(i) for i in self.phys_indices)
        if len(values) != len(phys_indices):
            raise ValueError("values and phys_indices must have the same length.")
        if len(values) == 0:
            raise ValueError("At least one local configuration value is required.")
        if len(set(values)) != len(values):
            raise ValueError("Local configuration values must be unique.")
        if any(i < 0 for i in phys_indices):
            raise ValueError("PEPS physical indices must be non-negative.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "phys_indices", phys_indices)

    @classmethod
    def spin_half(cls, *, up=0, down=1):
        """Return the standard NetKet spin-1/2 ``+1/-1`` map."""
        return cls(values=(1, -1), phys_indices=(up, down))


@dataclass(frozen=True)
class SpinOrbitalColumns:
    """NetKet occupation-column layout for spinful fermions."""

    up: tuple[int, ...]
    down: tuple[int, ...]

    def __post_init__(self):
        up = tuple(int(i) for i in self.up)
        down = tuple(int(i) for i in self.down)
        if len(up) == 0:
            raise ValueError("SpinOrbitalColumns requires at least one orbital.")
        if len(up) != len(down):
            raise ValueError("up and down columns must have the same length.")
        if any(i < 0 for i in (*up, *down)):
            raise ValueError("Spin-orbital columns must be non-negative.")
        if len(set(up)) != len(up) or len(set(down)) != len(down):
            raise ValueError("Spin-orbital columns must be unique per spin.")
        if set(up) & set(down):
            raise ValueError("up and down columns must be disjoint.")
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)

    @property
    def n_orbitals(self):
        """Number of spatial orbitals/sites represented by this layout."""
        return len(self.up)

    @property
    def n_columns(self):
        """Number of NetKet occupation columns in each configuration row."""
        return 2 * self.n_orbitals


@dataclass(frozen=True)
class PackedPEPS:
    """Packed PEPS data needed by a NetKet/Flax log-amplitude model."""

    params: Any
    skeleton: Any
    leaves: tuple[Any, ...]
    treedef: Any
    sites: tuple[Any, ...]
    orbital_sites: tuple[Any, ...]
    orb_to_site: tuple[int, ...]
    site_to_orb: tuple[int, ...]
    n_params: int
    site_inds: tuple[Any, ...] = ()
    uses_flat_symmray: bool | None = None
    phys_charges: tuple[Any, ...] = ()

    @property
    def n_sites(self):
        """Number of physical lattice sites/orbitals."""
        return len(self.orbital_sites)

    @property
    def config_sites(self):
        """NetKet configuration-site order."""
        return self.orbital_sites

    @property
    def config_to_site(self):
        """Map configuration-site order to packed PEPS site order."""
        return self.orb_to_site

    @property
    def site_to_config(self):
        """Map packed PEPS site order to configuration-site order."""
        return self.site_to_orb


@dataclass(frozen=True)
class PackedFermionicPEPS(PackedPEPS):
    """Packed fermionic PEPS data for backward-compatible type checks."""


@dataclass(frozen=True)
class NetKetChunkSettings:
    """Forward, sampler, and backward chunk sizes for NetKet VMC."""

    chunk_size: int | None
    sampler_chunk_size: int | None
    chunk_size_bwd: int | None


@dataclass(frozen=True)
class NetKetVMCSettings:
    """Conservative large-run NetKet settings suggested by Pepsy."""

    driver: str
    n_samples: int
    n_chains: int
    chunks: NetKetChunkSettings
    use_sr: bool
    sr_mode: str
    use_ntk: bool | None
    on_the_fly: bool | None
    auto_chunk: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class NetKetPEPSVMC:
    """Bundle returned by generic PEPS/NetKet VMC builders."""

    hilbert: Any
    graph: Any
    hamiltonian: Any
    sampler: Any
    vstate: Any
    model: Any
    ansatz: PackedPEPS
    config_map: NetKetLocalConfigMap | None
    preconditioner: Any | None

    @property
    def n_sites(self):
        """Number of PEPS/configuration sites in the variational ansatz."""
        return self.ansatz.n_sites

    @property
    def n_params(self):
        """Number of scalar variational parameters in the packed ansatz."""
        return self.ansatz.n_params

    def expect_energy(self):
        """Return NetKet's expectation estimate for this setup Hamiltonian."""
        return self.vstate.expect(self.hamiltonian)

    def make_driver(self, **kwargs):
        """Create a NetKet VMC driver for this setup.

        This is a convenience wrapper around :func:`make_netket_vmc_driver`.
        """
        return make_netket_vmc_driver(self, **kwargs)


@dataclass(frozen=True)
class NetKetFermiHubbardVMC(NetKetPEPSVMC):
    """Bundle returned by :func:`build_fermi_hubbard_vmc`."""

    columns: SpinOrbitalColumns | None = None


def configure_jax_for_vmc(
    *,
    preallocate=False,
    mem_fraction=0.65,
    platform=None,
    disable_netket_tips=True,
):
    """Set JAX/NetKet environment defaults for notebook VMC runs.

    Call this before importing ``jax`` or ``netket``. Existing environment
    values are preserved.
    """
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if preallocate else "false",
    )
    if mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    if platform is not None:
        os.environ.setdefault("JAX_PLATFORMS", str(platform))
    if disable_netket_tips:
        os.environ.setdefault("NETKET_NO_TIPS", "1")


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'jax'. "
            "Install a CPU or CUDA JAX build before using this module."
        ) from exc
    return jax, jnp


def _require_flax_linen():
    try:
        import flax.linen as nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'flax'. "
            "Install it with `pip install flax`."
        ) from exc
    return nn


def _require_netket():
    try:
        import netket as nk
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pepsy.vmc.netket requires optional dependency 'netket'. "
            "Install it with `pip install netket`."
        ) from exc
    return nk


def _require_symmray():
    try:
        import symmray as sr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "fermionic PEPS VMC requires optional dependency 'symmray'. "
            "Install it with `pip install symmray`."
        ) from exc
    return sr


def _check_positive_int(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer or None.")
    return int(value)


def _largest_power_of_two_at_most(n):
    if n < 1:
        raise ValueError("n must be positive.")
    return 1 << (int(n).bit_length() - 1)


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


def _validate_contraction(name, contraction, chi):
    key = str(contraction).replace("_", "-").lower()
    try:
        contraction = _CONTRACTION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"{name} must be 'exact', 'hotrg', 'ctmrg', or 'boundary'."
        ) from exc
    if contraction in {"hotrg", "ctmrg", "boundary"} and chi is None:
        raise ValueError(f"{name}={contraction!r} requires chi.")
    return contraction


def _contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


def _coerce_config_map(config_map):
    if config_map is None:
        return NetKetLocalConfigMap.spin_half()
    if isinstance(config_map, NetKetLocalConfigMap):
        return config_map
    if isinstance(config_map, dict):
        return NetKetLocalConfigMap(
            values=tuple(config_map.keys()),
            phys_indices=tuple(config_map.values()),
        )
    values, phys_indices = config_map
    return NetKetLocalConfigMap(values=tuple(values), phys_indices=tuple(phys_indices))


def _is_symmray_array(value):
    return type(value).__module__.split(".", 1)[0] == "symmray"


def _uses_flat_symmray_arrays(tn):
    """Return whether Symmray arrays in ``tn`` use flat JAX-friendly storage."""
    symmray_seen = False
    for tensor in tn:
        data = getattr(tensor, "data", None)
        if _is_symmray_array(data):
            symmray_seen = True
            if "Flat" not in type(data).__name__:
                return False
    return True if symmray_seen else None


def _param_leaf_size(leaf):
    size = getattr(leaf, "size", None)
    if size is not None:
        return int(size)
    shape = getattr(leaf, "shape", None)
    if shape is not None:
        return int(np.prod(shape, dtype=np.int64))
    return int(np.asarray(leaf).size)


def _peps_phys_charges(tn):
    """Return the ordered physical-index charges of a Symmray PEPS, else ``()``.

    For a spinful fermionic PEPS the physical index carries the local charge of
    each basis state. For ``U1U1`` these are ``(n_up, n_down)`` tuples in a
    definite order (e.g. ``(0, 0), (0, 1), (1, 0), (1, 1)``); for ``Z2`` they
    are parity integers. The ordering is authoritative for the
    occupation->physical-index fold used by the amplitude functions.
    """
    try:
        site = next(iter(tn.sites))
    except Exception:  # pragma: no cover - non-PEPS input
        return ()
    tensor = tn[site]
    data = getattr(tensor, "data", None)
    if not _is_symmray_array(data):
        return ()
    try:
        phys_ind = tn.site_ind(site)
        axis = tuple(tensor.inds).index(phys_ind)
        index = data.indices[axis]
        chargemap = getattr(index, "chargemap", None)
        if chargemap is None:
            return ()
        return tuple(chargemap.keys())
    except Exception:  # pragma: no cover - best-effort introspection
        return ()


def _spinful_phys_lookup(phys_charges):
    """Return a ``(2, 2)`` int lookup ``lut[n_up, n_down] -> phys index``.

    The lookup is derived from the PEPS physical-index charge order so the
    occupation->physical fold matches the symmetry actually stored in the
    ansatz. Returns ``None`` when the charges are not resolved per
    ``(n_up, n_down)`` tuple (e.g. ``Z2`` parity sectors), in which case callers
    use the legacy ``2 * (n_up != n_down) + n_down`` fold.
    """
    charges = tuple(phys_charges or ())
    if len(charges) != 4:
        return None
    if not all(isinstance(c, tuple) and len(c) == 2 for c in charges):
        return None
    lut = np.full((2, 2), -1, dtype=np.int32)
    for pos, (n_up, n_down) in enumerate(charges):
        if n_up in (0, 1) and n_down in (0, 1):
            lut[int(n_up), int(n_down)] = pos
    if bool((lut < 0).any()):
        return None
    return lut


def _require_jittable_fermionic_ansatz(ansatz):
    """Raise a clear error when NetKet's jitted VMC path cannot use ``ansatz``."""
    if ansatz.uses_flat_symmray is not False:
        return
    if _spinful_phys_lookup(getattr(ansatz, "phys_charges", ())) is not None:
        symmetry = "U1U1"
    else:
        symmetry = "non-flat Symmray"
    raise NotImplementedError(
        "NetKet MCState JIT-compiles the PEPS log-amplitude model, but this "
        f"{symmetry} fermionic PEPS uses block-sparse Symmray arrays rather "
        "than a flat JAX-friendly backend. Use "
        "make_fermionic_peps_batched_amplitude_function(..., jit=False) with "
        "contraction='exact', 'hotrg', 'ctmrg', or 'boundary' for validation, "
        "or use a flat Z2 fermionic PEPS for full NetKet VMC until Symmray "
        "provides a flat U1U1 fermionic backend."
    )


def choose_netket_chunk_size(
    total_size,
    *,
    target=None,
    n_devices=1,
    require_divisor=True,
):
    """Choose a conservative power-of-two NetKet chunk size.

    ``total_size`` is usually ``n_samples`` for the forward pass or
    ``n_chains`` for the sampler. The chosen size is bounded by ``target`` and
    by the per-device workload. When ``require_divisor=True`` the result also
    divides the per-device workload, matching the strict sampler requirement
    and the older MCState recommendation.
    """
    total_size = _check_positive_int("total_size", total_size)
    target = _check_positive_int("target", target)
    n_devices = _check_positive_int("n_devices", n_devices)

    per_device = max(total_size // n_devices, 1)
    cap = per_device if target is None else min(target, per_device)
    chunk = _largest_power_of_two_at_most(cap)
    if not require_divisor:
        return chunk

    while per_device % chunk != 0:
        chunk //= 2
    return chunk


def _resolve_netket_qgt(nk, qgt):
    if qgt is None or qgt == "auto":
        return None
    if not isinstance(qgt, str):
        return qgt

    qgt_name = qgt.replace("-", "_").lower()
    qgt_map = {
        "jacobian_dense": nk.optimizer.qgt.QGTJacobianDense,
        "jacobian_pytree": nk.optimizer.qgt.QGTJacobianPyTree,
        "onthefly": nk.optimizer.qgt.QGTOnTheFly,
        "on_the_fly": nk.optimizer.qgt.QGTOnTheFly,
    }
    try:
        return qgt_map[qgt_name]
    except KeyError as exc:
        names = ", ".join(sorted(qgt_map))
        raise ValueError(f"Unknown NetKet QGT {qgt!r}; expected one of {names}.") from exc


def make_netket_sr_preconditioner(
    *,
    qgt="auto",
    solver=None,
    diag_shift=0.01,
    diag_scale=None,
    solver_restart=False,
    **qgt_kwargs,
):
    """Create a NetKet ``SR`` preconditioner with explicit QGT selection."""
    nk = _require_netket()
    qgt = _resolve_netket_qgt(nk, qgt)
    kwargs = {
        "diag_shift": diag_shift,
        "diag_scale": diag_scale,
        "solver_restart": solver_restart,
        **qgt_kwargs,
    }
    if qgt is not None:
        kwargs["qgt"] = qgt
    if solver is not None:
        kwargs["solver"] = solver
    return nk.optimizer.SR(**kwargs)


def make_netket_autochunk_callback(
    *,
    sampler_chunk_size=None,
    chunk_size=None,
    chunk_size_bwd=None,
    minimum_chunk_size=1,
):
    """Create NetKet's auto-chunk callback for sampler/forward/backward OOMs."""
    nk = _require_netket()
    return nk.callbacks.AutoChunkSize(
        sampler_chunk_size=sampler_chunk_size,
        chunk_size=chunk_size,
        chunk_size_bwd=chunk_size_bwd,
        minimum_chunk_size=minimum_chunk_size,
    )


def recommend_netket_vmc_settings(
    *,
    n_params,
    n_samples=4096,
    n_chains=32,
    target_chunk_size=256,
    target_sampler_chunk_size=None,
    max_standard_sr_params=5_000,
    n_devices=1,
    auto_chunk=True,
):
    """Suggest first-pass NetKet settings for larger PEPS VMC runs.

    This is deliberately conservative: it does not guess physics parameters,
    but it does choose chunk sizes, SR driver mode, and NTK/minSR policy from
    the number of variational parameters and samples.
    """
    n_params = _check_positive_int("n_params", n_params)
    n_samples = _check_positive_int("n_samples", n_samples)
    n_chains = _check_positive_int("n_chains", n_chains)
    max_standard_sr_params = _check_positive_int(
        "max_standard_sr_params",
        max_standard_sr_params,
    )

    chunk_size = choose_netket_chunk_size(
        n_samples,
        target=target_chunk_size,
        n_devices=n_devices,
    )
    sampler_chunk_size = choose_netket_chunk_size(
        n_chains,
        target=target_sampler_chunk_size,
        n_devices=n_devices,
    )

    use_large_sr_driver = n_params > max_standard_sr_params
    driver = "vmc_sr" if use_large_sr_driver else "vmc"
    use_ntk = n_params > n_samples if use_large_sr_driver else None
    on_the_fly = True if use_ntk else None
    chunk_size_bwd = chunk_size if use_large_sr_driver else None

    notes = [
        "Use contraction='hotrg', 'ctmrg', or 'boundary' with a fixed chi for "
        "lattices where exact amplitude contraction is no longer feasible.",
        "Keep dense exact-energy and all-state column checks restricted to tiny "
        "systems.",
    ]
    if use_large_sr_driver:
        notes.append(
            "Use nk.driver.VMC_SR so NetKet can switch between QGT and "
            "NTK/minSR style updates."
        )
    else:
        notes.append(
            "The parameter count is still small enough for standard VMC plus "
            "an optional SR preconditioner."
        )
    if auto_chunk:
        notes.append(
            "Run with make_netket_autochunk_callback(...) on the first GPU "
            "attempt to tune memory-safe chunks."
        )

    return NetKetVMCSettings(
        driver=driver,
        n_samples=n_samples,
        n_chains=n_chains,
        chunks=NetKetChunkSettings(
            chunk_size=chunk_size,
            sampler_chunk_size=sampler_chunk_size,
            chunk_size_bwd=chunk_size_bwd,
        ),
        use_sr=True,
        sr_mode="real",
        use_ntk=use_ntk,
        on_the_fly=on_the_fly,
        auto_chunk=auto_chunk,
        notes=tuple(notes),
    )


def square_lattice_edges(Lx, Ly, *, pbc=False):
    """Return nearest-neighbor row-major edges for an ``Lx`` by ``Ly`` lattice."""
    if isinstance(pbc, bool):
        pbc_x = pbc_y = pbc
    else:
        pbc_x, pbc_y = pbc

    def site_index(i, j):
        return i * Ly + j

    edges = []
    for i in range(Lx):
        for j in range(Ly):
            if j + 1 < Ly:
                edges.append((site_index(i, j), site_index(i, j + 1)))
            elif pbc_y and Ly > 2:
                edges.append((site_index(i, j), site_index(i, 0)))

            if i + 1 < Lx:
                edges.append((site_index(i, j), site_index(i + 1, j)))
            elif pbc_x and Lx > 2:
                edges.append((site_index(i, j), site_index(0, j)))
    return tuple(edges)


def netket_spin_orbital_columns(hilbert):
    """Return NetKet occupation columns for ``(orbital, spin)`` modes."""
    n_orbitals = int(getattr(hilbert, "n_orbitals"))
    if hasattr(hilbert, "_get_index"):
        up = tuple(int(hilbert._get_index(i, +1)) for i in range(n_orbitals))
        down = tuple(int(hilbert._get_index(i, -1)) for i in range(n_orbitals))
    else:
        down = tuple(range(n_orbitals))
        up = tuple(range(n_orbitals, 2 * n_orbitals))
    return SpinOrbitalColumns(up=up, down=down)


def verify_netket_spin_columns(hilbert, columns=None, *, max_states=50_000):
    """Verify spin columns against NetKet number operators on small sectors."""
    _require_netket()
    columns = columns or netket_spin_orbital_columns(hilbert)
    if hilbert.n_states > max_states:
        raise ValueError(
            "Refusing to enumerate NetKet Hilbert sector with "
            f"{hilbert.n_states} states; raise max_states to force this check."
        )

    from netket.operator.fermion import number  # pylint: disable=import-outside-toplevel

    states = np.asarray(hilbert.all_states()).astype(int)

    def match_col(op):
        diag = np.round(np.real(np.diag(np.asarray(op.to_dense())))).astype(int)
        return next(c for c in range(hilbert.size) if np.array_equal(states[:, c], diag))

    n_orbitals = int(hilbert.n_orbitals)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"\s*WARNING: Initializing `netket\.operator\.FermionOperator2nd`",
            category=UserWarning,
        )
        detected_up = tuple(
            match_col(number(hilbert, i, +1)) for i in range(n_orbitals)
        )
        detected_down = tuple(
            match_col(number(hilbert, i, -1)) for i in range(n_orbitals)
        )
    if detected_up != columns.up or detected_down != columns.down:
        raise ValueError(
            "NetKet spin-column mismatch: "
            f"expected up={columns.up}, down={columns.down}; "
            f"detected up={detected_up}, down={detected_down}."
        )
    return columns


def occupation_to_phys_indices(occ_rows, columns, *, site_to_orb=None, phys_charges=None):
    """Map NetKet spin-orbital occupations to Symmray spinful physical indices.

    The occupation->physical-index fold follows the PEPS physical-index charge
    order supplied via ``phys_charges``. For ``U1U1`` the charges resolve each
    ``(n_up, n_down)`` state directly (typically
    ``0=(0,0)``, ``1=(0,1)``, ``2=(1,0)``, ``3=(1,1)`` -> ``2*n_up + n_down``).
    When ``phys_charges`` is ``None`` or only parity-resolved (``Z2``), the
    legacy fold ``0=(0,0)``, ``1=(1,1)``, ``2=(1,0)``, ``3=(0,1)`` is used.
    """
    occ_rows = np.asarray(occ_rows).astype(int).reshape(-1, 2 * len(columns.up))
    nu = occ_rows[:, np.asarray(columns.up)]
    nd = occ_rows[:, np.asarray(columns.down)]
    lut = _spinful_phys_lookup(phys_charges)
    if lut is not None:
        phys_orb = lut[nu, nd].astype(np.int32)
    else:
        phys_orb = 2 * (nu != nd).astype(np.int32) + nd
    if site_to_orb is None:
        return phys_orb.astype(np.int32)
    return phys_orb[:, np.asarray(site_to_orb)].astype(np.int32)


def config_to_phys_indices(config_rows, config_map=None, *, site_to_config=None):
    """Map NetKet local configurations to PEPS physical indices.

    NetKet spin-1/2 Hilbert spaces use local values ``+1`` and ``-1``. The
    default map sends ``+1 -> 0`` and ``-1 -> 1``. Supply ``config_map`` to use
    another local basis or physical-index order.
    """
    config_map = _coerce_config_map(config_map)
    rows = np.asarray(config_rows)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)

    phys = np.zeros(rows.shape, dtype=np.int32)
    matched = np.zeros(rows.shape, dtype=bool)
    for value, phys_index in zip(config_map.values, config_map.phys_indices):
        mask = rows == value
        phys[mask] = phys_index
        matched |= mask
    if not np.all(matched):
        bad = np.unique(rows[~matched])
        raise ValueError(
            "Configuration row contains value(s) not in config_map: "
            f"{bad.tolist()!r}."
        )

    if site_to_config is None:
        return phys
    return phys[:, np.asarray(site_to_config)].astype(np.int32)


def _row_major_sites(Lx, Ly):
    return tuple((i, j) for i in range(Lx) for j in range(Ly))


def _pack_peps_ansatz(
    peps,
    *,
    lattice_shape=None,
    config_sites=None,
    packed_cls=PackedPEPS,
):
    tn = getattr(peps, "tn", peps)
    if not hasattr(tn, "sites"):
        raise TypeError("peps must be a quimb PEPS-like object with a `sites` attribute.")

    sites = tuple(tn.sites)
    if config_sites is None:
        if lattice_shape is None:
            config_sites = sites
        else:
            config_sites = _row_major_sites(*lattice_shape)
    config_sites = tuple(config_sites)

    missing = [site for site in config_sites if site not in sites]
    if missing:
        raise ValueError(f"config_sites contains site(s) not in PEPS: {missing!r}")

    params, skeleton = qtn.pack(tn)
    leaves, treedef = _require_jax()[0].tree_util.tree_flatten(params)
    site_inds = tuple(tn.site_ind(site) for site in sites)
    orb_to_site = tuple(sites.index(site) for site in config_sites)
    site_to_orb = tuple(int(i) for i in np.argsort(np.asarray(orb_to_site)))
    n_params = int(sum(_param_leaf_size(leaf) for leaf in leaves))
    return packed_cls(
        params=params,
        skeleton=skeleton,
        leaves=tuple(leaves),
        treedef=treedef,
        sites=sites,
        orbital_sites=config_sites,
        site_inds=site_inds,
        orb_to_site=orb_to_site,
        site_to_orb=site_to_orb,
        n_params=n_params,
        uses_flat_symmray=_uses_flat_symmray_arrays(tn),
        phys_charges=_peps_phys_charges(tn),
    )


def pack_peps_ansatz(peps, *, lattice_shape=None, config_sites=None):
    """Pack a quimb/Symmray PEPS for use as a Flax parameter pytree."""
    return _pack_peps_ansatz(
        peps,
        lattice_shape=lattice_shape,
        config_sites=config_sites,
        packed_cls=PackedPEPS,
    )


def pack_fermionic_peps_ansatz(peps, *, lattice_shape=None, orbital_sites=None):
    """Pack a Symmray/quimb fermionic PEPS for use as a Flax pytree."""
    return _pack_peps_ansatz(
        peps,
        lattice_shape=lattice_shape,
        config_sites=orbital_sites,
        packed_cls=PackedFermionicPEPS,
    )


def _make_peps_batched_amplitude_apply(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")
    if ansatz.uses_flat_symmray is False:
        warnings.warn(
            "JAX-jitted Symmray PEPS amplitudes are fastest with flat=True "
            "PEPS data; repack a flat Symmray PEPS for large GPU runs.",
            RuntimeWarning,
            stacklevel=2,
        )

    config_map = _coerce_config_map(config_map)
    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    values = jnp.asarray(config_map.values)
    phys_indices = jnp.asarray(config_map.phys_indices, dtype=jnp.int32)
    site_to_config = jnp.asarray(ansatz.site_to_config, dtype=jnp.int32)
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    n_sites = ansatz.n_sites
    site_inds = tuple(ansatz.site_inds)

    def config_rows_to_phys_jax(config_rows):
        rows = jnp.asarray(config_rows).reshape((-1, n_sites))
        phys = jnp.zeros(rows.shape, dtype=jnp.int32)
        for value, phys_index in zip(values, phys_indices):
            phys = jnp.where(rows == value, phys_index, phys)
        return phys[:, site_to_config]

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: phys[k] for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })

    def contract_mantissa_exponent(tnx):
        if contraction == "hotrg":
            return tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "ctmrg":
            return tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "boundary":
            return tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        amp = tnx.contract(all)
        return amp, jnp.zeros((), dtype=real_dtype)

    def log_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def amplitude_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.asarray(mantissa).astype(complex_dtype)
            * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
        )

    def apply(config_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = config_rows_to_phys_jax(config_rows)

        def evaluate_phys(phys):
            mantissa, exponent = contract_mantissa_exponent(select_phys(tn, phys))
            if output == "mantissa_exponent":
                return mantissa, exponent
            if output == "amplitude":
                return amplitude_from_mantissa_exponent(mantissa, exponent)
            return log_from_mantissa_exponent(mantissa, exponent)

        return jax.vmap(evaluate_phys)(phys_rows)

    return apply


def _make_peps_batched_amplitude_nojit(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")

    config_map = _coerce_config_map(config_map)
    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    site_inds = tuple(ansatz.site_inds)

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: int(phys[k]) for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): int(phys[k])
            for k, site in enumerate(ansatz.sites)
        })

    def evaluate_one(tn, phys):
        tnx = select_phys(tn, phys)
        if contraction == "hotrg":
            mantissa, exponent = tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "ctmrg":
            mantissa, exponent = tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        else:
            mantissa = tnx.contract(all)
            exponent = jnp.zeros((), dtype=real_dtype)

        if output == "mantissa_exponent":
            return mantissa, exponent
        if output == "amplitude":
            return (
                jnp.asarray(mantissa).astype(complex_dtype)
                * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
            )
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def apply(config_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = config_to_phys_indices(
            config_rows,
            config_map,
            site_to_config=ansatz.site_to_config,
        )
        values = [evaluate_one(tn, phys) for phys in phys_rows]
        if output == "mantissa_exponent":
            mantissas, exponents = zip(*values)
            return (
                jnp.stack([jnp.asarray(x) for x in mantissas]),
                jnp.asarray(exponents, dtype=real_dtype),
            )
        return jnp.stack([jnp.asarray(x) for x in values])

    return apply


def make_peps_batched_amplitude_function(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="mantissa_exponent",
    jit=True,
):
    """Return a batched JAX amplitude function for NetKet local configs."""
    jax, _ = _require_jax()
    config_map = _coerce_config_map(config_map)
    if jit:
        apply = _make_peps_batched_amplitude_apply(
            ansatz,
            config_map,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )
        apply = jax.jit(apply)
    else:
        apply = _make_peps_batched_amplitude_nojit(
            ansatz,
            config_map,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )

    def evaluate(config_rows, params=None):
        if params is None:
            params = ansatz.params
        return apply(config_rows, params)

    return evaluate


def make_peps_log_amplitude_model(
    ansatz,
    config_map=None,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    param_dtype=None,
):
    """Build a Flax model returning batched ``log(psi(config_row))``."""
    _validate_contraction("contraction", contraction, chi)

    config_map = _coerce_config_map(config_map)
    jax, jnp = _require_jax()
    nn = _require_flax_linen()
    init_values = tuple(np.asarray(leaf) for leaf in ansatz.leaves)
    batched_log_amplitude = _make_peps_batched_amplitude_apply(
        ansatz,
        config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        output="log",
    )

    class PEPSLogAmplitude(nn.Module):
        """Flax module wrapping a packed PEPS log amplitude."""

        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x)
            leaves = [
                self.param(
                    f"t{k}",
                    lambda key, value=value: (
                        jnp.asarray(value)
                        if param_dtype is None
                        else jnp.asarray(value, dtype=param_dtype)
                    ),
                )
                for k, value in enumerate(init_values)
            ]
            params = jax.tree_util.tree_unflatten(ansatz.treedef, leaves)
            return batched_log_amplitude(x, params)

    return PEPSLogAmplitude()


def _make_fermionic_peps_batched_amplitude_apply(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")

    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    col_up = jnp.asarray(columns.up, dtype=jnp.int32)
    col_down = jnp.asarray(columns.down, dtype=jnp.int32)
    site_to_orb = jnp.asarray(ansatz.site_to_orb, dtype=jnp.int32)
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    n_sites = ansatz.n_sites
    site_inds = tuple(ansatz.site_inds)
    _phys_lut = _spinful_phys_lookup(getattr(ansatz, "phys_charges", ()))
    phys_lut = None if _phys_lut is None else jnp.asarray(_phys_lut, dtype=jnp.int32)

    def occ_rows_to_phys_jax(occ_rows):
        occ_rows = jnp.asarray(occ_rows, dtype=jnp.int32).reshape((-1, 2 * n_sites))
        nu = occ_rows[:, col_up]
        nd = occ_rows[:, col_down]
        if phys_lut is not None:
            phys_orb = phys_lut[nu, nd]
        else:
            phys_orb = 2 * (nu != nd).astype(jnp.int32) + nd
        return phys_orb[:, site_to_orb]

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: phys[k] for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): phys[k]
            for k, site in enumerate(ansatz.sites)
        })

    def contract_mantissa_exponent(tnx):
        if contraction == "hotrg":
            return tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "ctmrg":
            return tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        if contraction == "boundary":
            return tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        amp = tnx.contract(all)
        return amp, jnp.zeros((), dtype=real_dtype)

    def log_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def amplitude_from_mantissa_exponent(mantissa, exponent):
        return (
            jnp.asarray(mantissa).astype(complex_dtype)
            * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
        )

    def apply(occ_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = occ_rows_to_phys_jax(occ_rows)

        def evaluate_phys(phys):
            mantissa, exponent = contract_mantissa_exponent(select_phys(tn, phys))
            if output == "mantissa_exponent":
                return mantissa, exponent
            if output == "amplitude":
                return amplitude_from_mantissa_exponent(mantissa, exponent)
            return log_from_mantissa_exponent(mantissa, exponent)

        return jax.vmap(evaluate_phys)(phys_rows)

    return apply


def _make_fermionic_peps_batched_amplitude_nojit(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="log",
):
    contraction = _validate_contraction("contraction", contraction, chi)
    if output not in {"log", "amplitude", "mantissa_exponent"}:
        raise ValueError("output must be 'log', 'amplitude', or 'mantissa_exponent'.")

    method_opts = _contraction_options(contraction_opts)
    if contraction == "boundary":
        method_opts.setdefault("mode", "mps")
    jax, jnp = _require_jax()
    real_dtype = jnp.float64 if jax.config.x64_enabled else jnp.float32
    complex_dtype = jnp.complex128 if jax.config.x64_enabled else jnp.complex64
    log10 = jnp.log(jnp.asarray(10.0, dtype=real_dtype))
    site_inds = tuple(ansatz.site_inds)

    def select_phys(tn, phys):
        if site_inds:
            return tn.isel({ind: int(phys[k]) for k, ind in enumerate(site_inds)})
        return tn.isel({
            tn.site_ind(site): int(phys[k])
            for k, site in enumerate(ansatz.sites)
        })

    def evaluate_one(tn, phys):
        tnx = select_phys(tn, phys)
        if contraction == "hotrg":
            mantissa, exponent = tnx.contract_hotrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "ctmrg":
            mantissa, exponent = tnx.contract_ctmrg(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = tnx.contract_boundary(
                max_bond=chi,
                cutoff=cutoff,
                strip_exponent=True,
                **method_opts,
            )
        else:
            mantissa = tnx.contract(all)
            exponent = jnp.zeros((), dtype=real_dtype)

        if output == "mantissa_exponent":
            return mantissa, exponent
        if output == "amplitude":
            return (
                jnp.asarray(mantissa).astype(complex_dtype)
                * jnp.power(jnp.asarray(10.0, dtype=real_dtype), exponent)
            )
        return (
            jnp.log(jnp.asarray(mantissa).astype(complex_dtype))
            + jnp.asarray(exponent, dtype=real_dtype) * log10
        )

    def apply(occ_rows, params):
        tn = qtn.unpack(params, ansatz.skeleton)
        phys_rows = occupation_to_phys_indices(
            np.asarray(occ_rows),
            columns,
            site_to_orb=ansatz.site_to_orb,
            phys_charges=getattr(ansatz, "phys_charges", ()),
        )
        values = [evaluate_one(tn, phys) for phys in phys_rows]
        if output == "mantissa_exponent":
            mantissas, exponents = zip(*values)
            return (
                jnp.stack([jnp.asarray(x) for x in mantissas]),
                jnp.asarray(exponents, dtype=real_dtype),
            )
        return jnp.stack([jnp.asarray(x) for x in values])

    return apply


def make_fermionic_peps_batched_amplitude_function(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    output="mantissa_exponent",
    jit=True,
):
    """Return a batched JAX amplitude function for notebook/GPU profiling.

    The returned callable has signature ``fn(occupation_rows, params=None)``.
    ``occupation_rows`` uses NetKet's spin-orbital occupation columns. When
    ``params`` is omitted, the packed parameters from ``ansatz`` are used; pass
    an updated quimb-packed pytree to evaluate a changed PEPS.

    ``contraction`` can be ``"exact"``, ``"hotrg"``, ``"ctmrg"``, or
    ``"boundary"`` / ``"mps"``. Approximate Quimb contractions require ``chi``.
    ``contraction_opts`` is forwarded to the selected Quimb contraction method.

    ``output="mantissa_exponent"`` mirrors Symmray's batch-GPU example and is
    the numerically stable choice for approximate contractions: it returns
    ``(mantissa, exponent)`` with amplitudes represented as
    ``mantissa * 10**exponent``. Use ``output="log"`` for NetKet-style log
    amplitudes or ``output="amplitude"`` for direct scalar amplitudes on
    tiny/exact contractions.
    """
    jax, _ = _require_jax()
    if jit:
        _require_jittable_fermionic_ansatz(ansatz)
        apply = _make_fermionic_peps_batched_amplitude_apply(
            ansatz,
            columns,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )
        apply = jax.jit(apply)
    else:
        apply = _make_fermionic_peps_batched_amplitude_nojit(
            ansatz,
            columns,
            contraction=contraction,
            chi=chi,
            cutoff=cutoff,
            contraction_opts=contraction_opts,
            output=output,
        )

    def evaluate(occupation_rows, params=None):
        if params is None:
            params = ansatz.params
        return apply(occupation_rows, params)

    return evaluate


def make_fermionic_peps_log_amplitude_model(
    ansatz,
    columns,
    *,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    param_dtype=None,
):
    """Build a Flax model returning batched ``log(psi(occupation_row))``."""
    _validate_contraction("contraction", contraction, chi)
    _require_jittable_fermionic_ansatz(ansatz)

    jax, jnp = _require_jax()
    nn = _require_flax_linen()
    init_values = tuple(np.asarray(leaf) for leaf in ansatz.leaves)
    batched_log_amplitude = _make_fermionic_peps_batched_amplitude_apply(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        output="log",
    )

    class FermionicPEPSLogAmplitude(nn.Module):
        """Flax module wrapping a packed fermionic PEPS log amplitude."""

        @nn.compact
        def __call__(self, x):
            x = jnp.atleast_2d(x).astype(jnp.int32)
            leaves = [
                self.param(
                    f"t{k}",
                    lambda key, value=value: (
                        jnp.asarray(value)
                        if param_dtype is None
                        else jnp.asarray(value, dtype=param_dtype)
                    ),
                )
                for k, value in enumerate(init_values)
            ]
            params = jax.tree_util.tree_unflatten(ansatz.treedef, leaves)
            return batched_log_amplitude(x, params)

    return FermionicPEPSLogAmplitude()


def _maybe_make_sr_preconditioner(
    ansatz,
    *,
    use_sr,
    max_sr_params,
    sr_diag_shift,
    sr_diag_scale,
    sr_qgt,
    sr_solver,
    sr_solver_restart,
):
    if use_sr == "auto":
        use_sr = ansatz.n_params <= max_sr_params
    return (
        make_netket_sr_preconditioner(
            qgt=sr_qgt,
            solver=sr_solver,
            diag_shift=sr_diag_shift,
            diag_scale=sr_diag_scale,
            solver_restart=sr_solver_restart,
        )
        if use_sr
        else None
    )


def _build_netket_peps_vmc(
    peps,
    *,
    Lx,
    Ly,
    hilbert,
    graph,
    hamiltonian,
    sampler,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_discard_per_chain=32,
    chunk_size=256,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    config_map = _coerce_config_map(config_map)
    ansatz = pack_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    model = make_peps_log_amplitude_model(
        ansatz,
        config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    vstate = _require_netket().vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )
    preconditioner = _maybe_make_sr_preconditioner(
        ansatz,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
    )
    return NetKetPEPSVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        vstate=vstate,
        model=model,
        ansatz=ansatz,
        config_map=config_map,
        preconditioner=preconditioner,
    )


def build_ising_vmc(
    peps,
    *,
    Lx,
    Ly,
    h=1.0,
    J=1.0,
    pbc=False,
    edges=None,
    graph=None,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    """Create NetKet VMC objects for a spin-1/2 transverse-field Ising PEPS."""
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites)
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.Ising(
        hilbert,
        graph=graph,
        h=h,
        J=J,
        dtype=float,
    )
    sampler_kwargs = {"n_chains": n_chains}
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisLocal(hilbert, **sampler_kwargs)
    return _build_netket_peps_vmc(
        peps,
        Lx=Lx,
        Ly=Ly,
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        config_map=config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
        param_dtype=param_dtype,
    )


def build_heisenberg_vmc(
    peps,
    *,
    Lx,
    Ly,
    J=1.0,
    total_sz=0.0,
    sign_rule=None,
    pbc=False,
    edges=None,
    graph=None,
    d_max=1,
    config_map=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
):
    """Create NetKet VMC objects for a spin-1/2 Heisenberg PEPS."""
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    hilbert = nk.hilbert.Spin(s=1 / 2, N=n_sites, total_sz=total_sz)
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.Heisenberg(
        hilbert,
        graph=graph,
        J=J,
        sign_rule=sign_rule,
        dtype=float,
    )
    sampler_kwargs = {"graph": graph, "d_max": d_max, "n_chains": n_chains}
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisExchange(hilbert, **sampler_kwargs)
    return _build_netket_peps_vmc(
        peps,
        Lx=Lx,
        Ly=Ly,
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        config_map=config_map,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
        param_dtype=param_dtype,
    )


def _default_u1u1_flux_occupations(Lx, Ly, n_up, n_down):
    """Per-site ``(n_up, n_down)`` flux summing to ``(n_up, n_down)``.

    Up charges are placed on the first ``n_up`` sites (row-major) and down
    charges on the last ``n_down`` sites, spreading them apart. Only the *total*
    charge sector matters for VMC; local occupation still fluctuates through the
    virtual bonds of the random ansatz.
    """
    sites = [(i, j) for i in range(int(Lx)) for j in range(int(Ly))]
    n = len(sites)
    n_up = int(n_up)
    n_down = int(n_down)
    if not (0 <= n_up <= n and 0 <= n_down <= n):
        raise ValueError(
            f"n_fermions_per_spin=({n_up}, {n_down}) is out of range for "
            f"{n} sites."
        )
    ups = [0] * n
    downs = [0] * n
    for k in range(n_up):
        ups[k] = 1
    for k in range(n_down):
        downs[n - 1 - k] = 1
    return {site: (ups[k], downs[k]) for k, site in enumerate(sites)}


def fermionic_peps_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    *,
    n_fermions_per_spin=None,
    site_charge=None,
    seed=None,
    dtype="float64",
    flat="auto",
    **kwargs,
):
    """Build a random fermionic PEPS for VMC, symmetry-aware.

    ``"Z2"`` uses the flat (``jax.jit``/``vmap``-friendly) Symmray backend and a
    ``phys_dim=4`` parity-resolved physical index. ``"U1U1"`` uses the
    block-sparse backend (Symmray currently has no flat ``U1U1`` fermionic
    array) with a per-spin ``(n_up, n_down)`` physical charge map and a default
    site-charge summing to ``n_fermions_per_spin`` (half filling if omitted).

    Note
    ----
    A ``U1U1`` ansatz cannot yet be driven through the NetKet Monte-Carlo state
    (which JIT-compiles the model) because the flat backend is missing upstream.
    The block-sparse ``U1U1`` PEPS still evaluates correctly through the
    non-jitted amplitude functions (``jit=False``) for validation and exact/dense
    sums.
    """
    sr = _require_symmray()
    sym = str(symmetry).upper().replace("-", "").replace("_", "")
    n_sites = int(Lx) * int(Ly)

    if sym == "Z2":
        use_flat = True if flat == "auto" else bool(flat)
        return sr.networks.PEPS_fermionic_rand(
            "Z2",
            Lx,
            Ly,
            bond_dim,
            phys_dim=4,
            subsizes="equal",
            flat=use_flat,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    if sym == "U1U1":
        from pepsy.tensors import (
            default_physical_sectors,
            site_charge_from_occupations,
        )

        if site_charge is None:
            if n_fermions_per_spin is None:
                n_fermions_per_spin = (n_sites // 2, n_sites // 2)
            n_up, n_down = (int(x) for x in n_fermions_per_spin)
            site_charge = site_charge_from_occupations(
                _default_u1u1_flux_occupations(Lx, Ly, n_up, n_down)
            )
        use_flat = False if flat == "auto" else bool(flat)
        if use_flat:
            warnings.warn(
                "Symmray has no flat U1U1 fermionic backend; falling back to "
                "block-sparse (flat=False). NetKet MC sampling JIT-compiles the "
                "model and needs a flat backend, so use the non-jit amplitude "
                "functions for U1U1 until flat U1U1 lands upstream.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_flat = False
        return sr.networks.PEPS_fermionic_rand(
            "U1U1",
            Lx,
            Ly,
            bond_dim,
            phys_dim=default_physical_sectors(model="fermi_hubbard_u1u1"),
            site_charge=site_charge,
            flat=use_flat,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported symmetry {symmetry!r} for fermionic_peps_rand; "
        "use 'Z2' or 'U1U1'."
    )


def build_fermi_hubbard_vmc(
    peps,
    *,
    Lx,
    Ly,
    t=1.0,
    U=8.0,
    n_fermions_per_spin=None,
    pbc=False,
    edges=None,
    graph=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampler_chunk_size=None,
    seed=None,
    sampler_seed=None,
    use_sr="auto",
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
    verify_columns=False,
):
    """Create the first-pass NetKet VMC objects for a fermionic PEPS."""
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=tuple(n_fermions_per_spin),
    )
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = nk.operator.FermiHubbardJax(
        hilbert,
        graph=graph,
        t=t,
        U=U,
        dtype=float,
    )
    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    model = make_fermionic_peps_log_amplitude_model(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    sampler_kwargs = {
        "graph": graph,
        "n_chains": n_chains,
        "spin_symmetric": True,
    }
    if sampler_chunk_size is not None:
        sampler_kwargs["chunk_size"] = _check_positive_int(
            "sampler_chunk_size",
            sampler_chunk_size,
        )
    sampler = nk.sampler.MetropolisFermionHop(hilbert, **sampler_kwargs)
    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )

    preconditioner = _maybe_make_sr_preconditioner(
        ansatz,
        use_sr=use_sr,
        max_sr_params=max_sr_params,
        sr_diag_shift=sr_diag_shift,
        sr_diag_scale=sr_diag_scale,
        sr_qgt=sr_qgt,
        sr_solver=sr_solver,
        sr_solver_restart=sr_solver_restart,
    )
    return NetKetFermiHubbardVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        sampler=sampler,
        vstate=vstate,
        model=model,
        ansatz=ansatz,
        config_map=None,
        preconditioner=preconditioner,
        columns=columns,
    )


def make_netket_vmc_driver(
    setup,
    *,
    optimizer=None,
    learning_rate=0.02,
    driver="vmc",
    preconditioner="setup",
    use_sr=None,
    qgt="auto",
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_solver=None,
    sr_solver_restart=False,
    sr_mode="real",
    use_ntk=None,
    on_the_fly=None,
    chunk_size_bwd=None,
    proj_reg=None,
    momentum=None,
    linear_solver=None,
):
    """Create a NetKet VMC driver from a Pepsy/NetKet VMC setup.

    ``driver="vmc"`` returns standard NetKet ``VMC``. ``driver="vmc_sr"``
    returns NetKet's newer SR/minSR driver with explicit Jacobian mode,
    NTK/on-the-fly, and backward chunk controls.
    """
    nk = _require_netket()
    if optimizer is None:
        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    driver_name = str(driver).replace("-", "_").lower()
    if driver_name in {"vmc_sr", "vmcsr", "sr"}:
        default_preconditioner = (
            preconditioner == "setup"
            or preconditioner is None
            or preconditioner is False
        )
        if not default_preconditioner or use_sr is not None:
            warnings.warn(
                "preconditioner/use_sr is ignored when driver='vmc_sr'; "
                "VMC_SR computes the SR update internally.",
                stacklevel=2,
            )
        kwargs = {
            "diag_shift": sr_diag_shift,
            "variational_state": setup.vstate,
        }
        if proj_reg is not None:
            kwargs["proj_reg"] = proj_reg
        if momentum is not None:
            kwargs["momentum"] = momentum
        if linear_solver is not None:
            kwargs["linear_solver"] = linear_solver
        if chunk_size_bwd is not None:
            kwargs["chunk_size_bwd"] = _check_positive_int(
                "chunk_size_bwd",
                chunk_size_bwd,
            )
        if sr_mode is not None:
            kwargs["mode"] = sr_mode
        if use_ntk is not None:
            kwargs["use_ntk"] = use_ntk
        if on_the_fly is not None:
            kwargs["on_the_fly"] = on_the_fly

        return nk.driver.VMC_SR(
            setup.hamiltonian,
            optimizer=optimizer,
            **kwargs,
        )

    if driver_name != "vmc":
        raise ValueError("driver must be 'vmc' or 'vmc_sr'.")

    kwargs = {}
    if use_sr is not None:
        preconditioner = bool(use_sr)
    if preconditioner == "setup":
        preconditioner = setup.preconditioner
    elif preconditioner is True or preconditioner == "sr":
        preconditioner = make_netket_sr_preconditioner(
            qgt=qgt,
            solver=sr_solver,
            diag_shift=sr_diag_shift,
            diag_scale=sr_diag_scale,
            solver_restart=sr_solver_restart,
        )
    elif preconditioner is None or preconditioner is False:
        preconditioner = None

    if preconditioner is not None:
        kwargs["preconditioner"] = preconditioner
    return nk.VMC(
        setup.hamiltonian,
        optimizer=optimizer,
        variational_state=setup.vstate,
        **kwargs,
    )
