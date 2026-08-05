"""NetKet bridge for PEPS VMC.

This module keeps NetKet/JAX/Flax/Symmray optional. Importing it requires those
packages only when the concrete helpers are used.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
import os
import time
from typing import Any
import warnings

import autoray as ar
import numpy as np
import quimb.tensor as qtn

__all__ = [
    "NetKetLocalConfigMap",
    "NetKetChunkSettings",
    "NetKetBuildTiming",
    "NetKetAmplitudeTiming",
    "NetKetGPUUsage",
    "NetKetResourceUsage",
    "NetKetVMCConfig",
    "NetKetEtaPairObservable",
    "NetKetPEPSVMC",
    "NetKetFermiHubbardVMC",
    "NetKetVMCSetup",
    "NetKetSparseFermiHubbardVMC",
    "NetKetVMCSettings",
    "PackedPEPS",
    "PackedFermionicPEPS",
    "SpinOrbitalColumns",
    "build_heisenberg_vmc",
    "build_ising_vmc",
    "build_fermi_hubbard_vmc",
    "build_fermion_vmc",
    "build_netket_vmc",
    "build_sparse_fermi_hubbard_vmc",
    "fermionic_peps_rand",
    "fermion_model_terms",
    "netket_fermion_operator",
    "compile_operator_sum_netket",
    "standard_fermion_observables",
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
    "prepare_fermionic_peps_for_netket",
    "recommend_netket_vmc_settings",
    "square_lattice_edges",
    "verify_netket_spin_columns",
    "VMCOptimizeResult",
    "warmup_netket_vmc",
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
    symmray_symmetry: str | None = None

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
class NetKetBuildTiming:
    """Wall-clock breakdown for :func:`build_fermion_vmc`.

    The setup phase is separate from :meth:`NetKetPEPSVMC.warmup`: it creates
    the Hilbert space, packs the PEPS, builds the JAX model, and constructs the
    sampler/state.  ``warmup`` then triggers the lazy sampler/amplitude/energy
    compilations.  Keeping these timings separate makes the first-run cost
    visible instead of attributing it all to a VMC iteration.
    """

    settings_seconds: float
    geometry_seconds: float
    hamiltonian_seconds: float
    peps_seconds: float
    model_seconds: float
    sampler_seconds: float
    total_seconds: float
    preconditioner_seconds: float = 0.0
    state_seconds: float = 0.0

    def as_dict(self):
        """Return a JSON-friendly phase-to-seconds mapping."""
        return {
            "settings_seconds": float(self.settings_seconds),
            "geometry_seconds": float(self.geometry_seconds),
            "hamiltonian_seconds": float(self.hamiltonian_seconds),
            "peps_seconds": float(self.peps_seconds),
            "model_seconds": float(self.model_seconds),
            "sampler_seconds": float(self.sampler_seconds),
            "total_seconds": float(self.total_seconds),
            "preconditioner_seconds": float(self.preconditioner_seconds),
            "state_seconds": float(self.state_seconds),
        }

    @property
    def slowest_phase(self):
        """Return ``(name, seconds)`` for the slowest setup phase."""
        phases = self.as_dict()
        phases.pop("total_seconds")
        name = max(phases, key=phases.get)
        return name, phases[name]


@dataclass(frozen=True)
class NetKetAmplitudeTiming:
    """Wall-clock timing for a synchronized PEPS amplitude batch."""

    n_samples: int
    amplitude_seconds: float
    compile_seconds: float | None = None

    @property
    def amplitude_seconds_per_sample(self):
        """Return the measured average time per amplitude in the batch."""
        return float(self.amplitude_seconds) / int(self.n_samples)

    def as_dict(self):
        """Return a JSON-friendly timing mapping."""
        return {
            "n_samples": int(self.n_samples),
            "amplitude_seconds": float(self.amplitude_seconds),
            "amplitude_seconds_per_sample": float(
                self.amplitude_seconds_per_sample
            ),
            "compile_seconds": (
                None
                if self.compile_seconds is None
                else float(self.compile_seconds)
            ),
        }


@dataclass(frozen=True)
class NetKetGPUUsage:
    """One ``nvidia-smi`` snapshot for a GPU visible to the current process."""

    index: int
    name: str
    memory_used_mib: int | None
    memory_total_mib: int | None
    utilization_percent: int | None
    memory_utilization_percent: int | None
    process_memory_mib: int | None = None

    def as_dict(self):
        """Return a JSON-friendly representation of this GPU snapshot."""
        return {
            "index": self.index,
            "name": self.name,
            "memory_used_mib": self.memory_used_mib,
            "memory_total_mib": self.memory_total_mib,
            "utilization_percent": self.utilization_percent,
            "memory_utilization_percent": self.memory_utilization_percent,
            "process_memory_mib": self.process_memory_mib,
        }


@dataclass(frozen=True)
class NetKetResourceUsage:
    """Host and GPU resource snapshots collected around one VMC operation.

    ``gpu_peak`` is sampled while the operation runs. GPU utilization is an
    instantaneous NVML/``nvidia-smi`` quantity, so its peak is useful for
    observing activity, but is not a time-averaged utilization percentage.
    ``process_memory_mib`` distinguishes this notebook kernel's allocation
    from memory consumed by other processes sharing the device.
    """

    elapsed_seconds: float
    host_rss_before_mib: float | None
    host_rss_after_mib: float | None
    host_rss_peak_mib: float | None
    gpu_before: tuple[NetKetGPUUsage, ...] = ()
    gpu_after: tuple[NetKetGPUUsage, ...] = ()
    gpu_peak: tuple[NetKetGPUUsage, ...] = ()
    host_monitor: str | None = None

    def as_dict(self):
        """Return a JSON-friendly representation suitable for sample metadata."""
        return {
            "elapsed_seconds": float(self.elapsed_seconds),
            "host_rss_before_mib": self.host_rss_before_mib,
            "host_rss_after_mib": self.host_rss_after_mib,
            "host_rss_peak_mib": self.host_rss_peak_mib,
            "gpu_before": tuple(item.as_dict() for item in self.gpu_before),
            "gpu_after": tuple(item.as_dict() for item in self.gpu_after),
            "gpu_peak": tuple(item.as_dict() for item in self.gpu_peak),
            "host_monitor": self.host_monitor,
        }

    def summary(self, label="VMC resources"):
        """Format a compact human-readable summary for notebooks and logs."""
        def gib(value):
            return "n/a" if value is None else f"{value / 1024:.2f} GiB"

        parts = [
            f"{label}: {self.elapsed_seconds:.1f}s",
            "host RSS "
            f"{gib(self.host_rss_before_mib)} -> {gib(self.host_rss_after_mib)} "
            f"(peak {gib(self.host_rss_peak_mib)})",
        ]
        after = {item.index: item for item in self.gpu_after}
        before = {item.index: item for item in self.gpu_before}
        peak = {item.index: item for item in self.gpu_peak}
        for index in sorted(set(before) | set(after) | set(peak)):
            current = after.get(index) or before.get(index) or peak[index]
            highest = peak.get(index, current)
            process_memory = current.process_memory_mib
            process_peak = highest.process_memory_mib
            current_utilization = current.utilization_percent
            peak_utilization = highest.utilization_percent
            device_memory = (
                "n/a"
                if current.memory_used_mib is None
                or current.memory_total_mib is None
                else f"{current.memory_used_mib / 1024:.2f}/"
                f"{current.memory_total_mib / 1024:.2f} GiB"
            )
            parts.append(
                f"GPU {index} ({current.name}): process {gib(process_memory)} "
                f"(peak {gib(process_peak)}), device {device_memory}, "
                f"util {current_utilization if current_utilization is not None else 'n/a'}% "
                f"(peak {peak_utilization if peak_utilization is not None else 'n/a'}%)"
            )
        if not after and not before and not peak:
            parts.append("GPU metrics unavailable (nvidia-smi not visible)")
        return "; ".join(parts)


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
class NetKetVMCConfig:
    """Validated construction/runtime settings for :func:`build_fermion_vmc`.

    This bundle keeps physics inputs (PEPS, ``Fermion``, and Hamiltonian)
    separate from numerical settings without removing the builder's legacy
    keyword arguments. ``sampling`` controls the MCState defaults; call
    ``setup.sample(...)`` later to request a different retained batch.
    """

    contraction: Any = "exact"
    sampling: Any = None
    sampler_sweep_size: int | None = None
    conserving: bool | str = False
    use_sr: bool | str = False
    param_dtype: Any | None = None
    verify_columns: bool = False
    progress: bool = False

    def __post_init__(self):
        from .api import SamplingConfig

        if self.sampling is not None and not isinstance(self.sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if self.sampler_sweep_size is not None:
            _check_positive_int("sampler_sweep_size", self.sampler_sweep_size)
        if self.conserving not in {False, True, "auto"}:
            raise ValueError("conserving must be False, True, or 'auto'.")
        if self.use_sr not in {False, True, "auto"}:
            raise ValueError("use_sr must be False, True, or 'auto'.")
        if not isinstance(self.verify_columns, bool):
            raise TypeError("verify_columns must be a bool.")
        if not isinstance(self.progress, bool):
            raise TypeError("progress must be a bool.")


@dataclass(frozen=True)
class NetKetEtaPairObservable:
    r"""Declarative eta-pair observable resolved by NetKet measurement.

    This represents

    .. math::

        P_\eta(dx, dy) = \frac{1}{N}\sum_i
        \left(\Delta_i^\dagger \Delta_{i + (dx, dy)} + \mathrm{h.c.}\right).

    Pass an instance as a value in the ``observables`` mapping given to
    :meth:`NetKetPEPSVMC.measure_samples`.  The operator is compiled there
    from the packed PEPS lattice order, so it shares the retained NetKet
    configurations with every other requested observable.  At zero
    displacement it instead measures the mean double occupancy.

    ``periodic=False`` keeps only in-bounds pairs and normalizes by their
    number. ``staggered=True`` multiplies each nonzero-displacement pair by
    ``(-1)**(x_i + y_i + x_j + y_j)``.
    """

    dx: int
    dy: int
    periodic: bool = True
    staggered: bool = False

    def __post_init__(self):
        for name in ("dx", "dy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
            object.__setattr__(self, name, int(value))
        for name in ("periodic", "staggered"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool.")


def _make_progress_bar(*, total=None, desc=None, enabled=True):
    """Return a notebook-friendly ``tqdm`` bar, or ``None`` when unavailable."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover - tqdm ships with netket
        return None
    return tqdm(total=total, desc=desc, leave=True, dynamic_ncols=True)


def _block_until_ready(value):
    """Synchronize a JAX value or pytree without importing JAX eagerly."""
    wait = getattr(value, "block_until_ready", None)
    if callable(wait):
        wait()
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _block_until_ready(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _block_until_ready(item)
        return
    # ``np.asarray`` synchronizes JAX arrays and remains harmless for the
    # small NumPy/scalar stand-ins used in optional-dependency tests.
    _ = np.asarray(value)


def _tree_isfinite(value):
    """Return whether every numerical leaf in a JAX/Flax pytree is finite."""
    if isinstance(value, Mapping):
        return all(_tree_isfinite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_tree_isfinite(item) for item in value)
    array = np.asarray(value)
    try:
        return bool(np.all(np.isfinite(array)))
    except TypeError:
        # Test doubles and framework metadata are occasionally carried next
        # to numerical leaves; only numerical parameter arrays need checking.
        return True


def _process_rss_mib():
    """Return this Python process's resident memory, if ``psutil`` is present."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1 << 20)
    except Exception:
        return None


def _nvidia_smi_integer(value):
    """Parse a possibly unavailable ``nvidia-smi`` integer field."""
    value = value.strip()
    if value.lower() in {"", "n/a", "[not supported]"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _nvidia_smi_gpu_usage():
    """Return per-GPU memory/utilization snapshots without a Python NVML dep."""
    import subprocess

    try:
        gpu_process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.total,"
                "utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if gpu_process.returncode or not gpu_process.stdout.strip():
        return ()

    process_memory_by_uuid = {}
    try:
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if applications.returncode == 0:
            for line in applications.stdout.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) != 3 or _nvidia_smi_integer(fields[0]) != os.getpid():
                    continue
                process_memory_by_uuid[fields[1]] = _nvidia_smi_integer(fields[2])
    except (OSError, subprocess.SubprocessError):
        pass

    result = []
    for line in gpu_process.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        index = _nvidia_smi_integer(fields[0])
        if index is None:
            continue
        result.append(
            NetKetGPUUsage(
                index=index,
                name=fields[2],
                memory_used_mib=_nvidia_smi_integer(fields[3]),
                memory_total_mib=_nvidia_smi_integer(fields[4]),
                utilization_percent=_nvidia_smi_integer(fields[5]),
                memory_utilization_percent=_nvidia_smi_integer(fields[6]),
                process_memory_mib=process_memory_by_uuid.get(fields[1]),
            )
        )
    return tuple(result)


class _NetKetResourceMonitor:
    """Sample host RSS and ``nvidia-smi`` while one blocking VMC call runs."""

    def __init__(self, *, interval=0.25):
        interval = float(interval)
        if interval <= 0:
            raise ValueError("resource_interval must be positive.")
        self.interval = interval
        self._started = None
        self._host_before = None
        self._gpu_before = ()
        self._gpu_peak = {}
        self._host_monitor = None
        self._host_monitor_name = None
        self._stop_event = None
        self._thread = None

    def _record_gpu_usage(self, usage):
        for item in usage:
            previous = self._gpu_peak.get(item.index)
            if previous is None:
                self._gpu_peak[item.index] = item
                continue
            self._gpu_peak[item.index] = NetKetGPUUsage(
                index=item.index,
                name=item.name,
                memory_used_mib=max(
                    value
                    for value in (previous.memory_used_mib, item.memory_used_mib)
                    if value is not None
                )
                if any(
                    value is not None
                    for value in (previous.memory_used_mib, item.memory_used_mib)
                )
                else None,
                memory_total_mib=item.memory_total_mib
                or previous.memory_total_mib,
                utilization_percent=max(
                    value
                    for value in (previous.utilization_percent, item.utilization_percent)
                    if value is not None
                )
                if any(
                    value is not None
                    for value in (
                        previous.utilization_percent,
                        item.utilization_percent,
                    )
                )
                else None,
                memory_utilization_percent=max(
                    value
                    for value in (
                        previous.memory_utilization_percent,
                        item.memory_utilization_percent,
                    )
                    if value is not None
                )
                if any(
                    value is not None
                    for value in (
                        previous.memory_utilization_percent,
                        item.memory_utilization_percent,
                    )
                )
                else None,
                process_memory_mib=max(
                    value
                    for value in (
                        previous.process_memory_mib,
                        item.process_memory_mib,
                    )
                    if value is not None
                )
                if any(
                    value is not None
                    for value in (
                        previous.process_memory_mib,
                        item.process_memory_mib,
                    )
                )
                else None,
            )

    def _poll(self):
        while not self._stop_event.wait(self.interval):
            self._record_gpu_usage(_nvidia_smi_gpu_usage())

    def start(self):
        """Start non-intrusive resource sampling."""
        import threading

        self._host_before = _process_rss_mib()
        self._gpu_before = _nvidia_smi_gpu_usage()
        self._record_gpu_usage(self._gpu_before)
        # xyzpy already supplies a low-overhead RSS peak sampler.  It remains
        # optional so the NetKet bridge has no new hard runtime dependency.
        try:
            import xyzpy

            # Keep host RSS sampling responsive at teardown; GPU polling still
            # follows ``resource_interval`` and is the expensive operation.
            self._host_monitor = xyzpy.MemoryMonitor(
                interval=min(self.interval, 0.05)
            )
            self._host_monitor.start()
            self._host_monitor_name = "xyzpy.MemoryMonitor"
        except Exception:
            self._host_monitor = None
        # Exclude snapshot/monitor setup from the reported operation time.
        self._started = time.perf_counter()
        if self._gpu_before:
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        """Stop sampling and return the collected resource report."""
        if self._started is None:
            raise RuntimeError("Resource monitor has not been started.")
        ended = time.perf_counter()
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        gpu_after = _nvidia_smi_gpu_usage()
        self._record_gpu_usage(gpu_after)
        host_after = _process_rss_mib()
        host_peak = max(
            value
            for value in (self._host_before, host_after)
            if value is not None
        ) if any(value is not None for value in (self._host_before, host_after)) else None
        if self._host_monitor is not None:
            self._host_monitor.stop()
            if self._host_monitor.peak is not None:
                host_peak = max(host_peak or 0.0, self._host_monitor.peak * 1024)
        return NetKetResourceUsage(
            elapsed_seconds=ended - self._started,
            host_rss_before_mib=self._host_before,
            host_rss_after_mib=host_after,
            host_rss_peak_mib=host_peak,
            gpu_before=self._gpu_before,
            gpu_after=gpu_after,
            gpu_peak=tuple(
                self._gpu_peak[index] for index in sorted(self._gpu_peak)
            ),
            host_monitor=self._host_monitor_name,
        )


def _netket_sampling_diagnostics(vstate, sampler):
    """Read portable sampling metadata from a NetKet MCState."""
    state = getattr(vstate, "sampler_state", None)

    def scalar(value):
        if value is None:
            return None
        try:
            return float(np.asarray(value))
        except (TypeError, ValueError):
            return None

    acceptance = scalar(getattr(state, "acceptance", None))
    n_steps = scalar(getattr(state, "n_steps", None))
    n_accepted = scalar(getattr(state, "n_accepted", None))
    return {
        "n_samples": int(getattr(vstate, "n_samples", 0)),
        "n_chains": int(getattr(sampler, "n_chains", 0)),
        "burn_in": int(getattr(vstate, "n_discard_per_chain", 0) or 0),
        "sweep_size": getattr(sampler, "sweep_size", None),
        "chunk_size": getattr(vstate, "chunk_size", None),
        "acceptance_rate": acceptance,
        "n_steps": None if n_steps is None else int(n_steps),
        "n_accepted": None if n_accepted is None else int(n_accepted),
    }


def _format_sampling_postfix(diagnostics):
    """Format compact chain diagnostics for a progress-bar postfix."""
    parts = []
    acceptance = diagnostics.get("acceptance_rate")
    if acceptance is not None:
        parts.append(f"acc={acceptance:.2f}")
    n_samples = diagnostics.get("n_samples")
    n_chains = diagnostics.get("n_chains")
    if n_samples and n_chains:
        parts.append(f"samples={n_samples}/{n_chains}ch")
    burn_in = diagnostics.get("burn_in")
    sweep_size = diagnostics.get("sweep_size")
    if burn_in is not None:
        parts.append(f"burn={burn_in}")
    if sweep_size is not None:
        parts.append(f"sweep={sweep_size}")
    return " | ".join(parts)


@dataclass(frozen=True)
class VMCOptimizeResult:
    """Energy-optimization history returned by :meth:`NetKetPEPSVMC.optimize`.

    ``energies``/``errors``/``variances`` are per-step Monte-Carlo estimates
    (real energy mean, error of the mean, and sample variance). ``energy_shift``
    is added to every energy by :attr:`shifted_energies` and :meth:`plot` so a
    convention offset (for example ``-U/4`` for Fermi-Hubbard) can be applied
    without mutating the raw samples. ``compile_seconds`` measures the staged
    sampler/amplitude/energy warmup; ``optimization_seconds`` and
    ``total_seconds`` are wall-clock measurements for the run.
    """

    steps: Any
    energies: Any
    errors: Any
    variances: Any
    final_energy: float
    final_error: float
    compile_seconds: float | None = None
    energy_shift: float = 0.0
    optimization_seconds: float | None = None
    total_seconds: float | None = None

    @property
    def warmup_seconds(self):
        """Alias for the one-time sampler/JAX compilation duration."""
        return self.compile_seconds

    @property
    def optimization_seconds_per_step(self):
        """Return the measured optimization wall time per recorded step."""
        if self.optimization_seconds is None or len(np.asarray(self.steps)) == 0:
            return None
        return float(self.optimization_seconds) / len(np.asarray(self.steps))

    @property
    def shifted_energies(self):
        """Energies with ``energy_shift`` added."""
        return np.asarray(self.energies, dtype=float) + float(self.energy_shift)

    def plot(self, ax=None, *, per_site=None, reference=None, reference_label=None):
        """Plot a clean energy-vs-iteration curve with a Monte-Carlo error band."""
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        if ax is None:
            _, ax = plt.subplots(figsize=(7.0, 4.5))
        steps = np.asarray(self.steps)
        y = self.shifted_energies
        band = np.asarray(self.errors, dtype=float)
        scale = 1.0 if not per_site else float(per_site)
        y = y / scale
        band = band / scale
        ax.plot(steps, y, "-", color="#1f77b4", lw=1.8, label="VMC energy")
        ax.fill_between(steps, y - band, y + band, color="#1f77b4", alpha=0.2)
        if reference is not None:
            ax.axhline(
                float(reference),
                ls="--",
                color="k",
                lw=1.4,
                label=reference_label or "reference",
            )
        ax.set_xlabel("iteration", fontsize=13)
        ax.set_ylabel("energy / site" if per_site else "energy", fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        return ax


class _VMCProgressCallback:
    """NetKet callback: one live ``tqdm`` energy bar that records the history."""

    def __init__(self, n_iter, *, enabled=True, energy_shift=0.0, per_site=None):
        self.n_iter = int(n_iter)
        self.enabled = enabled
        self.energy_shift = float(energy_shift)
        self.per_site = per_site
        self._bar = None
        self._first_update_started = None
        self.steps = []
        self.energies = []
        self.errors = []
        self.variances = []

    def _ensure_bar(self):
        if self._bar is None and self.enabled:
            self._bar = _make_progress_bar(
                total=self.n_iter, desc="VMC energy", enabled=True
            )
        return self._bar

    def start(self, status="first update: sampling and compiling gradients"):
        """Show the VMC bar before NetKet's first (potentially JIT-heavy) step."""
        self._first_update_started = time.perf_counter()
        bar = self._ensure_bar()
        if bar is not None:
            bar.set_description_str(f"VMC 0/{self.n_iter}: preparing")
            bar.set_postfix_str(str(status))

    def set_status(self, status):
        """Update the pre-first-step status without advancing the bar."""
        bar = self._ensure_bar()
        if bar is not None and not self.steps:
            bar.set_postfix_str(str(status))

    def __call__(self, step, log_data, driver):
        # NetKet invokes legacy callbacks after ``update_parameters``, but JAX
        # may still have the backward pass and optimizer update queued on the
        # device. Synchronize the updated parameter pytree before advancing
        # tqdm: otherwise the first energy statistic can tick while the GPU is
        # still working on that update, and the apparent stall moves to the
        # following bar item. VMC updates are parameter-dependent, so this
        # does not remove useful inter-step GPU parallelism; it makes elapsed
        # time and ETA honest.
        vstate = getattr(driver, "variational_state", None)
        if vstate is None:
            vstate = getattr(driver, "_variational_state", None)
        if vstate is not None:
            parameters = getattr(vstate, "parameters", None)
            if parameters is not None:
                _block_until_ready(parameters)
            if parameters is not None and not _tree_isfinite(parameters):
                raise FloatingPointError(
                    "NetKet VMC produced non-finite PEPS parameters after "
                    f"update {int(step)}. Reduce the learning rate, restart "
                    "from the pre-update state, and verify a short run before "
                    "using a larger sample count."
                )

        name = getattr(driver, "_loss_name", "Energy")
        stats = None
        if isinstance(log_data, dict):
            stats = log_data.get(name)
        if stats is None:
            stats = getattr(driver, "_loss_stats", None)
        if stats is not None:
            mean = float(np.real(np.asarray(getattr(stats, "mean", stats))))
            err = float(getattr(stats, "error_of_mean", np.nan))
            var = float(getattr(stats, "variance", np.nan))
            if not np.isfinite(mean):
                raise FloatingPointError(
                    "NetKet VMC produced a non-finite local-energy estimate "
                    f"at update {int(step)}. The PEPS/sampler state is no "
                    "longer numerically usable; reduce the learning rate and "
                    "restart from the initial PEPS."
                )
            self.steps.append(int(step))
            self.energies.append(mean)
            self.errors.append(err)
            self.variances.append(var)
            bar = self._ensure_bar()
            if bar is not None:
                scale = 1.0 if not self.per_site else float(self.per_site)
                shown = (mean + self.energy_shift) / scale
                details = [f"E={shown:.6f}\u00b1{err / scale:.1e}"]
                if self._first_update_started is not None:
                    first_seconds = time.perf_counter() - self._first_update_started
                    details.insert(0, f"first update {first_seconds:.1f}s")
                    self._first_update_started = None
                if vstate is not None:
                    sampling = _netket_sampling_diagnostics(
                        vstate, getattr(vstate, "sampler", None)
                    )
                    sampling_text = _format_sampling_postfix(sampling)
                    if sampling_text:
                        details.append(sampling_text)
                bar.set_description_str("VMC energy")
                bar.set_postfix_str(" | ".join(details))
                bar.update(1)
        return True

    def close(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def result(
        self,
        compile_seconds=None,
        optimization_seconds=None,
        total_seconds=None,
    ):
        energies = np.asarray(self.energies, dtype=float)
        errors = np.asarray(self.errors, dtype=float)
        return VMCOptimizeResult(
            steps=np.asarray(self.steps, dtype=int),
            energies=energies,
            errors=errors,
            variances=np.asarray(self.variances, dtype=float),
            final_energy=float(energies[-1]) if energies.size else float("nan"),
            final_error=float(errors[-1]) if errors.size else float("nan"),
            compile_seconds=compile_seconds,
            optimization_seconds=optimization_seconds,
            total_seconds=total_seconds,
            energy_shift=self.energy_shift,
        )


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
    build_timing: NetKetBuildTiming | None = None

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

    def _mc_diagnostic_method(self, name):
        """Return a recent NetKet MC diagnostic or raise a clear version error."""
        method = getattr(self.vstate, name, None)
        if callable(method):
            return method
        raise RuntimeError(
            f"NetKet MC diagnostic {name}() is unavailable in this NetKet "
            "version. Install NetKet >= 3.22 to use it."
        )

    def check_mc_convergence(self, hamiltonian=None, **kwargs):
        """Diagnose final-state mixing without mutating the VMC state.

        This delegates to NetKet's ``MCState.check_mc_convergence``. It is
        intentionally an explicit post-optimization diagnostic because it
        draws long chains to estimate :math:`\\hat R` and autocorrelation time.
        """
        if hamiltonian is None:
            hamiltonian = self.hamiltonian
        return self._mc_diagnostic_method("check_mc_convergence")(
            hamiltonian, **kwargs
        )

    def thermalise(self, hamiltonian=None, **kwargs):
        """Advance chains in place until NetKet's mixing criterion is met."""
        if hamiltonian is None:
            hamiltonian = self.hamiltonian
        return self._mc_diagnostic_method("thermalise")(hamiltonian, **kwargs)

    def expect_to_precision(self, observable=None, **kwargs):
        """Sample an observable until NetKet reaches a requested tolerance."""
        if observable is None:
            observable = self.hamiltonian
        return self._mc_diagnostic_method("expect_to_precision")(
            observable, **kwargs
        )

    def make_driver(self, **kwargs):
        """Create a NetKet VMC driver for this setup.

        This is a convenience wrapper around :func:`make_netket_vmc_driver`.
        """
        return make_netket_vmc_driver(self, **kwargs)

    def warmup(
        self,
        *,
        progress=True,
        hamiltonian=None,
        verbose=None,
        resource_monitor=False,
        resource_interval=0.25,
    ):
        """Compile the VMC kernels up front with a small staged progress bar.

        Thin wrapper over :func:`warmup_netket_vmc`; returns the elapsed
        compile seconds so the following optimization ETA is meaningful.
        """
        return warmup_netket_vmc(
            self,
            hamiltonian=hamiltonian,
            progress=progress,
            verbose=verbose,
            resource_monitor=resource_monitor,
            resource_interval=resource_interval,
        )

    def to_peps(self, variables=None, *, device_get=True):
        """Return the current NetKet parameters as a quimb PEPS-like network.

        With no explicit ``variables``, this reconstructs the PEPS from the
        setup's current ``MCState`` parameters, including parameters updated by
        a NetKet VMC driver.  Pass either the full Flax variables mapping or
        its ``"params"`` collection to inspect a checkpoint without mutating
        the VMC setup.  By default leaves are copied from JAX devices to NumPy,
        yielding a regular quimb/Symmray network suitable for Pepsy methods;
        set ``device_get=False`` to retain JAX-backed leaves.

        The result is the underlying quimb network (the ``.tn`` of a
        :class:`pepsy.SymPEPS`), preserving the packed skeleton's tensor
        topology, indices, and fermionic metadata.
        """
        if variables is None:
            variables = getattr(self.vstate, "variables", None)
            if variables is None:
                variables = {"params": self.vstate.parameters}
        params = _peps_params_from_netket_variables(
            self.ansatz,
            variables,
            device_get=device_get,
        )
        return qtn.unpack(params, self.ansatz.skeleton)

    def benchmark_amplitude(self, configs=None, *, n_samples=1):
        """Time one synchronized jitted PEPS amplitude batch.

        If ``configs`` is omitted, the first ``n_samples`` configurations from
        the current NetKet sample cache are used. Sampling itself is outside
        the timed region, so the result isolates NetKet's jitted log-amplitude
        evaluation. The first synchronized call is treated as a compile probe
        and is reported separately in ``compile_seconds``. The returned
        ``amplitude_seconds`` is a second call with the same batch shape, so
        it measures steady-state evaluation rather than shape-specific JAX
        compilation. Choose ``n_samples`` to match a real sampler or forward
        chunk; timing an arbitrary small batch does not predict VMC throughput.
        """
        n_samples = int(n_samples)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        if configs is None:
            native = self.vstate.samples
            configs = native.reshape((-1, native.shape[-1]))
        else:
            configs = np.asarray(configs)
            if configs.ndim == 1:
                configs = configs.reshape((1, -1))
            if configs.ndim != 2:
                raise ValueError(
                    "configs must have shape (n_samples, n_orbitals)."
                )
        if int(configs.shape[0]) < n_samples:
            raise ValueError(
                f"configs contains {int(configs.shape[0])} rows, "
                f"but n_samples={n_samples} was requested."
            )
        configs = configs[:n_samples]
        log_value = getattr(self.vstate, "log_value", None)
        if not callable(log_value):
            raise TypeError(
                "NetKet amplitude benchmarking requires an MCState with a "
                "log_value(configs) method."
            )
        compile_started = time.perf_counter()
        log_amplitudes = log_value(configs)
        _block_until_ready(log_amplitudes)
        compile_seconds = time.perf_counter() - compile_started
        started = time.perf_counter()
        log_amplitudes = log_value(configs)
        _block_until_ready(log_amplitudes)
        return NetKetAmplitudeTiming(
            n_samples=n_samples,
            amplitude_seconds=time.perf_counter() - started,
            compile_seconds=compile_seconds,
        )

    def sample(
        self,
        sampling=None,
        *,
        fresh=False,
        progress=False,
        resource_monitor=False,
        resource_interval=0.25,
    ):
        """Collect samples using the shared :class:`SamplingConfig` contract.

        NetKet stores samples as ``(n_chains, n_samples_per_chain, sites)``;
        this façade canonicalizes them to the backend-neutral
        ``(n_samples_per_chain, n_chains, sites)`` layout used by Torch.  The
        sampler's chain count is fixed when the setup is built, so a config
        requesting another count raises a clear error instead of silently
        returning a different ensemble. With ``fresh=True``, reset NetKet's
        retained-sample cache before reading it, forcing one new batch from the
        current chain state. With ``progress=True``, the bar shows
        retained samples, burn-in, sweep size, elapsed time, and NetKet's
        acceptance rate after sampling. Set ``resource_monitor=True`` to also
        print host RSS plus this kernel's GPU-memory and GPU-utilization
        snapshots; the serializable report is retained in
        ``VMCSamples.diagnostics["resources"]``.
        """
        from .api import BackendCapabilityWarning, SamplingConfig, VMCSamples

        if sampling is not None and not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if not isinstance(fresh, bool):
            raise TypeError("fresh must be a bool.")
        bar = _make_progress_bar(total=1, desc="NetKet sampling", enabled=progress)
        monitor = (
            _NetKetResourceMonitor(interval=resource_interval)
            if resource_monitor
            else None
        )
        if monitor is not None:
            monitor.start()
        # Keep the sampling time independent of optional telemetry setup.
        started = time.perf_counter()
        try:
            if fresh:
                self.vstate.reset()
            if sampling is None:
                native = self.vstate.samples
            else:
                actual_chains = getattr(self.sampler, "n_chains", None)
                if (
                    actual_chains is not None
                    and int(actual_chains) != sampling.n_chains
                ):
                    raise ValueError(
                        "sampling.n_chains does not match the setup sampler: "
                        f"expected {int(actual_chains)}, got {sampling.n_chains}. "
                        "Rebuild the setup with n_chains=... to change it."
                    )
                if sampling.sweep_size != 1:
                    warnings.warn(
                        "NetKet MCState.sample has no per-sample thinning option; "
                        "sampling.sweep_size is ignored.",
                        BackendCapabilityWarning,
                        stacklevel=2,
                    )
                if sampling.proposal is not None:
                    warnings.warn(
                        "sampling.proposal is selected when the NetKet sampler is "
                        "built and is ignored by MCState.sample.",
                        BackendCapabilityWarning,
                        stacklevel=2,
                    )
                if sampling.seed is not None or sampling.sampler_seed is not None:
                    warnings.warn(
                        "NetKet MCState.sample cannot reseed an existing sampler; "
                        "seed settings are ignored. Rebuild the setup to reseed.",
                        BackendCapabilityWarning,
                        stacklevel=2,
                    )
                kwargs = sampling.netket_kwargs()
                kwargs.pop("n_chains", None)
                requested_chunk = sampling.chunk_size
                old_chunk = getattr(self.vstate, "chunk_size", None)
                if requested_chunk is not None:
                    self.vstate.chunk_size = requested_chunk
                try:
                    native = self.vstate.sample(**kwargs)
                finally:
                    if requested_chunk is not None:
                        self.vstate.chunk_size = old_chunk
        finally:
            resource_usage = monitor.stop() if monitor is not None else None

        diagnostics = _netket_sampling_diagnostics(self.vstate, self.sampler)
        # ``MCState.sample`` accepts a temporary sample count/discard value,
        # but does not consistently copy those values back onto the state.
        # Keep the progress text and returned metadata faithful to the request
        # rather than reporting the build-time defaults in that case.
        if sampling is not None:
            diagnostics["n_samples"] = sampling.n_samples
            diagnostics["n_chains"] = sampling.n_chains
            diagnostics["burn_in"] = sampling.n_discard_per_chain
            if sampling.chunk_size is not None:
                diagnostics["chunk_size"] = sampling.chunk_size
        diagnostics["elapsed_seconds"] = time.perf_counter() - started
        if resource_usage is not None:
            diagnostics["resources"] = resource_usage.as_dict()
            print(resource_usage.summary("NetKet sampling resources"), flush=True)
        if bar is not None:
            postfix = _format_sampling_postfix(diagnostics)
            if postfix:
                bar.set_postfix_str(postfix)
            bar.update(1)
            bar.close()

        shape = getattr(native, "shape", ())
        if len(shape) == 3:
            # NetKet's native order is chains first.
            configs = native.swapaxes(0, 1)
            n_samples_per_chain, n_chains = int(shape[1]), int(shape[0])
        elif len(shape) == 2:
            configs = native.reshape((int(shape[0]), 1, int(shape[1])))
            n_samples_per_chain, n_chains = int(shape[0]), 1
        else:
            raise ValueError(
                "NetKet samples must have shape (chains, samples, sites) or "
                f"(samples, sites), got {shape}."
            )
        diagnostics["n_samples_per_chain"] = n_samples_per_chain
        return VMCSamples(
            configs=configs,
            n_samples_per_chain=n_samples_per_chain,
            n_chains=n_chains,
            native=native,
            acceptance_rate=diagnostics.get("acceptance_rate"),
            diagnostics=diagnostics,
        )

    def optimize(
        self,
        n_iter=None,
        *,
        optimization=None,
        learning_rate=None,
        driver=None,
        optimizer=None,
        progress=None,
        warmup=None,
        energy_shift=None,
        per_site=None,
        extra_callbacks=None,
        driver_options=None,
        **run_kwargs,
    ):
        """Run VMC energy optimization with a single clean progress bar.

        Builds a driver (see :func:`make_netket_vmc_driver`), optionally warms
        up XLA compilation (:meth:`warmup`), then runs ``n_iter`` steps while a
        live ``tqdm`` bar shows the current energy. NetKet's own bar is
        disabled so there is exactly one bar.

        ``energy_shift``/``per_site`` only affect the displayed and returned
        energies (for example ``energy_shift=-U/4`` and ``per_site=n_sites``
        for the Fermi-Hubbard convention); the raw Monte-Carlo means are kept.
        Extra NetKet callbacks (early stopping, timeout, ...) can be passed via
        ``extra_callbacks``. Returns a :class:`VMCOptimizeResult`.
        """
        if optimization is not None:
            from .api import OptimizationConfig, VMCBackendCapabilityError
            if not isinstance(optimization, OptimizationConfig):
                raise TypeError("optimization must be an OptimizationConfig or None.")
            if optimization.method == "minsr":
                raise VMCBackendCapabilityError(
                    "OptimizationConfig(method='minsr') is Torch-specific. "
                    "Use method='sr' for portable stochastic reconfiguration, "
                    "or call make_netket_vmc_driver(driver='vmc_sr') directly."
                )
            if n_iter is not None and n_iter != optimization.n_steps:
                raise ValueError("n_iter conflicts with optimization.n_steps.")
            n_iter = optimization.n_steps
            if learning_rate is None:
                learning_rate = optimization.learning_rate
            if progress is None:
                progress = optimization.progress
            if warmup is None:
                warmup = optimization.warmup
            if energy_shift is None:
                energy_shift = optimization.energy_shift
            if per_site is None:
                per_site = optimization.per_site
            if driver is None:
                driver = "vmc" if optimization.method == "sgd" else "vmc_sr"
            driver_options = {} if driver_options is None else dict(driver_options)
            if driver != "vmc":
                driver_options.setdefault("sr_mode", optimization.sr_mode)
                driver_options.setdefault("sr_diag_shift", optimization.diag_shift)
        if n_iter is None:
            raise TypeError("n_iter is required unless optimization is supplied.")
        if learning_rate is None:
            learning_rate = 0.02
        if driver is None:
            driver = "vmc"
        if progress is None:
            progress = True
        if warmup is None:
            warmup = True
        if energy_shift is None:
            energy_shift = 0.0
        driver_options = {} if driver_options is None else dict(driver_options)
        total_started = time.perf_counter()
        cb = _VMCProgressCallback(
            n_iter,
            enabled=progress,
            energy_shift=energy_shift,
            per_site=per_site,
        )
        callbacks = [cb]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        # When a separate warmup has already completed (the usual notebook
        # path), even driver setup would otherwise be invisible.
        if not warmup:
            cb.start("building VMC driver")
        run_driver = make_netket_vmc_driver(
            self,
            optimizer=optimizer,
            learning_rate=learning_rate,
            driver=driver,
            **driver_options,
        )
        compile_seconds = None
        if warmup:
            compile_seconds = self.warmup(progress=progress)
        # NetKet invokes callbacks only after its first update.  Create this
        # bar now so the JIT-heavy first gradient/sampling pass is visible.
        cb.start()
        optimization_started = time.perf_counter()
        try:
            run_driver.run(
                n_iter,
                show_progress=False,
                callback=callbacks,
                **run_kwargs,
            )
        finally:
            cb.close()
        optimization_seconds = time.perf_counter() - optimization_started
        total_seconds = time.perf_counter() - total_started
        return cb.result(
            compile_seconds=compile_seconds,
            optimization_seconds=optimization_seconds,
            total_seconds=total_seconds,
        )

    def run(
        self,
        n_iter,
        *,
        learning_rate=0.02,
        driver="vmc",
        optimizer=None,
        warmup=True,
        progress=True,
        energy_shift=0.0,
        per_site=None,
        sr_mode="real",
        sr_diag_shift=0.01,
        use_sr=False,
        driver_options=None,
        **run_kwargs,
    ):
        """Run NetKet VMC with one timing-aware public entry point.

        This is the concise native setup API: contraction and sampler choices
        belong to :func:`build_fermion_vmc`, while optimization choices are
        passed directly here. Standard ``driver='vmc'`` uses no SR
        preconditioner unless ``use_sr=True`` or an explicit driver option is
        supplied. The returned :class:`VMCOptimizeResult` reports
        one-time warmup/JAX compilation, optimization wall time, total wall
        time, and the measured time per optimization step.
        """
        options = {} if driver_options is None else dict(driver_options)
        driver_name = str(driver).replace("-", "_").lower()
        if driver_name in {"vmc_sr", "vmcsr", "sr"}:
            options.setdefault("sr_mode", sr_mode)
            options.setdefault("sr_diag_shift", sr_diag_shift)
        elif "use_sr" not in options and "preconditioner" not in options:
            options["use_sr"] = use_sr
        return self.optimize(
            n_iter,
            learning_rate=learning_rate,
            driver=driver,
            optimizer=optimizer,
            progress=progress,
            warmup=warmup,
            energy_shift=energy_shift,
            per_site=per_site,
            driver_options=options,
            **run_kwargs,
        )

    def measure_samples(self, samples, observables=None):
        """Measure observables from a retained NetKet sample batch.

        ``samples`` must be the :class:`~pepsy.vmc.VMCSamples` returned by
        :meth:`sample` (or its unchanged native NetKet array). NetKet's local
        estimators consume the ``MCState`` sample cache rather than accepting
        configurations as an argument, so this method verifies that the batch
        is still that cache. It then evaluates every observable on precisely
        the same Markov-chain configurations, without drawing another batch.

        A fresh ``sample(...)`` call is required after a parameter update or
        any other operation that invalidates NetKet's cache. ``observables``
        may be a single NetKet operator, a ``{name: operator}`` mapping, a
        :class:`NetKetEtaPairObservable` specification, or ``None`` to use
        those stored on the setup.
        """
        from .api import VMCBackendCapabilityError

        native_samples = getattr(samples, "native", samples)
        cached_samples = getattr(self.vstate, "_samples", None)
        if native_samples is None:
            raise VMCBackendCapabilityError(
                "NetKet measurement needs a native sample batch. Call "
                "setup.sample(...) and pass its returned VMCSamples object."
            )
        if cached_samples is None or native_samples is not cached_samples:
            raise VMCBackendCapabilityError(
                "NetKet can measure only the current MCState sample cache. "
                "Call setup.sample(...) and pass that returned batch; external "
                "or stale configurations cannot be installed safely."
            )
        if observables is None:
            observables = getattr(self, "observables", None)
        if observables is None:
            raise ValueError(
                "No observables to measure: pass observables=... or build the "
                "setup with observables=... (see build_fermion_vmc)."
            )
        if isinstance(observables, dict):
            return {
                name: self.vstate.expect(
                    _resolve_netket_measurement_observable(
                        self.hilbert,
                        self.ansatz,
                        op,
                    )
                )
                for name, op in observables.items()
            }
        return self.vstate.expect(
            _resolve_netket_measurement_observable(
                self.hilbert,
                self.ansatz,
                observables,
            )
        )

    def measure(
        self,
        observables=None,
        *,
        samples=None,
        sampling=None,
        progress=False,
    ):
        """Measure observables, optionally from an explicitly retained batch.

        ``setup.measure_samples(samples, observables)`` is the explicit
        sample-once/measure-many form. This convenience wrapper preserves the
        direct API: it uses the current NetKet cache when available, or calls
        :meth:`sample` once when a batch must be drawn. Pass ``samples=`` to
        make the sampling boundary explicit.
        """
        if samples is not None and sampling is not None:
            raise ValueError("Pass either samples or sampling, not both.")
        if samples is None:
            cached_samples = getattr(self.vstate, "_samples", None)
            if sampling is not None or cached_samples is None:
                samples = self.sample(sampling, progress=progress)
            else:
                samples = cached_samples
        return self.measure_samples(samples, observables)


@dataclass(frozen=True)
class NetKetVMCSetup:
    """Backend-neutral façade over a native :class:`NetKetPEPSVMC` setup.

    The wrapped setup keeps its NetKet drivers, state, and statistics objects
    available through :attr:`native`. This façade is deliberately strict about
    shared settings which NetKet cannot honour after construction, rather than
    warning and silently changing their meaning.
    """

    setup: NetKetPEPSVMC
    problem: Any

    @property
    def backend(self):
        """Name of the numerical backend behind this setup."""
        return "netket"

    @property
    def native(self):
        """Return the native NetKet/Flax setup for advanced control."""
        return self.setup

    @property
    def n_sites(self):
        return self.setup.n_sites

    @property
    def n_params(self):
        return self.setup.n_params

    @property
    def build_timing(self):
        """Return the setup-phase timing breakdown, when available."""
        return self.setup.build_timing

    def check_mc_convergence(self, hamiltonian=None, **kwargs):
        """Forward NetKet's post-optimization mixing diagnostic."""
        return self.setup.check_mc_convergence(hamiltonian, **kwargs)

    def thermalise(self, hamiltonian=None, **kwargs):
        """Forward NetKet's in-place chain thermalisation helper."""
        return self.setup.thermalise(hamiltonian, **kwargs)

    def expect_to_precision(self, observable=None, **kwargs):
        """Forward NetKet's precision-targeted expectation helper."""
        return self.setup.expect_to_precision(observable, **kwargs)

    def benchmark_amplitude(self, configs=None, *, n_samples=1):
        """Time a synchronized amplitude batch on the native NetKet setup."""
        return self.setup.benchmark_amplitude(configs, n_samples=n_samples)

    def sample(
        self,
        sampling=None,
        *,
        fresh=False,
        progress=False,
        resource_monitor=False,
        resource_interval=0.25,
    ):
        """Collect samples as backend-neutral :class:`VMCSamples`.

        Chain count and seeds belong to an MCState at build time in NetKet.
        The portable sampling call therefore rejects requests which would be
        ignored by an existing state.
        """
        from .api import SamplingConfig, VMCBackendCapabilityError

        if sampling is None:
            return self.setup.sample(
                fresh=fresh,
                progress=progress,
                resource_monitor=resource_monitor,
                resource_interval=resource_interval,
            )
        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        unsupported = []
        if sampling.sweep_size != 1:
            unsupported.append("sweep_size (formerly thin)")
        if sampling.seed is not None or sampling.sampler_seed is not None:
            unsupported.append("seed/sampler_seed")
        if sampling.proposal is not None:
            unsupported.append("proposal")
        if unsupported:
            joined = ", ".join(unsupported)
            raise VMCBackendCapabilityError(
                f"NetKet cannot apply SamplingConfig.{joined} after setup "
                "construction. Supply it while building the native sampler, "
                "or use the native NetKet API explicitly."
            )
        return self.setup.sample(
            sampling,
            fresh=fresh,
            progress=progress,
            resource_monitor=resource_monitor,
            resource_interval=resource_interval,
        )

    def measure(
        self,
        observables=None,
        *,
        sampling=None,
        samples=None,
        weights=None,
        proposal_log_probs=None,
    ):
        """Measure energy and optional observables on the current VMC state."""
        from .api import VMCBackendCapabilityError, VMCMeasurement

        if weights is not None or proposal_log_probs is not None:
            raise VMCBackendCapabilityError(
                "NetKet's portable adapter does not accept weighted or "
                "proposal-distribution sample batches. Use the Torch adapter "
                "for importance sampling."
            )
        if samples is not None and sampling is not None:
            raise ValueError("Pass either samples or sampling, not both.")
        if samples is None:
            samples = self.sample(sampling)
        if observables is None:
            extra = dict(getattr(self.setup, "observables", None) or {})
        else:
            try:
                extra = dict(observables)
            except (TypeError, ValueError) as exc:
                raise TypeError("observables must be a mapping of names to operators.") from exc
        if "energy" in extra:
            raise ValueError(
                "'energy' is reserved for problem.hamiltonian; use a different "
                "observable name."
            )
        native = self.setup.measure_samples(
            samples,
            {"energy": self.setup.hamiltonian, **extra},
        )
        energy = native["energy"]
        return VMCMeasurement(
            energy_mean=getattr(energy, "mean", energy),
            energy_variance=getattr(energy, "variance", None),
            energy_stderr=getattr(energy, "error_of_mean", None),
            observables=native,
            effective_sample_size=getattr(energy, "effective_sample_size", None),
            diagnostics={
                "backend": self.backend,
                "samples": samples,
            },
            native=native,
        )

    def optimize(self, optimization=None, *, n_steps=None, **kwargs):
        """Optimize and return a backend-neutral VMC history."""
        from .api import (
            OptimizationConfig,
            VMCBackendCapabilityError,
            VMCOptimizationResult,
        )

        if any(
            kwargs.get(name) is not None
            for name in ("samples", "weights", "proposal_log_probs")
        ):
            raise VMCBackendCapabilityError(
                "NetKet's portable adapter does not yet optimize from an "
                "externally supplied weighted sample batch. Use its MCState "
                "sampling path, or use the Torch adapter for importance sampling."
            )

        if optimization is not None and not isinstance(optimization, OptimizationConfig):
            raise TypeError("optimization must be an OptimizationConfig or None.")
        if optimization is not None and optimization.method == "minsr":
            raise VMCBackendCapabilityError(
                "OptimizationConfig(method='minsr') is Torch-specific. "
                "Use method='sr' for the portable NetKet path."
            )
        if optimization is not None:
            if n_steps is not None and n_steps != optimization.n_steps:
                raise ValueError("n_steps conflicts with optimization.n_steps.")
            native = self.setup.optimize(optimization=optimization, **kwargs)
            energy_shift = optimization.energy_shift
            per_site = optimization.per_site
        else:
            if n_steps is None:
                raise TypeError("n_steps is required unless optimization is supplied.")
            native = self.setup.optimize(n_steps, **kwargs)
            energy_shift = native.energy_shift
            per_site = None
        return VMCOptimizationResult(
            steps=native.steps,
            energies=native.energies,
            errors=native.errors,
            variances=native.variances,
            final_energy=native.final_energy,
            final_error=native.final_error,
            energy_shift=energy_shift,
            per_site=per_site,
            diagnostics={
                "backend": self.backend,
                "build_timing": (
                    self.setup.build_timing.as_dict()
                    if self.setup.build_timing is not None
                    else None
                ),
                "compile_seconds": native.compile_seconds,
                "warmup_seconds": native.warmup_seconds,
                "optimization_seconds": native.optimization_seconds,
                "total_seconds": native.total_seconds,
                "optimization_seconds_per_step": native.optimization_seconds_per_step,
            },
            native=native,
        )


@dataclass(frozen=True)
class NetKetFermiHubbardVMC(NetKetPEPSVMC):
    """Bundle returned by :func:`build_fermi_hubbard_vmc`."""

    columns: SpinOrbitalColumns | None = None
    observables: dict | None = None


@dataclass(frozen=True)
class NetKetSparseFermiHubbardVMC:
    """Bundle for sparse-block fermionic PEPS VMC.

    NetKet supplies the Hilbert space, graph, and Hamiltonian metadata. The
    actual samples, amplitudes, and local energies use the non-jitted torch
    PEPS path so block-sparse ``U1``, ``U1U1``, and ``Z2Z2`` Symmray tensors
    can be evaluated today.
    """

    hilbert: Any
    graph: Any
    hamiltonian: Any
    model: Any
    ansatz: PackedFermionicPEPS
    columns: SpinOrbitalColumns
    torch_graph: Any
    encoding: Any
    configs: Any
    amplitudes: Any
    n_fermions_per_spin: tuple[int, int]
    t: Any
    U: Any
    mode_order: str = "down-up"
    generator: Any | None = None

    @property
    def n_sites(self):
        """Number of spatial orbitals/sites in the PEPS and Hilbert sector."""
        return self.ansatz.n_sites

    @property
    def n_params(self):
        """Number of scalar variational parameters in the packed ansatz."""
        return self.ansatz.n_params

    def amplitudes_for(self, configs):
        """Evaluate PEPS amplitudes for site-local fermion config rows."""
        return self.model(configs)

    def local_energy(self, configs=None, amplitudes=None):
        """Return per-sample Fermi-Hubbard local energies."""
        from .torch import (
            local_energy_from_connections,
            spinful_fermi_hubbard_connections,
        )

        configs = self.configs if configs is None else configs
        amplitudes = self.amplitudes_for(configs) if amplitudes is None else amplitudes
        connections = spinful_fermi_hubbard_connections(
            configs,
            self.torch_graph,
            t=self.t,
            U=self.U,
            encoding=self.encoding,
            mode_order=self.mode_order,
        )
        return local_energy_from_connections(
            configs,
            amplitudes,
            connections,
            self.model,
        )

    def energy_estimate(self, configs=None, amplitudes=None):
        """Return the Monte-Carlo mean of :meth:`local_energy`."""
        return self.local_energy(configs=configs, amplitudes=amplitudes).mean()

    def sample(self, sampling=None):
        """Collect chain-preserving samples with the torch sparse-block path.

        This is the non-jitted counterpart of :meth:`NetKetPEPSVMC.sample`.
        It is intended for ``U1``/``U1U1`` Symmray PEPS that cannot use
        NetKet's jitted ``MCState`` amplitude kernel.
        """
        from .api import BackendCapabilityWarning, SamplingConfig
        from .torch import TorchMetropolisSampler

        if sampling is None:
            sampling = SamplingConfig(
                n_samples_per_chain=128,
                n_chains=int(self.configs.shape[0]),
                burn_in=0,
                thin=1,
            )
        if not isinstance(sampling, SamplingConfig):
            raise TypeError("sampling must be a SamplingConfig or None.")
        if sampling.proposal is not None:
            warnings.warn(
                "SamplingConfig.proposal is ignored by the sparse fermion "
                "sampler; its symmetry-preserving spinful proposal is fixed.",
                BackendCapabilityWarning,
                stacklevel=2,
            )
        n_chains = sampling.n_chains
        if self.configs.shape[0] == 1 and n_chains > 1:
            initial_configs = self.configs.expand(n_chains, -1).clone()
            initial_amplitudes = self.amplitudes.expand(n_chains).clone()
        elif self.configs.shape[0] >= n_chains:
            initial_configs = self.configs[:n_chains].clone()
            initial_amplitudes = self.amplitudes[:n_chains].clone()
        else:
            raise ValueError(
                "sampling.n_chains exceeds the sparse setup's initial walker "
                f"count ({self.configs.shape[0]}). Rebuild with n_samples >= "
                "the requested chain count."
            )
        if sampling.seed is not None and sampling.sampler_seed is not None:
            raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
        sampler = TorchMetropolisSampler(
            self.model,
            self.torch_graph,
            initial_configs,
            amplitudes=initial_amplitudes,
            n_chains=n_chains,
            proposal="spinful",
            encoding=self.encoding,
            chunk_size=sampling.chunk_size,
            seed=(
                sampling.seed
                if sampling.seed is not None
                else sampling.sampler_seed
            ),
        )
        result = sampler.sample(
            n_samples=sampling.n_samples,
            n_discard_per_chain=sampling.n_discard_per_chain,
            sweep_size=sampling.sweep_size,
        )
        object.__setattr__(self, "configs", sampler.configs)
        object.__setattr__(self, "amplitudes", sampler.amplitudes)
        object.__setattr__(self, "generator", sampler.generator)
        return result.to_common()

    def sample_sweep(
        self,
        configs=None,
        amplitudes=None,
        *,
        hopping_rate=0.25,
        generator=None,
    ):
        """Run one nearest-neighbor Metropolis sweep with the torch sampler."""
        from .torch import metropolis_exchange_sweep

        configs = self.configs if configs is None else configs
        if amplitudes is None:
            amplitudes = (
                self.amplitudes
                if configs is self.configs
                else self.amplitudes_for(configs)
            )
        if generator is None:
            generator = self.generator
        return metropolis_exchange_sweep(
            configs,
            self.model,
            self.torch_graph,
            current_amplitudes=amplitudes,
            proposal="spinful",
            hopping_rate=hopping_rate,
            encoding=self.encoding,
            generator=generator,
        )


def configure_jax_for_vmc(
    *,
    preallocate=False,
    mem_fraction=0.65,
    platform=None,
    compilation_cache_dir=None,
    disable_netket_tips=True,
):
    """Set JAX/NetKet environment defaults for notebook VMC runs.

    Call this before importing ``jax`` or ``netket``. Existing environment
    values are preserved. When ``compilation_cache_dir`` is set, it must be a
    private, trusted location: JAX treats a persistent compilation cache as
    executable trusted input.
    """
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if preallocate else "false",
    )
    if mem_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    if platform is not None:
        os.environ.setdefault("JAX_PLATFORMS", str(platform))
    if compilation_cache_dir is not None:
        compilation_cache_dir = os.fspath(compilation_cache_dir)
        if not compilation_cache_dir:
            raise ValueError("compilation_cache_dir must not be empty.")
        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", compilation_cache_dir)
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


def _require_static_cutoff_for_jit(name, contraction, cutoff):
    """Reject a nonzero ``cutoff`` on an approximate jitted contraction.

    Approximate boundary/HOTRG/CTMRG contractions keep
    ``min(max_bond, rank-above-cutoff)`` singular values. Under
    ``jax.jit``/``jax.vmap`` that surviving count is data-dependent, so a
    nonzero ``cutoff`` produces dynamic tensor shapes that XLA cannot trace. A
    fixed ``max_bond`` with ``cutoff=0.0`` keeps every intermediate shape
    static; use the ``jit=False`` builder for adaptive truncation.
    """
    if contraction in {"hotrg", "ctmrg", "boundary"} and float(cutoff) != 0.0:
        raise ValueError(
            f"{name}={contraction!r} under jax.jit requires cutoff=0.0 "
            f"(data-dependent truncation is not traceable); got "
            f"cutoff={cutoff!r}. Use cutoff=0.0 with a fixed max_bond=chi, or "
            "build the amplitude with jit=False for adaptive truncation."
        )


def _maybe_register_stable_jax_svd(contraction):
    """Install Pepsy's truncation-safe JAX SVD backward rule for VMC gradients.

    HOTRG/CTMRG/boundary-MPS contractions compress with SVDs. Quimb may retain
    only the leading singular-vector columns, so Pepsy's registered thin-SVD
    pullback restores the omitted zero cotangents before using JAX's native
    derivative. Exact contraction has no SVD, so registration is skipped. This
    is a global autoray side effect and a soft no-op when JAX is unavailable.
    """
    from .api import ContractionConfig
    if isinstance(contraction, ContractionConfig):
        contraction = contraction.method
    key = str(contraction).replace("_", "-").lower()
    if _CONTRACTION_ALIASES.get(key) not in {"hotrg", "ctmrg", "boundary"}:
        return
    try:
        from ..tensors import reg_rel_svd_jax  # pylint: disable=import-outside-toplevel

        reg_rel_svd_jax()
    except ImportError:
        return


def _contraction_options(contraction_opts):
    return {} if contraction_opts is None else dict(contraction_opts)


_FLAT_SYMMRAY_BOUNDARY_FALLBACK_WARNED = False
_FLAT_SYMMRAY_CTMRG_FALLBACK_WARNED = False


def _is_flat_symmray_network(tn):
    """Return whether ``tn`` contains flat Symmray tensor data."""
    for tensor in tn:
        data = getattr(tensor, "data", None)
        if _is_symmray_array(data):
            return "Flat" in type(data).__name__
    return False


def _contract_boundary_for_vmc(tn, *, max_bond, cutoff, method_opts):
    """Contract a PEPS boundary, with a flat-Symmray compatibility retry.

    Quimb's ``max_separation=0`` path can ask its canonizer to unfuse an
    empty boundary axis. Current flat fermionic Symmray arrays cannot represent
    that intermediate operation, which surfaces as an ``align_axes`` or
    ``unfuse`` exception during JAX tracing. The requested sequence, chi,
    canonization, and other options are still honored; only this one stopping
    threshold is relaxed to Quimb's stable ``1`` fallback for flat Symmray.
    Dense/non-Symmray networks and all other failures are re-raised unchanged.
    """
    global _FLAT_SYMMRAY_BOUNDARY_FALLBACK_WARNED
    kwargs = dict(method_opts)
    try:
        return tn.contract_boundary(
            max_bond=max_bond,
            cutoff=cutoff,
            strip_exponent=True,
            **kwargs,
        )
    except (AttributeError, TypeError, ValueError):
        if (
            kwargs.get("max_separation", 1) == 0
            and _is_flat_symmray_network(tn)
        ):
            if not _FLAT_SYMMRAY_BOUNDARY_FALLBACK_WARNED:
                warnings.warn(
                    "Flat Symmray JAX boundary contraction does not support "
                    "the max_separation=0 intermediate axis path; retrying "
                    "with max_separation=1. The requested sequence, chi, and "
                    "canonization options remain active.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                _FLAT_SYMMRAY_BOUNDARY_FALLBACK_WARNED = True
            kwargs["max_separation"] = 1
            return tn.contract_boundary(
                max_bond=max_bond,
                cutoff=cutoff,
                strip_exponent=True,
                **kwargs,
            )
        raise


def _contract_ctmrg_for_vmc(tn, *, max_bond, cutoff, method_opts):
    """Contract CTMRG, retrying flat-Symmray ``max_separation=0`` safely.

    CTMRG delegates boundary compression to Quimb internally. On flat
    fermionic Symmray arrays, its zero-separation intermediate can produce a
    JAX block-matmul shape error (for example an environment axis of size
    ``chi`` paired with a Z2 block axis). Keep all requested options active and
    relax only this stopping threshold to Quimb's stable value ``1``.
    """
    from ..boundary.metrics import quimb_ctmrg_projector_compat

    global _FLAT_SYMMRAY_CTMRG_FALLBACK_WARNED
    kwargs = dict(method_opts)
    with quimb_ctmrg_projector_compat():
        try:
            return tn.contract_ctmrg(
                max_bond=max_bond,
                cutoff=cutoff,
                strip_exponent=True,
                **kwargs,
            )
        except (AttributeError, TypeError, ValueError):
            if (
                kwargs.get("max_separation", 1) == 0
                and _is_flat_symmray_network(tn)
            ):
                if not _FLAT_SYMMRAY_CTMRG_FALLBACK_WARNED:
                    warnings.warn(
                        "Flat Symmray JAX CTMRG does not support the "
                        "max_separation=0 intermediate axis path; retrying "
                        "with max_separation=1. The requested sequence, chi, "
                        "and canonization options remain active.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    _FLAT_SYMMRAY_CTMRG_FALLBACK_WARNED = True
                kwargs["max_separation"] = 1
                return tn.contract_ctmrg(
                    max_bond=max_bond,
                    cutoff=cutoff,
                    strip_exponent=True,
                    **kwargs,
                )
            raise


def _resolve_netket_contraction(contraction, chi, cutoff, contraction_opts):
    """Resolve a common ContractionConfig for NetKet public builders."""
    from .api import ContractionConfig
    if not isinstance(contraction, ContractionConfig):
        return contraction, chi, cutoff, contraction_opts
    if chi is not None and chi != contraction.chi:
        raise ValueError(
            f"chi={chi} conflicts with ContractionConfig.chi={contraction.chi}."
        )
    if contraction_opts is not None and dict(contraction_opts) != dict(contraction.options):
        raise ValueError("contraction options conflict with ContractionConfig.options.")
    return (
        contraction.method,
        contraction.chi,
        contraction.cutoff,
        dict(contraction.options),
    )


def _resolve_sampling_build_config(
    sampling,
    *,
    n_samples,
    n_chains,
    n_discard_per_chain,
    chunk_size,
    seed,
    sampler_seed,
):
    """Apply a common SamplingConfig before constructing a NetKet sampler."""
    from .api import SamplingConfig
    if sampling is None:
        return (
            n_samples,
            n_chains,
            n_discard_per_chain,
            chunk_size,
            seed,
            sampler_seed,
        )
    if not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if seed is not None and sampling.seed is not None and seed != sampling.seed:
        raise ValueError("seed conflicts with sampling.seed.")
    if (
        sampler_seed is not None
        and sampling.sampler_seed is not None
        and sampler_seed != sampling.sampler_seed
    ):
        raise ValueError("sampler_seed conflicts with sampling.sampler_seed.")
    if seed is not None and sampling.sampler_seed is not None:
        raise ValueError("Pass either seed or sampling.sampler_seed, not both.")
    if sampler_seed is not None and sampling.seed is not None:
        raise ValueError("Pass either sampler_seed or sampling.seed, not both.")
    if sampling.seed is not None and sampling.sampler_seed is not None:
        raise ValueError("Pass either sampling.seed or sampling.sampler_seed, not both.")
    return (
        sampling.n_samples,
        sampling.n_chains,
        sampling.n_discard_per_chain,
        sampling.chunk_size if sampling.chunk_size is not None else chunk_size,
        sampling.seed if sampling.seed is not None else seed,
        sampling.sampler_seed if sampling.sampler_seed is not None else sampler_seed,
    )


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


def _symmray_symmetry_name(tn):
    """Return the first Symmray tensor's symmetry name, if available."""
    for tensor in tn:
        data = getattr(tensor, "data", None)
        if not _is_symmray_array(data):
            continue
        symmetry = getattr(data, "symmetry", None)
        if symmetry is not None:
            return str(symmetry)
    return None


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


def _host_array_for_flatten(value):
    """Copy a Torch/CUDA block to a host NumPy array for Symmray packing."""
    return np.asarray(ar.to_numpy(value))


def _z2_flat_padded(data, sr):
    """Flatten a Z2 array whose charge blocks have unequal truncated sizes.

    Symmray's regular ``to_flat`` path stacks charge blocks directly.  A PEPS
    simple update can leave those blocks with different shapes after an SVD
    truncation, so the blocks need zero-padding to their flattened index sizes
    before they can share one dense JAX array.  Padding is exact: the newly
    added entries are outside the original charge-block support.
    """
    flat_indices = tuple(index.to_flat() for index in data.indices)
    target_shape = tuple(index.charge_size for index in flat_indices)
    padded_blocks = {}
    for sector, block in data.blocks.items():
        array = _host_array_for_flatten(block)
        if len(array.shape) != len(target_shape):
            raise ValueError(
                "Cannot flatten a Z2 block with rank "
                f"{array.ndim}; expected {len(target_shape)}."
            )
        if any(got > wanted for got, wanted in zip(array.shape, target_shape)):
            raise ValueError(
                "Cannot zero-pad a Z2 block larger than its flattened index: "
                f"block={array.shape}, target={target_shape}."
            )
        if tuple(array.shape) != target_shape:
            padded = np.zeros(target_shape, dtype=array.dtype)
            padded[tuple(slice(0, size) for size in array.shape)] = array
            array = padded
        padded_blocks[sector] = array
    return sr.Z2FermionicArrayFlat.from_blocks(
        padded_blocks,
        flat_indices,
        phases=data.phases,
        label=data.label,
        dummy_modes=data.dummy_modes,
        symmetry=data.symmetry,
    )


def prepare_fermionic_peps_for_netket(peps, *, device=None):
    """Prepare an evolved fermionic PEPS for NetKet's jitted JAX model.

    The returned copy uses the natural ``qtn.pack`` representation consumed by
    :func:`pack_fermionic_peps_ansatz`.  Z2 block-sparse tensors are flattened
    here, including the zero-padding required after unequal simple-update SVD
    blocks, and then moved to JAX.  ``device=None`` follows JAX's default
    device; ``PEPSY_FH_JAX_DEVICE`` can be used for a notebook-level override.

    Non-Z2 Symmray tensors are left block-sparse so the jitted NetKet model can
    raise a clear capability error.  Dense, already-flat, and non-Symmray
    tensors only go through the JAX backend conversion.
    """
    work = peps.copy()
    tn = getattr(work, "tn", work)
    padded_sites = []
    sr = None

    for site in tuple(tn.sites):
        data = tn[site].data
        type_name = type(data).__name__
        if (
            _is_symmray_array(data)
            and str(getattr(data, "symmetry", "")) == "Z2"
            and "Flat" not in type_name
        ):
            if sr is None:
                sr = _require_symmray()
            try:
                converted = data.to_flat()
            except (RuntimeError, ValueError):
                converted = _z2_flat_padded(data, sr)
                padded_sites.append(site)
            tn[site].modify(data=converted)

    if padded_sites:
        warnings.warn(
            "Zero-padded unequal Z2 charge blocks while preparing "
            f"{len(padded_sites)} PEPS tensor(s) for NetKet.",
            RuntimeWarning,
            stacklevel=2,
        )

    from ..tensors import backend_jax  # pylint: disable=import-outside-toplevel

    if device is None:
        device = os.environ.get("PEPSY_FH_JAX_DEVICE")
    work.apply_to_arrays(backend_jax(device=device, dtype=None))
    return work


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


def _fermion_site_encoding_from_phys_charges(phys_charges):
    """Return the local torch encoding matching a PEPS physical charge order."""
    from .torch import FermionSiteEncoding

    lut = _spinful_phys_lookup(phys_charges)
    if lut is None:
        return FermionSiteEncoding.symmray()
    return FermionSiteEncoding(
        empty=int(lut[0, 0]),
        down=int(lut[0, 1]),
        up=int(lut[1, 0]),
        double=int(lut[1, 1]),
    )


def _require_jittable_fermionic_ansatz(ansatz):
    """Raise a clear error when NetKet's jitted VMC path cannot use ``ansatz``."""
    if ansatz.uses_flat_symmray is not False:
        return
    symmetry = getattr(ansatz, "symmray_symmetry", None)
    if symmetry is None:
        if _spinful_phys_lookup(getattr(ansatz, "phys_charges", ())) is not None:
            symmetry = "U1U1 or Z2Z2"
        else:
            symmetry = "non-flat Symmray"
    raise NotImplementedError(
        "NetKet MCState JIT-compiles the PEPS log-amplitude model, but this "
        f"{symmetry} fermionic PEPS uses block-sparse Symmray arrays rather "
        "than a flat JAX-friendly backend. Use "
        "make_fermionic_peps_batched_amplitude_function(..., jit=False) with "
        "contraction='exact', 'hotrg', 'ctmrg', or 'boundary' for validation, "
        "or use a flat Z2 fermionic PEPS for full NetKet VMC until Symmray "
        "provides flat backends for the requested symmetry."
    )


def _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin):
    """Make the supported flat-Z2/fixed-U1U1 NetKet configuration explicit."""
    phys_charges = tuple(getattr(ansatz, "phys_charges", ()) or ())
    is_flat_z2 = (
        getattr(ansatz, "uses_flat_symmray", None) is True
        # Current Symmray exposes flat fermionic arrays only for Z2. FlatIndex
        # does not retain a ``chargemap``, hence ``phys_charges`` is empty for
        # normal flat Z2 PEPS after packing.
        and (
            not phys_charges
            or all(charge in (0, 1) for charge in phys_charges)
        )
    )
    if not is_flat_z2:
        return

    n_up, n_down = (int(value) for value in n_fermions_per_spin)
    from .api import SymmetryFallbackWarning

    warnings.warn(
        "NetKet VMC is using a flat Z2 fermionic PEPS ansatz with the fixed "
        f"U1U1 sampling sector (N_up, N_down)=({n_up}, {n_down}). This is "
        "intentional and supported: Z2 is the JAX-friendly tensor-storage "
        "symmetry, while NetKet's SpinOrbitalFermions Hilbert space and "
        "MetropolisFermionHop sampler preserve N_up and N_down separately. "
        "The PEPS itself does not carry block-sparse U1U1 charges.",
        SymmetryFallbackWarning,
        stacklevel=3,
    )


def _make_torch_generator(seed, device=None):
    if seed is None:
        return None
    from .torch import _require_torch  # pylint: disable=import-outside-toplevel

    torch = _require_torch()
    try:
        generator = (
            torch.Generator(device=device)
            if device is not None
            else torch.Generator()
        )
    except (RuntimeError, TypeError, ValueError):
        generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _nonzero_amplitude_mask(amplitudes, *, amplitude_floor=0.0):
    from .torch import _require_torch  # pylint: disable=import-outside-toplevel

    torch = _require_torch()
    amplitudes = torch.as_tensor(amplitudes)
    return torch.isfinite(amplitudes) & (
        amplitudes.abs() > float(amplitude_floor)
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
    callback_cls = getattr(getattr(nk, "callbacks", None), "AutoChunkSize", None)
    if callback_cls is None:
        raise RuntimeError(
            "NetKet AutoChunkSize requires NetKet >= 3.22; "
            f"found {getattr(nk, '__version__', 'an unknown version')!r}."
        )
    return callback_cls(
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


def _infer_lattice_shape_from_peps(peps):
    """Infer a rectangular ``(Lx, Ly)`` shape from a PEPS-like object."""
    tn = getattr(peps, "tn", peps)
    shape = (getattr(tn, "Lx", None), getattr(tn, "Ly", None))
    if all(value is not None for value in shape):
        return tuple(int(value) for value in shape)

    sites = tuple(getattr(tn, "sites", ()))
    if not sites or not all(
        isinstance(site, tuple) and len(site) == 2 for site in sites
    ):
        raise ValueError(
            "Could not infer a rectangular PEPS lattice; provide both Lx and Ly."
        )
    Lx = max(int(site[0]) for site in sites) + 1
    Ly = max(int(site[1]) for site in sites) + 1
    expected = {(i, j) for i in range(Lx) for j in range(Ly)}
    if set(sites) != expected:
        raise ValueError(
            "PEPS sites are not a complete row-major rectangular lattice; "
            "provide both Lx and Ly."
        )
    return Lx, Ly


def _infer_lattice_shape_from_fermi_terms(terms):
    """Infer ``(Lx, Ly)`` from coordinate-keyed native fermion terms.

    Integer site labels do not contain enough information to distinguish a
    rectangular lattice from a one-dimensional indexing scheme, so this
    fallback deliberately accepts only coordinate-keyed native terms.  The
    normal public path still prefers the PEPS geometry when it is available.
    """
    if terms is None or not hasattr(terms, "keys"):
        return None
    coordinates = []
    for key in terms.keys():
        if _is_coordinate_edge_key(key):
            coordinates.extend(tuple(site) for site in key)
        elif isinstance(key, tuple) and len(key) == 2:
            try:
                coordinates.append((int(key[0]), int(key[1])))
            except (TypeError, ValueError):
                continue
    if not coordinates:
        return None
    return (
        max(int(site[0]) for site in coordinates) + 1,
        max(int(site[1]) for site in coordinates) + 1,
    )


def _site_index_for_lattice(site, Lx, Ly):
    if isinstance(site, Integral):
        site = int(site)
        return site if 0 <= site < Lx * Ly else None
    try:
        i, j = tuple(site)
    except (TypeError, ValueError):
        return None
    i, j = int(i), int(j)
    if 0 <= i < Lx and 0 <= j < Ly:
        return i * Ly + j
    return None


def _is_coordinate_edge_key(key):
    try:
        left, right = tuple(key)
        return (
            isinstance(left, (tuple, list))
            and isinstance(right, (tuple, list))
            and len(left) == 2
            and len(right) == 2
        )
    except (TypeError, ValueError):
        return False


def _edges_from_fermi_terms(terms, Lx, Ly):
    """Extract integer graph edges from native coordinate-keyed terms."""
    if terms is None or not hasattr(terms, "keys"):
        return ()
    keys = tuple(terms.keys())
    coordinate_edges = tuple(key for key in keys if _is_coordinate_edge_key(key))
    use_integer_edges = not coordinate_edges
    found = []
    for key in keys:
        try:
            left, right = tuple(key)
        except (TypeError, ValueError):
            continue
        if coordinate_edges and not _is_coordinate_edge_key(key):
            continue
        if not coordinate_edges and not use_integer_edges:
            continue
        left_index = _site_index_for_lattice(left, Lx, Ly)
        right_index = _site_index_for_lattice(right, Lx, Ly)
        if left_index is None or right_index is None or left_index == right_index:
            continue
        edge = (min(left_index, right_index), max(left_index, right_index))
        if edge not in found:
            found.append(edge)
    return tuple(found)


def _edges_from_operator_sum(operator_sum, Lx, Ly):
    """Extract NetKet integer edges from the common operator IR."""
    sites = _row_major_sites(int(Lx), int(Ly))
    site_to_index = {site: index for index, site in enumerate(sites)}
    found = []
    for term in operator_sum:
        support = tuple(getattr(term, "support", ()))
        if len(support) != 2:
            continue
        mapped = []
        for site in support:
            if site in site_to_index:
                mapped.append(site_to_index[site])
            elif isinstance(site, Integral) and 0 <= int(site) < len(sites):
                mapped.append(int(site))
            else:
                mapped = []
                break
        if len(mapped) == 2 and mapped[0] != mapped[1]:
            edge = tuple(sorted(mapped))
            if edge not in found:
                found.append(edge)
    return tuple(found)


def _infer_pbc_from_fermi_terms(terms, Lx, Ly):
    """Infer periodic axes from coordinate-keyed nearest-neighbor terms."""
    if terms is None or not hasattr(terms, "keys"):
        return False
    pbc_x = False
    pbc_y = False
    for key in terms.keys():
        if _is_coordinate_edge_key(key):
            (i0, j0), (i1, j1) = key
        else:
            try:
                left, right = tuple(key)
                left, right = int(left), int(right)
            except (TypeError, ValueError):
                continue
            if not (0 <= left < Lx * Ly and 0 <= right < Lx * Ly):
                continue
            i0, j0 = divmod(left, Ly)
            i1, j1 = divmod(right, Ly)
        i0, j0, i1, j1 = int(i0), int(j0), int(i1), int(j1)
        pbc_x |= {i0, i1} == {0, Lx - 1} and j0 == j1 and Lx > 2
        pbc_y |= {j0, j1} == {0, Ly - 1} and i0 == i1 and Ly > 2
    return (pbc_x, pbc_y)


def _coerce_spin_sector(value):
    """Convert common Pepsy sector/setup forms to ``(N_up, N_down)``."""
    if value is None:
        return None
    if hasattr(value, "spin_occupations"):
        value = value.spin_occupations
    elif hasattr(value, "n_fermions_per_spin"):
        value = value.n_fermions_per_spin

    if isinstance(value, dict):
        if {"n_up", "n_down"}.issubset(value):
            return int(value["n_up"]), int(value["n_down"])
        if "n_fermions_per_spin" in value:
            return _coerce_spin_sector(value["n_fermions_per_spin"])
        values = tuple(value.values())
        if values and all(
            isinstance(item, (tuple, list)) and len(item) == 2
            for item in values
        ):
            return tuple(
                int(sum(int(item[spin]) for item in values))
                for spin in (0, 1)
            )

    try:
        values = tuple(value)
    except TypeError:
        return None
    if len(values) != 2 or any(
        isinstance(item, (tuple, list, dict)) for item in values
    ):
        return None
    return tuple(int(item) for item in values)


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


def _site_major_to_netket_jw_phase(
    occ_rows,
    columns,
    *,
    site_to_orb=None,
    xp=np,
):
    """Return the fermionic basis phase from PEPS to NetKet mode order.

    A fermionic PEPS contraction uses site-major local modes
    (up_0, down_0, up_1, down_1, ...). NetKet's SpinOrbitalFermions stores
    configurations as (down_0, ..., down_n, up_0, ..., up_n). Reordering the
    occupied creation operators gives the phase

    (-1) ** sum_j(n_down[j] * sum_{i <= j} n_up[i]).

    site_to_orb changes NetKet's orbital order into the packed PEPS site order
    before the phase is evaluated.
    """
    n_orbitals = len(columns.up)
    occ_rows = xp.asarray(occ_rows, dtype=xp.int32).reshape(
        (-1, 2 * n_orbitals)
    )
    col_up = xp.asarray(columns.up, dtype=xp.int32)
    col_down = xp.asarray(columns.down, dtype=xp.int32)
    n_up = occ_rows[:, col_up]
    n_down = occ_rows[:, col_down]
    if site_to_orb is not None:
        site_to_orb = xp.asarray(site_to_orb, dtype=xp.int32)
        n_up = n_up[:, site_to_orb]
        n_down = n_down[:, site_to_orb]
    inversions = xp.sum(xp.cumsum(n_up, axis=1) * n_down, axis=1)
    return 1 - 2 * (inversions % 2)


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
        symmray_symmetry=_symmray_symmetry_name(tn),
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
    """Pack a fermionic PEPS using quimb's natural params/skeleton pair.

    This is the package equivalent of ``params, skeleton = qtn.pack(peps)``;
    the returned ansatz retains ``skeleton`` so every JAX/Flax evaluation can
    reconstruct it with ``qtn.unpack(params, skeleton)``.
    """
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
    _require_static_cutoff_for_jit("contraction", contraction, cutoff)
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
            return _contract_ctmrg_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
            )
        if contraction == "boundary":
            return _contract_boundary_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
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
            mantissa, exponent = _contract_ctmrg_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = _contract_boundary_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
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
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
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
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
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


def _netket_variables_from_ansatz(ansatz, param_dtype=None):
    """Build Flax variables directly from an already-packed PEPS ansatz.

    ``MCState`` otherwise calls ``model.init`` with a dummy configuration.
    For these PEPS models that dummy call needlessly traces the contraction
    once, even though the packed leaves are already the desired parameters.
    """
    _, jnp = _require_jax()
    dtype = None if param_dtype is None else param_dtype
    return {
        "params": {
            f"t{k}": jnp.asarray(leaf, dtype=dtype)
            for k, leaf in enumerate(ansatz.leaves)
        }
    }


def _peps_params_from_netket_variables(ansatz, variables, *, device_get=True):
    """Rebuild quimb's packed parameter tree from Flax/NetKet variables.

    The PEPS Flax models deliberately name their leaves ``t0``, ``t1``, ...
    in :func:`_netket_variables_from_ansatz`.  Keeping the inverse conversion
    here makes that private model detail explicit and gives the public setup
    handoff a useful validation error instead of an opaque ``qtn.unpack``
    failure.
    """
    if not hasattr(variables, "get"):
        raise TypeError(
            "variables must be a Flax variables mapping or a parameters mapping."
        )
    parameters = variables.get("params", variables)
    if not hasattr(parameters, "__getitem__"):
        raise TypeError("variables['params'] must be a mapping of PEPS leaves.")

    expected = tuple(f"t{k}" for k in range(len(ansatz.leaves)))
    try:
        leaves = tuple(parameters[name] for name in expected)
    except KeyError as error:
        raise ValueError(
            "NetKet PEPS parameters are missing leaf "
            f"{error.args[0]!r}; expected {expected!r}."
        ) from error

    if device_get:
        jax, _ = _require_jax()
        leaves = tuple(np.asarray(jax.device_get(leaf)) for leaf in leaves)
    else:
        leaves = tuple(leaves)
    jax, _ = _require_jax()
    return jax.tree_util.tree_unflatten(ansatz.treedef, leaves)


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
    _require_static_cutoff_for_jit("contraction", contraction, cutoff)

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
        phase = _site_major_to_netket_jw_phase(
            occ_rows,
            columns,
            site_to_orb=site_to_orb,
            xp=jnp,
        )
        return phys_orb[:, site_to_orb], phase

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
            return _contract_ctmrg_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
            )
        if contraction == "boundary":
            return _contract_boundary_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
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
        phys_rows, phase = occ_rows_to_phys_jax(occ_rows)

        def evaluate_phys(phys):
            mantissa, exponent = contract_mantissa_exponent(select_phys(tn, phys))
            if output == "mantissa_exponent":
                return mantissa, exponent
            if output == "amplitude":
                return amplitude_from_mantissa_exponent(mantissa, exponent)
            return log_from_mantissa_exponent(mantissa, exponent)

        values = jax.vmap(evaluate_phys)(phys_rows)
        if output == "mantissa_exponent":
            mantissas, exponents = values
            return mantissas * phase.astype(mantissas.dtype), exponents
        if output == "amplitude":
            return values * phase.astype(values.dtype)
        return values + jnp.log(phase.astype(complex_dtype))

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
            mantissa, exponent = _contract_ctmrg_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
            )
        elif contraction == "boundary":
            mantissa, exponent = _contract_boundary_for_vmc(
                tnx,
                max_bond=chi,
                cutoff=cutoff,
                method_opts=method_opts,
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
        occ_rows = np.asarray(occ_rows)
        phys_rows = occupation_to_phys_indices(
            occ_rows,
            columns,
            site_to_orb=ansatz.site_to_orb,
            phys_charges=getattr(ansatz, "phys_charges", ()),
        )
        phase = _site_major_to_netket_jw_phase(
            occ_rows,
            columns,
            site_to_orb=ansatz.site_to_orb,
        )
        values = [evaluate_one(tn, phys) for phys in phys_rows]
        if output == "mantissa_exponent":
            mantissas, exponents = zip(*values)
            mantissas = jnp.stack([jnp.asarray(x) for x in mantissas])
            return (
                mantissas * jnp.asarray(phase, dtype=mantissas.dtype),
                jnp.asarray(exponents, dtype=real_dtype),
            )
        values = jnp.stack([jnp.asarray(x) for x in values])
        if output == "amplitude":
            return values * jnp.asarray(phase, dtype=values.dtype)
        return values + jnp.log(jnp.asarray(phase, dtype=complex_dtype))

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
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
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
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
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
        variables=_netket_variables_from_ansatz(ansatz, param_dtype),
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
    sampling=None,
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
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
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
    sampling=None,
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
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
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
    ``phys_dim=4`` parity-resolved physical index. ``"U1"``, ``"U1U1"``, and
    ``"Z2Z2"`` use block-sparse backends. ``U1`` receives the total local
    occupation charge, while the latter two receive a per-spin
    ``(n_up, n_down)`` physical charge map. The default site-charge map sums
    to ``n_fermions_per_spin`` (half filling if omitted).

    Note
    ----
    A ``U1``, ``U1U1``, or ``Z2Z2`` ansatz cannot yet be driven through the NetKet
    Monte-Carlo state (which JIT-compiles the model) because the corresponding
    flat backend is missing upstream. The block-sparse PEPS still evaluates
    correctly through the non-jitted amplitude functions (``jit=False``) for
    validation and exact/dense sums.
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

    if sym in {"U1", "U1U1", "Z2Z2"}:
        from pepsy.tensors import (
            default_physical_sectors,
            site_charge_from_occupations,
        )

        if site_charge is None:
            if n_fermions_per_spin is None:
                n_fermions_per_spin = (n_sites // 2, n_sites // 2)
            n_up, n_down = (int(x) for x in n_fermions_per_spin)
            occupations = _default_u1u1_flux_occupations(
                Lx, Ly, n_up, n_down
            )
            if sym == "U1":
                occupations = {
                    site: sum(charge)
                    for site, charge in occupations.items()
                }
            site_charge = site_charge_from_occupations(occupations)
        use_flat = False if flat == "auto" else bool(flat)
        if use_flat:
            warnings.warn(
                f"Symmray has no flat {sym} fermionic backend; falling back to "
                "block-sparse (flat=False). NetKet MC sampling JIT-compiles the "
                "model and needs a flat backend, so use the non-jit amplitude "
                f"functions for {sym} until a flat {sym} backend lands upstream.",
                RuntimeWarning,
                stacklevel=2,
            )
            use_flat = False
        return sr.networks.PEPS_fermionic_rand(
            sym,
            Lx,
            Ly,
            bond_dim,
            phys_dim=default_physical_sectors(sym, 4),
            site_charge=site_charge,
            flat=use_flat,
            seed=seed,
            dtype=dtype,
            **kwargs,
        )

    raise ValueError(
        f"Unsupported symmetry {symmetry!r} for fermionic_peps_rand; "
        "use 'Z2', 'U1', 'Z2Z2', or 'U1U1'."
    )


def _maybe_conserving_fermion_operator(operator, *, strict=False):
    """Convert to a particle-number/spin-conserving fermion operator if valid.

    NetKet's ``ParticleNumberAndSpinConservingFermioperator2nd`` exposes far
    fewer connected configurations, which makes VMC local energies cheaper. It
    only applies to operators that conserve both particle number and spin, so
    ``strict=False`` silently returns the input operator when the conversion is
    not applicable (for example a spin-flipping observable).
    """
    try:
        import netket.experimental as nkx  # pylint: disable=import-outside-toplevel

        cls = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd
    except (ImportError, AttributeError):
        if strict:
            raise
        return operator
    try:
        return cls.from_fermionoperator2nd(operator)
    except Exception:  # pylint: disable=broad-except
        if strict:
            raise
        return operator


def netket_fermion_operator(hilbert, terms, *, constant=0.0, conserving=False):
    r"""Build a NetKet fermion operator from symbolic second-quantized terms.

    This is the general "build from terms" primitive: it turns a list of
    fermionic monomials into a jittable NetKet operator usable both as a VMC
    Hamiltonian (``vstate.expect_and_grad``) and as a measurement observable
    (``vstate.expect``).

    Parameters
    ----------
    hilbert:
        A NetKet ``SpinOrbitalFermions`` Hilbert space.
    terms:
        Iterable of ``(coefficient, ops)`` pairs. ``ops`` is an iterable of
        ``(site, sz, dagger)`` tuples where ``site`` is the orbital index,
        ``sz`` is ``+1`` (spin up), ``-1`` (spin down), or ``None`` (spinless),
        and ``dagger`` is ``True`` for a creation operator
        :math:`c^{\dagger}` and ``False`` for an annihilation operator
        :math:`c`. Factors in ``ops`` are applied left to right, so a number
        operator :math:`n = c^{\dagger} c` is
        ``((site, sz, True), (site, sz, False))``.
    constant:
        Scalar identity shift added to the operator.
    conserving:
        When ``True`` or ``"auto"``, convert to NetKet's
        particle-number/spin-conserving fermion operator (cheaper in VMC).
        ``"auto"`` falls back to the plain operator when the conversion is not
        applicable; ``True`` raises on failure.

    Returns
    -------
    A jittable NetKet fermion operator.
    """
    _require_netket()
    from netket.operator import (  # pylint: disable=import-outside-toplevel
        fermion as nkf,
    )

    total = None
    for coeff, ops in terms:
        term_op = None
        for site, sz, dagger in ops:
            builder = nkf.create if dagger else nkf.destroy
            factor = builder(hilbert, int(site), sz=sz)
            term_op = factor if term_op is None else term_op @ factor
        if term_op is None:
            continue
        term_op = coeff * term_op
        total = term_op if total is None else total + term_op

    if total is None:
        total = nkf.zero(hilbert)
    if constant:
        total = total + constant * nkf.identity(hilbert)

    if conserving:
        total = _maybe_conserving_fermion_operator(
            total, strict=conserving != "auto"
        )
    return total


def _fermi_hubbard_terms(edges, n_sites, *, t, U):
    """Return second-quantized terms for the spinful Hubbard Hamiltonian."""
    terms = []
    for site in range(int(n_sites)):
        terms.append(
            (
                U,
                (
                    (site, +1, True),
                    (site, +1, False),
                    (site, -1, True),
                    (site, -1, False),
                ),
            )
        )
    for left, right in edges:
        left, right = int(left), int(right)
        for spin in (+1, -1):
            terms.extend(
                (
                    (-t, ((left, spin, True), (right, spin, False))),
                    (-t, ((right, spin, True), (left, spin, False))),
                )
            )
    return terms


def _build_netket_fermi_hubbard_operator(hilbert, graph, *, n_sites, t, U):
    """Build Hubbard metadata across NetKet's native API generations.

    NetKet releases before the removal of ``FermiHubbardJax`` expose the
    equivalent second-quantized operator instead. The fallback preserves the
    fixed spin sector and uses NetKet's conserving implementation when
    available; it is not a dense or Jordan--Wigner conversion.
    """
    nk = _require_netket()
    native = getattr(nk.operator, "FermiHubbardJax", None)
    if native is not None:
        return native(hilbert, graph=graph, t=t, U=U, dtype=float)

    edges = tuple(graph.edges())
    return netket_fermion_operator(
        hilbert,
        _fermi_hubbard_terms(edges, n_sites, t=t, U=U),
        conserving="auto",
    )


def compile_operator_sum_netket(hilbert, terms, *, site_order=None, conserving=False):
    """Compile a backend-neutral :class:`OperatorSum` for NetKet.

    Symbolic fermion products are lowered to the existing
    :func:`netket_fermion_operator` primitive. Local matrix terms are lowered
    to ``nk.operator.LocalOperator``. The identity constant is included in the
    returned native operator, so callers must not add it again. Pass
    ``conserving=\"auto\"`` to use NetKet's reduced fixed-sector fermion
    operator when the symbolic terms preserve particle number and spin; terms
    that do not preserve the sector retain the generic operator.
    """
    from .api import (
        LocalMatrixTerm,
        ProductTerm,
        _expand_fermion_factor,
        normalize_operator_sum,
    )

    nk = _require_netket()
    operator_sum = normalize_operator_sum(terms)
    site_order = None if site_order is None else tuple(site_order)

    def map_site(site):
        if site_order is None:
            return int(site) if isinstance(site, Integral) else site
        if site in site_order:
            return site_order.index(site)
        if isinstance(site, Integral) and 0 <= int(site) < len(site_order):
            return int(site)
        raise ValueError(f"Term site {site!r} is not present in site_order.")

    symbolic = []
    matrix_ops = []
    for term in operator_sum:
        if isinstance(term, ProductTerm):
            factors = []
            for factor in term.factors:
                for site, name in _expand_fermion_factor(factor):
                    if name in {"create", "annihilate"}:
                        sz = None
                    elif name in {"create_u", "annihilate_u"}:
                        sz = 1
                    elif name in {"create_d", "annihilate_d"}:
                        sz = -1
                    else:
                        raise ValueError(
                            "NetKet ProductTerm factors must be fermionic "
                            "creation, annihilation, or number operators; "
                            f"got {name!r}."
                        )
                    factors.append(
                        (map_site(site), sz, name.startswith("create"))
                    )
            symbolic.append((term.coefficient, tuple(factors)))
            continue
        if not isinstance(term, LocalMatrixTerm):  # pragma: no cover
            raise TypeError(f"Unsupported operator term {type(term).__name__}.")
        matrix = term.matrix
        detach = getattr(matrix, "detach", None)
        if callable(detach):
            matrix = detach()
        cpu = getattr(matrix, "cpu", None)
        if callable(cpu):
            matrix = cpu()
        matrix = np.asarray(matrix) * term.coefficient
        n_sites = len(term.support)
        # The common term contract uses output axes followed by input axes,
        # while NetKet LocalOperator takes a flattened square local matrix.
        local_dim = int(np.prod(matrix.shape[:n_sites]))
        matrix = matrix.reshape(local_dim, local_dim)
        matrix_ops.append(
            nk.operator.LocalOperator(
                hilbert,
                matrix,
                acting_on=[map_site(site) for site in term.support],
            )
        )

    total = None
    if symbolic:
        total = netket_fermion_operator(
            hilbert,
            symbolic,
            constant=operator_sum.constant,
            conserving=conserving,
        )
    elif operator_sum.constant:
        total = nk.operator.LocalOperator(
            hilbert,
            constant=operator_sum.constant,
        )
    for matrix_op in matrix_ops:
        total = matrix_op if total is None else total + matrix_op
    if total is None:
        total = nk.operator.LocalOperator(hilbert, constant=0.0)
    return total


def fermion_model_terms(
    fermion,
    edges,
    *,
    t,
    U,
    V=0.0,
    mu=0.0,
    n_sites=None,
):
    r"""Return symbolic Hamiltonian terms for a spinful :class:`pepsy.Fermion`.

    Reconstructs the hopping (:math:`-t`), on-site Hubbard
    (:math:`U\,n_\uparrow n_\downarrow`), nearest-neighbor density
    (:math:`V\,n_i n_j`) and chemical-potential (:math:`-\mu\,n`) terms of
    explicit coefficients over the integer ``edges`` as a list of
    ``(coefficient, ops)`` pairs (see :func:`netket_fermion_operator`). The
    coefficients are arguments rather than state on ``fermion``, so the
    symbolic NetKet operator cannot diverge from the native Hamiltonian.

    Only uniform scalar parameters are supported; supply non-uniform couplings
    as a native ``fermion.hamiltonian({...})`` mapping to
    :func:`build_fermion_vmc`.
    """
    if not getattr(fermion, "spinful", True):
        raise NotImplementedError(
            "fermion_model_terms supports spinful fermions; build spinless "
            "operators directly with netket_fermion_operator."
        )
    edges = tuple((int(i), int(j)) for i, j in edges)
    if n_sites is None:
        n_sites = 1 + max((max(i, j) for i, j in edges), default=-1)
    sites = range(int(n_sites))

    def _scalar(name, value):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} must be a real scalar for fermion_model_terms; "
                f"got {value!r}. Pass explicit terms for non-uniform couplings."
            ) from exc

    t = _scalar("t", t)
    U = _scalar("U", U)
    V = _scalar("V", V)
    mu = _scalar("mu", mu)

    terms = []
    for i, j in edges:
        for sz in (1, -1):
            terms.append((-t, ((i, sz, True), (j, sz, False))))
            terms.append((-t, ((j, sz, True), (i, sz, False))))
    if U:
        for i in sites:
            terms.append(
                (
                    U,
                    (
                        (i, 1, True),
                        (i, 1, False),
                        (i, -1, True),
                        (i, -1, False),
                    ),
                )
            )
    if V:
        for i, j in edges:
            for sz in (1, -1):
                for sz2 in (1, -1):
                    terms.append(
                        (
                            V,
                            (
                                (i, sz, True),
                                (i, sz, False),
                                (j, sz2, True),
                                (j, sz2, False),
                            ),
                        )
                    )
    if mu:
        for i in sites:
            for sz in (1, -1):
                terms.append((-mu, ((i, sz, True), (i, sz, False))))
    return terms


def _native_term_support(where, *, coordinate_sites):
    """Return the ordered one- or two-site support encoded by a term key."""
    if coordinate_sites and (
        isinstance(where, (tuple, list))
        and len(where) == 2
        and all(isinstance(value, Integral) for value in where)
    ):
        return (tuple(int(value) for value in where),)
    if isinstance(where, (tuple, list)):
        support = tuple(where)
    else:
        support = (where,)
    if len(support) not in {1, 2}:
        raise ValueError(
            "Native Fermion Hamiltonian terms must have one-site or two-site keys; "
            f"got {where!r}."
        )
    return support


def _native_term_to_numpy(term):
    """Transfer one small native local term to a host dense matrix."""
    dense = term.to_dense() if hasattr(term, "to_dense") else term
    return np.asarray(ar.to_numpy(dense), dtype=np.complex128)


def _project_native_term(matrix, candidates, *, where):
    """Expand a native local matrix in a small Fermi-Hubbard operator basis."""
    columns = np.stack([candidate.reshape(-1) for candidate in candidates], axis=1)
    coefficients, _, _, _ = np.linalg.lstsq(columns, matrix.reshape(-1), rcond=None)
    reconstructed = (columns @ coefficients).reshape(matrix.shape)
    residual = np.linalg.norm(matrix - reconstructed)
    scale = max(1.0, float(np.linalg.norm(matrix)))
    if residual > 5.0e-6 * scale:
        raise ValueError(
            "Native term at "
            f"{where!r} is not in the supported spinful Fermi-Hubbard local "
            "operator span. Supply an explicit symbolic OperatorSum to NetKet "
            "for a custom interaction."
        )
    return coefficients


def _native_fermi_hubbard_terms_to_netket(fermion, terms, *, site_order):
    """Compile explicit native Hubbard terms into NetKet fermion monomials.

    The native Symmray terms are the authoritative Hamiltonian. Their local
    matrices are decomposed into the neutral Fermi-Hubbard basis (identity,
    spin-resolved number, doublon, hopping, and density terms), then emitted
    as NetKet creation/annihilation products. This retains NetKet's fermionic
    Jordan-Wigner signs while allowing the native and VMC paths to share the
    exact same explicit term mapping.
    """
    if fermion is None or not getattr(fermion, "spinful", False):
        raise TypeError(
            "Compiling native terms for NetKet requires a spinful Fermion helper."
        )
    terms = _native_fermion_terms_mapping(terms)
    coordinate_sites = any(
        isinstance(where, (tuple, list))
        and len(where) == 2
        and all(
            isinstance(site, (tuple, list))
            and len(site) == 2
            and all(isinstance(value, Integral) for value in site)
            for site in where
        )
        for where in terms
    )
    site_order = tuple(site_order)
    site_to_orbital = {site: orbital for orbital, site in enumerate(site_order)}

    def orbital(site):
        if site in site_to_orbital:
            return site_to_orbital[site]
        if isinstance(site, Integral) and 0 <= int(site) < len(site_order):
            return int(site)
        raise ValueError(
            f"Native term site {site!r} is not present in the PEPS site order."
        )

    # Build basis operators on CPU: native input terms can live on JAX/Torch,
    # but compilation only needs tiny 4x4 / 16x16 host matrices.
    from ..tensors import Fermion  # pylint: disable=import-outside-toplevel

    reference = Fermion(
        spinful=True,
        symmetry=fermion.symmetry,
        dtype=fermion.dtype,
    )
    one_site_basis = (
        _native_term_to_numpy(reference.observable("identity")),
        _native_term_to_numpy(reference.observable("number_up")),
        _native_term_to_numpy(reference.observable("number_down")),
        _native_term_to_numpy(reference.interaction_operator()),
    )
    two_site_basis = (
        _native_term_to_numpy(reference.hopping_operator(spin="up")),
        _native_term_to_numpy(
            reference.hopping_operator(spin="up", peierls_angle=np.pi / 2)
        ),
        _native_term_to_numpy(reference.hopping_operator(spin="down")),
        _native_term_to_numpy(
            reference.hopping_operator(spin="down", peierls_angle=np.pi / 2)
        ),
        _native_term_to_numpy(reference.density_operator()),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((0, "number_up"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((0, "number_down"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((1, "number_up"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((1, "number_down"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((0, "double"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term(
                [(1.0, ((1, "double"),))], sites=(0, 1)
            )
        ),
        _native_term_to_numpy(
            reference.operator_term([(1.0, ())], sites=(0, 1))
        ),
    )

    symbolic = []
    constant = 0.0j
    for where, term in dict(terms).items():
        support = _native_term_support(where, coordinate_sites=coordinate_sites)
        matrix = _native_term_to_numpy(term)
        if len(support) == 1:
            (site,) = support
            coeff_identity, coeff_up, coeff_down, coeff_double = _project_native_term(
                matrix,
                one_site_basis,
                where=where,
            )
            constant += coeff_identity
            target = orbital(site)
            if abs(coeff_up) > 1.0e-10:
                symbolic.append((coeff_up, ((target, 1, True), (target, 1, False))))
            if abs(coeff_down) > 1.0e-10:
                symbolic.append((coeff_down, ((target, -1, True), (target, -1, False))))
            if abs(coeff_double) > 1.0e-10:
                symbolic.append(
                    (
                        coeff_double,
                        (
                            (target, 1, True),
                            (target, 1, False),
                            (target, -1, True),
                            (target, -1, False),
                        ),
                    )
                )
            continue

        left, right = (orbital(site) for site in support)
        (
            up_real,
            up_imag,
            down_real,
            down_imag,
            density,
            left_up,
            left_down,
            right_up,
            right_down,
            left_double,
            right_double,
            identity,
        ) = _project_native_term(matrix, two_site_basis, where=where)
        constant += identity
        for sz, real, imag in (
            (1, up_real, up_imag),
            (-1, down_real, down_imag),
        ):
            forward = real + 1.0j * imag
            backward = real - 1.0j * imag
            if abs(forward) > 1.0e-10:
                symbolic.append((forward, ((left, sz, True), (right, sz, False))))
            if abs(backward) > 1.0e-10:
                symbolic.append((backward, ((right, sz, True), (left, sz, False))))
        if abs(density) > 1.0e-10:
            for left_sz in (1, -1):
                for right_sz in (1, -1):
                    symbolic.append(
                        (
                            density,
                            (
                                (left, left_sz, True),
                                (left, left_sz, False),
                                (right, right_sz, True),
                                (right, right_sz, False),
                            ),
                        )
                    )
        for target, sz, coefficient in (
            (left, 1, left_up),
            (left, -1, left_down),
            (right, 1, right_up),
            (right, -1, right_down),
        ):
            if abs(coefficient) > 1.0e-10:
                symbolic.append(
                    (coefficient, ((target, sz, True), (target, sz, False)))
                )
        for target, coefficient in (
            (left, left_double),
            (right, right_double),
        ):
            if abs(coefficient) > 1.0e-10:
                symbolic.append(
                    (
                        coefficient,
                        (
                            (target, 1, True),
                            (target, 1, False),
                            (target, -1, True),
                            (target, -1, False),
                        ),
                    )
                )
    return symbolic, constant


def standard_fermion_observables(hilbert):
    """Return common spinful observables for :meth:`NetKetPEPSVMC.measure_samples`.

    The returned ``{name: operator}`` mapping contains the total particle
    number per spin (``"n_up"``, ``"n_down"``), the total particle number
    (``"n_total"``), and the total on-site double occupancy
    (``"double_occupancy"``). Each value is a jittable NetKet operator.
    """
    n = int(hilbert.n_orbitals)
    n_up = [(1.0, ((i, 1, True), (i, 1, False))) for i in range(n)]
    n_down = [(1.0, ((i, -1, True), (i, -1, False))) for i in range(n)]
    doub = [
        (1.0, ((i, 1, True), (i, 1, False), (i, -1, True), (i, -1, False)))
        for i in range(n)
    ]
    return {
        "n_up": netket_fermion_operator(hilbert, n_up, conserving="auto"),
        "n_down": netket_fermion_operator(hilbert, n_down, conserving="auto"),
        "n_total": netket_fermion_operator(
            hilbert, n_up + n_down, conserving="auto"
        ),
        "double_occupancy": netket_fermion_operator(
            hilbert, doub, conserving="auto"
        ),
    }


def _eta_pair_lattice_sites(ansatz):
    """Return a rectangular zero-origin coordinate lattice from an ansatz."""
    sites = tuple(getattr(ansatz, "orbital_sites", ()))
    if not sites:
        raise ValueError(
            "Eta-pair measurement needs a packed PEPS with coordinate "
            "orbital_sites."
        )
    if len(set(sites)) != len(sites):
        raise ValueError("Eta-pair measurement requires unique orbital_sites.")
    if any(
        not isinstance(site, tuple)
        or len(site) != 2
        or any(
            isinstance(coord, bool) or not isinstance(coord, Integral)
            for coord in site
        )
        for site in sites
    ):
        raise ValueError(
            "Eta-pair measurement requires two-dimensional integer "
            "orbital_sites."
        )

    sites = tuple((int(x), int(y)) for x, y in sites)
    xs = {x for x, _ in sites}
    ys = {y for _, y in sites}
    if min(xs) != 0 or min(ys) != 0:
        raise ValueError(
            "Eta-pair measurement requires a zero-origin rectangular lattice."
        )
    Lx, Ly = max(xs) + 1, max(ys) + 1
    expected = {(x, y) for x in range(Lx) for y in range(Ly)}
    if set(sites) != expected:
        raise ValueError(
            "Eta-pair measurement requires orbital_sites to cover a complete "
            "rectangular lattice."
        )
    return sites, Lx, Ly


def _netket_eta_pair_operator(hilbert, ansatz, specification):
    """Compile a declarative eta-pair specification for a packed PEPS."""
    sites, Lx, Ly = _eta_pair_lattice_sites(ansatz)
    site_to_orbital = {site: orbital for orbital, site in enumerate(sites)}

    if specification.dx == 0 and specification.dy == 0:
        coefficient = 1.0 / len(sites)
        terms = [
            (
                coefficient,
                (
                    (orbital, 1, True),
                    (orbital, 1, False),
                    (orbital, -1, True),
                    (orbital, -1, False),
                ),
            )
            for orbital in range(len(sites))
        ]
        return netket_fermion_operator(hilbert, terms, conserving="auto")

    pairs = []
    site_set = set(sites)
    for left in sites:
        x, y = left
        if specification.periodic:
            right = ((x + specification.dx) % Lx, (y + specification.dy) % Ly)
        else:
            right = (x + specification.dx, y + specification.dy)
            if right not in site_set:
                continue
        pairs.append((left, right))
    if not pairs:
        raise ValueError("The requested eta-pair displacement has no valid pairs.")

    normalizer = len(sites) if specification.periodic else len(pairs)
    terms = []
    for left, right in pairs:
        phase = (
            -1.0
            if specification.staggered and (sum(left) + sum(right)) % 2
            else 1.0
        )
        coefficient = phase / normalizer
        left_orbital = site_to_orbital[left]
        right_orbital = site_to_orbital[right]
        terms.extend(
            (
                (
                    coefficient,
                    (
                        (left_orbital, 1, True),
                        (left_orbital, -1, True),
                        (right_orbital, -1, False),
                        (right_orbital, 1, False),
                    ),
                ),
                (
                    coefficient,
                    (
                        (right_orbital, 1, True),
                        (right_orbital, -1, True),
                        (left_orbital, -1, False),
                        (left_orbital, 1, False),
                    ),
                ),
            )
        )
    return netket_fermion_operator(hilbert, terms, conserving="auto")


def _resolve_netket_measurement_observable(hilbert, ansatz, observable):
    """Compile declarative NetKet measurement observables on demand."""
    if isinstance(observable, NetKetEtaPairObservable):
        return _netket_eta_pair_operator(hilbert, ansatz, observable)
    return observable


def _normalize_fermion_observables(hilbert, observables, *, site_order=None):
    """Resolve an ``{name: operator_or_terms}`` mapping to NetKet operators.

    Declarative symbolic operators request NetKet's conserving fermion
    specialization when possible. Non-conserving observables safely fall back
    to the generic operator, so pairing or spin-changing measurements retain
    their physical meaning.
    """
    from .api import OperatorSum
    if observables is None:
        return None
    if not isinstance(observables, dict):
        raise TypeError(
            "observables must be a {name: operator_or_terms} mapping."
        )
    resolved = {}
    for name, spec in observables.items():
        if hasattr(spec, "hilbert"):
            resolved[str(name)] = spec
        elif isinstance(spec, OperatorSum):
            resolved[str(name)] = compile_operator_sum_netket(
                hilbert,
                spec,
                site_order=site_order,
                conserving="auto",
            )
        else:
            resolved[str(name)] = netket_fermion_operator(
                hilbert,
                spec,
                conserving="auto",
            )
    return resolved


def _looks_like_native_fermion_terms(obj):
    """True for a native Pepsy ``SymHamiltonian`` or coordinate-keyed mapping."""
    if obj is None or hasattr(obj, "hilbert"):
        return False
    terms_map = getattr(obj, "terms", None)
    if terms_map is not None and hasattr(terms_map, "keys"):
        return True
    return hasattr(obj, "keys")


def _native_fermion_terms_mapping(obj):
    """Return the coordinate/int-keyed terms mapping from a native source."""
    terms_map = getattr(obj, "terms", None)
    if terms_map is not None and hasattr(terms_map, "keys"):
        return terms_map
    return obj


def build_fermi_hubbard_vmc(
    peps,
    *,
    Lx=None,
    Ly=None,
    t=None,
    U=None,
    fermion=None,
    terms=None,
    n_fermions_per_spin=None,
    sector=None,
    pbc=None,
    edges=None,
    graph=None,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    register_stable_svd=True,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
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
    """Create NetKet VMC objects for a fixed-sector Fermi-Hubbard PEPS.

    ``fermion`` and ``terms`` may be the native Pepsy objects used during
    imaginary-time evolution. ``terms`` is inspected only to infer the
    lattice edges and which Hamiltonian axes are periodic; its operator
    coefficients (including any chemical-potential or on-site shifts) are NOT
    used to build the Hamiltonian.  The Hamiltonian is always NetKet's
    ``FermiHubbardJax`` with the resolved ``t``/``U`` (see below), so pass the
    hopping/interaction through explicit ``t`` and ``U`` rather than through
    ``terms``. ``sector`` or
    ``n_fermions_per_spin`` accepts either ``(N_up, N_down)`` or Pepsy's
    ``setup.spin_occupations`` mapping.  The evolved PEPS is prepared for JAX
    internally, and its variational parameters use quimb's native
    ``qtn.pack``/``qtn.unpack`` representation.

    When available, NetKet's ``FermiHubbardJax`` Hamiltonian is used. Newer
    NetKet releases that removed that convenience constructor use the
    equivalent conserving second-quantized operator instead. No
    chemical-potential term is added, which is equivalent to setting ``MU=0``
    once the spin sector is fixed.

    When ``register_stable_svd`` is True (default) and ``contraction`` is an
    SVD-based approximation (``hotrg``/``ctmrg``/``boundary``), Pepsy installs
    a JAX thin-SVD pullback that safely handles Quimb's fixed-rank truncation.
    Set it False to keep your own autoray SVD registration.
    """
    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    nk = _require_netket()
    if (Lx is None) != (Ly is None):
        raise ValueError("Lx and Ly must be supplied together or both omitted.")
    if Lx is None:
        Lx, Ly = _infer_lattice_shape_from_peps(peps)
    Lx, Ly = int(Lx), int(Ly)
    n_sites = int(Lx) * int(Ly)
    if sector is not None:
        if n_fermions_per_spin is not None:
            raise ValueError(
                "Pass only one of sector and n_fermions_per_spin."
            )
        n_fermions_per_spin = sector
    n_fermions_per_spin = _coerce_spin_sector(n_fermions_per_spin)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_fermions_per_spin = tuple(int(value) for value in n_fermions_per_spin)
    if len(n_fermions_per_spin) != 2:
        raise ValueError("n_fermions_per_spin must contain (N_up, N_down).")
    if fermion is not None and not getattr(fermion, "spinful", True):
        raise NotImplementedError(
            "build_fermi_hubbard_vmc supports spinful fermions only."
        )
    t = 1.0 if t is None else t
    U = 8.0 if U is None else U
    if pbc is None:
        pbc = _infer_pbc_from_fermi_terms(terms, Lx, Ly)

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=tuple(n_fermions_per_spin),
    )
    if graph is None:
        if edges is None:
            edges = _edges_from_fermi_terms(terms, Lx, Ly)
            if not edges:
                edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = _build_netket_fermi_hubbard_operator(
        hilbert,
        graph,
        n_sites=n_sites,
        t=t,
        U=U,
    )
    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    if register_stable_svd:
        _maybe_register_stable_jax_svd(contraction)

    peps = prepare_fermionic_peps_for_netket(peps)
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin)
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
        variables=_netket_variables_from_ansatz(ansatz, param_dtype),
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


def build_fermion_vmc(
    peps,
    *,
    fermion=None,
    hamiltonian=None,
    terms=None,
    observables=None,
    config=None,
    Lx=None,
    Ly=None,
    n_fermions_per_spin=None,
    sector=None,
    pbc=None,
    edges=None,
    graph=None,
    conserving=False,
    contraction="exact",
    chi=None,
    cutoff=0.0,
    contraction_opts=None,
    register_stable_svd=True,
    n_samples=1024,
    n_chains=16,
    n_discard_per_chain=32,
    chunk_size=256,
    sampling=None,
    sampler_chunk_size=None,
    sampler_sweep_size=None,
    seed=None,
    sampler_seed=None,
    use_sr=False,
    max_sr_params=5_000,
    sr_diag_shift=0.01,
    sr_diag_scale=None,
    sr_qgt="auto",
    sr_solver=None,
    sr_solver_restart=False,
    param_dtype=None,
    verify_columns=False,
    progress=False,
):
    """Create a NetKet VMC setup for a general spinful-fermion model.

    Unlike :func:`build_fermi_hubbard_vmc`, the Hamiltonian is not restricted to
    NetKet's ``FermiHubbardJax``. Define the model in one of these ways:

    * ``fermion`` **plus** native ``terms=`` / ``hamiltonian=`` -- pass the
      authoritative native Pepsy ``SymHamiltonian`` (from
      ``fermion.hamiltonian(...)``) or its coordinate-keyed ``.terms`` mapping.
      The builder infers integer lattice ``edges`` and periodic axes (``pbc``)
      directly from those terms, then compiles the supplied local operators to
      a matching NetKet fermion operator. Couplings never live on ``fermion``.
    * ``terms`` -- an explicit list of symbolic ``(coefficient, ops)`` terms
      (see :func:`netket_fermion_operator`) for a custom fermionic model.
    * ``OperatorSum`` -- the backend-neutral term representation shared with
      Torch VMC; it is compiled to a NetKet fermion/local operator.
    * ``hamiltonian`` -- an already-built NetKet fermion operator.

    ``edges`` / ``graph`` / ``pbc``, when given explicitly, take precedence over
    any geometry inferred from native terms.

    When omitted, ``Lx``/``Ly`` are inferred from the PEPS rectangular site
    layout, with a coordinate-keyed native Hamiltonian as a fallback.  Native
    coordinate or row-major integer edge keys provide the graph and periodic
    boundary inference.  Pass :class:`~pepsy.vmc.ContractionConfig` and
    :class:`~pepsy.vmc.SamplingConfig` through ``contraction=`` and
    ``sampling=`` to keep those settings in one validated object.

    ``progress=True`` shows an eight-stage setup bar (settings, Hilbert/graph,
    Hamiltonian, PEPS packing, JAX model, sampler, MCState, and optional SR
    preconditioner). The returned
    setup stores the corresponding :class:`NetKetBuildTiming` in
    ``build_timing``. This setup timing is distinct from the lazy JAX
    sampler/amplitude/energy compilation reported by :meth:`warmup`.

    ``conserving="auto"`` option asks NetKet to convert the operator to its
    experimental particle-number/spin-conserving representation. That can
    trigger a one-time Numba compilation during the first build; it is an
    optional runtime optimization, not a physics change. The default
    ``conserving=False`` builds the ordinary exact NetKet fermion operator
    immediately. ``sampler_sweep_size`` is passed to NetKet's Metropolis sampler as the
    number of proposals between retained samples. When omitted, NetKet uses
    the Hilbert-space size (``2 * Lx * Ly`` for this spinful model), so set it
    explicitly when you want the Markov-chain work to be obvious and
    reproducible. ``SamplingConfig.burn_in`` remains the per-chain discard
    count before retained samples.

    For a compact call, pass a :class:`NetKetVMCConfig` as ``config``. Its
    numerical fields override the corresponding legacy keywords, while the
    explicit ``fermion``/``hamiltonian`` inputs remain the model definition.

    ``observables`` is an optional ``{name: operator_or_terms}`` mapping stored
    on the returned setup; call :meth:`NetKetPEPSVMC.measure` to evaluate them
    (see :func:`standard_fermion_observables` for common choices). All
    sampler / state / SR / contraction options match
    :func:`build_fermi_hubbard_vmc`, and the returned setup exposes the same
    :meth:`NetKetPEPSVMC.warmup` and :meth:`NetKetPEPSVMC.optimize` helpers.
    """
    if config is not None:
        if not isinstance(config, NetKetVMCConfig):
            raise TypeError("config must be a NetKetVMCConfig or None.")
        contraction = config.contraction
        sampling = config.sampling
        sampler_sweep_size = config.sampler_sweep_size
        conserving = config.conserving
        use_sr = config.use_sr
        param_dtype = config.param_dtype
        verify_columns = config.verify_columns
        progress = config.progress

    build_started = time.perf_counter()
    build_bar = _make_progress_bar(
        total=8, desc="Build NetKet VMC", enabled=progress
    )
    stage_started = build_started
    stage_seconds = {}

    def mark_stage(name):
        nonlocal stage_started
        now = time.perf_counter()
        elapsed = now - stage_started
        stage_seconds[name] = elapsed
        stage_started = now
        if build_bar is not None:
            build_bar.set_postfix_str(f"{name}: {elapsed:.1f}s")
            build_bar.update(1)

    (
        n_samples,
        n_chains,
        n_discard_per_chain,
        chunk_size,
        seed,
        sampler_seed,
    ) = _resolve_sampling_build_config(
        sampling,
        n_samples=n_samples,
        n_chains=n_chains,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=chunk_size,
        seed=seed,
        sampler_seed=sampler_seed,
    )
    contraction, chi, cutoff, contraction_opts = _resolve_netket_contraction(
        contraction,
        chi,
        cutoff,
        contraction_opts,
    )
    nk = _require_netket()
    mark_stage("settings")
    from .api import OperatorSum
    common_hamiltonian = hamiltonian if isinstance(hamiltonian, OperatorSum) else None
    common_terms = terms if isinstance(terms, OperatorSum) else None
    if common_hamiltonian is not None and common_terms is not None:
        raise ValueError("Pass the common OperatorSum as hamiltonian or terms, not both.")
    common_operator_sum = (
        common_hamiltonian
        if common_hamiltonian is not None
        else common_terms
    )
    if fermion is not None and not getattr(fermion, "spinful", True):
        raise NotImplementedError(
            "build_fermion_vmc supports spinful fermions; use the sparse/torch "
            "path for spinless models."
        )

    # Classify native inputs before resolving the lattice shape.  This lets a
    # coordinate-keyed SymHamiltonian provide geometry when the PEPS wrapper
    # does not expose rectangular ``sites`` metadata.
    prebuilt_operator = None
    symbolic_terms = None
    native_terms = None
    if common_operator_sum is not None:
        native_terms = common_operator_sum
    elif hamiltonian is not None:
        if hasattr(hamiltonian, "hilbert"):
            prebuilt_operator = hamiltonian
        elif _looks_like_native_fermion_terms(hamiltonian):
            native_terms = _native_fermion_terms_mapping(hamiltonian)
        else:
            raise ValueError(
                "hamiltonian must be a NetKet operator or a native Pepsy "
                "SymHamiltonian / coordinate-keyed terms mapping."
            )
    if common_operator_sum is None and terms is not None:
        if _looks_like_native_fermion_terms(terms):
            native_terms = _native_fermion_terms_mapping(terms)
        else:
            symbolic_terms = terms

    if (Lx is None) != (Ly is None):
        raise ValueError("Lx and Ly must be supplied together or both omitted.")
    if Lx is None:
        try:
            Lx, Ly = _infer_lattice_shape_from_peps(peps)
        except ValueError as peps_error:
            inferred_shape = _infer_lattice_shape_from_fermi_terms(native_terms)
            if inferred_shape is None:
                raise peps_error
            Lx, Ly = inferred_shape
    Lx, Ly = int(Lx), int(Ly)
    n_sites = Lx * Ly
    if sector is not None:
        if n_fermions_per_spin is not None:
            raise ValueError(
                "Pass only one of sector and n_fermions_per_spin."
            )
        n_fermions_per_spin = sector
    n_fermions_per_spin = _coerce_spin_sector(n_fermions_per_spin)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_fermions_per_spin = tuple(int(value) for value in n_fermions_per_spin)
    if len(n_fermions_per_spin) != 2:
        raise ValueError("n_fermions_per_spin must contain (N_up, N_down).")

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=n_fermions_per_spin,
    )

    # Infer lattice geometry (edges) and periodicity from native terms when
    # they were not given explicitly, mirroring build_fermi_hubbard_vmc.
    if pbc is None:
        pbc = (
            False
            if common_operator_sum is not None
            else _infer_pbc_from_fermi_terms(native_terms, Lx, Ly)
        )
    if graph is None:
        if edges is None:
            edges = (
                _edges_from_operator_sum(common_operator_sum, Lx, Ly)
                if common_operator_sum is not None
                else _edges_from_fermi_terms(native_terms, Lx, Ly)
            )
            if not edges:
                edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)
    elif edges is None:
        edges = tuple(tuple(edge) for edge in graph.edges())
    mark_stage("Hilbert/graph")

    # Build the NetKet Hamiltonian.
    if common_operator_sum is not None:
        hamiltonian = compile_operator_sum_netket(
            hilbert,
            common_operator_sum,
            site_order=_row_major_sites(Lx, Ly),
            conserving=conserving,
        )
    elif prebuilt_operator is not None:
        hamiltonian = prebuilt_operator
    elif symbolic_terms is not None:
        hamiltonian = netket_fermion_operator(
            hilbert, symbolic_terms, conserving=conserving
        )
    elif native_terms is not None:
        if fermion is None:
            raise ValueError(
                "Native Pepsy terms require fermion=... so their local "
                "symmetry and basis can be validated for NetKet."
            )
        native_symbolic_terms, native_constant = _native_fermi_hubbard_terms_to_netket(
            fermion,
            native_terms,
            site_order=_row_major_sites(Lx, Ly),
        )
        hamiltonian = netket_fermion_operator(
            hilbert,
            native_symbolic_terms,
            constant=native_constant,
            conserving=conserving,
        )
    elif fermion is not None:
        raise ValueError(
            "build_fermion_vmc requires explicit native hamiltonian=... or "
            "terms=... with fermion=.... Fermion stores local symmetry and "
            "backend conventions, not t/U/V/mu couplings."
        )
    else:
        raise ValueError(
            "Provide fermion=... plus native terms=/hamiltonian=..., a "
            "symbolic terms=..., or an already-built NetKet hamiltonian=... "
            "to define the model for build_fermion_vmc."
        )
    mark_stage("Hamiltonian")

    observable_ops = _normalize_fermion_observables(
        hilbert,
        observables,
        site_order=_row_major_sites(Lx, Ly),
    )

    if register_stable_svd:
        _maybe_register_stable_jax_svd(contraction)

    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    peps = prepare_fermionic_peps_for_netket(peps)
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    _warn_flat_z2_ansatz_fixed_u1u1_sector(ansatz, n_fermions_per_spin)
    mark_stage("PEPS packing")
    model = make_fermionic_peps_log_amplitude_model(
        ansatz,
        columns,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        param_dtype=param_dtype,
    )
    mark_stage("JAX model")
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
    if sampler_sweep_size is not None:
        sampler_kwargs["sweep_size"] = _check_positive_int(
            "sampler_sweep_size",
            sampler_sweep_size,
        )
    sampler = nk.sampler.MetropolisFermionHop(hilbert, **sampler_kwargs)
    mark_stage("sampler")
    vstate = nk.vqs.MCState(
        sampler,
        model,
        variables=_netket_variables_from_ansatz(ansatz, param_dtype),
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        chunk_size=_check_positive_int("chunk_size", chunk_size),
        seed=seed,
        sampler_seed=sampler_seed,
    )
    mark_stage("MCState")
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
    mark_stage("SR preconditioner")
    if build_bar is not None:
        build_bar.set_postfix_str(
            f"done: {time.perf_counter() - build_started:.1f}s"
        )
        build_bar.close()
    build_timing = NetKetBuildTiming(
        settings_seconds=stage_seconds["settings"],
        geometry_seconds=stage_seconds["Hilbert/graph"],
        hamiltonian_seconds=stage_seconds["Hamiltonian"],
        peps_seconds=stage_seconds["PEPS packing"],
        model_seconds=stage_seconds["JAX model"],
        sampler_seconds=stage_seconds["sampler"],
        total_seconds=time.perf_counter() - build_started,
        preconditioner_seconds=stage_seconds["SR preconditioner"],
        state_seconds=stage_seconds["MCState"],
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
        build_timing=build_timing,
        columns=columns,
        observables=observable_ops,
    )


def build_netket_vmc(
    problem,
    *,
    fermion=None,
    contraction=None,
    sampling=None,
    **kwargs,
):
    """Build the portable NetKet façade from a :class:`VMCProblem`.

    The portable fermion path targets NetKet's fixed ``U1U1`` spin-orbital
    Hilbert space. Native builders remain available for spin models and for
    backend-specific sampler/driver choices.
    """
    from .api import (
        ContractionConfig,
        SamplingConfig,
        VMCBackendCapabilityError,
        VMCProblem,
    )

    if not isinstance(problem, VMCProblem):
        raise TypeError("problem must be a VMCProblem.")
    if sampling is not None and not isinstance(sampling, SamplingConfig):
        raise TypeError("sampling must be a SamplingConfig or None.")
    if problem.symmetry not in {None, "U1U1"}:
        raise VMCBackendCapabilityError(
            "The portable NetKet fermion setup currently supports only the "
            "fixed U1U1 sector; use build_torch_vmc for "
            f"symmetry={problem.symmetry!r}."
        )
    if problem.site_order is not None:
        raise VMCBackendCapabilityError(
            "build_netket_vmc uses NetKet's fixed row-major site order. "
            "Use build_fermion_vmc directly for an explicitly remapped model."
        )
    if sampling is not None:
        unsupported = []
        if sampling.sweep_size != 1:
            unsupported.append("sweep_size (formerly thin)")
        if sampling.proposal is not None:
            unsupported.append("proposal")
        if unsupported:
            raise VMCBackendCapabilityError(
                "NetKet cannot honour portable SamplingConfig."
                f"{', '.join(unsupported)}; configure its native sampler directly."
            )
    if contraction is None:
        contraction = ContractionConfig()
    setup = build_fermion_vmc(
        problem.peps,
        fermion=fermion,
        hamiltonian=problem.hamiltonian,
        observables=problem.observables,
        contraction=contraction,
        sampling=sampling,
        **kwargs,
    )
    return NetKetVMCSetup(setup=setup, problem=problem)


def build_sparse_fermi_hubbard_vmc(
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
    initial_configs=None,
    init_max_attempts=16,
    init_max_states=50_000,
    amplitude_floor=0.0,
    seed=None,
    sampler_seed=None,
    dtype=None,
    device=None,
    mode_order="down-up",
    verify_columns=False,
):
    """Create a sparse-block PEPS VMC setup for the Fermi-Hubbard model.

    This path is intended for Symmray block-sparse fermionic PEPS, including
    ``U1``, ``U1U1``, and ``Z2Z2`` tensors that cannot yet be used by NetKet's
    jitted ``MCState``. It is a Pepsy VMC loop with NetKet Hilbert/graph/Hamiltonian
    metadata rather than an ``nk.driver.VMC`` instance: Pepsy's torch kernels
    do Metropolis sweeps and local-energy evaluation with exact, HOTRG, CTMRG,
    or boundary contractions.
    """
    nk = _require_netket()
    n_sites = int(Lx) * int(Ly)
    if n_fermions_per_spin is None:
        n_fermions_per_spin = (n_sites // 2, n_sites // 2)
    n_up, n_down = (int(x) for x in n_fermions_per_spin)

    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_sites,
        s=1 / 2,
        n_fermions_per_spin=(n_up, n_down),
    )
    if graph is None:
        if edges is None:
            edges = square_lattice_edges(Lx, Ly, pbc=pbc)
        graph = nk.graph.Graph(edges=tuple(edges), n_nodes=n_sites)

    hamiltonian = _build_netket_fermi_hubbard_operator(
        hilbert,
        graph,
        n_sites=n_sites,
        t=t,
        U=U,
    )
    columns = netket_spin_orbital_columns(hilbert)
    if verify_columns:
        verify_netket_spin_columns(hilbert, columns)

    from .torch import (  # pylint: disable=import-outside-toplevel
        TorchPEPSAmplitude,
        _require_torch,
        random_spinful_configs,
    )

    torch = _require_torch()
    ansatz = pack_fermionic_peps_ansatz(peps, lattice_shape=(Lx, Ly))
    encoding = _fermion_site_encoding_from_phys_charges(ansatz.phys_charges)
    model = TorchPEPSAmplitude(
        peps,
        contraction=contraction,
        chi=chi,
        cutoff=cutoff,
        contraction_opts=contraction_opts,
        dtype=dtype,
        device=device,
        site_order=ansatz.config_sites,
    )
    init_generator = _make_torch_generator(seed, device=device)
    sampler_generator = _make_torch_generator(sampler_seed, device=device)
    n_samples = _check_positive_int("n_samples", n_samples)
    init_max_attempts = _check_positive_int("init_max_attempts", init_max_attempts)
    init_max_states = _check_positive_int("init_max_states", init_max_states)
    if n_samples is None:
        raise ValueError("n_samples must be a positive integer.")

    if initial_configs is not None:
        configs = torch.as_tensor(initial_configs, dtype=torch.long, device=device)
        if configs.ndim == 1:
            configs = configs.reshape(1, -1)
        if tuple(configs.shape) != (n_samples, n_sites):
            raise ValueError(
                "initial_configs must have shape "
                f"({n_samples}, {n_sites}), got {tuple(configs.shape)}."
            )
        amplitudes = model(configs)
        keep = _nonzero_amplitude_mask(
            amplitudes,
            amplitude_floor=amplitude_floor,
        )
        if not bool(keep.all()):
            raise ValueError(
                "initial_configs include zero, non-finite, or below-floor "
                "PEPS amplitudes; pass configurations inside the ansatz support."
            )
    else:
        kept_configs = []
        kept_amplitudes = []
        n_kept = 0
        for _ in range(init_max_attempts):
            candidate = random_spinful_configs(
                n_samples,
                n_sites,
                n_up,
                n_down,
                encoding=encoding,
                device=device,
                generator=init_generator,
            )
            candidate_amplitudes = model(candidate)
            keep = _nonzero_amplitude_mask(
                candidate_amplitudes,
                amplitude_floor=amplitude_floor,
            )
            if not bool(keep.any()):
                continue
            kept_configs.append(candidate[keep])
            kept_amplitudes.append(candidate_amplitudes[keep])
            n_kept += int(keep.sum().item())
            if n_kept >= n_samples:
                break

        if (
            n_kept < n_samples
            and init_max_states is not None
            and hilbert.n_states <= init_max_states
        ):
            states = np.asarray(hilbert.all_states()).astype(np.int32)
            phys = occupation_to_phys_indices(
                states,
                columns,
                site_to_orb=ansatz.site_to_orb,
                phys_charges=ansatz.phys_charges,
            )
            candidate = torch.as_tensor(phys, dtype=torch.long, device=device)
            candidate_amplitudes = model(candidate)
            keep = _nonzero_amplitude_mask(
                candidate_amplitudes,
                amplitude_floor=amplitude_floor,
            )
            if bool(keep.any()):
                support_configs = candidate[keep]
                support_amplitudes = candidate_amplitudes[keep]
                choice = torch.randint(
                    support_configs.shape[0],
                    (n_samples,),
                    device=support_configs.device,
                    generator=init_generator,
                )
                kept_configs = [support_configs[choice]]
                kept_amplitudes = [support_amplitudes[choice]]
                n_kept = n_samples

        if n_kept < n_samples:
            raise RuntimeError(
                "Could not initialize enough non-zero PEPS walkers in the "
                "requested Fermi-Hubbard sector. Pass initial_configs from the "
                "ansatz support, increase init_max_attempts, or raise "
                "init_max_states for small exact sectors."
            )

        configs = torch.cat(kept_configs, dim=0)[:n_samples]
        amplitudes = torch.cat(kept_amplitudes, dim=0)[:n_samples]

    return NetKetSparseFermiHubbardVMC(
        hilbert=hilbert,
        graph=graph,
        hamiltonian=hamiltonian,
        model=model,
        ansatz=ansatz,
        columns=columns,
        torch_graph=graph,
        encoding=encoding,
        configs=configs,
        amplitudes=amplitudes,
        n_fermions_per_spin=(n_up, n_down),
        t=t,
        U=U,
        mode_order=mode_order,
        generator=sampler_generator,
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


def warmup_netket_vmc(
    setup,
    *,
    hamiltonian=None,
    progress=True,
    verbose=None,
    resource_monitor=False,
    resource_interval=0.25,
):
    """Force XLA compilation of a NetKet VMC setup before ``driver.run(...)``.

    The first optimization step compiles the sampler, the jitted
    log-amplitude model (including the CTMRG / boundary-MPS contraction), and
    the local-energy/gradient kernel. That one-time compile cost is folded
    into the first ``tqdm`` tick, so the NetKet progress-bar ETA is misleading
    until it clears. Calling this once runs those same paths up front without
    updating parameters, so the reported ETA is meaningful.

    Parameters
    ----------
    setup:
        A NetKet VMC setup (for example from :func:`build_fermi_hubbard_vmc`)
        exposing ``vstate`` and ``hamiltonian`` attributes, or a bare NetKet
        variational state.
    hamiltonian:
        Operator to evaluate; defaults to ``setup.hamiltonian``.
    progress:
        When True (default), show a small three-stage ``tqdm`` bar (sampler,
        one jitted log-amplitude chunk, then local energy plus gradient) while
        compiling.
    verbose:
        Print a short text message instead of / in addition to the bar. When
        ``None`` (default) it prints only if the progress bar is unavailable.
    resource_monitor:
        When True, use :class:`xyzpy.MemoryMonitor` when available to sample
        host RSS, and ``nvidia-smi`` to sample GPU memory and utilization.
        A compact report is printed after warmup. This remains opt-in because
        GPU utilization sampling launches a lightweight subprocess.
    resource_interval:
        Seconds between GPU samples while ``resource_monitor=True``. Host RSS
        is sampled more frequently by ``xyzpy`` so peak tracking has a quick
        teardown.

    Returns
    -------
    float
        Elapsed wall-clock seconds spent compiling and evaluating once.
    """
    import time  # pylint: disable=import-outside-toplevel

    vstate = getattr(setup, "vstate", setup)
    if hamiltonian is None:
        hamiltonian = getattr(setup, "hamiltonian", None)
    if hamiltonian is None:
        raise ValueError(
            "warmup_netket_vmc needs a Hamiltonian: pass hamiltonian=... or a "
            "setup exposing a .hamiltonian attribute."
        )
    bar = _make_progress_bar(
        total=3, desc="Warmup 1/3: sampler", enabled=progress
    )
    if verbose is None:
        verbose = bar is None
    if verbose:
        print(
            "Compiling NetKet VMC kernels (sampler, log amplitude, energy gradient)...",
            flush=True,
        )
    monitor = (
        _NetKetResourceMonitor(interval=resource_interval)
        if resource_monitor
        else None
    )
    if monitor is not None:
        monitor.start()
    started = time.perf_counter()
    try:
        vstate.reset()
        # Stage 1: compile the Metropolis sampler and collect the retained
        # batch used by the following two warmup stages.
        stage_started = time.perf_counter()
        samples = vstate.samples
        sampler_seconds = time.perf_counter() - stage_started
        n_amplitude_rows = int(np.prod(samples.shape[:-1]))
        sample_chains = int(samples.shape[0]) if samples.ndim >= 3 else 1
        samples_per_chain = n_amplitude_rows // sample_chains
        sample_summary = (
            f"{n_amplitude_rows} retained = {sample_chains} chains x "
            f"{samples_per_chain}/chain"
        )
        if bar is not None:
            bar.set_postfix_str(f"{sampler_seconds:.1f}s | {sample_summary}")
            bar.update(1)
        # Stage 2: compile NetKet's public JIT log-amplitude route. Use the
        # configured forward chunk shape rather than the whole retained batch:
        # VMC's chunked local-energy/gradient kernels use that shape in
        # production.
        chunk_size = getattr(vstate, "chunk_size", None)
        if chunk_size is None:
            amplitude_rows = n_amplitude_rows
        else:
            amplitude_rows = min(n_amplitude_rows, int(chunk_size))
        amplitude_configs = samples.reshape((-1, samples.shape[-1]))[:amplitude_rows]
        if bar is not None:
            bar.set_description_str(
                f"Warmup 2/3: JIT log amplitudes ({amplitude_rows} rows)"
            )
        amplitude_started = time.perf_counter()
        log_value = getattr(vstate, "log_value", None)
        if not callable(log_value):
            raise TypeError(
                "warmup_netket_vmc requires an MCState with a "
                "log_value(configs) method."
            )
        _block_until_ready(log_value(amplitude_configs))
        amplitude_seconds = time.perf_counter() - amplitude_started
        if bar is not None:
            bar.set_postfix_str(
                f"{amplitude_seconds:.1f}s | {amplitude_rows}-row JIT chunk | "
                f"{sample_summary}"
            )
            bar.update(1)
        # Stage 3: compile the driver-dominant local-energy and gradient route.
        # ``expect_and_grad`` is what NetKet's ordinary VMC driver calls before
        # updating parameters, so this deliberately leaves the PEPS unchanged.
        if bar is not None:
            bar.set_description_str("Warmup 3/3: local energy + gradient")
        energy_started = time.perf_counter()
        stats, gradient = vstate.expect_and_grad(hamiltonian)
        # Resolve lazy device arrays so compilation is finished before timing.
        mean = getattr(stats, "mean", stats)
        _block_until_ready(mean)
        _block_until_ready(gradient)
        energy_seconds = time.perf_counter() - energy_started
        elapsed = time.perf_counter() - started
        if bar is not None:
            energy_text = (
                f"{energy_seconds:.1f}s | E={float(np.asarray(mean).real):+.6f} | "
                "local-energy and gradient kernels"
            )
            bar.set_postfix_str(energy_text)
            bar.update(1)
    finally:
        # Resource polling must not outlive a failed JIT/compilation attempt.
        try:
            if bar is not None:
                bar.close()
        finally:
            resource_usage = monitor.stop() if monitor is not None else None
    if verbose:
        print(
            "Warmup complete: "
            f"{elapsed:.1f}s total (sampler {sampler_seconds:.1f}s, "
            f"JIT log amplitudes {amplitude_seconds:.1f}s, energy + gradient "
            f"{energy_seconds:.1f}s); {sample_summary}; "
            f"stage 2 compiles a representative {amplitude_rows}-row forward "
            "chunk. Stage 3 uses Hamiltonian-connected configurations and "
            "compiles the backward pass used by VMC.",
            flush=True,
        )
    if resource_usage is not None:
        print(resource_usage.summary("NetKet warmup resources"), flush=True)
    return elapsed
