"""Samplers for MPS and PEPS tensor networks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

sample_d2bp = None
build_optimizer = None

__all__ = [
    "MpsSampleResult",
    "MpsSampler",
    "PEPSSampleResult",
    "PepsBpSampler",
    "VecSampler",
]


def _validate_one_d_to_two_d(
    one_d_to_two_d: dict[int, tuple[int, int]],
    *,
    expected_L: int | None = None,
) -> int:
    """Validate a complete 0..L-1 site map and return ``L``."""
    if not one_d_to_two_d:
        raise ValueError("one_d_to_two_d must contain at least one site.")
    L = int(expected_L) if expected_L is not None else len(one_d_to_two_d)
    if L < 1:
        raise ValueError("expected_L must be >= 1.")
    expected = set(range(L))
    got = set(one_d_to_two_d)
    if got != expected:
        raise ValueError(
            "one_d_to_two_d keys must be exactly consecutive site indices "
            f"0..{L - 1}; got {sorted(got)!r}."
        )
    for site, coord in one_d_to_two_d.items():
        if not (
            isinstance(coord, tuple)
            and len(coord) == 2
            and all(isinstance(value, int) for value in coord)
        ):
            raise TypeError(
                "one_d_to_two_d values must be (x, y) integer tuples; "
                f"site {site!r} maps to {coord!r}."
            )
    return L


@dataclass
class PEPSSampleResult:
    """Container for PEPS BP importance samples.

    Attributes
    ----------
    configs
        Sampled physical configurations in row-major ``[x * Ly + y]`` order.
    omegas
        Pair ``(mantissas, exponents)`` for BP proposal probabilities.
    ps
        Pair ``(mantissas, exponents)`` for sampled PEPS amplitudes.
    """

    configs: list[list[int]]
    omegas: tuple[list[float], list[int]]
    ps: tuple[list[Any], list[Any]]

    def __len__(self):
        """Return the number of sampled configurations."""
        return len(self.configs)


@dataclass
class MpsSampleResult:
    """Container for MPS samples with 2D coordinate mapping.

    Attributes
    ----------
    configs_1d : list[list[int]]
        Each entry is a list of length L with spin indices (0 or 1).
    configs_2d : list[np.ndarray]
        Each entry is a (Ly, Lx) int array with spin indices on the 2D lattice.
    probs : list[float]
        Born probability ``|⟨config|ψ⟩|²`` for each sample.
    Lx : int
        Lattice width.
    Ly : int
        Lattice height.
    """

    configs_1d: list[list[int]]
    configs_2d: list[np.ndarray]
    probs: list[float]
    Lx: int
    Ly: int

    def __len__(self):
        return len(self.configs_1d)

    def magnetizations(self) -> np.ndarray:
        """Per-sample magnetization ⟨M⟩ = (1/L) Σ (1 - 2·spin_i)."""
        L = self.Lx * self.Ly
        return np.array([
            np.sum(1 - 2 * np.array(c)) / L for c in self.configs_1d
        ])


class MpsSampler:
    """Sample from an MPS using quimb's canonical-form sampler.

    Handles GPU→CPU conversion and maps 1D MPS site indices to 2D lattice
    coordinates via a ``one_d_to_two_d`` mapping.

    Parameters
    ----------
    psi : MatrixProductState
        The MPS to sample from (can be on any backend).
    one_d_to_two_d : dict[int, tuple[int, int]]
        Mapping from 1D site index to (x, y) lattice coordinate.
    """

    def __init__(self, psi, one_d_to_two_d: dict[int, tuple[int, int]]):
        _validate_one_d_to_two_d(one_d_to_two_d, expected_L=getattr(psi, "L", None))
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1
        # Convert to numpy for quimb sampling compatibility
        self._psi = psi.copy()
        self._psi.apply_to_arrays(
            lambda x: x.get() if hasattr(x, "get") else np.asarray(x)
        )

    def sample(self, n_samples: int = 1, seed: int | None = None) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the MPS.

        Uses quimb's ``MatrixProductState.sample()`` which internally
        right-canonicalizes the MPS and sweeps left-to-right.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        configs_1d = []
        configs_2d = []
        probs = []

        for config, prob in self._psi.sample(n_samples, seed=seed):
            configs_1d.append(list(config))
            grid = np.zeros((self.Ly, self.Lx), dtype=int)
            for site_1d, spin in enumerate(config):
                x, y = self.one_d_to_two_d[site_1d]
                grid[y, x] = spin
            configs_2d.append(grid)
            probs.append(float(prob))

        return MpsSampleResult(
            configs_1d=configs_1d,
            configs_2d=configs_2d,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
        )


class VecSampler:
    """Sample from a dense state vector (e.g. from MpsOptimizer mode='exact').

    Computes Born probabilities ``p_i = |ψ_i|²`` and samples configurations
    from the resulting categorical distribution.

    Parameters
    ----------
    state : TensorNetwork, Tensor, or array-like
        The dense state. If a quimb TensorNetwork/Tensor, the physical indices
        are assumed to follow ``ind_id`` format (default ``'k{}'``). If a raw
        array, it is reshaped to a 1D vector of length 2^L.
    one_d_to_two_d : dict[int, tuple[int, int]]
        Mapping from 1D site index to (x, y) lattice coordinate.
    ind_id : str
        Format string for physical index names (default ``'k{}'``).
    """

    def __init__(
        self,
        state,
        one_d_to_two_d: dict[int, tuple[int, int]],
        ind_id: str = "k{}",
    ):
        L = _validate_one_d_to_two_d(one_d_to_two_d)
        self.one_d_to_two_d = one_d_to_two_d
        self.Lx = max(x for x, y in one_d_to_two_d.values()) + 1
        self.Ly = max(y for x, y in one_d_to_two_d.values()) + 1

        # Extract the state vector with correct index ordering
        vec = self._to_vector(state, L, ind_id)

        # Move to CPU if needed (cupy)
        if hasattr(vec, "get"):
            vec = vec.get()
        vec = np.asarray(vec, dtype=complex).ravel()
        expected_size = 2 ** L
        if vec.size != expected_size:
            raise ValueError(
                f"state vector size must be 2**L={expected_size} for L={L}; "
                f"got {vec.size}."
            )

        # Compute and cache Born probabilities
        probs = np.abs(vec) ** 2
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("state vector must have a finite non-zero norm.")
        probs /= total
        self._probs = probs
        self._L = L

    @staticmethod
    def _to_vector(state, L, ind_id):
        """Convert state to a flat vector with sites in 0..L-1 order."""
        import quimb.tensor as qtn  # noqa: F811

        if isinstance(state, qtn.TensorNetwork):
            # Contract to a single tensor with ordered physical indices
            inds = [ind_id.format(i) for i in range(L)]
            t = state.contract(all, output_inds=inds)
            return t.data.reshape(-1)
        if isinstance(state, qtn.Tensor):
            inds = [ind_id.format(i) for i in range(L)]
            return state.transpose(*inds).data.reshape(-1)
        # Raw array
        return np.asarray(state).reshape(-1)

    def sample(self, n_samples: int = 1, seed: int | None = None) -> MpsSampleResult:
        """Draw ``n_samples`` configurations from the state vector.

        Returns
        -------
        MpsSampleResult
            Contains 1D configs, 2D grids, and Born probabilities.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._probs), size=n_samples, p=self._probs)

        configs_1d = []
        configs_2d = []
        probs = []

        for idx in indices:
            # Convert flat index to binary configuration (big-endian: site 0 is MSB)
            config = [(idx >> (self._L - 1 - site)) & 1 for site in range(self._L)]
            configs_1d.append(config)

            grid = np.zeros((self.Ly, self.Lx), dtype=int)
            for site_1d, spin in enumerate(config):
                x, y = self.one_d_to_two_d[site_1d]
                grid[y, x] = spin
            configs_2d.append(grid)
            probs.append(float(self._probs[idx]))

        return MpsSampleResult(
            configs_1d=configs_1d,
            configs_2d=configs_2d,
            probs=probs,
            Lx=self.Lx,
            Ly=self.Ly,
        )


class PepsBpSampler:
    """Sample PEPS configurations and amplitudes with BP proposals.

    The sampler draws configurations from :func:`sample_d2bp`, records the BP
    proposal probability ``omega(x)``, contracts the projected PEPS amplitude
    ``p(x)``, and returns the ingredients for the PEPS norm estimator
    ``E_q[|p(x)|^2 / omega(x)]``.
    """

    def __init__(self, tn, *, optimizer=None, sample_kwargs: dict[str, Any] | None = None):
        self.tn = tn
        self.Lx = tn.Lx
        self.Ly = tn.Ly
        self.optimizer = optimizer
        self.sample_kwargs = dict(sample_kwargs or {})

    @staticmethod
    def mantissa_exponent10(w: float) -> tuple[float, int]:
        """Represent ``w`` as ``mantissa * 10**exponent``."""
        if w == 0:
            return 0.0, 0
        exponent = math.floor(math.log10(abs(w)))
        mantissa = w / (10 ** exponent)
        return mantissa, exponent

    def _get_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        global build_optimizer  # pylint: disable=global-statement
        if build_optimizer is None:
            from ..tensors.core import build_optimizer as _build_optimizer  # pylint: disable=import-outside-toplevel

            build_optimizer = _build_optimizer

        self.optimizer = build_optimizer(
            progbar=False,
            directory="cash",
            parallel=False,
            max_time="rate:1e8",
        )
        return self.optimizer

    def _config_list(self, config: dict[str, Any]) -> list[int]:
        out = [None] * (self.Lx * self.Ly)
        for i in range(self.Lx):
            for j in range(self.Ly):
                out[i * self.Ly + j] = int(config[f"k{i},{j}"])
        return out

    def _sample_d2bp(self, sample_seed, bp_kwargs=None):
        kwargs = {
            "max_iterations": 100,
            "tol": 1.0e-2,
            "seed": sample_seed,
            "optimize": "auto-hq",
            "damping": 0.0,
            "diis": False,
            "update": "parallel",
            "local_convergence": True,
            "progbar": False,
        }
        kwargs.update(self.sample_kwargs)
        if bp_kwargs:
            kwargs.update(bp_kwargs)
        kwargs["seed"] = sample_seed

        global sample_d2bp  # pylint: disable=global-statement
        if sample_d2bp is None:
            from quimb.tensor.belief_propagation import sample_d2bp as _sample_d2bp  # pylint: disable=import-outside-toplevel

            sample_d2bp = _sample_d2bp

        return sample_d2bp(self.tn, **kwargs)

    def _contract_sample(
        self,
        tn_flat,
        *,
        chi: int,
        method: str,
        max_separation: int,
        equalize_norms: bool,
        cutoff: float,
    ):
        optimizer = self._get_optimizer()

        if method == "mps":
            return tn_flat.contract_boundary(
                max_bond=int(chi),
                mode="mps",
                final_contract_opts={"optimize": optimizer, "strip_exponent": True},
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=["xmin", "xmax", "ymin", "ymax"],
                equalize_norms=equalize_norms,
                progbar=False,
            )

        if method == "ctmrg":
            return tn_flat.contract_ctmrg(
                max_bond=int(chi),
                final_contract_opts={"optimize": optimizer, "strip_exponent": True},
                max_separation=max_separation,
                cutoff=cutoff,
                inplace=False,
                equalize_norms=equalize_norms,
                progbar=False,
            )

        if method == "exact":
            return tn_flat.contract(all, optimize=optimizer, strip_exponent=True)

        raise ValueError(f"Unknown contraction method: {method!r}")

    def sample(
        self,
        *,
        chi: int = 12,
        samples: int = 1,
        method: str = "exact",
        seed: int | None = None,
        max_separation: int = 1,
        equalize_norms: bool = True,
        progbar: bool = False,
        cutoff: float = 0.0,
        bp_kwargs: dict[str, Any] | None = None,
    ) -> PEPSSampleResult:
        """Draw samples and contract the corresponding PEPS amplitudes.

        Parameters
        ----------
        bp_kwargs : dict, optional
            Override BP sampling parameters. Supported keys:
            max_iterations, tol, optimize, damping, diis, update,
            local_convergence, progbar.
        """
        configs: list[list[int]] = []
        omegas_mantissa: list[float] = []
        omegas_exponent: list[int] = []
        ps_mantissa: list[Any] = []
        ps_exponent: list[Any] = []

        sample_range: Iterable[int] = tqdm(
            range(int(samples)),
            desc="Sampling configs",
            disable=not progbar,
        )

        for sample_idx in sample_range:
            sample_seed = None if seed is None else int(seed) + sample_idx
            config_i, tn_flat, omega_i = self._sample_d2bp(sample_seed, bp_kwargs)
            mantissa, exponent = self._contract_sample(
                tn_flat,
                chi=chi,
                method=method,
                max_separation=max_separation,
                equalize_norms=equalize_norms,
                cutoff=cutoff,
            )

            configs.append(self._config_list(config_i))

            omega_mantissa, omega_exponent = self.mantissa_exponent10(float(omega_i))
            omegas_mantissa.append(omega_mantissa)
            omegas_exponent.append(omega_exponent)

            ps_mantissa.append(mantissa)
            ps_exponent.append(exponent)

        return PEPSSampleResult(
            configs=configs,
            omegas=(omegas_mantissa, omegas_exponent),
            ps=(ps_mantissa, ps_exponent),
        )
