"""PEPS sampling with belief-propagation proposals."""

from __future__ import annotations

from pepsy._internal.quimb import call_quimb_2d
import math
from typing import Any, Iterable
import autoray as ar
import numpy as np
from tqdm import tqdm
from .results import (
    PEPSSampleResult,
)
from ._common import (
    _fermion_code_order,
    _infer_fermion_code_order,
)

__all__ = ['PepsBpSampler']


sample_d2bp = None

build_optimizer = None


def _to_dense_numpy(array):
    """Convert a dense or Symmray array to a NumPy array for BP sampling."""
    if hasattr(array, "to_dense"):
        array = array.to_dense()
    return np.asarray(ar.to_numpy(array))


def _prepare_bp_binary_network(tn, *, site_order=None, encoding=None):
    """Prepare a binary-output copy for Quimb's binary ``sample_d2bp``.

    Quimb's public D2BP sampler currently samples each output index from
    ``[0, 1]``. A four-state fermionic physical leg is therefore represented
    as two occupation legs, while all tensor data are converted to dense
    NumPy arrays in the private BP copy. The source network is never mutated.
    """
    network = getattr(tn, "tn", tn)
    if not hasattr(network, "sites") or not hasattr(network, "copy"):
        return network, {}, tuple(), (0, 1, 2, 3)

    sites = tuple(network.sites if site_order is None else site_order)
    missing = [site for site in sites if site not in network.sites]
    if missing:
        raise ValueError(f"site_order contains site(s) not in PEPS: {missing!r}")

    work = network.copy()
    code_order = (
        _fermion_code_order(encoding)
        if encoding is not None
        else _infer_fermion_code_order(network, sites)
    )
    split_inds = {}
    existing_inds = set(work.ind_map)

    for site in sites:
        tensor = work[site]
        site_ind = work.site_ind(site)
        try:
            axis = tensor.inds.index(site_ind)
        except ValueError as exc:
            raise ValueError(
                f"Could not locate physical index for PEPS site {site!r}."
            ) from exc

        data = _to_dense_numpy(tensor.data)
        physical_dim = int(data.shape[axis])
        if physical_dim == 2:
            if hasattr(tensor.data, "to_dense"):
                tensor.modify(data=data)
            continue
        if physical_dim != 4:
            raise ValueError(
                "Quimb BP sampling supports binary physical legs directly "
                "and spinful fermion legs through a four-state adapter; got "
                f"dimension {physical_dim} at site {site!r}."
            )

        # ``code_order`` maps flattened (up, down) bits back to the PEPS
        # physical-index order. Reshaping after this permutation gives BP two
        # binary output indices with the intended fermion convention.
        data = np.take(data, code_order, axis=axis)
        data = data.reshape(
            data.shape[:axis] + (2, 2) + data.shape[axis + 1:]
        )
        up_ind = f"{site_ind}_bp_up"
        down_ind = f"{site_ind}_bp_down"
        suffix = 0
        while up_ind in existing_inds or down_ind in existing_inds:
            suffix += 1
            up_ind = f"{site_ind}_bp_up_{suffix}"
            down_ind = f"{site_ind}_bp_down_{suffix}"
        existing_inds.update((up_ind, down_ind))
        inds = list(tensor.inds)
        inds[axis:axis + 1] = [up_ind, down_ind]
        tensor.modify(data=data, inds=inds)
        split_inds[site] = (up_ind, down_ind)

    return work, split_inds, sites, code_order


class PepsBpSampler:
    """Sample PEPS configurations and amplitudes with BP proposals.

    The sampler draws configurations from :func:`sample_d2bp`, records the BP
    proposal probability ``omega(x)``, contracts the projected PEPS amplitude
    ``p(x)``, and returns the ingredients for the PEPS norm estimator
    ``E_q[|p(x)|^2 / omega(x)]``.

    Quimb's public D2BP sampler currently samples binary output indices. For a
    four-state spinful PEPS this class transparently samples two binary
    occupation legs per site and maps them back to the requested local
    fermion encoding. Symmray inputs are densified only in the private BP
    proposal copy; the original network remains block-sparse.
    """

    def __init__(
        self,
        tn,
        *,
        optimizer=None,
        sample_kwargs: dict[str, Any] | None = None,
        encoding=None,
        site_order=None,
    ):
        self.tn = getattr(tn, "tn", tn)
        self.Lx = self.tn.Lx
        self.Ly = self.tn.Ly
        self.optimizer = optimizer
        self.sample_kwargs = dict(sample_kwargs or {})
        self.encoding = encoding
        (
            self._bp_tn,
            self._split_inds,
            self.site_order,
            self._code_order,
        ) = _prepare_bp_binary_network(
            self.tn,
            site_order=site_order,
            encoding=encoding,
        )

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
            from ..tensors.contractions import build_optimizer as _build_optimizer  # pylint: disable=import-outside-toplevel

            build_optimizer = _build_optimizer

        self.optimizer = build_optimizer(
            progbar=False,
            directory="cash",
            parallel=False,
            max_time="rate:1e8",
        )
        return self.optimizer

    def _config_list(self, config: dict[str, Any]) -> list[int]:
        if self._split_inds:
            out = []
            code_order = self._code_order
            # Invert the (up, down) -> physical-code map built above.
            code_from_bits = {
                (0, 0): code_order[0],
                (0, 1): code_order[1],
                (1, 0): code_order[2],
                (1, 1): code_order[3],
            }
            for site in self.site_order:
                up_ind, down_ind = self._split_inds[site]
                bits = (int(config[up_ind]), int(config[down_ind]))
                out.append(code_from_bits[bits])
            return out

        if self.site_order:
            out = []
            for site in self.site_order:
                try:
                    site_ind = self.tn.site_ind(site)
                except (AttributeError, KeyError):
                    site_ind = f"k{site[0]},{site[1]}"
                out.append(int(config[site_ind]))
            return out

        # Keep the small dummy-network/testing protocol backwards compatible.
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

        return sample_d2bp(self._bp_tn, **kwargs)

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

        def scaled_is_finite(value):
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return bool(np.isfinite(value[0]) and np.isfinite(value[1]))
            return bool(np.isfinite(value))

        def as_scaled(value):
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return value
            return self.mantissa_exponent10(value)

        if method == "mps":
            opts = {
                "optimize": optimizer,
                "strip_exponent": True,
            }
            result = call_quimb_2d(
                tn_flat.contract_boundary,
                max_bond=int(chi),
                mode="mps",
                final_contract_opts=opts,
                max_separation=max_separation,
                cutoff=cutoff,
                sequence=["xmin", "xmax", "ymin", "ymax"],
                equalize_norms=equalize_norms,
                progbar=False,
            )
            if not scaled_is_finite(result):
                opts["strip_exponent"] = False
                result = call_quimb_2d(
                    tn_flat.contract_boundary,
                    max_bond=int(chi),
                    mode="mps",
                    final_contract_opts=opts,
                    max_separation=max_separation,
                    cutoff=cutoff,
                    sequence=["xmin", "xmax", "ymin", "ymax"],
                    equalize_norms=equalize_norms,
                    progbar=False,
                )
            return as_scaled(result)

        if method == "ctmrg":
            opts = {
                "optimize": optimizer,
                "strip_exponent": True,
            }
            result = call_quimb_2d(
                tn_flat.contract_ctmrg,
                max_bond=int(chi),
                final_contract_opts=opts,
                max_separation=max_separation,
                cutoff=cutoff,
                inplace=False,
                equalize_norms=equalize_norms,
                progbar=False,
            )
            if not scaled_is_finite(result):
                opts["strip_exponent"] = False
                result = call_quimb_2d(
                    tn_flat.contract_ctmrg,
                    max_bond=int(chi),
                    final_contract_opts=opts,
                    max_separation=max_separation,
                    cutoff=cutoff,
                    inplace=False,
                    equalize_norms=equalize_norms,
                    progbar=False,
                )
            return as_scaled(result)

        if method == "exact":
            result = tn_flat.contract(all, optimize=optimizer, strip_exponent=True)
            if not scaled_is_finite(result):
                result = tn_flat.contract(all, optimize=optimizer, strip_exponent=False)
            return as_scaled(result)

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
