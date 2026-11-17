"""Gate-stream evolution of density operators stored as PEPOs."""

import math
from numbers import Integral

import autoray as ar

from ..operators.gates import gate_simple
from ..backends import infer_backend_converter_from_sample
from ..backends.config import _patch_backend_string_namespace_key
from .peps import PepsOptimizer
from .sweep.environments import uses_symmray_arrays


class PepoOptimizer:
    """Evolve a PEPO by a stream of density-operator gate sandwiches.

    Each entry ``(G, where)`` applies ``G @ rho @ G.conj().T``.  The
    supported modes are local ``svd``, ``simple-update`` and boundary-MPS
    ``full-update``, and experimental single-layer ``dmt``.
    """

    def __init__(self, state, gates=None, chi=None, *, mode="svd",
                 boundary_chi=None, fit_mode="direct", cutoff="auto",
                 cutoff_mode="auto", contraction_opt="greedy",
                 boundary_kwargs=None, full_update_kwargs=None,
                 regions="L", dmt_tolerance=1e-8, dmt_max_nfev=200,
                 dmt_max_map_elements=1000000, dmt_cache=True,
                 dmt_cache_max_elements=1000000):
        if not all(hasattr(state, name) for name in ("upper_ind", "lower_ind", "Lx", "Ly")):
            raise TypeError("state must be a two-dimensional PEPO")
        if uses_symmray_arrays(state):
            raise NotImplementedError("PepoOptimizer currently requires dense PEPO arrays")
        if isinstance(chi, bool) or not isinstance(chi, Integral) or chi < 1:
            raise ValueError("chi must be a positive integer")
        if mode not in {"svd", "simple-update", "full-update", "dmt"}:
            raise ValueError("mode must be 'svd', 'simple-update', 'full-update', or 'dmt'")
        if fit_mode not in {"direct", "dmrg"}:
            raise ValueError("fit_mode must be 'direct' or 'dmrg'")
        _patch_backend_string_namespace_key()
        self.state = state.copy()
        self.gates = list(gates) if gates is not None else []
        self.chi, self.mode = int(chi), mode
        self.boundary_chi, self.fit_mode = boundary_chi, fit_mode
        self.cutoff, self.cutoff_mode = cutoff, cutoff_mode
        self.contraction_opt = contraction_opt
        self.boundary_kwargs = dict(boundary_kwargs or {})
        self.full_update_kwargs = dict(full_update_kwargs or {})
        self.regions = regions
        self.dmt_tolerance, self.dmt_max_nfev = float(dmt_tolerance), int(dmt_max_nfev)
        self.dmt_max_map_elements = int(dmt_max_map_elements)
        if not isinstance(dmt_cache, bool):
            raise TypeError("dmt_cache must be a bool")
        if (isinstance(dmt_cache_max_elements, bool) or not isinstance(dmt_cache_max_elements, Integral)
                or dmt_cache_max_elements < 1):
            raise ValueError("dmt_cache_max_elements must be a positive integer")
        self.dmt_cache = dmt_cache
        self.dmt_cache_max_elements = int(dmt_cache_max_elements)
        if mode == "dmt":
            import numpy as np
            if any(not isinstance(t.data, np.ndarray) for t in state):
                raise NotImplementedError("PEPO DMT prototype requires NumPy arrays")
            if any(t.data.dtype != np.dtype("complex128") for t in state):
                raise NotImplementedError("PEPO DMT prototype requires complex128 tensors")
            if state.is_cyclic_x() or state.is_cyclic_y():
                raise NotImplementedError("PEPO DMT prototype requires open boundaries")
            if boundary_chi == "auto":
                self.boundary_chi = max(4, 2 * self.chi)
                boundary_chi = self.boundary_chi
            if boundary_chi is not None and (isinstance(boundary_chi, bool)
                    or not isinstance(boundary_chi, Integral) or boundary_chi < 1):
                raise ValueError("DMT boundary_chi must be a positive integer or None")
            if boundary_chi is None and boundary_kwargs:
                raise ValueError("DMT boundary_kwargs requires boundary_chi")
            if set(self.boundary_kwargs) - {"cutoff", "n_iter"}:
                raise ValueError("DMT boundary_kwargs supports cutoff and n_iter")
            bc = float(self.boundary_kwargs.get("cutoff", 0.))
            if not math.isfinite(bc) or bc < 0:
                raise ValueError("DMT boundary cutoff must be nonnegative")
            ni = self.boundary_kwargs.get("n_iter", 10)
            if isinstance(ni, bool) or not isinstance(ni, Integral) or ni < 1:
                raise ValueError("DMT boundary n_iter must be a positive integer")
            if fit_mode == "direct" and "n_iter" in self.boundary_kwargs:
                raise ValueError("DMT boundary n_iter applies only to dmrg fitting")
            if not (1e-14 < self.dmt_tolerance < 1) or self.dmt_max_nfev < 1 or self.dmt_max_map_elements < 1:
                raise ValueError("invalid DMT tolerance or solver/map limit")
        self._records = []

    def _entries(self):
        entries = []
        for entry in self.gates:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("each gate must be a (gate, sites) pair")
            payload, sites = entry
            if len(sites) == 2 and all(isinstance(x, Integral) for x in sites):
                sites = (tuple(sites),)
            else:
                sites = tuple(tuple(site) for site in sites)
            if len(sites) not in (1, 2) or any(len(s) != 2 for s in sites):
                raise ValueError("gates must act on one or two 2D coordinate sites")
            if any(not (0 <= x < self.state.Lx and 0 <= y < self.state.Ly) for x, y in sites):
                raise ValueError("gate site is outside the PEPO")
            if len(sites) == 2 and sum(abs(a-b) for a, b in zip(*sites)) != 1:
                raise ValueError("two-site gates must act on adjacent sites")
            entries.append((payload, sites))
        return entries

    def run(self):
        """Replay the stream on the current state and return the updated PEPO."""
        entries = self._entries()
        self._records = []
        if not entries:
            return self.state
        convert = infer_backend_converter_from_sample(next(iter(self.state)).data)
        entries = [(convert(g), sites) for g, sites in entries]
        if self.mode == "dmt":
            from ._pepo_dmt import RegionalMapCache, compress_pair, l_regions

            if isinstance(self.regions, str):
                if self.regions != "L":
                    raise ValueError("regions must be 'L' or a sequence of coordinate regions")
                regions = l_regions((self.state.Lx, self.state.Ly))
            else:
                regions = [tuple(tuple(s) for s in region) for region in self.regions]
            if not regions:
                raise ValueError("DMT requires at least one protected region")
            valid_sites = set(self.state.gen_site_coos())
            if any(not region or len(set(region)) != len(region) or not set(region) <= valid_sites
                   for region in regions):
                raise ValueError("DMT regions must contain distinct lattice sites")
            regions = list(dict.fromkeys(tuple(sorted(region)) for region in regions))
            work, records = self.state.copy(), []
            cache = RegionalMapCache(self.dmt_cache_max_elements) if self.dmt_cache else None
            for g, sites in entries:
                if len(sites) == 1:
                    gauges = {}
                    gate_simple(work, g, where=sites, gauges=gauges, renorm=False, inplace=True)
                    work.gauge_simple_insert(gauges)
                else:
                    work, record = compress_pair(work, g, sites, self.chi, regions,
                        tol=self.dmt_tolerance, max_nfev=self.dmt_max_nfev,
                        optimize=self.contraction_opt, max_map_elements=self.dmt_max_map_elements,
                        boundary_chi=self.boundary_chi, boundary_method=self.fit_mode,
                        boundary_cutoff=float(self.boundary_kwargs.get("cutoff", 0.)),
                        boundary_max_iterations=int(self.boundary_kwargs.get("n_iter", 10)),
                        map_cache=cache)
                    records.append(record)
                if cache is not None:
                    cache.invalidate(sites)
                    if len(sites) == 2:
                        record["map_cache"] = cache.report()
            self.state, self._records = work, records
        elif self.mode == "full-update":
            actions = []
            for g, sites in entries:
                actions.extend(((g, sites, "upper"), (ar.do("conj", g), sites, "lower")))
            optimizer = PepsOptimizer(
                self.state, actions, chi=self.chi, mode="full-update",
                boundary_chi=self.boundary_chi, fit_mode=self.fit_mode,
                boundary_kwargs=self.boundary_kwargs, full_update_kwargs=self.full_update_kwargs,
                contraction_opt=self.contraction_opt, boundary_convergence=False,
            )
            self.state = optimizer.run(fuse_pepo_legs=True, cutoff=self.cutoff,
                                       cutoff_mode=self.cutoff_mode)
            self._records = optimizer.get_step_records()
        elif self.mode == "svd":
            work = self.state.copy()
            for g, sites in entries:
                gauges = {}
                gate_simple(work, g, where=sites, gauges=gauges, max_bond=self.chi,
                            cutoff=self.cutoff, cutoff_mode=self.cutoff_mode,
                            renorm=False, inplace=True)
                work.gauge_simple_insert(gauges)
            self.state = work
        else:
            work, gauges = self.state.copy(), {}
            for g, sites in entries:
                gate_simple(work, g, where=sites, gauges=gauges, max_bond=self.chi,
                            cutoff=self.cutoff, cutoff_mode=self.cutoff_mode,
                            renorm=False, inplace=True)
            work.gauge_simple_insert(gauges)
            self.state = work
        return self.state

    def get_step_records(self):
        """Return full-update or DMT records from the most recent run."""
        return list(self._records)
