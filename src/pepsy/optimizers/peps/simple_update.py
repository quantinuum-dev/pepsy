"""Routed simple-update evolution for PEPS-like tensor networks."""

from __future__ import annotations

import quimb.tensor as qtn

from ...operators import gate_simple as pepsy_gate_simple

__all__ = ["SimpleUpdateGen"]


_DEFAULT_ROUTE_OPTS = {
    "cutoff_mode": "rsum2",
    "sequence": "auto",
    "path_canonize": True,
    "path_compress": False,
}

_PEPSY_GATE_SIMPLE_KEYS = frozenset({
    "which",
    "ind_id",
    "renorm",
    "smudge",
    "max_bond",
    "cutoff",
    "cutoff_mode",
    "sequence",
    "path_canonize",
    "path_canonize_distance",
    "path_canonize_opts",
    "path_compress",
    "path_compress_max_bond",
    "path_compress_cutoff",
    "path_compress_canonize_distance",
    "path_compress_opts",
})


class SimpleUpdateGen(qtn.SimpleUpdateGen):
    """Quimb ``SimpleUpdateGen`` with Pepsy routed gate application.

    This class preserves Quimb's arbitrary-geometry simple-update lifecycle:
    state and gauge storage, sweep ordering, Trotter gate construction, energy
    bookkeeping, normalization, equilibration, callbacks, and stopping logic.
    The only changed hook is :meth:`gate`, which calls :func:`pepsy.gate_simple`
    rather than raw ``tn.gate_simple_(...)``.

    The Pepsy gate wrapper handles non-adjacent two-site gates by routing them
    through adjacent SWAPs and swapping back. This makes long-range PEPS terms
    usable in sequential simple-update sweeps while keeping the SWAP tensors
    dimension-aware, backend-aligned, and Symmray-aware.

    Notes
    -----
    ``update="parallel"`` is intentionally restricted to direct-neighbor gates.
    Quimb's parallel update bookkeeping swaps only the two endpoint tensors and
    their direct gauge, while routed gates touch every site and gauge along the
    SWAP path.
    """

    def __init__(
        self,
        psi0,
        ham,
        tau=0.01,
        D=None,
        cutoff=1e-10,
        imag=True,
        gate_opts=None,
        gauge_smudge=1e-6,
        ordering=None,
        second_order_reflect=False,
        update="sequential",
        compute_energy_every=None,
        compute_energy_final=True,
        compute_energy_opts=None,
        compute_energy_fn=None,
        compute_energy_per_site=False,
        tol=None,
        tol_energy_diff=None,
        equilibrate_every=None,
        equilibrate_start=True,
        equilibrate_opts=None,
        gauge_diff_period=None,
        callback=None,
        keep_best=False,
        plot_every=None,
        progbar=True,
        route_opts=None,
    ):
        """Initialize the routed simple-update driver.

        Parameters match ``quimb.tensor.SimpleUpdateGen`` with one addition:
        ``route_opts`` supplies Pepsy-specific routing controls forwarded to
        :func:`pepsy.gate_simple`. Values in ``route_opts`` override any
        duplicate entries in ``gate_opts``.
        """
        merged_gate_opts = self._merge_gate_opts(gate_opts, route_opts)
        super().__init__(
            psi0,
            ham,
            tau=tau,
            D=D,
            cutoff=cutoff,
            imag=imag,
            gate_opts=merged_gate_opts,
            gauge_smudge=gauge_smudge,
            ordering=ordering,
            second_order_reflect=second_order_reflect,
            update=update,
            compute_energy_every=compute_energy_every,
            compute_energy_final=compute_energy_final,
            compute_energy_opts=compute_energy_opts,
            compute_energy_fn=compute_energy_fn,
            compute_energy_per_site=compute_energy_per_site,
            tol=tol,
            tol_energy_diff=tol_energy_diff,
            equilibrate_every=equilibrate_every,
            equilibrate_start=equilibrate_start,
            equilibrate_opts=equilibrate_opts,
            gauge_diff_period=gauge_diff_period,
            callback=callback,
            keep_best=keep_best,
            plot_every=plot_every,
            progbar=progbar,
        )

    def gate(self, G, where):
        """Apply one Trotter gate through Pepsy's routed simple update."""
        if self.update == "parallel" and self._is_nonlocal_two_site_where(where):
            raise ValueError(
                "Pepsy SimpleUpdateGen update='parallel' currently supports "
                "only direct-neighbor two-site terms. Long-range routed terms "
                "require route-aware layer scheduling; use update='sequential'."
            )

        pepsy_gate_simple(
            self._psi,
            G,
            where=where,
            gauges=self._gauges,
            inplace=True,
            **self._pepsy_gate_opts(),
        )

    @classmethod
    def supported_gate_option_names(cls):
        """Return ``gate_opts`` names accepted by the Pepsy routed gate hook."""
        return tuple(sorted(_PEPSY_GATE_SIMPLE_KEYS))

    @staticmethod
    def default_route_opts():
        """Return the default Pepsy routing controls."""
        return dict(_DEFAULT_ROUTE_OPTS)

    @staticmethod
    def _merge_gate_opts(gate_opts, route_opts):
        opts = dict(gate_opts or {})
        for key, value in _DEFAULT_ROUTE_OPTS.items():
            opts.setdefault(key, value)
        if route_opts is not None:
            opts.update(dict(route_opts))
        return opts

    def _pepsy_gate_opts(self):
        opts = dict(self.gate_opts)
        unsupported = sorted(set(opts) - _PEPSY_GATE_SIMPLE_KEYS)
        if unsupported:
            unsupported_s = ", ".join(repr(name) for name in unsupported)
            supported_s = ", ".join(self.supported_gate_option_names())
            raise TypeError(
                "Pepsy SimpleUpdateGen gate_opts contains unsupported option(s) "
                f"{unsupported_s}. Supported options are: {supported_s}."
            )
        return opts

    def _is_nonlocal_two_site_where(self, where):
        if not self._is_two_site_where(where):
            return False
        site_a, site_b = tuple(where)
        return not self._sites_are_adjacent(site_a, site_b)

    def _is_two_site_where(self, where):
        if not isinstance(where, (tuple, list)):
            return False
        if self._has_site(where):
            return False
        return len(where) == 2

    def _has_site(self, site):
        has_site = getattr(self._psi, "has_site", None)
        if callable(has_site):
            return bool(has_site(site))
        return False

    def _sites_are_adjacent(self, site_a, site_b):
        tag_a = self._psi.site_tag(site_a)
        tag_b = self._psi.site_tag(site_b)
        return bool(qtn.bonds(self._psi[tag_a], self._psi[tag_b]))
