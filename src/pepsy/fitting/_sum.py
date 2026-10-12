"""Termwise overlap environments for a variational sum target.

Only effective local tensors are added. Each ordinary FIT owns one immutable
target and its contraction metadata; sweep caches contain tuples of its
environments. The existing sweep kernels still perform a single local update.
"""

import functools
import inspect

import autoray as ar
import quimb.tensor as qtn

from .local import FIT, _native_fermionic_bra_fit


def _add_tensors(tensors):
    iterator = iter(tensors)
    result = next(iterator)
    for tensor in iterator:
        result = result + tensor
    return result


class _SumFIT(FIT):
    @functools.wraps(FIT.__init__)
    def __init__(self, tn, *args, **kwargs):
        if not tn:
            raise ValueError("FIT requires a non-empty list of target networks.")
        if any(not isinstance(term, qtn.TensorNetwork) for term in tn):
            raise TypeError("Each FIT sum term must be a TensorNetwork.")
        # Validate common physical spaces before ownership transfer/reindexing.
        guess = args[0] if args else kwargs.get("p")
        if guess is not None:
            if not isinstance(guess, (qtn.MatrixProductState, qtn.MatrixProductOperator)):
                raise TypeError("Initial `p` must be MatrixProductState or MatrixProductOperator.")
            physical = set(guess.outer_inds())
            for term in tn:
                if set(term.outer_inds()) != physical:
                    raise ValueError("Every sum term and p must have the same outer indices.")
                if any(term.ind_size(ix) != guess.ind_size(ix) for ix in physical):
                    raise ValueError("Every sum term and p must have the same physical dimensions.")

        parameters = dict(inspect.signature(FIT.__init__).bind(
            self, tn, *args, **kwargs,
        ).arguments)
        parameters.pop("self")
        parameters.pop("tn")
        first = FIT(tn[0], **parameters)
        parameters.update(p=first.p, inplace=True)
        # Repeated input references share one owned target, so transferring
        # ownership cannot invalidate an earlier term's reindexing cache.
        by_identity = {id(tn[0]): first}
        terms = [first]
        for target in tn[1:]:
            if id(target) not in by_identity:
                by_identity[id(target)] = FIT(target, **parameters)
            terms.append(by_identity[id(target)])
        terms = tuple(terms)
        self.__dict__.update(first.__dict__)
        self._terms = terms
        self.tn = tuple(term.tn for term in terms)
        site_tags = {self.site_tag_id.format(i) for i in range(self.L)}
        for term in terms:
            term.p = self.p
            term._explicit_exterior = True
            if any(len(site_tags.intersection(t.tags)) != 1 for t in term.tn.tensors):
                raise ValueError("Each sum-target tensor must carry exactly one site tag; use retag=True if needed.")
            if term.tn.exponent:
                tensor = term.tn.tensors[0]
                tensor.modify(data=tensor.data * 10**term.tn.exponent)
                term.tn.exponent = 0.0
        self._allow_sweep_environment_reuse = all(
            term._allow_sweep_environment_reuse for term in terms
        )
        strategies = {term.environment_strategy for term in terms}
        self.environment_strategy = (
            next(iter(strategies)) if len(strategies) == 1 else "generic"
        )
        # Only cache immutable routing here, never contracted target blocks:
        # a layered block can be much larger than its uncontracted factors.
        self._component_groups = {}
        self._sum_target_norm = None

    def _components_for(self, sites):
        """Select/order each visited target block once for this owned target."""
        sites = tuple(sites)
        try:
            return self._component_groups[sites]
        except KeyError:
            groups = tuple(
                tuple(term._target_components(sites)) for term in self._terms
            )
            self._component_groups[sites] = groups
            return groups

    def _reset_run_traces(self):
        super()._reset_run_traces()
        # A new run must build fresh diagnostics/autodiff graphs even if
        # callers updated owned target data between runs.
        self._sum_target_norm = None

    @functools.cached_property
    def prepared_target(self):
        return tuple(term.prepared_target for term in self._terms)

    def _target_isfermionic(self):
        return any(term._target_isfermionic() for term in self._terms)

    def _resolve_cutoff(self, cutoff):
        value = max(term._resolve_cutoff(cutoff) for term in self._terms)
        if cutoff == "auto":
            self.info.update(cutoff_requested="auto", cutoff_resolved=value)
        return value

    def _reference_effective_tensor(self, bra, output_inds):
        return _add_tensors(
            term._reference_effective_tensor(bra, output_inds) for term in self._terms
        )

    def _target_fidelity(self, psi, *, contraction_opt=None):
        optimize = contraction_opt or self.contraction_opt

        def overlap(a, b):
            return (a.H & b).contract(all, optimize=optimize, output_inds=())

        projection = sum(overlap(psi, term) for term in self.tn)
        if self._sum_target_norm is None:
            # Hermitian cross terms occur in conjugate pairs. The target is
            # fixed during a run, so compute this diagnostic only once.
            self._sum_target_norm = sum(
                ar.do("real", overlap(a, a)) + sum(
                    2 * ar.do("real", overlap(a, b)) for b in self.tn[i + 1:]
                )
                for i, a in enumerate(self.tn)
            )
        return ar.do("abs", projection)**2 / ar.do(
            "abs", self._sum_target_norm * overlap(psi, psi)
        )

    def visual(self, *args, **kwargs):
        """Draw each target with the common fitted network separately."""
        return tuple(term.visual(*args, **kwargs) for term in self._terms)

    def _overlap_environment_site(self, psi, site, start, stop, prior=None):
        # All terms use the same fitted bra. Rebuild it once for this site
        # update; caching it across updates would retain stale fitted data.
        bra = psi[site] if self._fermionic_bra_working else psi[site].H
        groups = self._components_for((site,))
        environments = []
        for i, term in enumerate(self._terms):
            components = (bra, *groups[i])
            if prior is not None:
                components += (prior[i],)
            environments.append(term._contract_components(components))
        return tuple(environments)

    def _effective_tensor(
        self, psi, sites, start, stop, *, left_environment=None,
        right_environment=None, output_inds,
    ):
        groups = self._components_for(sites)

        def contributions():
            for i, term in enumerate(self._terms):
                components = groups[i]
                if left_environment is not None:
                    components += (left_environment[i],)
                if right_environment is not None:
                    components += (right_environment[i],)
                yield term._contract_components(components, output_inds=output_inds)

        return _add_tensors(contributions())

    def _prepare_fermionic_effective_tensor(
        self, tensor, left_tensor, right_tensor, left_environment, right_environment,
    ):
        return super()._prepare_fermionic_effective_tensor(
            tensor, left_tensor, right_tensor,
            None if left_environment is None else left_environment[0],
            None if right_environment is None else right_environment[0],
        )

    @_native_fermionic_bra_fit
    @functools.wraps(FIT.run_eff)
    def run_eff(self, *args, **kwargs):
        return super().run_eff(*args, **kwargs)

    def _run_eff_fermionic_sweeps(
        self, psi, L, site_tag_id, contraction_opt, *, n_iter, verbose,
        sweep_sequence,
    ):
        # Use the common bra-gauge one-site kernel, which accepts tuples of
        # environments, rather than the legacy single-target ket-gauge path.
        return self._run_eff_one_site_sweeps(
            psi, L, contraction_opt, n_iter=n_iter, verbose=verbose,
            sweep_sequence=sweep_sequence, min_iter=1, rtol=None, patience=1,
        )

    def _run_gate_sweeps(self, psi, start, stop, **kwargs):
        if not self._fermionic_bra_working:
            # The outside norm metric must already be canonical, as for
            # ordinary window FIT. Contract the actual outside overlaps for
            # each term while leaving the outside fitted tensors untouched.
            prior = None
            for site in range(start):
                prior = self._overlap_environment_site(psi, site, start, stop, prior)
            self._fermionic_left_exterior_environment = prior
            prior = None
            for site in range(self.L - 1, stop, -1):
                prior = self._overlap_environment_site(psi, site, start, stop, prior)
            self._fermionic_right_exterior_environment = prior
        try:
            return super()._run_gate_sweeps(psi, start, stop, **kwargs)
        finally:
            self._fermionic_left_exterior_environment = None
            self._fermionic_right_exterior_environment = None
