"""Prepared dense Torch PEPO cluster replay, with fresh numerical environments.

Plans retain geometry, layouts and Cotengra programs, never tensor values.
The original gate order and every pair truncation remain explicit.
"""

from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
import math


@dataclass(frozen=True)
class _Slot:
    sites: tuple
    which: str
    tids: tuple
    physical_axis: int | None = None
    output_permutation: tuple = ()


class _EnvironmentProgram:
    def __init__(self, pair, spectators, plans, optimize, budget):
        import cotengra as ct
        from ._compression_utils import ContractionCost, enforce_cost_budget

        network, output, _, boundary = pair._cluster_environment_network_from_tids(spectators)
        tree, self.cost, _ = plans.plan_for(network, output_inds=output, optimize=optimize)
        enforce_cost_budget(ContractionCost(**self.cost), label='prepared loop-cluster region', **budget)
        tensors = network.tensors
        self.boundary = boundary
        self.plan_key = plans._topology_key(network, output, optimize)
        self.shapes = tuple(t.shape for t in tensors)
        self.function = partial(ct.array_contract_expression(
            [t.inds for t in tensors], output, shapes=self.shapes,
            optimize=tree, cache=False), backend='torch')

    def contract(self, pair, spectators):
        import torch

        ket = [pair.q_left.data, pair.q_right.data]
        ket.extend(pair.tn.tensor_map[tid].data for tid in spectators)
        weights, checks = [], []
        for index in self.boundary:
            gauge = pair.su_gauges[index]
            if gauge.shape != (pair.tn.ind_size(index),):
                raise ValueError(f'SU gauge shape changed for {index!r}')
            valid = torch.isfinite(gauge).all() & (gauge.real >= 0).all()
            if gauge.is_complex():
                valid = valid & (gauge.imag.abs().max() <= 1e-12)
            checks.append(valid)
            weights.append(torch.diag(gauge.real).to(ket[0].dtype))
        # Keep all reference boundary checks, but transfer one combined
        # verdict instead of synchronizing the GPU for every cut bond.
        if checks and not bool(torch.stack(checks).all().detach()):
            raise ValueError('SU boundary gauges must be finite, real and nonnegative')
        arrays = (*ket, *(value.conj() for value in ket), *weights)
        if tuple(value.shape for value in arrays) != self.shapes:
            raise ValueError('prepared environment operand shapes do not match the program')
        # Cotengra's generated program already binds the contraction order.
        # Keep optional AOT on the shared ALS kernels: capturing a separate
        # environment graph for every bond would churn its bounded chart cache.
        return self.function(*arrays)


class PreparedLoopClusterUpdate:
    """Replay an ordered dense Torch PEPO stream with prepared local geometry.

    ``schedule`` contains ``(sites, leg)`` entries with one/two lattice sites
    and ``leg='upper'`` or ``'lower'``. Gate arrays supplied to :meth:`apply`
    use the same native convention as ``gate_loop_cluster``: lower action is
    ``O G.T``. Prepare a right-applied lower gate's transpose once beforehand.

    The plan binds topology/dtype/device, not an initial numerical state.
    Each call reads current cores and gauges. Bond growth makes a new bounded
    contraction program; it never pads ranks or reuses an old environment.
    ALS, its conditioning checks and final gauge extraction use the existing
    reduced-update routines. No gate fusion or reordering is performed.

    This first prepared path supports the direct six-site loop-union patch
    with SU boundary gauges, dense complex Torch qubit PEPOs and Torch ALS.
    Other cluster/message/backend routes retain ``gate_loop_cluster``.
    """

    def __init__(self, network, schedule, *, max_bond=6, max_iterations=10,
                 tol=1e-8, psd_project=False, torch_backend=None, optimize=None,
                 max_peak_memory_log2=None, max_flops_log10=None,
                 max_programs=512):
        import torch
        import quimb.tensor as qtn
        from ._compression_utils import ContractionPlanCache, validate_cost_options
        from .reduced_update import _single_tid
        from ..tensors import build_optimizer

        for name, value in (('max_bond', max_bond), ('max_iterations', max_iterations),
                            ('max_programs', max_programs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        if not isinstance(psd_project, bool):
            raise TypeError('psd_project must be a bool')
        if torch_backend not in {None, 'aot_eager'}:
            raise ValueError('torch_backend must be None or aot_eager')
        prototype = network.tensors[0].data
        if (not hasattr(network, 'upper_ind') or not hasattr(network, 'lower_ind')
                or not isinstance(prototype, torch.Tensor)
                or prototype.dtype not in {torch.complex64, torch.complex128}):
            raise TypeError('prepared cluster replay requires a dense complex Torch PEPO')
        self.dtype, self.device = prototype.dtype, prototype.device
        self._topology = self._signature(network)
        self._validate_network(network)
        self._inner_inds = tuple(network.inner_inds())
        self._inner_set = set(self._inner_inds)
        self.slots = []
        for sites, which in schedule:
            sites = tuple(tuple(site) for site in sites)
            if which not in {'upper', 'lower'} or len(sites) not in {1, 2}:
                raise ValueError('schedule requires one/two sites and an upper/lower leg')
            tids = tuple(_single_tid(network, site) for site in sites)
            if len(set(tids)) != len(tids):
                raise ValueError('two-site gates require distinct sites')
            for site, tid in zip(sites, tids):
                if any(network.ind_size(fn(*site)) != 2
                       for fn in (network.upper_ind, network.lower_ind)):
                    raise ValueError('prepared cluster replay requires qubit operator legs')
            axis, permutation = None, ()
            if len(tids) == 2:
                if len(qtn.bonds(*(network.tensor_map[tid] for tid in tids))) != 1:
                    raise ValueError('prepared cluster replay requires adjacent single-bond pairs')
            else:
                tensor = network.tensor_map[tids[0]]
                axis = tensor.inds.index(getattr(network, which + '_ind')(*sites[0]))
                rest = tuple(i for i in range(tensor.ndim) if i != axis)
                permutation = tuple(0 if i == axis else rest.index(i)+1 for i in range(tensor.ndim))
            self.slots.append(_Slot(sites, which, tids, axis, permutation))
        self.slots = tuple(self.slots)
        if not self.slots:
            raise ValueError('prepared cluster schedule cannot be empty')
        self.max_bond, self.max_iterations, self.tol = max_bond, max_iterations, tol
        self.psd_project, self.torch_backend = psd_project, torch_backend
        self.optimize = (build_optimizer(parallel=False, progbar=False, max_repeats=8, max_time=.5)
                         if optimize is None else optimize)
        max_flops_log10, max_peak_memory_log2, on_budget = validate_cost_options(
            max_flops_log10, max_peak_memory_log2, 'raise')
        self.budget = dict(max_flops_log10=max_flops_log10,
                           max_peak_memory_log2=max_peak_memory_log2, on_budget=on_budget)
        self.max_programs = max_programs
        self.plan_cache = ContractionPlanCache()
        self._geometry = {}
        self._programs = OrderedDict()
        self._counts = dict(applications=0, pair_updates=0, one_site_updates=0,
                            program_hits=0, program_misses=0, program_evictions=0)

    @staticmethod
    def _signature(network):
        return (type(network), tuple(network.sites), network.upper_ind_id, network.lower_ind_id,
                tuple((tid, tuple(t.inds), tuple(t.tags))
                                     for tid, t in network.tensor_map.items()))

    def _validate_network(self, network):
        import torch

        if self._signature(network) != self._topology:
            raise ValueError('prepared cluster plan requires the original tensor topology and axis order')
        if any(not isinstance(t.data, torch.Tensor) or t.data.dtype != self.dtype
               or t.data.device != self.device for t in network.tensors):
            raise TypeError('prepared cluster replay must retain its Torch dtype and device')
        if any(network.ind_size(fn(*site)) != 2 for site in network.sites
               for fn in (network.upper_ind, network.lower_ind)):
            raise ValueError('prepared physical dimensions cannot change')

    @property
    def info(self):
        """Scalar execution/cache counts; no tensor data or environments."""
        return dict(**self._counts, geometry_plans=len(self._geometry),
                    contraction_programs=len(self._programs), ordering='input; no fusion or reordering')

    def _problem(self, pair, gate, which):
        from .reduced_update import (
            LoopClusterReducedUpdateProblem, LoopClusterTerm, _LazyReducedMetric,
            _hermitian_open_environment, _metric_views, _open_environment_tensor,
            _psd_project_open_environment, _reduced_gate_target,
        )

        if pair.where not in self._geometry:
            spectators = pair._cluster_tids(4, mode='loopunion', fillin=True)
            self._geometry[pair.where] = spectators, pair.full_cluster_radius()
        spectators, radius = self._geometry[pair.where]
        ket = (pair.q_left, pair.q_right, *(pair.tn.tensor_map[tid] for tid in spectators))
        shape_key = tuple((t.inds, t.shape) for t in ket)
        key = pair.where, shape_key
        if key not in self._programs:
            self._counts['program_misses'] += 1
            self._programs[key] = _EnvironmentProgram(
                pair, spectators, self.plan_cache, self.optimize, self.budget)
            if len(self._programs) > self.max_programs:
                _, evicted = self._programs.popitem(last=False)
                self.plan_cache.plans.pop(evicted.plan_key, None)
                self.plan_cache.costs.pop(evicted.plan_key, None)
                self._counts['program_evictions'] += 1
        else:
            self._counts['program_hits'] += 1
            self._programs.move_to_end(key)
        program = self._programs[key]
        raw = _hermitian_open_environment(pair, _open_environment_tensor(
            pair, program.contract(pair, spectators)))
        environment, minimum, clipped = (
            _psd_project_open_environment(pair, raw, 0.) if self.psd_project else (raw, None, 0))
        target = _reduced_gate_target(pair, gate, which)
        metric, linear = _metric_views(pair, environment, target, materialize_metric=False)
        term = LoopClusterTerm(frozenset((*spectators, pair.left_tid, pair.right_tid)),
                               1, spectators, program.boundary)
        return LoopClusterReducedUpdateProblem(
            pair=pair, gate=gate, base_radius=4, max_loop_size=0, full_radius=radius,
            loop_regions=(), terms=(term,), environment=environment, metric=metric,
            raw_metric=_LazyReducedMetric(pair, raw), linear_term=linear, target=target,
            psd_projected=self.psd_project, raw_min_eigenvalue=minimum,
            clipped_eigenvalues=clipped, contraction_costs=(program.cost,),
            contraction_cost=program.cost, cluster_mode='loopunion', fillin=True, which=which)

    def apply(self, network, gates, *, gauges, start=0, stop=None, tracker=None,
              step_callback=None, inplace=True, return_results=False):
        """Replay ``[start:stop]`` of the original full gate bank in order.

        Gauges are updated in place. ``step_callback(index, result)`` receives
        the original zero-based gate index; result is None for one-site gates.
        Reuse a tracker across calls to retain local products across source
        barriers. Full pair results are retained only when requested.
        """
        import torch
        from .cluster_updates import LoopClusterUpdateTracker
        from .reduced_update import (
            ReducedLoopClusterGateResult, _finish_reduced_gate, _identity_bond_gauge,
            prepare_reduced_bond_pair,
        )
        from ..operators import renorm_gauge

        if stop is None:
            stop = len(self.slots)
        if (not isinstance(start, int) or not isinstance(stop, int)
                or not 0 <= start <= stop <= len(self.slots) or len(gates) != len(self.slots)):
            raise ValueError('gate bank and [start:stop] must match the prepared full schedule')
        self._validate_network(network)
        if tracker is not None and not isinstance(tracker, LoopClusterUpdateTracker):
            raise TypeError('tracker must be a LoopClusterUpdateTracker or None')
        if set(gauges).difference(self._inner_set):
            raise ValueError('SU gauges contain unknown bonds')
        # Validate before mutation, rather than partially evolving a stream
        # before discovering that a later gate came from the wrong backend.
        for index in range(start, stop):
            gate = gates[index]
            width = len(self.slots[index].sites)
            if (not isinstance(gate, torch.Tensor) or gate.dtype != self.dtype
                    or gate.device != self.device or tuple(gate.shape) not in
                    {(2**width, 2**width), (2,)*(2*width)}):
                raise ValueError('gate arrays must match the prepared dimensions, Torch dtype and device')
        real_dtype = network.tensors[0].data.real.dtype
        for index, gauge in gauges.items():
            if (not isinstance(gauge, torch.Tensor) or gauge.device != self.device
                    or gauge.dtype not in {self.dtype, real_dtype}
                    or gauge.shape != (network.ind_size(index),)):
                raise ValueError('SU gauges must be matching Torch vectors on the prepared device')
        for index in self._inner_inds:
            if index not in gauges:
                gauges[index] = _identity_bond_gauge(network, index)
        work = network if inplace else network.copy()
        results = []
        self._counts['applications'] += 1
        for index in range(start, stop):
            slot, gate = self.slots[index], gates[index]
            result = None
            if len(slot.sites) == 1:
                tensor = work.tensor_map[slot.tids[0]]
                data = torch.tensordot(gate.reshape(2, 2), tensor.data,
                                       dims=([1], [slot.physical_axis]))
                tensor.modify(data=data.permute(slot.output_permutation))
                self._counts['one_site_updates'] += 1
            else:
                pair = prepare_reduced_bond_pair(work, gauges, where=slot.sites,
                                                 input_mode='su_core', run_bp=False)
                problem = self._problem(pair, gate, slot.which)
                result = _finish_reduced_gate(
                    work, gauges, pair, problem, max_bond=self.max_bond,
                    als_opts=dict(solver='torch', max_iterations=self.max_iterations,
                                  tol=self.tol, monitor_convergence=True,
                                  measure_fidelity=tracker is not None, torch_backend=self.torch_backend),
                    regauge_opts={}, inplace=True, result_type=ReducedLoopClusterGateResult)
                renorm_gauge(work, gauges, slot.sites, smudge=0.)
                if tracker is not None:
                    tracker.record(result)
                self._counts['pair_updates'] += 1
                if return_results:
                    results.append(result)
            if step_callback is not None:
                step_callback(index, result)
        return (work, tuple(results)) if return_results else work
