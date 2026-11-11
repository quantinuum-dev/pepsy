"""Prepared dense Torch PEPO cluster replay, with fresh numerical environments.

Plans retain geometry, layouts and Cotengra programs, never tensor values.
The original gate order and every pair truncation remain explicit.
"""

from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import replace
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
        labels = {}
        inputs = tuple(tuple(labels.setdefault(ix, len(labels)) for ix in t.inds) for t in tensors)
        self.structure = inputs, tuple(labels[ix] for ix in output)
        self.boundary = boundary
        self.plan_key = plans._topology_key(network, output, optimize)
        self.shapes = tuple(t.shape for t in tensors)
        self.function = partial(ct.array_contract_expression(
            [t.inds for t in tensors], output, shapes=self.shapes,
            optimize=tree, cache=False), backend='torch')

    def operands(self, pair, spectators):
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
        return arrays

    def contract(self, pair, spectators):
        return self.function(*self.operands(pair, spectators))


class PreparedLoopClusterUpdate:
    """Replay an ordered dense Torch PEPO stream with prepared local geometry.

    ``schedule`` contains ``(sites, leg)`` entries with one/two lattice sites
    and ``leg='upper'`` or ``'lower'``. A two-site ``'both'`` entry takes
    ``(upper_gate, lower_gate)``. Gate arrays supplied to :meth:`apply`
    use the same native convention as ``gate_loop_cluster``: lower action is
    ``O G.T``. Prepare a right-applied lower gate's transpose once beforehand.

    The plan binds topology/dtype/device, not an initial numerical state.
    Each call reads current cores and gauges. Bond growth makes a new bounded
    contraction program; it never pads ranks or reuses an old environment.
    ALS, its conditioning checks and final gauge extraction use the existing
    reduced-update routines. Gate/truncation dependencies and commit order
    are preserved by default. ``fuse_pepo_legs=True`` opts into one fit for
    adjacent opposite-leg gates on the same bond, without reordering and
    without crossing any apply boundary. Automatic pairing requires batch size one.

    The default is the direct six-site loop-union patch. Use
    ``cluster_mode='graphdistance', base_radius=1, fillin=True`` for a
    rotated twelve-site rectangle in the bulk. Both use SU boundary gauges,
    dense complex Torch qubit PEPOs and Torch ALS.
    Other cluster/message/backend routes retain ``gate_loop_cluster``.

    ``batch_size>1`` opts into eager batches of consecutive shape-compatible,
    disjoint patches. Environment contraction and ALS use batch kernels;
    preparation and final local gauge extraction remain sequential. Individual
    stopping decisions, original callback order and apply boundaries are kept.
    ``batch_size='auto'`` discovers independent ready patches with bounded
    lookahead and sizes their batches from topology, shapes and workspace.
    """

    def __init__(self, network, schedule, *, max_bond=6, max_iterations=10,
                 tol=1e-8, psd_project=False, torch_backend=None, optimize=None,
                 max_peak_memory_log2=None, max_flops_log10=None,
                 max_programs=512, batch_size=1, cluster_mode='loopunion',
                 base_radius=4, fillin=True, fuse_pepo_legs=False):
        import torch
        import quimb.tensor as qtn
        from ._compression_utils import ContractionPlanCache, validate_cost_options
        from .reduced_update import _single_tid
        from ..tensors import build_optimizer

        for name, value in (('max_bond', max_bond), ('max_iterations', max_iterations),
                            ('max_programs', max_programs)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        automatic = isinstance(batch_size, str) and batch_size == 'auto'
        if not automatic and (isinstance(batch_size, bool)
                              or not isinstance(batch_size, int) or batch_size < 1):
            raise ValueError("batch_size must be a positive integer or 'auto'")
        if not math.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative')
        if not isinstance(psd_project, bool):
            raise TypeError('psd_project must be a bool')
        if not isinstance(fuse_pepo_legs, bool):
            raise TypeError('fuse_pepo_legs must be a bool')
        if fuse_pepo_legs and batch_size != 1:
            raise ValueError('adjacent joint updates currently require batch_size=1')
        if cluster_mode not in {'loopunion', 'graphdistance'}:
            raise ValueError("cluster_mode must be 'loopunion' or 'graphdistance'")
        if isinstance(base_radius, bool) or not isinstance(base_radius, int) or base_radius < 0:
            raise ValueError('base_radius must be a nonnegative integer')
        if not isinstance(fillin, (bool, int)) or fillin < 0:
            raise ValueError('fillin must be a nonnegative integer or bool')
        if torch_backend not in {None, 'aot_eager'}:
            raise ValueError('torch_backend must be None or aot_eager')
        if batch_size != 1 and torch_backend is not None:
            raise ValueError('independent cluster batches currently require eager Torch')
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
            if which not in {'upper', 'lower', 'both'} or len(sites) not in {1, 2}:
                raise ValueError('schedule requires upper/lower or joint both on two sites')
            if which == 'both' and len(sites) != 2:
                raise ValueError('joint upper/lower cluster gates require two sites')
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
        self.cluster_mode, self.base_radius, self.fillin = cluster_mode, base_radius, fillin
        self.fuse_pepo_legs = fuse_pepo_legs
        self.optimize = (build_optimizer(parallel=False, progbar=False, max_repeats=8, max_time=.5)
                         if optimize is None else optimize)
        max_flops_log10, max_peak_memory_log2, on_budget = validate_cost_options(
            max_flops_log10, max_peak_memory_log2, 'raise')
        self.budget = dict(max_flops_log10=max_flops_log10,
                           max_peak_memory_log2=max_peak_memory_log2, on_budget=on_budget)
        self.max_programs = max_programs
        self.batch_size = batch_size
        self.plan_cache = ContractionPlanCache()
        self._geometry = {}
        self._batch_regions = {}
        self._auto_batch_limit = None
        self._auto_workspace_bytes = None
        self._auto_member_bytes = None
        self._batch_histogram = {}
        if automatic:
            sizes = [len(self._region_for(network, slot)) for slot in self.slots
                     if len(slot.sites) == 2]
            # Bound preparation copies as well as geometric parallelism. Actual
            # widths are chosen later from ready gates, ranks and workspace.
            self._auto_batch_limit = (min(16, len(sizes), len(network.tensor_map)//min(sizes))
                                      if sizes else 1)
            self._auto_lookahead = max(64, 8*len(network.tensor_map))
        self._programs = OrderedDict()
        self._counts = dict(applications=0, pair_updates=0, one_site_updates=0,
                            physical_gate_actions=0, joint_pair_updates=0,
                            program_hits=0, program_misses=0, program_evictions=0,
                            cluster_batches=0, batched_pair_updates=0, largest_batch=0,
                            lookahead_pair_updates=0)

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
                    cluster_mode=self.cluster_mode, base_radius=self.base_radius, fillin=self.fillin,
                    physical_reduction='acted leg retained; untouched leg on Q',
                    fuse_pepo_legs=self.fuse_pepo_legs,
                    contraction_programs=len(self._programs), batch_size=self.batch_size,
                    auto_batch_limit=self._auto_batch_limit,
                    auto_workspace_bytes=self._auto_workspace_bytes,
                    auto_estimated_member_bytes=self._auto_member_bytes,
                    batch_histogram=dict(self._batch_histogram),
                    ordering=('input; adjacent opposite-leg pairs; no reordering' if self.fuse_pepo_legs else
                              'input commits; dependency-ready preparation' if self.batch_size == 'auto'
                              else 'input; no fusion or reordering'))

    def _region_for(self, work, slot):
        """Topology-only read footprint; one-site actions read/write one tensor."""
        if len(slot.sites) == 1:
            return frozenset(slot.tids)
        if slot.sites not in self._batch_regions:
            patch = work.select_local(
                tuple(work.site_tag(s) for s in slot.sites), which='any',
                max_distance=self.base_radius, mode=self.cluster_mode, fillin=int(self.fillin),
                grow_from='all', virtual=True)
            self._batch_regions[slot.sites] = frozenset(patch.tensor_map)
        return self._batch_regions[slot.sites]

    def _available_batch_memory(self):
        """Reserve a conservative fraction of currently available host/device RAM."""
        if self.device.type == 'cuda':
            import torch

            free, _ = torch.cuda.mem_get_info(self.device)
            return min(2*1024**3, free//4)
        import os

        try:
            available = os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
        except (AttributeError, OSError, ValueError):
            available = 4*1024**3
        return min(512*1024**2, available//8)

    def _auto_capacity(self, program, pair):
        """Shrink automatic groups to explicit costs and estimated ALS workspace."""
        # Lookahead can leave earlier fits pending while another ready group
        # runs. Refresh available memory instead of reusing the apply-start
        # allowance after those pending snapshots have consumed workspace.
        self._auto_workspace_bytes = min(self._auto_workspace_bytes, self._available_batch_memory())
        a, p, q, b = pair.theta_shape
        rank = min(self.max_bond, a*p, q*b)
        # In addition to Cotengra intermediates, allow for environment/target
        # stacks and both normal equations, factors and inverse certificates.
        # Preparation currently gauge-inserts a PRIVATE FULL NETWORK, not just
        # the patch. Its live arrays scale with lattice size; counting only the
        # local contraction would badly underestimate large-lattice batches.
        snapshot = sum(math.prod(t.shape) for t in pair.tn.tensors)
        snapshot_factor = 6 if any(t.data.requires_grad for t in pair.tn.tensors) else 2
        elements = (snapshot_factor*snapshot + 2**program.cost['peak_memory_log2'] + 4*(a*b)**2
                    + 8*a*p*q*b + 6*((a*rank)**2+(b*rank)**2)
                    + sum(math.prod(shape) for shape in program.shapes))
        per_member = elements*pair.q_left.data.element_size()
        self._auto_member_bytes = math.ceil(per_member)
        for width in range(self._auto_batch_limit, 1, -1):
            if width*per_member > self._auto_workspace_bytes:
                continue
            peak = self.budget['max_peak_memory_log2']
            flops = self.budget['max_flops_log10']
            if peak is not None and program.cost['peak_memory_log2']+math.log2(width) > peak:
                continue
            if flops is not None and program.cost['flops_log10']+math.log10(width) > flops:
                continue
            return width
        return 1

    def _has_ready_peer(self, work, index, end, automatic, excluded):
        """Reject serial-only windows before building a numerical pair copy."""
        first = self._region_for(work, self.slots[index])
        reads, writes = set(), set()
        for k in range(index+1, end):
            slot = self.slots[k]
            region = self._region_for(work, slot)
            active = set(slot.tids)
            if (k not in excluded and len(slot.sites) == 2 and not first & region
                    and not reads & active and not writes & region):
                return True
            if not automatic:
                return False
            reads.update(region)
            writes.update(active)
        return False

    def _program_for(self, pair):
        if pair.where not in self._geometry:
            spectators = pair._cluster_tids(
                self.base_radius, mode=self.cluster_mode, fillin=self.fillin)
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
        return self._programs[key], spectators, radius

    def _problem(self, pair, gate, which, *, prepared=None, data=None):
        from .reduced_update import (
            LoopClusterReducedUpdateProblem, LoopClusterTerm, _LazyReducedMetric,
            _hermitian_open_environment, _metric_views, _open_environment_tensor,
            _psd_project_open_environment, _reduced_gate_target,
        )

        program, spectators, radius = self._program_for(pair) if prepared is None else prepared
        if data is None:
            data = program.contract(pair, spectators)
        raw = _hermitian_open_environment(pair, _open_environment_tensor(
            pair, data))
        environment, minimum, clipped = (
            _psd_project_open_environment(pair, raw, 0.) if self.psd_project else (raw, None, 0))
        target = _reduced_gate_target(pair, gate, which)
        metric, linear = _metric_views(pair, environment, target, materialize_metric=False)
        term = LoopClusterTerm(frozenset((*spectators, pair.left_tid, pair.right_tid)),
                               1, spectators, program.boundary)
        return LoopClusterReducedUpdateProblem(
            pair=pair, gate=gate, base_radius=self.base_radius, max_loop_size=0, full_radius=radius,
            loop_regions=(), terms=(term,), environment=environment, metric=metric,
            raw_metric=_LazyReducedMetric(pair, raw), linear_term=linear, target=target,
            psd_projected=self.psd_project, raw_min_eigenvalue=minimum,
            clipped_eigenvalues=clipped, contraction_costs=(program.cost,),
            contraction_cost=program.cost, cluster_mode=self.cluster_mode, fillin=self.fillin, which=which)

    def _batch(self, work, gates, gauges, index, stop, measure_fidelity, *, excluded=frozenset()):
        """Fit independent patches; publish all mutations in original input order."""
        import torch
        from ._compression_utils import ContractionCost, enforce_cost_budget
        from .reduced_update import prepare_reduced_bond_pair
        from ._reduced_torch_batch import solve

        automatic = self.batch_size == 'auto'
        limit = self._auto_batch_limit if automatic else self.batch_size
        if limit < 2:
            return ()
        end = min(stop, index+(self._auto_lookahead if automatic else limit))
        if not self._has_ready_peer(work, index, end, automatic, excluded):
            return ()
        selected, occupied, signature = [], set(), None
        skipped_reads, skipped_writes = set(), set()
        for k in range(index, end):
            slot = self.slots[k]
            region = self._region_for(work, slot)
            writes = set(slot.tids)
            # Whole patches, including the active factors, must be disjoint.
            # Disjoint gate endpoints alone do not make their metrics independent.
            # A looked-ahead fit must also commute with EVERY skipped action:
            # neither its writes nor its metric may cross a preceding dependency.
            blocked = (k in excluded or len(slot.sites) != 2 or occupied & region
                       or skipped_reads & writes or skipped_writes & region)
            if blocked:
                if not automatic:
                    break
                skipped_reads.update(region)
                skipped_writes.update(writes)
                continue
            pair = prepare_reduced_bond_pair(work, gauges, where=slot.sites,
                                             input_mode='su_core', run_bp=False, which=slot.which)
            prepared = self._program_for(pair)
            program, tids, _ = prepared
            key = program.structure, program.shapes, pair.theta_shape
            if signature is not None and key != signature:
                if not automatic:
                    break
                skipped_reads.update(region)
                skipped_writes.update(writes)
                continue
            if not selected and automatic:
                limit = self._auto_capacity(program, pair)
                if limit < 2:
                    return ()
            signature = key
            selected.append((k, pair, prepared))
            occupied.update(region)
            if len(selected) == limit:
                break
        if len(selected) < 2:
            return ()
        # The user budget also bounds the simultaneous environment workspace.
        cost = selected[0][2][0].cost
        enforce_cost_budget(ContractionCost(
            flops_log10=cost['flops_log10']+math.log10(len(selected)),
            peak_memory_log2=cost['peak_memory_log2']+math.log2(len(selected))),
            label='independent cluster batch', **self.budget)
        operands = [program.operands(pair, tids) for _, pair, (program, tids, _) in selected]
        stacks = tuple(torch.stack(xs) for xs in zip(*operands))
        data = torch.vmap(selected[0][2][0].function)(*stacks)
        problems = [self._problem(pair, gates[k], self.slots[k].which,
                                  prepared=prepared, data=data[n])
                    for n, (k, pair, prepared) in enumerate(selected)]
        solutions = solve(problems, max_bond=self.max_bond,
                          max_iterations=self.max_iterations, tol=self.tol,
                          measure_fidelity=measure_fidelity)
        self._counts['cluster_batches'] += 1
        self._counts['batched_pair_updates'] += len(selected)
        self._counts['largest_batch'] = max(self._counts['largest_batch'], len(selected))
        self._counts['lookahead_pair_updates'] += sum(k != index+n
                                                     for n, (k, _, _) in enumerate(selected))
        self._batch_histogram[len(selected)] = self._batch_histogram.get(len(selected), 0)+1
        return tuple((k, pair, problem, solution)
                     for (k, pair, _), problem, solution in zip(selected, problems, solutions))

    def apply(self, network, gates, *, gauges, start=0, stop=None, tracker=None,
              step_callback=None, inplace=True, return_results=False,
              materialize_physical=True):
        """Replay ``[start:stop]`` of the original full gate bank in order.

        Gauges are updated in place. ``step_callback(index, result)`` receives
        the original zero-based gate index; result is None for one-site gates.
        Reuse a tracker across calls to retain local products across source
        barriers. Full pair results are retained only when requested.
        ``materialize_physical=False`` omits their unused physical snapshots;
        the updated core and gauge dictionary are identical.
        A callback for a fused pair uses its last original input index;
        ``result.su_info['original_steps']`` retains both one-based indices.
        """
        import torch
        from .cluster_updates import LoopClusterUpdateTracker
        from .reduced_update import (
            ReducedLoopClusterGateResult, _finish_reduced_gate, _identity_bond_gauge,
            prepare_reduced_bond_pair,
        )
        from ..operators import renorm_gauge

        if not isinstance(materialize_physical, bool):
            raise TypeError('materialize_physical must be a bool')
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
            if self.slots[index].which == 'both':
                if not isinstance(gate, (tuple, list)) or len(gate) != 2:
                    raise ValueError('joint gates require (upper_gate, lower_gate)')
                payloads = gate
            else:
                payloads = (gate,)
            for payload in payloads:
                if (not isinstance(payload, torch.Tensor) or payload.dtype != self.dtype
                        or payload.device != self.device or tuple(payload.shape) not in
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
        if self.batch_size == 'auto':
            self._auto_workspace_bytes = self._available_batch_memory()
        pending = {}
        def updates():
            if not self.fuse_pepo_legs:
                for i in range(start, stop):
                    yield i, self.slots[i], gates[i], (i,)
                return
            from ..operators.gates import _pair_pepo_gate_entries

            entries = [(gates[i], self.slots[i].sites, self.slots[i].which) for i in range(start, stop)]
            for end, (payload, _, side), steps in _pair_pepo_gate_entries(
                    entries, work, fuse_pepo_legs=True):
                yield start+end-1, replace(self.slots[start+steps[0]-1], which=side), payload, tuple(
                    start+s-1 for s in steps)

        for index, slot, gate, original_steps in updates():
            result = None
            if len(slot.sites) == 1:
                tensor = work.tensor_map[slot.tids[0]]
                data = torch.tensordot(gate.reshape(2, 2), tensor.data,
                                       dims=([1], [slot.physical_axis]))
                tensor.modify(data=data.permute(slot.output_permutation))
                self._counts['one_site_updates'] += 1
            else:
                if self.batch_size != 1 and index not in pending:
                    pending.update({k: (pair, problem, solution) for k, pair, problem, solution in
                        self._batch(work, gates, gauges, index, stop, tracker is not None,
                                    excluded=frozenset(pending))})
                solution = None
                if index in pending:
                    pair, problem, solution = pending.pop(index)
                else:
                    pair = prepare_reduced_bond_pair(work, gauges, where=slot.sites,
                                                     input_mode='su_core', run_bp=False, which=slot.which)
                    problem = self._problem(pair, gate, slot.which)
                result = _finish_reduced_gate(
                    work, gauges, pair, problem, max_bond=self.max_bond,
                    als_opts=dict(solver='torch', max_iterations=self.max_iterations,
                                  tol=self.tol, monitor_convergence=True,
                                  measure_fidelity=tracker is not None, torch_backend=self.torch_backend),
                    regauge_opts={}, inplace=True, result_type=ReducedLoopClusterGateResult,
                    solution=solution, materialize_physical=materialize_physical)
                renorm_gauge(work, gauges, slot.sites, smudge=0.)
                if tracker is not None:
                    tracker.record(result)
                self._counts['pair_updates'] += 1
                self._counts['joint_pair_updates'] += int(slot.which == 'both')
                result.su_info['original_steps'] = tuple(i+1 for i in original_steps)
                if return_results:
                    results.append(result)
            self._counts['physical_gate_actions'] += 2 if slot.which == 'both' else 1
            if step_callback is not None:
                step_callback(index, result)
        return (work, tuple(results)) if return_results else work
