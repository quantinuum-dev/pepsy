"""Frozen virtual channel projections assembled from sparse cluster blocks.

Preparation is an explicit host calculation using pivoted QR. Replay never
factorizes a tensor or selects a rank. The reduced operator is an approximation
of the chosen cluster expansion, not a claim of minimal or exact bond rank.
"""

from dataclasses import dataclass
from math import prod
from numbers import Integral

import autoray as ar
import numpy as np

from pepsy.backends.convert import _array_namespace
from .mpo_automaton import _as_backend, _backend_reference


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _budget(shape, limit, *, itemsize=16):
    if prod(shape) * itemsize > limit:
        raise MemoryError(f"channel preparation/work tensor {tuple(shape)} exceeds memory_budget={limit}.")


def _source(builder):
    from .mpo_product import CompiledMPOClusterProduct
    from .pepo_product import CompiledPEPOClusterProduct
    from .pepo_basis import CompiledPEPOExp

    if isinstance(builder, (CompiledMPOClusterProduct, CompiledPEPOExp)):
        return builder.basis
    if isinstance(builder, CompiledPEPOClusterProduct):
        return builder.expansion
    return builder


@dataclass(frozen=True)
class _Layout:
    kind: str
    sites: tuple
    keys: tuple
    # Each edge stores (site-position, local-axis) at both endpoints.
    ends: tuple
    sectors: tuple
    charges: tuple
    dimension: int
    physical_space: object = None
    shape: tuple = ()
    cyclic: tuple = ()
    degree: int = 1
    index_ids: tuple = ('k{}', 'b{}', 'I{}')


def _capture(value):
    from .mpo_semantic import FirstDegreeMPO
    from .pepo_active import ActivePEPOBlocks, GraphActivePEPOBlocks, _site_after, _OPPOSITE_DIRECTION

    if isinstance(value, FirstDegreeMPO):
        from ._mpo_sparse import SparseVirtualTensor, _combine_level_charge

        if not all(isinstance(a, SparseVirtualTensor) for a in value._arrays):
            raise ValueError("channel capture requires sparse fixed MPO histories.")
        blocks = tuple(a.blocks for a in value._arrays)
        sites = tuple(range(value.L))
        ends = tuple(((i, 1), (i+1, 0)) for i in range(value.L-1))
        sectors = tuple(tuple(range(a.shape[1])) for a in value._arrays[:-1])
        charges = tuple((0,)*len(cut) for cut in sectors)
        if value.symmetry is not None:
            import symmray as sr

            sym = sr.get_symmetry(value.symmetry)
            charges = tuple(tuple(_combine_level_charge(level.charge, value.symmetry, sym)
                                  for level in cut) for cut in value.levels[1:-1])
        layout = _Layout('mpo', sites, tuple(tuple(sorted(b)) for b in blocks), ends, sectors,
                         charges, value.phys_dim, value.physical_space, degree=value.degree,
                         index_ids=(value.upper_ind_id, value.lower_ind_id, value.site_tag_id))
    elif isinstance(value, (ActivePEPOBlocks, GraphActivePEPOBlocks)):
        if value.charge_symmetry is not None:
            raise NotImplementedError("native PEPO channel projection needs a sector-aware materializer.")
        graph = isinstance(value, GraphActivePEPOBlocks)
        sites = tuple(value.sites) if graph else tuple(value.blocks)
        positions = {site: i for i, site in enumerate(sites)}
        blocks = tuple(value.blocks[s] for s in sites)
        ends = []
        if graph:
            for edge, (a, b) in enumerate(value.edges):
                ends.append(((positions[a], value.site_directions[a].index(edge)),
                             (positions[b], value.site_directions[b].index(edge))))
        else:
            for site in sites:
                for direction in value.site_directions[site]:
                    if direction in ('u', 'r'):
                        other = _site_after(site, direction, value.lx, value.ly, value.cyclic)
                        ends.append(((positions[site], value.site_directions[site].index(direction)),
                                     (positions[other], value.site_directions[other].index(
                                         _OPPOSITE_DIRECTION[direction]))))
        sectors = []
        for endpoints in ends:
            used = {0}
            for site, axis in endpoints:
                used.update(key[axis] for key in blocks[site])
            sectors.append(tuple(sorted(used)))
        layout = _Layout('graph' if graph else 'square', sites,
                         tuple(tuple(sorted(b)) for b in blocks), tuple(ends), tuple(sectors),
                         tuple((0,)*len(s) for s in sectors), value.physical_dim,
                         shape=() if graph else (value.lx, value.ly),
                         cyclic=() if graph else value.cyclic)
    else:
        raise TypeError("expected a fixed cluster MPO or active PEPO builder.")
    return layout, tuple(tuple(blocks[i][key] for key in keys) for i, keys in enumerate(layout.keys))


def _same_layout(left, right):
    # Physical-space dataclasses contain only static dimensions/charges.
    if left != right:
        raise ValueError("cluster channel structure changed; prepare a new channel plan.")


def _host_values(values, limit):
    _budget((sum(len(v) for v in values), *values[0][0].shape), limit)
    return tuple(np.stack([np.asarray(ar.to_numpy(v)) for v in site]) for site in values)


def _unfold(layout, endpoint, sectors, samples, limit, conjugate):
    site, axis = endpoint
    keys = layout.keys[site]
    remaining = tuple(sorted({key[:axis]+key[axis+1:] for key in keys}))
    cols = {key: i for i, key in enumerate(remaining)}
    rows = {sector: i for i, sector in enumerate(sectors)}
    width = len(remaining)*layout.dimension**2
    _budget((len(sectors), width*len(samples)), limit)
    matrix = np.zeros((len(sectors), width*len(samples)), dtype=np.result_type(*(s[site] for s in samples)))
    for sample_index, sample in enumerate(samples):
        for k, key in enumerate(keys):
            start = sample_index*width + cols[key[:axis]+key[axis+1:]]*layout.dimension**2
            matrix[rows[key[axis]], start:start+layout.dimension**2] = sample[site][k].reshape(-1)
    return matrix.conj() if conjugate else matrix


def _qr_space(matrix, charges, cap, rtol):
    """Keep the rail exactly; allocate remaining channels within charge groups."""
    from scipy.linalg import qr

    dimension = len(charges)
    if cap is None or cap >= dimension:
        return np.eye(dimension, dtype=matrix.dtype), charges, 0.0
    candidates, groups = [], []
    for charge in dict.fromkeys(charges):
        rows = [i for i, q in enumerate(charges) if q == charge and i != 0]
        if not rows:
            continue
        block = matrix[rows]
        q, r, _ = qr(block, mode='economic', pivoting=True)
        diagonal = np.abs(np.diag(r))
        threshold = rtol*diagonal.max(initial=0.)
        index = len(groups)
        groups.append((charge, rows, q))
        candidates.extend((float(score), index, j) for j, score in enumerate(diagonal) if score > threshold)
    candidates.sort(key=lambda item: -item[0])
    retained = candidates[:cap-1]
    basis = np.zeros((dimension, 1+len(retained)), dtype=matrix.dtype)
    basis[0, 0] = 1
    output_charges = [charges[0]]
    for column, (_, group, j) in enumerate(retained, 1):
        charge, rows, q = groups[group]
        basis[rows, column] = q[:, j]
        output_charges.append(charge)
    norm = np.linalg.norm(matrix)
    defect = np.linalg.norm(matrix-basis@(basis.conj().T@matrix)) / max(norm, np.finfo(float).tiny)
    return basis, tuple(output_charges), float(defect)


class ClusterChannelPlan:
    """Experimental compact MPO/PEPO evaluator with fixed virtual channel maps.

    Use :func:`prepare_cluster_channels`. ``arrays`` returns only tensors;
    ``exp`` wraps them as an MPO/PEPO, and ``trace_exp`` measures that result.
    ``pack_residuals`` and ``bind_assembler`` expose the complete compact
    assembly kernel. ``pack`` and ``bind_projector`` retain the earlier
    expanded-block projection as an explicit diagnostic path.

    ``bases`` are effective maps at the first endpoints. With structural
    sharing the opposite endpoints use separate sum/select maps, rather than
    simply their conjugates; use the bound assembly/projection interfaces.
    """

    def __init__(self, source, layout, bases, charges, defects, max_bond, memory_budget,
                 *, binding, arguments, dual_bases=None, structural_report=None,
                 reference_evaluations=0, projection_applied=False):
        from ._cluster_channel_assembly import ChannelAssembly

        self._source = source
        self._binding = binding
        self._layout = layout
        self.bases = tuple(_readonly(b) for b in bases)
        self._dual_bases = tuple(_readonly(b) for b in
                                 (dual_bases if dual_bases is not None else (b.conj() for b in bases)))
        self.structural_report = structural_report
        self.charges = tuple(charges)
        self.reference_local_defects = tuple(defects)
        self.reference_evaluations = reference_evaluations
        self.projection_applied = projection_applied
        self.max_bond = max_bond
        self.memory_budget = memory_budget
        self._assembly = ChannelAssembly(binding, arguments, layout, self.bases, memory_budget,
                                         dual_bases=self._dual_bases)

    def _projection_weights(self):
        """Build the legacy diagnostic gather only when explicitly requested."""
        layout, bases, memory_budget = self._layout, self.bases, self.memory_budget
        weights = [[None]*len(keys[0]) for keys in layout.keys]
        for edge, endpoints in enumerate(layout.ends):
            lookup = {sector: i for i, sector in enumerate(layout.sectors[edge])}
            for side, (site, axis) in enumerate(endpoints):
                _budget((len(layout.keys[site]), bases[edge].shape[1]), memory_budget)
                rows = [lookup[key[axis]] for key in layout.keys[site]]
                matrix = (bases[edge] if side == 0 else self._dual_bases[edge])[rows]
                weights[site][axis] = _readonly(matrix)
        gathered = tuple(tuple(np.ones((len(keys), 1)) if w is None else w for w in site)
                         for site, keys in zip(weights, layout.keys))
        equations = tuple(
            ','.join('z'+chr(97+i) for i in range(len(site))) +
            (',' if site else '') + 'zyx->' + ''.join(chr(97+i) for i in range(len(site))) + 'yx'
            for site in weights)
        for site in gathered:
            _budget((*[w.shape[1] for w in site], layout.dimension, layout.dimension), memory_budget)
        return gathered, equations

    @property
    def report(self):
        """Static storage counts and local training defects, not global error bounds."""
        shape = self._layout
        raw = [[1]*len(keys[0]) for keys in shape.keys]
        for sectors, ends in zip(shape.sectors, shape.ends):
            for site, axis in ends:
                raw[site][axis] = len(sectors)
        projected = self.projection_applied
        preparation = self._assembly.report["preparation"]
        exact_methods = {
            "reference": "exact-reference-channels",
            "frontier": "exact-frontier-channels",
            "automaton": "fixed-pauli-automaton",
            "symbolic": "exact-symbolic-channels",
            "algebraic": "fixed-pauli-algebra",
        }
        method = "frozen-channel-qr" if projected else exact_methods[preparation]
        return dict(method=method, representation=shape.kind,
                    input_bond_dimensions=tuple(len(s) for s in shape.sectors),
                    bond_dimensions=tuple(q.shape[1] for q in self.bases),
                    input_dense_entries=sum(prod(d)*shape.dimension**2 for d in raw),
                    output_dense_entries=sum(prod(s) for s in self._assembly.output_shapes),
                    sparse_blocks=sum(map(len, shape.keys)),
                    reference_local_defects=self.reference_local_defects,
                    reference_evaluations=self.reference_evaluations,
                    projection_applied=projected,
                    rank_selection="reference-qr" if projected else "none",
                    structural_reuse=self.structural_report,
                    rank_selection_differentiated=False, replay_factorizations=0,
                    **self._assembly.report)

    def pack(self, step=1.0, parameters=None, *, coefficients=None):
        """Diagnostic sparse-block evaluation; normal replay uses residuals."""
        layout, values = _capture(self._binding.evaluate(dict(step=step, parameters=parameters,
                                                              coefficients=coefficients)))
        _same_layout(self._layout, layout)
        result = []
        for site in values:
            reference = _backend_reference(site)
            result.append(ar.do('stack', tuple(_as_backend(v, like=reference) for v in site)))
        return tuple(result)

    def validate_source(self, builder):
        """Reject accidental replay through a different parameter/geometry owner."""
        if _source(builder) is not self._source:
            raise ValueError('channel plan belongs to a different cluster builder or binding mode.')

    def bind_projector(self, like):
        """Bind constant bases to a dtype/device outside a compiled array kernel.

        Returns ``project(packed_blocks) -> tuple[site_tensor, ...]``. It owns
        constants only; it never captures a parameter graph or input values.
        """
        namespace = _array_namespace(like)
        gathered, equations = self._projection_weights()
        # A real input needs a complex output if its reference basis is complex.
        complex_basis = any(np.iscomplexobj(w) and np.any(w.imag) for s in gathered for w in s)
        if complex_basis and 'complex' not in str(like.dtype):
            like = like + 0j
        weights = tuple(tuple(ar.do('astype', _as_backend(np.array(w, copy=True), like=like, dtype=like.dtype), like.dtype)
                              for w in site) for site in gathered)

        def project(values):
            if len(values) != len(equations):
                raise ValueError('packed blocks must contain one array per site.')
            # Torch einsum requires identical operand dtypes. Adding a scalar
            # zero promotes a real input when a frozen basis is complex.
            return tuple(namespace.einsum(equation, *site, value + site[0][0, 0]*0 if site else value)
                         for equation, site, value in zip(equations, weights, values))

        return project

    def arrays(self, step=1.0, parameters=None, *, coefficients=None):
        """Fuse local cluster contributions directly into compact site tensors."""
        residuals = self.pack_residuals(step, parameters, coefficients=coefficients)
        return self.bind_assembler(residuals[0])(residuals)

    def pack_residuals(self, step=1.0, parameters=None, *, coefficients=None):
        """Evaluate local residuals only, without constructing channel histories."""
        return self._assembly.pack(dict(step=step, parameters=parameters, coefficients=coefficients))

    def bind_assembler(self, like):
        """Bind a static residual-to-compact-tensors kernel to dtype/device.

        The returned callable owns constants only. Refresh residuals on every
        evaluation; binding never retains their values or autodiff graphs.
        Use fixed shapes (``dynamic=False``) for explicit Torch compilation.
        """
        return self._assembly.bind(like)

    def exp(self, step=1.0, parameters=None, *, coefficients=None, return_report=False,
            delinearize=False, delinearize_opts=None):
        """Construct the compact MPO/PEPO; projection is part of this operator.

        ``delinearize=True`` additionally removes numerical dependencies in
        dense NumPy MPOs at this evaluation's parameters, using QR without
        SVD. ``delinearize_opts`` accepts rtol, preserve_zeros and max_sweeps.
        It does not change the reusable plan or its array/autodiff kernels.
        The returned report describes final bonds and includes the separate
        delinearisation report; ``self.report`` remains the static plan report.
        """
        import quimb.tensor as qtn

        if delinearize is not False or delinearize_opts is not None:
            from .mpo_delinearize import _resolve_delinearization_options

            delinearize_opts = _resolve_delinearization_options(delinearize, delinearize_opts)
        if delinearize and (self._layout.kind != 'mpo'
                            or self._layout.physical_space.symmetry is not None):
            raise NotImplementedError("delinearize=True requires a dense NumPy MPO channel plan.")
        arrays = self.arrays(step, parameters, coefficients=coefficients)
        layout = self._layout
        if layout.kind == 'mpo':
            from .mpo_semantic import FirstDegreeMPO
            from ._cluster_native import charge_levels

            native = layout.physical_space.symmetry is not None
            if native:
                import symmray as sr

                zero = sr.get_symmetry(layout.physical_space.symmetry).combine()
            else:
                zero = 0
            levels = charge_levels(((zero,), *self.charges, (zero,))) if native else None
            result = FirstDegreeMPO(arrays, levels=levels, physical_space=layout.physical_space,
                                    degree=layout.degree, upper_ind_id=layout.index_ids[0],
                                    lower_ind_id=layout.index_ids[1], site_tag_id=layout.index_ids[2],
                                    metadata={'history_valid': False, '_native_charge_validated': native,
                                              'channel_report': self.report}).to_mpo()
        elif layout.kind == 'square':
            arranged = dict(zip(layout.sites, arrays))
            result = qtn.PEPO([[ar.do('swapaxes', arranged[i, j], -1, -2)
                                for j in range(layout.shape[1])] for i in range(layout.shape[0])],
                              shape='urdlbk', cyclic=layout.cyclic)
        else:
            indices = [[None]*len(keys[0]) for keys in layout.keys]
            for edge, ends in enumerate(layout.ends):
                for site, axis in ends:
                    indices[site][axis] = ('graph-bond', edge)
            result = qtn.TensorNetwork([qtn.Tensor(array,
                inds=(*inds, ('graph-bra', site), ('graph-ket', site)), tags={'GRAPH_PEPO', f'site={site!r}'})
                for array, inds, site in zip(arrays, indices, layout.sites)])
        report = self.report
        if delinearize:
            from .mpo_delinearize import delinearize_mpo

            result, reduction = delinearize_mpo(
                result, return_report=True, **delinearize_opts)
            report = dict(report, channel_bond_dimensions=report['bond_dimensions'],
                          bond_dimensions=reduction.final_bond_dimensions,
                          output_dense_entries=sum(a.size for a in result.arrays),
                          rank_selection='evaluation-qr', delinearization=reduction,
                          method=report['method']+'+delinearisation-qr')
        result.pepsy_channel_report = report
        return (result, report) if return_report else result

    def trace_exp(self, step=1.0, parameters=None, *, coefficients=None, normalized=False,
                  state_budget=100000, contract_opts=None,
                  delinearize=False, delinearize_opts=None):
        """Trace the constructed compact operator with explicit contraction controls."""
        from .pepo_trace import _trace_options, trace_pepo

        if self._layout.kind == "mpo" and contract_opts is not None:
            raise ValueError("contract_opts requires a graph or square PEPO channel plan.")
        trace_options = _trace_options(state_budget, contract_opts, materialized=True)
        result = self.exp(step, parameters, coefficients=coefficients,
                          delinearize=delinearize, delinearize_opts=delinearize_opts)
        if self._layout.kind == 'mpo':
            value = result.trace()
            return value / self._layout.dimension**len(self._layout.sites) if normalized else value
        return trace_pepo(result, normalized=normalized, **trace_options)

    __call__ = exp


def prepare_cluster_channels(builder, step=1.0, parameters=None, *, coefficients=None,
                             max_bond=None, samples=(), tangent_pairs=(), rtol=1e-12,
                             memory_budget=256*1024**2, structural_reuse=True, preparation='reference',
                             frontier_state_budget=None):
    """Prepare fixed virtual channel bases from explicit reference evaluations.

    ``samples`` are additional dictionaries of ``step``, ``parameters`` and/or
    ``coefficients``; omitted entries inherit the primary evaluation. Optional
    ``tangent_pairs`` contain ``(plus, minus, denominator)`` dictionaries and
    add finite-difference derivative snapshots to the QR selection. No sample
    tensors/graphs survive preparation. Frozen-map rank decisions happen only
    here; optional ``exp(..., delinearize=True)`` is separate numerical
    postprocessing at each evaluation and does not modify these maps.

    MPO-only ``preparation='frontier'`` directly shares completed histories
    by tracking only unfinished clusters crossing each cut. This avoids full
    collection enumeration for exact direct graph or uncapped recursive
    sources. It does not change the local cluster expansion. The default
    ``'reference'`` retains the earlier preparation and collection policy.
    ``frontier_state_budget`` caps active-cluster sets per cut; None inherits
    the source's ``assembly_state_budget``. Exceeding a budget raises, without
    dropping terms. Frontier size can still grow exponentially with cutwidth.

    Dense qubit MPO-only ``preparation='automaton'`` adds operator-aware
    weighted-state reduction to the frontier. Fixed Pauli product terms
    certify each residual's closed Pauli span. Exact rational elimination
    removes constant linear dependencies between complete transition slices,
    retaining the singleton rail. No SVD or QR is used without a smaller cap.
    Replay uses a fixed projection onto the certified span (identity on the
    declared family); arbitrary raw residual inputs outside it are projected.
    Operator inventory changes require a new plan. This can reduce channels
    beyond equality sharing, but neither improves every model nor guarantees
    a globally minimal automaton. Native charge sectors use 'frontier'.

    PEPO-only ``preparation='symbolic'`` compiles independent residual entries
    into a reusable linear template. It shares equal graph-edge slices before
    square routing, then equal square-edge slices. Each slice includes all
    other virtual legs and physical entries, so loops and branching remain
    valid. Reference samples evaluate this smaller template; they never call
    the expanded numerical PEPO builder. Remaining routed blocks are still
    enumerated once, and the quotient is not globally minimal.

    PEPO-only ``preparation='algebraic'`` restricts residuals to the closed
    algebra generated by the declared fixed qubit Pauli product terms. Exact
    rational elimination then removes constant linear dependencies between
    complete edge slices, protecting the singleton rail. It never divides by
    live coefficients or merges directions because reference values agree.
    Full channels need neither QR nor SVD; optional caps use reference QR.
    Replay includes a fixed Pauli-span projection, which is the identity on
    this declared family. Arbitrary ``bind_assembler`` inputs outside that
    span are projected. Operator inventory changes require a new plan.

    ``structural_reuse=True`` first merges formally equal MPO transitions
    within charge sectors, independently of reference values. Every residual
    entry is a separate symbolic atom; this conservative quotient preserves
    the whole parameter family. It is not a global automaton minimizer and
    does not apply one-dimensional rules to graph/square PEPOs. In symbolic
    or algebraic PEPO preparation this switch controls the edge-slice quotient instead.
    Set False to disable sharing. Symbolic preparation still removes formally
    zero identity blocks, never reference-value zeros. Different exact gauges
    can yield different approximations at the same cap.

    ``max_bond=None`` keeps the exact structural quotient. A positive cap
    includes the protected singleton rail and is applied after exact sharing.
    Reduced spaces are approximations, even if training defects are zero.
    ``rtol`` tests QR pivots, not singular values or a global operator error.
    ``memory_budget`` guards symbolic entry estimates, explicit preparation matrices, constant maps and
    output buffers. It excludes source cluster/residual enumeration, backend
    contraction workspace, autodiff storage and total process memory.
    """
    if max_bond is not None and (isinstance(max_bond, bool) or not isinstance(max_bond, Integral) or max_bond < 1):
        raise ValueError('max_bond must be a positive integer or None.')
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError('rtol must be finite and nonnegative.')
    if not isinstance(structural_reuse, bool):
        raise TypeError('structural_reuse must be a bool.')
    if preparation not in ('reference', 'frontier', 'automaton', 'symbolic', 'algebraic'):
        raise ValueError("preparation must be 'reference', 'frontier', 'automaton', 'symbolic' or 'algebraic'.")
    if frontier_state_budget is not None:
        if isinstance(frontier_state_budget, bool) or not isinstance(frontier_state_budget, Integral) or frontier_state_budget < 1:
            raise ValueError('frontier_state_budget must be a positive integer or None.')
        if preparation not in ('frontier', 'automaton'):
            raise ValueError("frontier_state_budget requires preparation='frontier' or 'automaton'.")
    if isinstance(memory_budget, bool) or not isinstance(memory_budget, Integral) or memory_budget < 1:
        raise ValueError('memory_budget must be a positive integer.')
    from ._cluster_channel_assembly import ChannelSource

    builder = _source(builder)
    binding = ChannelSource(builder, preparation=preparation, memory_budget=memory_budget,
                            frontier_state_budget=frontier_state_budget)
    base = dict(step=step, parameters=parameters, coefficients=coefficients)
    samples = tuple(samples)
    tangent_pairs = tuple(tangent_pairs)

    def validate_arguments(arguments):
        if set(arguments)-set(base):
            raise ValueError('channel samples accept only step, parameters and coefficients.')

    for arguments in samples:
        validate_arguments(arguments)
    for plus, minus, denominator in tangent_pairs:
        validate_arguments(plus)
        validate_arguments(minus)
        if not np.isfinite(denominator) or denominator == 0:
            raise ValueError('tangent denominator must be finite and nonzero.')

    first_maps = second_maps = None
    structural_report = None
    reference_evaluations = 0
    if preparation in ('symbolic', 'algebraic'):
        from ._cluster_channel_pepo import SymbolicPEPO

        source, _ = binding.resolve(base)
        template = binding.pepo = SymbolicPEPO(source, memory_budget, structural_reuse,
                                             algebraic=preparation == 'algebraic')
        raw_layout, layout = template.layout, template.reduced_layout
        first_maps, second_maps = template.first, template.second
        structural_report = template.structural_report
        needs_projection = max_bond is not None and any(
            max_bond < len(sectors) for sectors in layout.sectors)
        if needs_projection:
            values = template.values(binding.residuals(base), reduced=True)
            reference_evaluations += 1
        else:
            values = None
    else:
        layout, values = _capture(binding.evaluate(base))
        raw_layout = layout
        reference_evaluations += 1
    needs_references = values is not None and max_bond is not None and any(
        max_bond < len(sectors) for sectors in layout.sectors)
    references = [_host_values(values, memory_budget)] if needs_references else []

    def snapshot(arguments):
        nonlocal reference_evaluations
        reference_evaluations += 1
        if binding.pepo is not None:
            blocks = binding.pepo.values(binding.residuals({**base, **arguments}), reduced=True)
            return _host_values(blocks, memory_budget)
        structure, blocks = _capture(binding.evaluate({**base, **arguments}))
        _same_layout(layout, structure)
        return _host_values(blocks, memory_budget)

    if needs_references:
        references.extend(snapshot(arguments) for arguments in samples)
        for plus, minus, denominator in tangent_pairs:
            a, b = snapshot(plus), snapshot(minus)
            references.append(tuple((x-y)/denominator for x, y in zip(a, b)))
    if structural_reuse and binding.mpo:
        from ._cluster_channel_structure import share_mpo_channels

        source, _ = binding.resolve(base)
        layout, references, first_maps, second_maps, structural_report = share_mpo_channels(
            source, layout, references, memory_budget,
            assemble=None if binding.frontier is None else binding.frontier.arrays,
            algebra=binding.algebra)
    bases, charges, defects = [], [], []
    projection_applied = False
    for ends, sectors, sector_charges in zip(layout.ends, layout.sectors, layout.charges):
        if max_bond is None or max_bond >= len(sectors):
            _budget((len(sectors), len(sectors)), memory_budget)
            bases.append(np.eye(len(sectors)))
            charges.append(sector_charges)
            defects.append(0.)
            continue
        projection_applied = True
        candidates = []
        for side, endpoint in enumerate(ends):
            matrix = _unfold(layout, endpoint, sectors, references, memory_budget, conjugate=side == 0)
            candidates.append(_qr_space(matrix, sector_charges, max_bond, rtol))
        # Either endpoint's contained subspace suffices for an exact bond
        # contraction on the training tensors. Smaller local defect wins;
        # there is no claim of environment-optimal PEPO compression.
        basis, charge, defect = min(candidates, key=lambda item: (item[2], item[0].shape[1]))
        bases.append(basis)
        charges.append(charge)
        defects.append(defect)
    dual_bases = None
    if first_maps is not None:
        dual_bases = tuple(matrix @ basis.conj() for matrix, basis in zip(second_maps, bases))
        bases = tuple(matrix @ basis for matrix, basis in zip(first_maps, bases))
    return ClusterChannelPlan(builder, raw_layout, bases, charges, defects, max_bond, memory_budget,
                              binding=binding, arguments=base, dual_bases=dual_bases,
                              structural_report=structural_report,
                              reference_evaluations=reference_evaluations,
                              projection_applied=projection_applied)
