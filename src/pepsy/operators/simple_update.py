"""Prepared dense Torch simple-update schedules with independent gate batches."""

from collections import defaultdict
from functools import partial
from math import isfinite, prod


class PreparedSimpleUpdate:
    """Snapshot a dense PEPO and replay a fixed ordered gate schedule.

    ``schedule`` contains ``(sites, leg)`` pairs, with leg upper/lower. Gates
    passed to :meth:`apply` have one leading batch axis, for independent
    copies of the network. Only operations on disjoint sites may share a
    layer. The per-site order of the supplied schedule is never changed.

    The fast array path retains the input bond shapes, checks every member's
    actual cutoff decision, and replays the original adaptive algorithm if
    any rank changes. Call inside the desired TorchLinalgConfig scope.
    """

    def __init__(self, network, gauges, schedule, *, max_bond=4, cutoff='auto',
                 cutoff_mode='auto', smudge=1e-12, renorm_bonds=True,
                 torch_backend=None):
        import torch
        from .gates import _resolve_gate_cutoff, _resolve_gate_cutoff_mode
        cutoff = _resolve_gate_cutoff(network, cutoff)
        cutoff_mode = _resolve_gate_cutoff_mode(cutoff_mode)
        if isinstance(max_bond, bool) or not isinstance(max_bond, int) or max_bond < 1:
            raise ValueError('max_bond must be a positive integer')
        if not isfinite(cutoff) or cutoff < 0 or not isfinite(smudge) or smudge < 0:
            raise ValueError('cutoff and smudge must be finite and nonnegative')
        if cutoff_mode not in ('abs', 'rel', 'sum1', 'sum2', 'rsum1', 'rsum2'):
            raise ValueError('unsupported cutoff mode')
        if torch_backend not in (None, 'aot_eager'):
            raise ValueError('torch_backend must be None or aot_eager; Inductor is not validated for complex SU derivatives')
        if network.get_multibonds():
            raise ValueError('prepared SU requires one bond per tensor pair')
        if not all(isinstance(t.data, torch.Tensor) and t.data.is_complex() for t in network):
            raise TypeError('prepared SU requires dense complex Torch tensors')
        prototype = network.tensors[0].data
        if any(t.data.dtype != prototype.dtype or t.data.device != prototype.device for t in network):
            raise TypeError('all target arrays must share a Torch dtype and device')
        if any(network.ind_size(fn(*site)) != 2 for site in network.sites
               for fn in (network.upper_ind, network.lower_ind)):
            raise ValueError('prepared SU requires qubit physical dimensions')
        if any(not isinstance(g, torch.Tensor) or g.device != prototype.device or g.is_complex()
               for g in gauges.values()):
            raise TypeError('SU weights must be real Torch vectors on the target device')
        self.network = network.copy()
        self.network.apply_to_arrays(lambda t: t.detach().clone())
        self.network.exponent = (network.exponent.detach().clone()
            if isinstance(network.exponent, torch.Tensor) else network.exponent)
        self.sites = tuple(network.sites)
        self.tensors = tuple(self.network[self.network.site_tag(s)] for s in self.sites)
        self.site_ids = {s: i for i, s in enumerate(self.sites)}
        self.gauges = {k: v.detach().clone() for k, v in gauges.items()}
        self.options = dict(max_bond=max_bond, cutoff=cutoff, cutoff_mode=cutoff_mode,
                            smudge=smudge, renorm_bonds=renorm_bonds)
        self.torch_backend = torch_backend
        self._kernels = {}
        self.schedule = tuple((tuple(sites), leg) for sites, leg in schedule)
        self.operations = []
        self._requires_adaptive = False
        levels = defaultdict(list)
        last = defaultdict(lambda: -1)
        for k, (sites, leg) in enumerate(self.schedule):
            if leg not in ('upper', 'lower') or len(sites) not in (1, 2):
                raise ValueError('schedule requires one/two sites and an upper/lower leg')
            ids = tuple(self.site_ids[s] for s in sites)
            if len(set(ids)) != len(ids):
                raise ValueError('two-site updates require distinct sites')
            indices = tuple((network.upper_ind if leg == 'upper' else network.lower_ind)(*s)
                            for s in sites)
            bond = None
            if len(ids) == 2:
                shared = set(self.tensors[ids[0]].inds) & set(self.tensors[ids[1]].inds)
                if len(shared) != 1:
                    raise ValueError('prepared SU supports adjacent two-site gates')
                bond = next(iter(shared))
            layouts = []
            for i, physical in zip(ids, indices):
                t = self.tensors[i]
                tail = (physical,) if bond is None else (bond, physical)
                outer = tuple(ix for ix in t.inds if ix not in tail)
                order = outer + tail
                permutation = tuple(t.inds.index(ix) for ix in order)
                shape = tuple(t.ind_size(ix) for ix in order)
                layouts.append((permutation, tuple(permutation.index(j) for j in range(t.ndim)),
                                shape, outer))
            if bond is not None:
                dimension = layouts[0][2][-2]
                # An overcomplete input bond can exceed the reduced core's
                # entire spectrum. Replay adaptively before fixed-shape views.
                self._requires_adaptive |= any(
                    min(prod(layout[2][:-2]), dimension*layout[2][-1])*layout[2][-1] < dimension
                    for layout in layouts)
            key = (len(ids), tuple(v[2] for v in layouts))
            self.operations.append((ids, leg, bond, tuple(layouts), key))
            level = max(last[i] for i in ids) + 1
            for i in ids:
                last[i] = level
            levels[level].append(k)
        self.groups = []
        for level in sorted(levels):
            buckets = defaultdict(list)
            for k in levels[level]:
                buckets[self.operations[k][-1]].append(k)
            self.groups.extend(tuple(ks) for ks in buckets.values())
        self.info = dict(operations=len(self.operations), batches=len(self.groups),
                         compiled=False, fallback=False)

    def _kernel(self, inputs):
        from ._simple_update_torch import two_site
        from pepsy.backends import get_torch_linalg_config
        key = (repr(get_torch_linalg_config()), tuple(
            (tuple(x.shape), x.stride(), x.dtype, x.device, x.requires_grad) for x in inputs))
        if key not in self._kernels:
            function = partial(two_site, **self.options)
            if self.torch_backend is not None:
                import torch
                from torch._functorch.aot_autograd import aot_function
                from functorch.compile import make_boxed_func

                def compiler(graph, examples):
                    # AOT may provide a lazy FX module; materialize its array
                    # program before Dynamo tries to trace its Python wrapper.
                    graph = torch.fx.GraphModule(graph, graph.graph)
                    return make_boxed_func(graph.forward)

                # Trace forward and the registered stabilized QR/SVD pullbacks
                # together; compiling a forward-only FX graph would lose them.
                function = aot_function(function, fw_compiler=compiler, bw_compiler=compiler)
            self._kernels[key] = function
        self.info['compiled'] = self.torch_backend is not None
        return self._kernels[key]

    @staticmethod
    def _outer_many(layouts, conditioned, batch, prototype):
        import torch
        shape = layouts[0][2][:-2]
        total = batch*len(layouts)
        output = prototype.real.new_ones((total,) + shape)
        for axis, dimension in enumerate(shape):
            indices = [layout[3][axis] for layout in layouts]
            if not any(ix in conditioned for ix in indices):
                continue
            factors = [conditioned[ix] if ix in conditioned else prototype.real.new_ones((batch, dimension))
                       for ix in indices]
            view = [total] + [1]*len(shape)
            view[axis+1] = dimension
            output = output*torch.cat(factors).reshape(view)
        return output.reshape(total, -1)

    def apply(self, gates):
        """Return independent ``(PEPO, gauges)`` pairs, preserving autograd."""
        steps = self._steps(gates)
        try:
            inputs = next(steps)
            while True:
                inputs = steps.send(self._kernel(inputs)(*inputs))
        except StopIteration as completed:
            return completed.value

    @staticmethod
    def apply_many(plans, gate_batches):
        """Replay independent schedules, batching compatible ready updates.

        Each gate list has the same contract as :meth:`apply`. Results retain
        plan order and each plan's network-batch order. Only independent
        networks share kernels; every schedule keeps its own update order,
        truncation policy and adaptive fallback.
        """
        import torch
        plans, gate_batches = tuple(plans), tuple(gate_batches)
        if len(plans) != len(gate_batches):
            raise ValueError('one gate list is required per prepared plan')
        outputs = [None] * len(plans)
        steps = [plan._steps(gates) for plan, gates in zip(plans, gate_batches)]
        ready = []
        for i, iterator in enumerate(steps):
            try:
                ready.append((i, next(iterator)))
            except StopIteration as completed:
                outputs[i] = completed.value
        while ready:
            compatible = defaultdict(list)
            for i, inputs in ready:
                plan = plans[i]
                key = (tuple(plan.options.items()), plan.torch_backend, tuple(
                    (tuple(x.shape[1:]), x.dtype, x.device) for x in inputs))
                compatible[key].append((i, inputs))
            ready = []
            for members in compatible.values():
                plan = plans[members[0][0]]
                sizes = [inputs[0].shape[0] for _, inputs in members]
                inputs = (members[0][1] if len(members) == 1 else
                          tuple(torch.cat(parts) for parts in zip(*(x for _, x in members))))
                updated = plan._kernel(inputs)(*inputs)
                pieces = tuple(x.split(sizes) for x in updated)
                for n, (i, _) in enumerate(members):
                    plans[i].info['compiled'] = plan.info['compiled']
                    plans[i].info['shared_batches'] += len(members) > 1
                    try:
                        ready.append((i, steps[i].send(tuple(x[n] for x in pieces))))
                    except StopIteration as completed:
                        outputs[i] = completed.value
        return outputs

    def _steps(self, gates):
        """Yield ready numeric updates while retaining per-network state."""
        import torch
        self.info['shared_batches'] = 0
        if len(gates) != len(self.operations) or not gates:
            raise ValueError('one batched matrix is required for every scheduled gate')
        batch = gates[0].shape[0]
        if batch < 1 or any(g.shape[0] != batch or g.dtype != self.tensors[0].data.dtype
                           or g.device != self.tensors[0].data.device for g in gates):
            raise ValueError('gate batches must share batch size, dtype and device with the target')
        if self._requires_adaptive:
            self.info['fallback'] = True
            return self._adaptive(gates)
        arrays = [t.data.unsqueeze(0).expand((batch,)+t.shape) for t in self.tensors]
        gauges = {k: v.unsqueeze(0).expand(batch, -1) for k, v in self.gauges.items()}
        def condition(g):
            return g + self.options['smudge']*g.amax(-1, keepdim=True)
        conditioned = {k: condition(v) for k, v in gauges.items()}
        exponent = gates[0].real.new_zeros(batch) + self.network.exponent
        valid = []
        for group in self.groups:
            aa, bb, la, lb, ws, gg = [], [], [], [], [], []
            for k in group:
                ids, leg, bond, layouts, _ = self.operations[k]
                gate = gates[k].reshape(batch, 2**len(ids), 2**len(ids))
                gg.append(gate.transpose(-1, -2) if leg == 'lower' else gate)
                for j, (i, layout) in enumerate(zip(ids, layouts)):
                    permutation, _, shape, _ = layout
                    arr = arrays[i].permute((0,)+tuple(x+1 for x in permutation))
                    (aa if j == 0 else bb).append(arr.reshape(batch, -1, *shape[-len(ids):]))
                    if len(ids) == 2:
                        (la if j == 0 else lb).append(layout)
                if bond is not None:
                    ws.append(gauges[bond] if bond in gauges else
                              arrays[ids[0]].real.new_ones((batch, layouts[0][2][-2])))
            g = torch.cat(gg)
            if len(ids) == 1:
                updated = (torch.einsum('nou,nxu->nxo', g, torch.cat(aa)),)
            else:
                inputs = (torch.cat(aa).contiguous(), torch.cat(bb).contiguous(),
                          self._outer_many(la, conditioned, batch, aa[0]),
                          self._outer_many(lb, conditioned, batch, bb[0]), torch.cat(ws), g)
                updated = yield inputs
                valid.append(updated[4])
                exponent = exponent + updated[3].reshape(len(group), batch).sum(0)
                # Condition all independent weight vectors together before
                # distributing views, avoiding one reduction/dispatch per gate.
                conditioned_update = condition(updated[2])
            for n, k in enumerate(group):
                ids, _, bond, layouts, _ = self.operations[k]
                for j, (i, layout) in enumerate(zip(ids, layouts)):
                    _, inverse, shape, _ = layout
                    arrays[i] = updated[j][n*batch:(n+1)*batch].reshape((batch,)+shape).permute(
                        (0,)+tuple(x+1 for x in inverse))
                if bond is not None:
                    gauges[bond] = updated[2][n*batch:(n+1)*batch]
                    conditioned[bond] = conditioned_update[n*batch:(n+1)*batch]
        if valid and not bool(torch.cat(valid).all()):
            self.info['fallback'] = True
            return self._adaptive(gates)
        self.info['fallback'] = False
        outputs = []
        for n in range(batch):
            network = self.network.copy()
            for site, arr in zip(self.sites, arrays):
                network[network.site_tag(site)].modify(data=arr[n])
            network.exponent = exponent[n]
            outputs.append((network, {ix: g[n] for ix, g in gauges.items()}))
        return outputs

    def _adaptive(self, gates):
        from .gates import gate_simple, renorm_gauge
        outputs = []
        for n in range(gates[0].shape[0]):
            network = self.network.copy()
            gauges = dict(self.gauges)
            for gate, (sites, leg) in zip(gates, self.schedule):
                if len(sites) == 1:
                    network.gate_(gate[n], sites, which=leg, transpose=leg == 'lower', contract=True)
                else:
                    gate_simple(network, gate[n], where=sites, gauges=gauges,
                        which=leg, transpose=leg == 'lower', renorm=False,
                        **{k: v for k, v in self.options.items() if k != 'renorm_bonds'})
                    if self.options['renorm_bonds']:
                        renorm_gauge(network, gauges, sites, self.options['smudge'])
            outputs.append((network, gauges))
        return outputs
