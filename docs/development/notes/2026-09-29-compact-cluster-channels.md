# 2026-09-29 — Compact spin-cluster channels: literature and first implementation

The user requested a careful, literature-informed way to reduce MPO/PEPO
channel growth while retaining efficient autodiff without SVD. The scope
remains Pauli/spin. This is an explicit experimental route; it does not
replace existing exact, recursive, native or frozen-tree policies.

## Literature and decision

- [Crosswhite and Bacon, finite automata for matrix products](https://arxiv.org/html/0708.1221)
  relates operator paths to weighted automata and repeated partial contractions.
  **Adopt:** static channel maps and separating combinatorial compilation from
  numerical replay. **Defer:** a full symbolic automaton minimizer for the
  cluster family, particularly PEPO connectivity and disjoint collections.
  Equal values at a single parameter point are not proof of symbolic identity.
- [Hubig, McCulloch and Schollwöck, efficient MPO construction](https://arxiv.org/html/1611.02498)
  describes direct-sum/product bond growth and SVD, deparallelisation and
  delinearisation. **Adopt:** reducing virtual representations locally and
  preserving sparse assembly. **Defer:** applying numerical equality tests as
  runtime autodiff channel-merging rules; that could delete derivative directions.
  No claim that this implementation reproduces their delinearisation algorithm.
- [Van Schie, Kramer and Hwang, weighted POD for optimization](https://arxiv.org/html/2508.09084v1)
  discusses reduced spaces enriched by derivative information and the cost
  of parameter sensitivities. **Adopt:** multiple explicit snapshots and
  optional finite-difference block tangents. **Defer:** moving bases or
  differentiating basis selection. The implementation uses local pivoted QR,
  not that paper's weighted POD or its error bound.

The chosen bounded design is an exact edge relabeling plus optional fixed
virtual projection. Preparing an approximate basis is outside the objective.
Sparse physical blocks are evaluated afresh and contracted directly with the
constant endpoint maps into small site tensors. This avoids the dense tensor
with the original expanded virtual dimensions. There is no numerical
decomposition or rank selection during replay.

## Implemented behavior

Public `prepare_cluster_channels` returns `ClusterChannelPlan`. Fixed direct
MPOs, native bosonic MPOs, graph PEPOs and square PEPOs share the plan. Graph
materialization independently now uses per-edge structural ids, matching the
existing square convention and retaining tensor zeros.

For each virtual edge, preparation considers local unfolding snapshots at
both endpoints (conjugating the first endpoint to match the bilinear bond
contraction). Within each charge group it computes a pivoted-QR basis, keeping
the singleton rail explicitly. It selects the endpoint with the smaller local
relative projection defect. The two endpoint maps insert the fixed projector
on that bond. `max_bond=None` instead uses identity maps and is exact for any
parameters with that source layout. Training tensors/graphs are not retained.

This local criterion is gauge-dependent and not environment-optimal. A
vanishing training defect does not certify an entire parameter family.
Projecting multiple bonds, especially on loops, does not turn local defects
into a global error bound. General symbolic merging and frozen projection
inside recursive MPO subproblem assembly are future work, not hidden features.

`pack` and `bind_projector` expose a pure array contraction kernel. Torch
`compile(backend='aot_eager', fullgraph=True)` captures that kernel and its
backward pass; complete Python builder capture is not claimed. JAX jit/grad
checks include the actual compact operator trace, including native U1 MPOs.
The new native JAX path has fixed preselected sectors/ranks; it does not add
adaptive native JAX compression.

The single-precision probes exposed a fixed square Pauli transform using
complex128 constants with complex64 Torch operands. Pauli expansion and
reconstruction now align their constant basis to the live complex precision.
The larger regression selection also found an old test still expecting native
fixed construction to be rejected; it now checks its independent dense target,
matching the earlier native implementation.

## Measured storage, error and time

A three-qubit, three-factor spin test uses adjacent `XX+YY` generators and
an onsite Z factor. Primary parameters `(a,b,c,t)=(.2,-.3,.1,.7)`, step
`-.2j*t`; QR snapshots include four central finite-difference directions
with displacement `1e-4`. Held-out parameters are `(.24,-.27,.08,.74)`.
Dense/operator gradients below concern `sum(real(U)+.37*imag(U))`.

| Layout | cap | retained bonds | dense entries | relative operator error | absolute gradient L2 error |
| --- | ---: | --- | ---: | ---: | ---: |
| Exact source | none | 9,9 | 396 | 0 | 0 |
| MPO | 3 | 3,3 | 60 | 4.01e-2 | 2.27e-1 |
| MPO | 5 | 5,5 | 140 | 2.84e-3 | 7.04e-2 |
| MPO | 7 | 5,7 | 188 | 1.13e-4 | 1.77e-3 |
| Graph/square PEPO | 3 | 3,3 | 60 | 3.57e-2 | 2.37e-1 |
| Graph/square PEPO | 5 | 5,5 | 140 | 2.84e-3 | 7.04e-2 |
| Graph/square PEPO | 7 | 7,5 | 188 | 1.01e-4 | 1.30e-3 |

Native U1 matches the MPO figures at caps 5 and 7; its cap-3 charge allocation
has relative operator error 5.65e-2 and gradient error 5.08e-1. Tight caps
therefore cannot be presented as accurate merely because autodiff is finite.
The tests separately compare autodiff against finite differences of each
*projected* operator and against independent uncompressed SciPy products in
the exact mode.

Single-thread CPU development timings use double-precision Torch and actual
trace plus backward, median of the final three of five evaluations. At cap 7,
MPO baseline/compact was 10.6/10.3 ms, native MPO 20.6/28.9 ms, graph PEPO
16.2/10.3 ms and square PEPO 15.3/13.0 ms. Preparation including tangent
snapshots took 13–37 ms in these cases. These small probes ran in the active
development workspace and are not a controlled throughput benchmark.
Storage reduction is demonstrated; a general speedup is not.

For the same graph fixture, exact edge relabeling changes padded dimension
17 to local dimension 9 with identical output. It is independent of the
optional approximate channel cap. Dense storage counts exclude constant maps,
temporary contractions, source residuals and autodiff buffers.

## Upstream audit and validation

Installed versions rechecked: Quimb 1.15.1.dev66+ge927f06e1,
Autoray 0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f,
Cotengrust 0.2.1, Symmray 0.4.1.dev7+g83fb22865, Torch 2.6.0+cu124,
JAX 0.10.2. Inspected SciPy `qr(..., mode='economic', pivoting=True)`, public
Quimb MPO compression signature and Autoray QR dispatch. Reviewed the
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra docs](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray source](https://github.com/jcmgray/symmray). The historical
[Symmray array URL](https://symmray.readthedocs.io/en/latest/abelian_arrays.html)
is unavailable; the installed source and earlier same-task audit supply that
capability check. **Adopt:** public QR/Autoray/Quimb APIs. No dependency change,
installed-library patch or compatibility shim was introduced.

Final selected-suite counts and commit state are recorded in the
[handoff](../../../history/2026-09-29-compact-cluster-channels.md).
The new tests include independent dense products, actual projected traces,
Torch CPU/CUDA derivatives, native charges, JAX JIT, Torch kernel capture,
zero parameters, dtype/device, reference-graph independence, periodic crossing
geometry, no-SVD spies and the sparse-to-dense allocation guard.
