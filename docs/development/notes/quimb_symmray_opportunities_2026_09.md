# Quimb and Symmray: remaining adoption opportunities

Audit date: **2026-09-24**. This assessment follows the implemented
[Symmray compatibility work](symmray_2026_09.md). It records recommendations
and small capability probes; no production behavior or installed dependency
was changed in this pass.

## Sources and actual environment

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
  including the **unreleased** 1.16 section.
- The requested Symmray ReadTheDocs changelog and Abelian-array page could
  not be retrieved. The official
  [Symmray release notes](https://github.com/jcmgray/symmray/releases)
  and installed implementations were checked instead.
- The Autoray repository and Cotengra documentation/changelog were also
  rechecked as required by the repository's upstream audit policy.

The activated `genpy` environment contains Quimb
`1.15.1.dev66+ge927f06e1`, Symmray `0.4.1.dev7+g83fb22865`, Autoray
`0.11.1.dev3+g1b476b305`, and Cotengra `0.8.3.dev7+g1d7fd333f`.
The development packages already expose many of the advertised additions.
Recommendations below concern Pepsy integration, not a blanket upgrade.

## Recommended order

### 1. Compatibility shim: measurement and compression options

Pepsy still forwards `mode=` to 2D contraction/measurement APIs in
`bp/observables.py`, `tensors/symmetric.py`, `optimizers/energy/peps.py`,
`optimizers/global_opt.py`, and boundary/VMC helpers. Current Quimb uses
`method=` for compression and `route=` for environment selection; the old
keyword warns. Introduce a narrow capability-aware adapter while preserving
Pepsy's public options and support for older Quimb. Do not mechanically replace
every `mode`: graph-distance, optimizer, and Torch compile modes are unrelated.

Preserve explicit normalization, bond limits, and return formats. In particular,
the BP wrapper forwards `normalized` without coercion, while symmetric and
energy helpers convert it to Boolean. Supporting Quimb's `normalized="return"`
would require a deliberate new result contract, not simply forwarding a string.

The default boundary route on a 1x4 PEPS still raises `IndexError`; explicitly
selecting `route="envs"` instead raises `KeyError('Y0')` in the tiny probe.
Use a tested one-dimensional or generic contraction route for degenerate
geometry. Do not assume the new environment dispatcher fixes this case.

Also audit cutoff policy across compressors. `boundary/sweeps.py` currently
changes cumulative `sdcr` cutoffs to `rel`. These controls are not numerically
equivalent. Expose the effective policy, or use a deterministic final sweep
that enforces the requested cumulative bound, following the new MPS guess
fallback. Preserve existing defaults until this is explicitly validated.

### 2. Adopt after validation: independent oversampling and final compression

The installed `tensor_network_1d_compress_sdc_oversample` now explicitly
accepts `max_bond_oversample`, `cutoff_oversample`, `cutoff_mode_oversample`,
and `compress_opts_final`. These were unavailable in the earlier
[MPS compression audit](mps_quimb_compression_audit.md).

Expose these consistently through MPS/MPO updates and PEPS boundary sweeps.
Tree and tree-PEPS code already has separate oversampling controls, so this is
an integration gap rather than a new algorithm. This allows a cheap intermediate
compression followed by a controlled final truncation. Validate reconstruction,
requested cutoff, dtype/device, reproducible random seeds, and FIT target
preservation. Do not change the default algorithm or advertise an unmeasured
speedup.

### 3. Prototype: native fermionic PEPS boundary compression

`boundary/sweeps.py` still rejects all native Symmray direct boundary targets.
Current upstream fixes justify testing deterministic `direct`, `sdc`, and
`sdc-oversample` there, followed by explicitly selected randomized methods.

A four-site complex128 U1 fermionic MPS, bond dimension 4 and alternating
site charges, retained native arrays through upstream `direct`, `sdc`,
`sdc-oversample`, and `sdcr`. At an untruncated cap of 64, relative reconstruction
errors were between `3.9e-16` and `5.7e-16`. This is a chain capability check;
it does not yet validate the PEPS boundary's bra/ket layers, odd dummy modes,
active truncation, or Torch gradients.

`boundary/metrics.py` also blocks native CTMRG with BP canonicalization, and
`SymPEPS.measure` unconditionally falls back from native projector compression
to MPS compression. Revisit these restrictions selectively. A 3x3 Z2 fermionic
norm probe using upstream `method="projector", canonize="bp"` matched the
exact norm at cap 64; cap 2 returned a finite result and invoked two D2BP
initializations. U1/U1U1, absent sectors, periodic seams, and backend gradients
still need validation before lifting Pepsy's gates.

### 4. Prototype: shared environments and periodic measurements

The installed APIs include `EnvironmentPlan`, `gen_compressed_environments`,
`PEPS.compute_partial_traces`, `PEPS.gen_block_environments`, and periodic
MPS `compute_partial_traces(route="envs")`. Pepsy does not directly use the
new planning APIs. Its BP boundary wrapper already batches expectation terms,
and Torch VMC already caches directional environments, so neither should be
described as currently lacking all reuse.

Start with many-observable PEPS measurements and periodic MPS reduced density
matrices. A six-site cyclic complex128 MPS returned batched one-site and
separated two-site density matrices agreeing with exact contractions within
`3e-16`. Next compare the new environment route with existing Pepsy caches on
representative measurements. Reuse plans only with compatible geometry and
invalidate numerical environments when state data changes. VMC walker-specific
conditioning and gradients need separate validation; this is not a drop-in
replacement for its compiled reuse path.

### 5. Prototype: upstream edge-loop enumeration

`bp/series.py::_enumerate_edge_loops` performs local connected-edge search.
Quimb's `gen_gloops_edge_induced` returns `NetworkPatch` objects with tensor
IDs and explicit bond indices. A 2x3 PEPS probe produced the same four closed
edge sets as Pepsy when using `max_size=6`, `num_joins=2`, and filtering to
seven edges.

An adapter may reduce local maintenance. Preserve Pepsy's edge-count cutoff:
Quimb's `max_size` counts tensors, and `num_joins` changes enumeration coverage.
Prove completeness for joined loops and parallel periodic bonds before
replacement. Open support-connecting paths, corridor budgets, deterministic
ordering, and native fermionic restrictions remain Pepsy responsibilities.

### 6. Adopt selectively: additional flat-array batching

The scalar-phase and flat-Z2 vmap fixes are already adopted and regression
tested. Extend validation to actual local-energy connected configurations and
selected truncating boundary geometries before expanding automatic batching.
Report executed batching and fallback reasons, and compare every parameter
gradient. The earlier roughly 10x CPU result compares serial and vmap on the
new stack; it is not a measured dependency-upgrade speedup or a GPU result.

## Already present or intentionally deferred

- SDC/SDCR selectors, higher-order Trotter scheduling, and dense MPO sandwich
  auto-swaps already exist in Pepsy. Extend their coverage instead of adding
  duplicate APIs.
- Symmray 0.4 scalar readout, native BP message conventions, and cumulative
  cutoff fallback for native MPS guesses were implemented in the previous pass.
- `gram(axes=-1)` is useful for individual positive native metrics and already
  supports the BP capability probe. It is not a replacement for arbitrary
  environments; fermionic Gram currently supports one open axis.
- Keep global sparse SVD and degenerate-sector behavior as the accuracy
  baseline. Eager sector allocation and randomized SVD can retain less spectral
  weight at the same bond cap, as the earlier measurements demonstrate.
- Do not replace Pepsy's graded FIT implementation with upstream one-site FIT
  wholesale: upstream still warns about odd-parity tensors.
- Preserve the structural-zero native QR safeguards. These release notes do
  not establish that the underlying issue is fixed.

## Validation boundary

This pass inspected actual callable signatures and the installed compression
registry and ran the small numerical probes described above. It did not rerun
the full suite or benchmark the proposed integrations. The preceding combined
verification remains **420 passed, 1 skipped, 2 upstream-reference failures**;
see the Symmray audit for exact scope. Only this assessment and its index link
were added in this pass.

## Implementation follow-up: priorities 1 and 2

Implemented on 2026-09-24 following explicit user authorization. Priorities
3–6 remain deferred. The environment was rechecked: Quimb, Autoray, and
Cotengra retain the versions above; Symmray had independently advanced to
`0.4.1.dev8+gc45f91457`. No installed dependency was changed by this task.

### Measurement compatibility shim

- `_internal/quimb.py` adapts explicit `mode`/`method` capabilities, including
  legacy full-bond similarity options. The adapter is used at PEPS boundary,
  energy, sampling, symmetric-tensor, and VMC entry points; graph-distance and
  optimizer mode selectors retain their own meanings.
- `boundary/_measurements.py` owns local measurement dispatch. Compression
  method and environment route remain separate. The environment route
  translates `contract_optimize` to `optimize` and rejects incompatible
  precomputed boundary caches instead of leaking those options into SVD.
- Single-row/column PEPS use exact operator-inserted contractions, including
  native graded arrays, without constructing a dense state vector or RDM.
  Boundary cutoffs are inapplicable on this path. Torch parameter gradients
  match exact references, and U1 long-range hopping retains fermionic signs.
- Boolean normalization still means raw versus normalized expectations.
  BP `return_all=True` now consistently returns scalars across Quimb revisions;
  `normalized="return"` explicitly requests numerator/norm pairs. SymPEPS and
  energy interfaces keep their existing Boolean/scalar contracts.
- Two existing one-dimensional BP tests now use `partial_trace_exact` for
  their exact oracle. This removes their dependence on upstream's broken
  degenerate 2D boundary sweep without relaxing the numerical assertions.

### Adopted compression controls

MPS/MPO dense Quimb replay accepts `compression_opts`; PEPS boundary workers,
metrics, and optimizers forward `fit_compression_opts`. Both expose separate
intermediate cap/cutoff controls and an explicit final decomposition/cutoff
policy. The final bond cap remains authoritative. Options are copied and
capability-validated before replay or boundary retuning. Unsupported native,
channel, or non-Quimb paths reject explicit controls. Existing FIT targets and
guess policies are unchanged. Base SDCR's cumulative-to-relative compatibility
coercion now warns rather than describing the policies as equivalent.

See [compression API and examples](../../api/boundary/compression.md).
Tests cover dense gates, interior sub-MPOs, both MPO layers, PEPS norms,
active truncation, copied shot replay, caller ownership, and seeded SRC
reproducibility. Small additional Torch CPU probes retained float64 arrays and
finite parameter gradients through SDC/zip-up oversampling. No GPU performance
or speedup claim is made, and native PEPS direct-compression gates remain in
place.

### Validation and outstanding repository failures

The final focused selection (`test_quimb_adoption`, `test_quimb_compat`,
`test_bp_open_series`, `test_public_api`, `test_package_layout`) passed
**136 tests**, including **49 new adoption regressions**. The final adoption
selection against isolated Quimb revision `d0591eb70` and Symmray 0.4.0 passed
**48 tests, 1 skipped**; the missing explicit environment route is skipped.
An earlier broader legacy selection passed 123 tests, 2 skipped.

Additional domain checks passed 218 tests, 1 skipped (native symmetric
tensors, Torch VMC compilation, direct PEPS sampling). The MPS/MPO/boundary
selection passed 980 tests with one existing FIT cache assertion failure.
Ruff and `git diff --check` passed.

The full repository run completed with **4,566 passed, 121 skipped, 13 failed**
before the final option-validation tightening and focused regression additions.
It is not a clean full-suite result:

| Remaining failures | Evidence |
| --- | --- |
| Three MPS FIT/cache/replay metadata assertions | Reproduced using the original `HEAD` MPS optimizer, including unexpected eager target caches and a mixed-replay backend choice. |
| Two Torch VMC batching/compiled-boundary assertions | Pass in isolated domain runs. Reproduced using the original `HEAD` amplitude implementation after registering the existing safe real-QR handler; its data-dependent rank check prevents batching/export in that process configuration. The QR safeguard was retained. |
| Native MPO DMRG product compression | Reproduces in isolation with a Quimb NumPy eig-SVD division by zero. No change was made to that fitting path. |
| Five JAX tree/stabilizer backend tests | Outside this change's implementation paths; remain unresolved. |
| Tree PEPO validation count and MPI-only configuration tests | Outside this change's implementation paths; remain unresolved. |

The full-run log is `/tmp/pepsy-priority12-full-suite.log`; focused and legacy
logs use the `/tmp/pepsy-priority12-` prefix. Baseline probes used temporary
modules rather than replacing working-tree files. Nothing was staged,
committed, or published.
