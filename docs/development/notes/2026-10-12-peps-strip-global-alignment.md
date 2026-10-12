# PEPS strip and global fitting alignment

## Implemented

Follow-up default policy: `mode="sweep"` selects strip batching, column
ordering (row for row style), sequential SU guesses, lazy targets, ALS
`linear_solver="auto"`, and `reuse-direct` boundary guesses. Caps are
`(2*D**2, 4*D**2)` for sweep/global norm and overlap, with matching automatic
normalization/evaluation settings. Strip sweeps keep fixed caps.

Each fit starts at the gated strip and proceeds to the last parallel strip,
then performs two complete backward/forward axis round trips. Each visit has
an independent budget of two inner ALS round trips. `optimize_axis` now accepts
`start_index`; `optimize_strip` retains its original frozen-exterior semantics.
Moving boundaries update from changed tensors, and local directional cursors
rebuild at reversal. The driver retains per-visit infidelity/cache histories.

- Fixed row/column gate targets, sequential SU guesses, and active-strip ALS
  round trips. Global mode also accepts two parallel strips or the full queue.
  Gate ordering remains conservative; targets are untruncated.
- Strip ALS defaults to explicit split-gate factors with site/row/column tags;
  local dense and matrix-free RHS contractions retain every target factor.
  Global fitting can opt into this representation. Direct bra/ket compression
  absorbs gates with KET; it does not compress after each individual gate.
- Independent norm/overlap caps, known unitary target norm one, and candidate
  normalization. Global constructor chi now also reaches the loss.
- Small Hermitian positive-support spectral ALS solves and large matrix-free
  CG through opt-in `linear_solver="auto"`. Existing explicit solvers remain.
- Boundary `reuse-direct/src/sdc` seeds once and refits thereafter. Compatible
  guesses cross strip batches; numerical contraction caches do not.
- Explicit bra/ket layer ordering for global norm and overlap. Torch/JAX rules
  register before global differentiation; JAX defaults retain fixed SVD ranks
  and avoid scalar exponent stripping in the differentiated loss.

## Upstream audit

Installed: Quimb `1.15.1.dev90+g6a3906cbe`, Autoray
`0.11.1.dev14+g014a3f69a`, Cotengra `0.8.3.dev8+g8954240f2`, Cotengrust
`0.2.1`, Symmray `0.4.1.dev15+g0374aaa3c`, Torch `2.6.0+cu124`.
Checked Quimb changelog/signatures, Autoray repository, Cotengra docs/changelog,
and Symmray repository. The Symmray array-documentation page was unavailable;
installed source supplied the capability check.

During validation the shared environment changed externally to Autoray
`0.8.4`, Quimb `1.15.1.dev91+gb3bf45a50`, and Symmray
`0.4.1.dev18+g44fc5ae53`. New Quimb calls `get_namespace(array)` and
`random.array`; that older Autoray lacks the required semantics and caused
mixed NumPy/Torch arrays and JAX tracing failures. The shared installation was
not modified by this task. Final isolated checks use the original Autoray
commit `014a3f69a` plus a snapshot of the new Quimb/Symmray/Cotengra under
`/tmp/pepsy-strip-validation-deps`. This is a validation environment, not a
package compatibility patch for mismatched upstream installations.

**Adopt:** public Quimb direct boundary compression and layer ordering; generic
ALS dense/matrix-free concepts, keeping Pepsy's PEPS environment schedule.
**Compatibility shim:** use the existing warned direct-guess fallback when
installed SRC cannot generate backend-native noise. Native Symmray keeps its
existing FIT route rather than entering a dense guess compressor.
**Defer:** new JAX/Symmray ALS support and paper-specific preconditioners.

[Zhang et al., arXiv:2511.09546](https://arxiv.org/abs/2511.09546) studies
preconditioning iPEPS energy optimization with approximate local norm metrics
and regularized iterative solves. It motivates conditioning strategies, but
does not establish a preferred solver or performance gain for these finite
PEPS gate-target ALS fits. The supplied Google Scholar citation was unavailable.

## Interpretation

The old FullUpdate sweep campaign used a single outer column/row pass, with
inner site round trips. It did not test this fixed strip-batch protocol. New
correctness tests must not be presented as long-run fidelity benchmarks. The
existing campaign snapshots, results, and notebook curves remain historical.

## Lazy-target measurements and numerical checks

A small CPU Torch complex128 probe used a normalized 3x3 random PEPS
(seed 322), six vertical ZZ(0.25) gates, Quimb direct BRA/KET compression,
caps `(16, 24)`, cutoff zero, and canonicalization enabled. Times are medians
of three calls; value/gradient timing excludes the first call. These are
single-process observations, not scaling benchmarks or peak-memory profiles.

| D | Target | Stored elements | Largest tensor | Build ms | Value/gradient ms |
|---|---|---:|---:|---:|---:|
| 2 | Absorbed PEPS | 384 | 128 | 12.56 | 30.91 |
| 2 | Explicit gates | 224 | 32 | 2.82 | 31.56 |
| 3 | Absorbed PEPS | 1,440 | 648 | 15.51 | 51.33 |
| 3 | Explicit gates | 546 | 162 | 2.79 | 52.93 |

Dense target vectors agreed to `1e-10`. At D=2 both approximate losses were
0.3362990656. At D=3 they were 0.3231560647 and 0.3240868173: different
truncation paths need not agree at a fixed cap. The probe shows reduced
target storage/build time, but no improvement in repeated gradient time.
It does not measure peak boundary workspace or long-run fitting quality.

An initial Torch finite-difference gradient check failed when a diagonal
qubit gate retained redundant rank-four channels. Using its exact algebraic
rank bound of two resolved that case. An initial JAX case passed its gradient
check but worsened independently measured fidelity; applying the same bound
to concrete JAX gates resolved that case. Traced/trainable gates do not use
value-based rank detection. Disabling canonicalization alone did not resolve
the JAX failure. These checks justify regression coverage, not a claim of
stable derivatives for arbitrary rank-deficient circuits.

**Prototype/opt-in:** lazy global targets remain optional. Quimb compresses
the first spatial boundary jointly and later absorbs all gates in the KET
group. Keeping the target factored does not guarantee a smaller largest
temporary tensor. A separate per-gate absorption schedule and peak-workspace
comparison are deferred; the current implementation does not claim either.

Validation results are recorded in the
[session handoff](../../../history/2026-10-12-peps-strip-global-alignment.md).

## Deeper compatibility review

**Compatibility shim:** the installed Quimb sequential MPS boundary routine
expects a single incoming tensor per site/layer; default split-gate targets
violate this and previously raised `ValueError` in `contract_between`.
Metrics, strip environments, and global MPS contractions now use a shared
exact within-site/layer preparation on temporary network copies. Ordinary
PEPS layers need no extra contraction. This preserves target storage and
gradients, but can enlarge temporary site tensors. The earlier timing table
predates this preparation and is not a benchmark of the revised path.

Quimb's environment builder also indexes a second plane for singleton axes.
Those axes now correctly have no boundary cuts. Global norm/loss evaluation
owns its temporary tensor metadata to prevent Quimb layer unions from
renaming caller indices when candidate and target share bond labels.

The upstream audit and installed versions above were reused in the unchanged
validation environment. Dense per-visit fidelity checks, Torch gradients,
and JAX finite differences cover these paths; see the
[deep-review handoff](../../../history/2026-10-12-peps-strip-deep-review.md)
for final validation scope and limitations.

## Publication review

The final review rechecked the official upstream sources linked in
[`AGENTS.md`](../../../AGENTS.md#upstream-compatibility-work) and the installed
boundary/environment and gate-application signatures. The Symmray array
documentation remained unavailable. **Defer:** no further upstream adoption
or dependency changes are needed for this patch.

Fresh checks use the same isolated Quimb/Autoray/Cotengra/Symmray snapshot
recorded in the deep-review handoff. Its shared backend packages currently
report Torch `2.14.1+cu126` and JAX `0.11.2`; these checks are separate from
the earlier results. Publication results are recorded in the
[final handoff](../../../history/2026-10-12-peps-strip-publication.md).
