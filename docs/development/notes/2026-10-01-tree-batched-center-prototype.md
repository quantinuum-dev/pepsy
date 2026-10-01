# 2026-10-01 — Batch carried through a conditional tree

Classification: **prototype**, outside the production sampler. The user
proposed attaching a sample batch to the tree and carrying it through the
sampling traversal as in MPS sampling. This investigation follows that
proposal rather than the earlier prefix-grouping suggestion.

## Exact construction

The experiment starts from TreeSampler's captured root-canonical arrays.
It moves the canonical center to the next physical site, samples that site's
Born weights, selects and normalizes one physical slice per shot, then
absorbs the resulting batch into adjacent tensors. Fully measured,
physical-free degree-one branches are absorbed and removed. Center moves
through surviving branch nodes use reduced QR and absorption of R.

The sample axis remains a leading batch dimension in QR: `(B, rows, bond)`.
It is never folded into the QR row space, contracted between shots, or
included as an ordinary isometry index. Shared, unconditioned arrays remain
unbatched until a conditional factor reaches them. This performs independent
exact canonicalization for every shot, without numerical rank truncation,
prefix grouping or sample caches. It uses the existing tree draw order and
uniform streams. Source tensors and canonical-center metadata are preserved.

Unlike a chain, a branching center can couple several unmeasured bonds.
Materializing that center replicates their joint tensor for each shot; batched
Q tensors can also persist along the traversal. Thus attaching a batch to
every visited tensor is not inherently a memory or GPU performance improvement.

## New validation

An isolated subclass in `/tmp/pepsy_tree_batched_center.py` overrides only
the dense array sampling implementation. No production numerical code was
changed during this experiment; earlier working-tree changes were preserved.

`/tmp/pepsy_tree_batched_center_checks.py` completed **144 sampling calls**:
NumPy, Torch CPU, Torch CUDA and CuPy; complex64 and complex128; seven qutrit
sites with physical or virtual root; chunks unset, one and seven; explicit
seeds and successive persistent-RNG calls; non-C-contiguous input arrays.
Configurations matched the current sampler exactly, and returned probabilities
matched both that sampler and independently normalized dense probabilities.
Maximum absolute dense-reference probability error was `6.22e-9`, across
both precisions. Source array identities and center metadata were preserved.

Four five-qubit Torch complex128 derivative comparisons used a physical or
virtual root and chunks unset or three. All free source-tensor derivatives
matched independently scored, explicitly normalized probabilities; maximum
absolute derivative difference was `6.11e-16`.

The first check-driver run stopped on its CuPy input conversion, after the
NumPy and Torch cases passed. Correcting the harness to convert the NumPy
array before calling `cupy.asfortranarray` allowed the complete run to pass.

Native Symmray, fermionic arrays, JAX, zero-weight/rank-deficient adversarial
cases, production checkpoints and larger topologies were not validated for
this prototype. No production pytest suite was rerun in this experiment;
the 186-test result in the preceding vector-propagation note is earlier
validation of that separate implementation.

## Measured cost

Serial comparisons used a synthetic 30-qubit balanced tree, root arity three,
chi=32, complex128, 256 shots in chunks of 128, state seed 19, sample seed 2,
and `threads=1`. Both samplers were warmed up; capture/setup was excluded;
GPU timings synchronized before and after sampling. All timed configurations
matched exactly, and all probabilities agreed at `rtol=1e-10`, `atol=1e-22`.
Each table entry lists three elapsed times in seconds.

| Backend | Current sampler | Batched canonical tree | Ratio of median times |
| --- | --- | --- | --- |
| NumPy | 0.893, 0.861, 0.965 | 0.661, 0.577, 0.632 | 1.41x faster |
| CuPy, RTX A5000 | 0.0775, 0.0911, 0.0751 | 2.064, 2.052, 2.065 | 26.6x slower |

There were 52 center-move QR calls per chunk, each possibly batched. The
largest conditional tensor had shape `(128, 32, 32, 32)`, occupying 64 MiB.
Maximum summed tensor payload was about 68.6 MiB. This payload count excludes
QR workspace, temporaries, the sampler's captured source and allocator pools;
it is not a measured device-memory peak.

A guarded probe of the same 30-site construction with chi=256 and chunk=2048
stopped before allocating a conditional tensor of shape
`(2048, 1, 256, 256)`: **2 GiB for that tensor alone**. The experiment's guard
is 256 MiB per new tensor, not a total-memory bound. This is an observed shape
on that synthetic tree, not an estimate for the user's checkpoint.

Temporary reproduction/check scripts, logs and JSON use the stems
`/tmp/pepsy_tree_batched_center_checks` and
`/tmp/pepsy_tree_batched_center_benchmark`; the benchmark reuses the synthetic
builder in `/tmp/pepsy_tree30_audit.py`. These files may disappear; the
algorithm, validation scope and numerical results above are durable evidence.

## Capabilities and next design question

The unchanged installed dependency audit from the
[vector-propagation investigation](2026-10-01-tree-sampling-vectors.md#upstream-audit-and-classification)
was reused. Additional local probes verified leading-batch reduced QR in
NumPy 2.5.2, Torch 2.6.0+cu124 and CuPy 14.1.1. The installed public QR
implementations/signatures were inspected; the CUDA allocations succeeded.
No installed dependency was edited or upgraded.

**Proposed, unmeasured:** preserve the user's batch-carrying architecture but
represent conditional centers with compact per-shot factors and shared tree
tensors. Branch correlations still need to be represented exactly: simply
treating every incoming message as a pure `(B, chi)` vector is insufficient
when unmeasured siblings remain. An implementation must bound intermediate
factor sizes, decide when branch factors need compression or an exact density
representation, and avoid materializing whole batched branch tensors when
that is more expensive. GPU center-move QR costs should be profiled separately.
The measured literal prototype is not ready to replace the production sampler.
