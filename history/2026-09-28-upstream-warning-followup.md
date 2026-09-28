# 2026-09-28 — Upstream warning and GPU follow-up

- Scope: the user's request to handle the remaining warnings and hardware
  validation after the previous publication.
- Branch / baseline: `develop` / `3525bc1`.
- Status: investigation complete locally; this handoff accompanies the
  documentation commit. Runtime dependencies and Pepsy's numerical
  implementation are unchanged. Upstream adoption and unavailable GPU
  hardware remain external limits.

## Correction to the preceding record

Commit `3525bc1` was pushed; hosted
[CI run 36492994369](https://github.com/quantinuum-dev/pepsy/actions/runs/36492994369)
passed both core and package jobs. The preceding handoff was written before
that push.

The full-run log also corrects the worker-warning attribution in the
[preceding evidence](../docs/development/notes/2026-09-28-remaining-maintenance.md):
its **one** notice was attached to
`test_tree_optimizer_dmrg_uses_tree_fit_engine[dmrg1-3-2]`, not the NumPy MPS
phase-pass test. The earlier two-notice run was attached to the MPS test.
Pytest attribution does not establish which operation created the shared pool.

## Upstream candidates — prototyped, not installed or submitted

Fresh source reads confirm that both operations remain in upstream `main`:

- [Quimb Vectorizer](https://github.com/jcmgray/quimb/blob/main/quimb/tensor/optimize.py):
  replace `array.shape = info.shape` with `array = array.reshape(info.shape)`.
  An isolated copy of the method passes **26** cases covering four real/complex
  dtypes, scalar/empty/multidimensional shapes, nested trees, contiguous and
  supported strided vector storage. Values, dtype, shape, and memory-sharing
  behavior match the installed method, with no shape warnings.
- [Cotengra Contractor](https://github.com/jcmgray/cotengra/blob/main/cotengra/contract.py):
  replace `if check_zero and float(factor) == 0.0:` with
  `if check_zero and factor == 0.0:`. **Six** eager NumPy/Torch/JAX real/complex
  cases preserve reconstructed contractions and exact-zero early exits.
  Torch gradients match direct einsum at `atol=rtol=1e-10`. The original
  method emits one gradient-scalar warning; the candidate emits none.
  This does not establish JIT, TensorFlow, or GPU compatibility.

Both patch files pass `git apply --check` against fresh upstream `main` source
reads. That confirms applicability, not upstream acceptance or complete
upstream test coverage.

Temporary patches and the isolated verification script are under `/tmp`:
`quimb-numpy-shape.patch`, `cotengra-scalar-check.patch`, and
`pepsy_upstream_prototypes.py`. The substitutions above retain the proposal
if those temporary artifacts expire. No upstream messages or pull requests
were posted, installed libraries modified, or internal implementations copied
into Pepsy. The repository's upstream policy forbids site-package edits and
vendoring; these fixes require adoption in the owning upstream projects.

Installed versions remain NumPy 2.5.2, Quimb `1.15.1.dev66+ge927f06e1`,
Cotengra `0.8.3.dev7+g1d7fd333f`, Autoray `0.11.1.dev3+g1b476b305`,
Symmray `0.4.1.dev8+gc45f91457`, Loky 3.6.0, and Torch 2.9.1.
The Quimb/Cotengra changelogs and documentation, Autoray repository, and
Symmray repository were rechecked. The Symmray array-documentation endpoint
again failed; installed source and the repository remain the available evidence.

## Worker investigation

A temporary full-suite harness enables multiprocessing INFO logs and records
Cotengra pool creation without changing search settings or timeout values.
The observed `auto-hq` pool has **14 workers and a 10-second idle timeout**.
Its first creation is under the BP corridor test. The completed trace records
**115 distinct workers over the whole suite**, each logging an idle timeout
and exit code zero. This is repeated pool replenishment, not 115 workers
running together. No memory-leak shutdown or nonzero exit was logged, and
the worker-stop warning did not recur. The trace establishes clean timeout
churn in this run; timing changed by instrumentation means it does not prove
the cause of the earlier asynchronous notice. No global pool policy, timeout,
or warning filter was changed to make it disappear.

Full diagnostic suite: **5,185 passed, 129 skipped, 680 warnings in 511.47s**.
The warning difference from the previous 685 consists of Quimb shape counts
varying from 605 to 601 and the absent worker notice. The unapplied upstream
proposals did not contribute to that difference. The remaining gradient-scalar
warning still originates in the installed Cotengra implementation.

The [contraction guide](../docs/api/tensors/contractions.md#controlling-search-workers)
now explains explicit serial search and diagnostic logging. Its explicit
serial-search recipe matches a direct matrix product in an isolated check.
Local Markdown links and `git diff --check` pass. These are documentation-only
repository changes; the full run was for warning investigation.

## GPU validation

The user confirmed that no NVIDIA GPU machine/runner is available. CUDA/CuPy
validation is therefore blocked by hardware. A fresh host-level run of the
backend/linalg and Metal norm-ledger selections reports **4 passed, 1 skipped
in 1.20s**. The actual Metal ledger test passes; native Metal QR remains
unsupported (`aten::linalg_qr.out`). CPU fallback was not enabled.

## Publication

The user-authorized follow-up is prepared for commit and push to `develop`.
The final session report records its revision and hosted CI result; this
entry records the local evidence available before that publication.
