# 2026-10-05 — Stabilizer MPS compression parity review

- Scope: compare MpsStabOptimizer/StabilizerMpsSimulator with MpsOptimizer
  and align coefficient-MPS updates where their contracts agree.
- Branch / baseline: develop / d7d509b.
- Commit status: working-tree changes only; no commit or push.

## Findings and change

Both engines already share the Quimb compression registry, seeded dispatch,
interior-sub-MPO workaround, and Pepsy FIT kernel. The stabilizer simulator
owns a tableau plus a coefficient MPS, not a nested MpsOptimizer. Physical
Clifford gates only change the tableau; other physical gates are mapped into
the coefficient frame before compression. Norm/measurement diagnostics remain
specific to that representation.

Found and removed the outdated implicit adjacent-pair dmrg2 shortcut. A
baseline-method probe requested five fixed sweeps and performed one; the
updated simulator performs five, matching ordinary MpsOptimizer. Explicit
fit_single_pair_fast_path=True still selects one update. API guide and
changelog describe the correction.

Remaining differences: STN dmrg/fit still defaults to two-site growth, retains
legacy dmrg1, has no fit_single_pair_n_iter cap, and does not expose ordinary
MPS compression_opts. These were not silently changed or added. For one-site
STN FIT, specify fit_block_size=1. Sharing more stateless preparation/schedule
helpers would require a separate scoped refactor; tableau, coefficient-frame
conversion, canonical metadata and diagnostic ownership must stay on STN.

## Fresh validation

Local genpy Python 3.12; Quimb 1.15.1.dev66+ge927f06e1, Autoray
0.11.1.dev3+g1b476b305, Cotengra 0.8.3.dev7+g1d7fd333f, Symmray
0.4.1.dev8+gc45f91457, Stim 1.16.0. Inspected installed FIT.run_gate and
MatrixProductState.gate_with_submpo_ signatures. No dependency changes.

- New parity tests compare actual coefficient-state directions for direct,
  zipup, SRC, and adjacent dmrg2/dmrg3, with default/explicit fast paths and
  complex64/complex128, under chi truncation.
- test_stabilizer_mps_compression_parity.py, test_mps_fit_window_budget.py,
  test_stabilizer_tn.py: 426 passed, 1 skipped, seven compatibility warnings.
  The skip is the optional array-backend coverage; no GPU parity is claimed.
- python -m ruff check src tests: passed.
- git diff --check: passed. No full repository suite run.

Checked official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
Symmray's abelian_arrays documentation URL returned an internal error;
used the official repository and installed implementation as fallback.
Disposition: adopt ordinary MPS's opt-in adjacent-pair shortcut policy;
defer other API/default changes. No upstream numerical algorithm was altered.

## Follow-through after explicit request to fix the remaining gaps

The user authorized implementing the remaining controls/default alignment.
STN dmrg/fit now fixes one-site FIT and rejects multi-site overrides, matching
ordinary MpsOptimizer. dmrg2/dmrg3 remain the explicit growth modes; legacy
STN dmrg1 keeps its existing latch for compatibility.

Added fit_single_pair_n_iter with the ordinary None/inherit and positive/min
semantics for both one-site and two-site adjacent windows. Longer windows keep
the general budget. The control propagates through shot run_kwargs and is
restored after replay.

Added dense native compression_opts using Pepsy's existing capability-checking
helper and interior-sub-MPO adapter. Caller options are copied, unsupported
FIT/native graded/dense branch-sum paths reject explicit settings, and per-run
controls restore after compressor failure. Invalid later configuration cannot
install the new controls. Direct compression optimizer forwarding and SDCR
cutoff compatibility now follow the ordinary MPS policy.

Updated the public API guide, changelog, and stabilizer skill in the same
change. Existing catalog and upload paths are unchanged.

Validation after this extension:

- Initial stabilizer/parity/MPS budget selection: 459 passed, one CuPy skip.
- Final expanded parity file: 48 passed, including shot option forwarding,
  long-window budget isolation, failure restoration, and invalid controls.
- Public API, package layout and stabilizer backend selection: 80 passed,
  22 skips (CUDA unavailable or CuPy missing). CPU Torch/JAX checks ran.
- The selections cover 543 distinct passing tests and 23 skipped cases.
- Ruff, individual skill quick_validate, skill catalog and diff whitespace
  checks passed. No full repository suite, GPU validation, commit or push.

The first one-site comparison used different random SRC starting states and
showed finite-sweep directional differences. Kept the tight comparison
tolerance and selected the same deterministic guess-direct policy to isolate
target/schedule parity. Randomized seed streams and tableau-specific
diagnostics remain engine-specific; matching method names alone does not
promise identical finite-sweep stochastic results.
