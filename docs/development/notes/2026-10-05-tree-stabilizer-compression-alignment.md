# Tree stabilizer compression alignment audit

Reviewed on 5 October 2026 against Pepsy `develop`, baseline `6ab177b`.
The user requested a final review, commit and push to `origin/develop`.
The containing Git commit identifies the finalized implementation.

## Findings and implementation

TreeOptimizer's direct/DM update losslessly routes the complete operator with
QR before canonical edge compression. Its DMRG path keeps the exact
operator/state layers separate from a disposable compressed guess and passes
the configured iteration budget, stopping policy and block schedule to
TreeFIT. These numerical kernels required no changes in this review.

StabilizerTreeSimulator already used the same direct/DM coefficient engine,
but rejected DMRG and other explicit algorithm selectors, omitted their
constructor controls, and reconstructed its engine from incomplete lists of
settings after frame-layout changes and physical caps.

- **Adopt:** use TreeOptimizer's mode normalizer and expose its FIT,
  randomized/oversampling compression and unitary stabilization controls.
  Keep the historical TreeStab default `tree_mpo_direct` and numeric cutoff.
- **Adopt:** reuse the live TreeOptimizer configuration snapshot on rebuilds,
  explicitly retaining the named DMRG schedule. Rebind the norm-events view
  after frame-layout replacement. Copies and shot templates retain the
  existing ordinary-tree copy implementation.
- **Adopt:** delegate `get_fit_diagnostics()` to the coefficient engine.
  Generic `dmrg`/`fit` is one-node; `dmrg2` and `dmrg3` preserve their ordinary
  schedules, including the default `(3, 3, 2, 1)` transition for DMRG3.
- **Compatibility shim:** cold concurrent Stim matrix classification stalled
  independently of FIT in a fresh two-thread probe. Tracebacks stopped inside
  `stim.Tableau.from_unitary_matrix` and Python import initialization. Serial
  first-use initialization before dispatch made public threaded TreeStab shot
  replay complete. Initialize the identity classifier only for multi-worker
  thread replay; this touches neither the tableau nor the coefficient state.
  Do not infer that every direct use of the shared shot runner is protected.
- **Adopt:** coefficient localizers delegate to `TreeOptimizer.apply_gate`,
  sharing compact SubTreeMPO construction, limits, caching and update ledgers.
  Remove the separate legacy `mpo` two-factor route; that spelling now follows
  ordinary TreeOptimizer gate behavior. Explicit full/compact tree events use
  the canonical `apply_sub_mpotree` dispatcher.
- **Adopt:** physical caps reconstruct an exact reduced TTN, restore its
  backend, then apply a full bond-one identity through the selected engine.
  The previous independent absolute SVD cutoff could erase small nonzero
  states even with a relative cutoff. Truncation now uses the selected engine,
  including DMRG, without a dense identity or replacement operator.
  A zero cap already produces an exact rank-one TTN and skips unnecessary
  compression: an extra probe found the DM splitter divides by zero there.
- **Adopt:** forward intermediate bond limits, subtree workers, history and
  profiling/bond diagnostics constructor controls unchanged.
- **Retain:** stabilizer frame mapping, exact cooling and basis-updating
  measurement are representation-specific. Clifford disentangling scores
  copied states and applies untruncated gauge moves through the existing
  TreeOptimizer two-qubit kernel. An approximate selected compressor cannot
  replace that gauge move while preserving `C|p>`; no new splitter is used.
  Physical caps still require a guarded dense physical-state reconstruction.

## Installed dependency and upstream audit

The activated checkout environment contains Quimb
`1.15.1.dev66+ge927f06e1`, Autoray `0.11.1.dev3+g1b476b305`, Cotengra
`0.8.3.dev7+g1d7fd333f`, Cotengrust `0.2.1`, Symmray
`0.4.1.dev8+gc45f91457`, and Stim `1.16.0`.

Inspected installed `Tensor.split`, `TensorNetwork.compress_between`,
TreeOptimizer's compact operator application and both constructor signatures.
The split API accepts explicit cutoff modes and the edge-compression wrapper
accepts `canonize_distance`; the wrapper continues to use those existing
contracts. No installed dependency was changed or vendored.

Checked the [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray repository](https://github.com/jcmgray/symmray).
The Symmray `abelian_arrays.html` page could not be retrieved; the installed
implementation and official repository remained available. Upstream split
defaults have changed, so this wrapper continues forwarding the existing
explicit tree cutoff policy rather than assuming an upstream default.

## Validation

Resumed audit: the final combined tree/TreeStab/FIT/compression/trajectory/
replay/public-API selection passed **773 tests**, with JAX x64 enabled.
The final TreeStab/parity selection separately passed **217 tests**, including
the zero-cap guard and backend-preserving cap compression. New operator cases compare actual
coefficient states for nonunitary dense gates, compact SubTreeMPO and full
TreeMPO payloads across fifteen modes. Small scaled cap tests distinguish
relative from the old absolute cutoff; zero caps remain exact rank-one trees.
Torch and JAX tests also compare capped states with independent dense results.
The parity module now contains **133 regression cases**. Ruff and whitespace
checks passed after the final changes.

Publication review repeated the **773-test** selection successfully, then
rebased onto updated `origin/develop` (`bec773a`). The numerical TreeStab
changes were unchanged by that rebase; both changelog additions were retained.
The integrated selection, including upstream zero-weight tests, passed
**788 tests** with no skips. Ruff and both skill/catalog validators passed.

- Earlier ordinary tree, TreeStab, FIT priorities/messages, successive compression,
  zipup, public API and package layout selection: **481 passed**.
- Earlier TreeStab parity module, complete trajectory-noise module and tree
  replay-composition module, with JAX x64 enabled: **205 passed**. This includes
  **46 new regression cases** covering path and branched supports across
  fifteen algorithms, independent dense references, DM local splits with SRC
  guesses, failure/retry semantics, rebuilds, independent/coalesced shots,
  cold threaded startup, Torch and JAX coefficient backends.
- `python -m ruff check src tests` and `git diff --check`: passed.

The first parity test mistakenly demanded a global cap from an input whose
untouched exterior bonds already exceeded that cap. Corrected the fixture to
start within the cap; the tests still exercise truncating active updates.
The earlier JAX-default run skipped x64; the explicit x64 rerun passed.
Cold threaded probes had bounded timeouts; the initial stalled test process
was terminated and replaced by the passing fresh-process regression.

No full repository suite, accelerator execution, multi-rank MPI test or
performance claim is implied by these focused checks. Native graded kernels
were not changed; the ordinary tree selection includes existing native cases,
while the stabilizer coefficient representation remains dense.
