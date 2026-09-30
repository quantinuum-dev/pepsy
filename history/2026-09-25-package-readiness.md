# 2026-09-25 — Package readiness validation

- Scope: verify the cleaned-up package, installation, consistency, and weight.
- Branch / baseline: `develop` / `46eb601`, with the preceding cleanup edits
  preserved in the working tree.
- Commit status: this record accompanies the cleanup commit authorized for
  push to `develop`. No tag update or package publication.

## Result

The package builds and passes the checks below. These are local validation
results; they do not claim that every optional platform was exercised or
that hosted CI has completed for this commit.

| Check | Result |
| --- | --- |
| Source API, layout, imports, and legacy hooks | 78 passed |
| Installed wheel on minimum core dependencies | 71 API, layout, constructor, and compatibility tests passed |
| Installed numerical examples | MPS and tree Bell-state replay matched the exact vector; README PEPS contraction passed |
| Dependency-free wheel discovery | Root and all public domain namespaces imported with site packages disabled |
| MPI integration | All 25 cases passed on every rank with both two- and three-rank execution |
| Distribution build | Source archive and wheel built; wheel built from the source archive |
| Metadata | Both artifacts passed Twine validation |
| Packaged source | All 187 Python modules matched working-tree source byte for byte; typing marker and license present |
| Package contents | Wheel contains no tests, docs, history, bytecode, or build directories; neither artifact contains device-local policy |
| Types, lint, and guidance | Focused mypy checks, Ruff, and skill catalog passed |
| Documentation | Strict Sphinx build passed |

The earlier full local run remains applicable to the implementation cleanup:
**4,607 passed, 121 skipped**. Its scope and final installation-message checks
are recorded in the [cleanup handoff](2026-09-25-final-cleanup-polish.md).
This readiness pass adds installed-artifact, minimum-version, and real MPI
coverage rather than repeating the same full source run.

## Weight and dependencies

- Wheel: **2,060,648 bytes (1.97 MiB)**; uncompressed archive contents:
  **9,201,265 bytes (8.78 MiB)**. Installed Pepsy files including generated
  bytecode occupy about **18.17 MiB** in the temporary installation.
- Five direct runtime requirements: NumPy, Quimb, Cotengra, Autoray, and tqdm.
  Optional Torch, JAX, NetKet, Symmray, Stim, and Nevergrad were absent from
  the isolated core numerical process.
- The complete minimum core dependency closure contains 12 distributions,
  including Pepsy, and occupies **337.54 MiB** here including bytecode.
  NumPy, SciPy, Numba, and their compiled dependencies contribute separately
  from Pepsy. Test tools are excluded from this measurement.
- Five fresh-process samples per profile measured root import at **20.77 ms**
  median and all core namespace imports at **22.06 ms**. Sampling, optimizer,
  and experimental discovery were 21.85, 22.32, and 22.17 ms. None loaded the
  optional roots monitored by the profiler. Timings are machine-specific.

Minimum versions exercised together: NumPy 1.26.0, Quimb 1.15.0, Cotengra
0.8.0, Autoray 0.9.0, and tqdm 4.65.0. Their transitive requirements were
checked against the isolated installation. Tests used the designated Python
interpreter with site packages disabled and a temporary install directory;
the shared environment was not modified.

## Consistency clarification

An initial probe replayed the tree circuit twice. `TreeOptimizer` runs on
construction by default; `MpsOptimizer` waits for `run()`. The corrected
comparison explicitly used `run=False` for the tree, then ran each stream
once. Both matched the dense Bell-state reference. Added this existing
lifecycle distinction to the [tree replay guide](../docs/api/optimizers/tree_replay.md#when-replay-starts);
no runtime default or numerical implementation was changed.

## Built artifacts

Temporary output: `/tmp/pepsy-ready-saQ6EF/dist/`. These files are a local
working-tree build with version 0.5.0, not a replacement for the existing
`v0.5.0` release tag.

| File | SHA-256 |
| --- | --- |
| `pepsy-0.5.0-py3-none-any.whl` | `5e10d7311ee8496daf672f409e47a7bc7825558d21e71269ac0b9cd5188686f7` |
| `pepsy-0.5.0.tar.gz` | `332db35323cbf6021dddeb9b7a2889e3a23377c944693f6e2c191b61774c1782` |

Skipped optional cases and hardware backends remain outside the validated
scope. GitHub-only distribution and the existing release tag are preserved.
