# Repository organization audit — 2026-09-26

This assessment examined the largest 30 Python modules, their top-level and
class responsibilities, import dependencies, existing module maps, and focused
tests. It concerns source organization and the complex sampling defect found
during validation; it does not claim that every subsystem received a
functional audit.

## Completed batch

Four files combined independently useful responsibilities with mature test
coverage. They were split together, preserving signatures, implementations,
subclass hooks, public imports, and historical serialized class paths.
The four Born-weight contractions corrected below are the only intentional
numerical changes in this batch.

| Previous owner | Before | After | Extracted responsibilities |
| --- | ---: | ---: | --- |
| `optimizers/mps/optimizer.py` | 10,508 | 8,978 | Measurement/reset/cap/conditional execution; local normalization and norm diagnostics |
| `optimizers/stabilizer_tn/mps_stab_optimizer.py` | 9,193 | 7,705 | Stream advice, coefficient-frame layout planning, shared parsing/localizers |
| `bp/series.py` | 8,420 | 7,023 | Loop terms, bounded enumeration, corridor geometry |
| `sampling/samplers.py` | 5,876 | 116 | Separate MPS, vector, direct-PEPS, and BP engines; shared records and conversion helpers |

Line counts describe this snapshot, not performance improvements. Public
method documentation stays on optimizer classes; helper functions receive the
live instance or class and retain calls through its hooks. No mixin hierarchy
or duplicate state container was introduced. The sampler facade preserves old
names; public exports and internal callers use the owning modules directly.
BP and stabilizer package facades also resolve public names lazily. An import
boundary test exposed that their previous eager exports loaded the entire
contraction or simulator implementation even when only geometry or advice
helpers were requested. Deprecated stabilizer names retain their warnings.

The preceding symmetric MPO extraction is retained. Its evidence is recorded
separately in the [Symmray notes](symmray.md).

## Other large modules reviewed

Size alone does not establish a useful split. The remaining review candidates
below need a design boundary and tests for that boundary before extraction;
they are not known defects or approved numerical changes.

| Area | Finding and useful next review boundary |
| --- | --- |
| MPS and tree replay | FIT target/guess/rollback and mode execution share live canonical metadata; review an explicit execution interface before moving more methods |
| Stabilizer replay | Measurement, cooling, and immediate/deferred injection coordinate both tableau and coefficient state; preserve their shared frame and ancilla-lifetime contracts |
| BP contractions | Open scalar/rho routes share budgets, projector caches, and native cyclic routing; a diagnostics layer is plausible but must not cache numerical values as topology |
| `mpo_semantic.py` | The large `FirstDegreeMPO` implements one algebra; dense/native materialization is a stronger prospective boundary than arbitrary method groups |
| `sym_dmrg.py` | Solver and effective-Hamiltonian machinery share charge-block environments; separate only with native reconstruction and sector tests |
| `noise.py` | Event/record definitions, Stim conversion, and trajectory branching could be separated after auditing constructor and serialization dependencies |
| `vmc/netket.py` | Packing, monitoring, and driver orchestration have potential boundaries; JAX tracing and optional-import behavior need dedicated checks |
| Tree layout/operators and PEPO builders | Geometry, structural plans, and numerical materialization are already partially separate; preserve those existing interfaces when evaluating further splits |
| FIT, boundary, and remaining optimizers | Their stateful kernels are cohesive; no extraction was selected solely to meet a line-count threshold |

Use the [optimizer](../modules/optimizers.md), [sampling](../modules/sampling.md),
and [BP](../modules/bp.md) maps to find the new owners. The session handoff
records the completed validation and working-tree status.

## Complex native MPS sampling correction

An installed-wheel smoke check compared native sampling probabilities with
independent dense Born weights on a three-site complex MPS. The mismatch
also reproduced with the preceding wheel, establishing that it predates the
module extraction. Real/product-state coverage had not exposed it.

Cached right environments have bra rows and ket columns. The old contraction
`conj(amps) * (amps @ right_env)` reversed that orientation for complex
off-diagonal entries. Sampling and probability evaluation now both use
`(conj(amps) @ right_env) * amps`, summed over the bond index. This corrects
the Torch path and the shared NumPy/CuPy path without changing the cache or
normalization policy. Complex sample distributions can change.

New NumPy/Torch regressions enumerate every configuration of complex qubit
and three-level MPS states, compare against normalized dense probabilities,
check returned sample weights, seeded reproducibility, and source preservation.
The CuPy expression is shared with NumPy, but hardware coverage remains
dependent on an available GPU/CuPy runtime.

## Upstream compatibility check

Installed versions were unchanged: Quimb `1.15.1.dev66+ge927f06e1`, Autoray
`0.11.1.dev3+g1b476b305`, Cotengra `0.8.3.dev7+g1d7fd333f`, Symmray
`0.4.1.dev8+gc45f91457`, and Stim `1.16.0`.

Reviewed the official [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
and [Symmray repository](https://github.com/jcmgray/symmray). The Symmray array
documentation was unavailable; installed source and signatures were used.
Checked MPS canonicalization, canonical expectation and norm, D2BP, and
`sample_d2bp` signatures in the executing environment.

- **Compatibility shim:** historical imports and pickle globals resolve to
  the new owners; optimizer wrappers and aliases preserve method binding.
- **Defer:** dependency upgrades and upstream cutoff, expectation, or sampling
  changes. This batch retains the existing options, native grading, seed
  handling, normalization policy, and contraction routes.
- **Adopt:** the corrected complex Born-weight contraction above, validated
  against independent dense amplitudes rather than an upstream replacement.
