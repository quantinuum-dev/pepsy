# 2026-10-06 — Native stabilizer measurement review

- Scope: review native shortcut correctness, generality, API routing and speed.
- Branch / baseline: `develop`, `d41ea9f`.
- Commit status: this review handoff is uncommitted; no production edits or push.

## Findings

Reviewed `_tableau_measurement.py`, measurement/basis-absorption dispatch,
sampler conditioning, focused tests and the October 5 benchmark/profile notes.
No numerical correctness defect found within the checked paths. The whole-state
certificate recognizes exact separated Pauli eigenstates; regional routing
requires separated mapped support and verifies identity tableau action on its
complement. It does not recognize arbitrary entangled stabilizer coefficient
states. Exact predicates reject tiny magic residues and trainable/block arrays.

Native collapse uses Pepsy RNG and Stim postselection, preserves normalized
physical states up to global phase, and avoids coefficient projector/localizer
compression. Eligibility is recomputed from live arrays rather than cached
across caller edits. Repeated scans, CPU conversion, simulator copies and norm/
bond diagnostics remain costs. Default/explicit-true `measure` tries native;
true falls back to localization, omitted flags to fixed-frame projection.
The sampler defaults to fixed frame. True enables native conditioning, but
sampling probabilities still go through `_pauli_expectation` first.

API caveat: a fifth stream field of `None` is coerced to `False`, unlike direct
`measure(..., disentangle=None)`. This is existing tested behavior. Omit the
field for automatic selection or explicitly provide `True` for absorption.

## Fresh validation

- Activated device-local genpy interpreter.
- `pytest -q -o addopts='' tests/test_stabilizer_tableau_measurement.py
  tests/test_stabilizer_sampler.py`: 101 passed in 3.41 seconds.
- Temporary deterministic four-qubit dense-reference probe: 50 random Clifford
  frames for full certificates and 50 for regional certificates with an
  entangled magic complement. 99 whole and 97 regional nonzero branches matched
  dense Born probabilities and normalized projected states to 1e-11; whole
  coefficient bonds remained one. The initial probe omitted a conjugated
  Pauli's sign when calling the unsigned public axis API; corrected the probe
  outcome sign before obtaining these results. No production fix was needed.
- Reran `examples/stabilizer_measurement_benchmark.py`: six qubits, 16 shots,
  one worker, complex128, direct mode, cutoff zero, chi 64, one warm-up and
  three timed repetitions per row. Same outcomes and final-state fidelity checks
  passed. Compilation excluded; resets retain normal routing in both variants.

| Workload | Backend | Auto median (s) | Fixed median (s) | Fixed / auto |
| --- | --- | ---: | ---: | ---: |
| Clifford | NumPy | 0.202896 | 0.809569 | 3.99 |
| Clifford | Torch CPU | 0.252333 | 1.295273 | 5.13 |
| Mixed | NumPy | 0.153576 | 0.238268 | 1.55 |
| Mixed | Torch CPU | 0.194269 | 0.334248 | 1.72 |

Clifford auto: all 576 collapse events native, coefficient bond one; fixed
comparison ends at bond eight. Mixed auto: 96 regional and 96 MPS events,
coefficient bond two. These are local small-workload timings, not general
scaling, GPU timing, a fresh Helix benchmark or a plain-Stim comparison.
No full suite rerun; no implementation changes requiring Ruff.

## Scoped performance candidates, not implemented

Reuse eligibility/probabilities only within a state-unchanged operation;
avoid duplicate failed certificate checks; use certified Stim probabilities
in absorbed bitstring sampling. Broader recognition and persistent caching
need separate correctness and mutation-invalidation design.

## Follow-up decision

The user initially requested fixes and compiled Clifford routing. Preliminary
implementation edits were started but not validated. After discussing the
measured performance and added state-tracking complexity, the user explicitly
selected keeping the reviewed implementation. Reverted all edits from that
attempt in the three production files; `git diff --stat` is empty and
`git diff --check` passes. Only this untracked review handoff remains. No
implementation, test, API or default changes were retained; no commit or push.
