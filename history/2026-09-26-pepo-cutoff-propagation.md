# 2026-09-26 — Propagate PEPO construction and contraction cutoff policies

- Scope: shared API changes required to finish Gaugy's local PEPO audit fixes.
- Branch / baseline: `develop` / `966831c`.
- Commit status: included with this implementation commit; not published.

## Changes and rationale

- `build_pepo_from_gates` accepts `cutoff_mode` and forwards it to per-gate
  splits and safety compression; the latter also respects the supplied cutoff.
- `contract_hypercompressed_tn` accepts an optional cutoff mode, preserving
  its previous Quimb default when omitted.
- Five regressions cover numerical absolute/relative rank differences,
  oversized-input safety compression, and hyper-contraction forwarding.
- Reused the unchanged upstream audit and inspected installed builder,
  gauge-insertion, boundary, and compressed-contraction signatures/source.
  Classification: adopt existing public Quimb compression options; no shim.

## Validation

- New cutoff tests and existing gate suite: 115 passed, 1 skipped.
- Ruff across `src tests`: passed.
- Strict documentation (`sphinx -W --keep-going`): passed.
- Full suite: 4688 passed, 121 skipped, 731 warnings in 330.18 seconds.
- Paired Gaugy full suite: 246 passed, 8 warnings. The real Torch/NLopt sweep
  also improved its cost and produced finite signed diagnostics.

## Integration

- Paired Gaugy changes preserve the original mean-absolute cost and expose
  signed diagnostics, retain helper SU weights, and handle sweep edge cases.
- No unrelated work is included; publication requires integrating existing
  unpublished work and remote changes separately.
