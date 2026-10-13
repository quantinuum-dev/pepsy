# 2026-10-13 — PEPO startup to native NetKet PEPS TDVP

- Scope: user requested a few careful implementation and simulation steps;
  later requested a fast wrap-up. No further trajectory studies launched.
- Branch / baseline: `develop` / `43dfc4a`.
- Commit status: working-tree edits only; nothing staged, committed or published.

## Implemented and measured

- Added `NetKetTDVPDriver`, `make_netket_tdvp_driver` and setup convenience
  method; native NetKet owns sampling, forces, QGT, solve and integration.
- Real/imaginary Flax parameter leaves support nonholomorphic contraction;
  reconstruction restores updated complex tensors. Existing complex defaults
  remain unchanged. Generic complex nonholomorphic TDVP is rejected.
- Accepted-state synchronization and final-stop handling fix intermediate
  stage handoff and floating-point extra-step errors found by exact checks.
- Added [bounded TFIM example](../examples/tfim_netket_tdvp.py), API docs,
  module map, changelog and [numerical tests](../tests/test_netket_tdvp.py).
- [Retained evidence](../docs/development/notes/2026-10-13-netket-peps-tdvp.md):
  NetKet 3.22.4/JAX 0.8.3 CPU, 4×4/D=3 fixed amplitude chi nine, two
  midpoint steps after PEPO startup. Final infidelities `3.1631e-9` with
  128 samples and `1.7501e-11` with 256 samples. All bounded trajectories
  completed. Shared environment untouched; dependencies used `/tmp` overlays.

## Validation

- Initial combined selection: 110 passed, one failed, 80.55 s. The failure
  was the new flat-Z2 gradient probe requesting an unrepresentable cap one;
  it was corrected to cap two, retaining actual compression and both sectors.
- Earlier 10-test dense TDVP selection passed with native NetKet 3.21.0.
- Final expanded TDVP selection: **12 passed**, 85.16 s, native NetKet
  3.22.4/JAX 0.8.3. Covers dense/native-flat-Z2 exact and actually truncated
  coordinate finite differences, full-sum D=3 ED dynamics in both charts,
  sampled boundary dynamics, endpoint timing and parameter validation.
- Ruff `src tests` and example, and `git diff --check` passed.
- No full-suite claim. Unrelated concurrent MPS edits preserved.

## Remaining work

Paper-specific gauge/local derivative storage and sustained convergence
studies remain. Native Z2 amplitudes/coordinate derivatives are checked,
but no native Z2 NetKet dynamics or dense-to-Z2 startup handoff was added.
