# 2026-10-13 — Native MPS replay alignment

**Implemented, working tree:** `MpsOptimizer.supported_modes()` exposes native
dispatch names and aliases. DMRG sub-MPO windows and disposable guesses now use
the full support's minimum/maximum endpoints. The previous pair-only parsing
failed for a three-site support; exact target construction and FIT schedules
remain unchanged.

Gaugy now delegates U/V replay vocabulary and numerical run controls to Pepsy.
Its permutation adapter relabels contraction views, preserving represented
scale and physical-chain order without extra swaps or dense conversion.
Derivative and compilation capabilities remain independently restricted.

## Upstream audit

Installed: Pepsy 0.5.0, Quimb 1.15.1.dev91+gb3bf45a50,
Autoray 0.11.1.dev14+g014a3f69a, Cotengra 0.8.3.dev8+g8954240f2,
Symmray 0.4.1.dev18+g44fc5ae53. Inspected public optimizer construction,
logical readout, Quimb `from_TN`, index relabeling and swap signatures.

Reviewed [Quimb changes](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra documentation](https://cotengra.readthedocs.io/en/latest/) and
[changelog](https://cotengra.readthedocs.io/en/latest/changelog.html), and the
[Symmray repository](https://github.com/jcmgray/symmray).
Symmray's requested abelian-array documentation returned an internal error;
installed source and repository evidence were used instead. SDCR consumes the
backend's shared RNG; its randomness is not controlled by Pepsy's SRC seed.
Native/adapter tests reset and restore that RNG to compare identical draws.
No installed libraries or dependencies were modified.

Classification: **adopt** public mode discovery and exact logical-index views;
**compatibility shim** for older Pepsy uses an empty public constructor probe.

## Validation and limits

- 226 MPS compression/normalization/layout/metadata checks passed.
- 23 selected native Symmray/fermionic and sub-MPO checks passed.
- 55 public API/package-layout checks passed.
- Gaugy: 410 MPS/MPO replay checks passed; four actual-cell checks passed.
  A reduced notebook used U `dmrg3`, V `dmrg2` and retained fixed trace cost.
- Ruff over Pepsy `src tests` and whitespace checks passed.

An additional Gaugy selection had 167 passes and one flat-PEPS failure:
Quimb arbitrary-geometry compression rejected `direct`. It reproduces with the
pre-task evaluation configuration and is separate from MPS replay.
No full-repository suite or new GPU benchmark was run; no commits or pushes.
