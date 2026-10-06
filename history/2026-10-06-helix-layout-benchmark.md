# 2026-10-06 — Helix layouts at chi 256 without native collapse

- Scope: user requested several MPS layouts, direct mode, chi 256, and
  disentangling True/False with the native shortcut disabled.
- Branch / baseline: `develop`, `d41ea9f`; existing default-change working-tree
  edits were preserved. No implementation or notebook edits in this task.
- Commit status: this handoff is uncommitted; nothing staged or pushed.

## Method

Same archived noiseless 52-qubit Helix circuit as the
[previous benchmark](2026-10-06-helix-no-shortcut-benchmark.md): 3,818 translated
entries, 550 visible measurements, and 530 resets. Torch CPU complex128,
`chi=256`, `mode="direct"`, cutoff 1e-12, seed 42. One shot per case, fresh
processes. Replay timing excludes imports, translation, conversion, compilation,
and layout installation. Single observations, not median throughput estimates.

Installed each explicit coefficient order using `sim.apply_layout(order,
layout_report=False)` while the initial coefficient MPS was product. Physical
qubit labels and the gate stream remain the same. Layouts:

- Input: `range(52)`.
- Even/odd: `list(range(0,52,2)) + list(range(1,52,2))`.
- Interaction RCM: SciPy reverse Cuthill–McKee on the symmetric physical
  two-qubit interaction adjacency of the flattened archived circuit. This is
  a physical-graph heuristic, not Pepsy's coefficient-frame layout finder.
  Order: `[49,33,43,9,48,32,40,41,25,24,8,13,5,4,51,35,50,34,27,26,14,42,6,7,
  10,29,19,16,18,17,31,30,11,45,47,46,12,23,22,2,3,15,37,39,38,28,44,21,20,
  36,1,0]`.

In each process, native `measure` and `probabilities` returned `None`, and
native fallback diagnostics returned `disabled_for_benchmark`. This disables
native collapse and unused eligibility scans, including resets. Visible
measurement entries explicitly select True/False; resets use their normal
basis-updating MPS path in both cases. No installed/package source was patched.
The 600-second and 2-GiB pre-entry guards did not fire.

## Results

| Layout | Disentangle | Replay seconds | Peak after entries | Peak internal updates | Final bond | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Input | False | 54.852929 | 128 | 128 | 128 | Complete |
| Input | True | 110.614772 | 32 | 64 | 32 | Complete |
| Even/odd | False | 0.242727 before error | 8 | 8 | — | Failed at entry 75 |
| Even/odd | True | 94.157187 | 32 | 32 | 16 | Complete |
| Interaction RCM | False | 63.581516 | 128 | 128 | 128 | Complete |
| Interaction RCM | True | 363.240894 before error | 128 | 256 | — | Failed at entry 3809 |

Internal peaks observe completed `_evolve_p` updates, including localizing
Cliffords; they do not measure temporary MPO/SVD workspace dimensions. Error
rows are partial replays, not seconds per completed shot. Their last stored
bond is 8 and 128, respectively, but neither has a valid final-shot bond.

Input True is about twice as slow as Input False despite smaller bonds.
Even/odd True is about 15% faster than Input True and halves its internal
peak. Interaction RCM False is about 16% slower than Input False, with the
same peak. No completed case reaches the chi cap; RCM True reaches 256
internally before failing. A physical interaction order can be a poor order
for basis-updating coefficient operations.

## Checks and failures

All four completed runs finish with norm one within 1e-10, all 550 visible
outcomes, and exactly 1,080 MPS collapses with zero native/regional collapses.
Their visible outcome arrays are identical for seed 42. Recorded cumulative
compression-infidelity proxies are 3.06e-12, 5.73e-12, 6.98e-12, and 6.62e-12
for Input False, Input True, Even/odd True, and RCM False. These are loss
proxies, not full-state fidelity assertions; no dense 52-qubit reconstruction
or logical/statistical accuracy test was performed.

- Even/odd False reproducibly fails measuring Z on physical qubit 26, after
  74 completed entries and six visible outcomes. Frame terms are Z on logical
  coefficient sites 12, 13, 14, 15, and 26. The diagnostic replay reports
  outcome +1 with Born probability 1, finite tensors before `_apply_projector`,
  and nonfinite tensors after it. `_renorm_p_at` raises for centre norm NaN.
  This is a projector/compression-path numerical failure at bond 8, not a
  cap-induced loss or an impossible sampled branch. Ownership/root cause
  beyond this call path is unverified; no numerical fix was attempted here.
- RCM True fails after 3,808 completed entries and 540 visible outcomes.
  Torch `linalg.svd` reports convergence failure from an ill-conditioned
  matrix or repeated singular values (error code 1). Terminal readout grows
  the public bond to 64 at outcome 535 and 128 at 536, then internal updates
  reach 256. This failure was observed once, not rerun for reproducibility.
- A brief failure probe overlapped the exploratory Even/odd True run
  (95.621 seconds). The table uses an isolated fresh repeat (94.157 seconds)
  after all other cases/probes; its bonds and visible outcomes match.

Environment: arm64 Mac, Python 3.12.14, Torch 2.9.1 (10 intra-op / 14 inter-op
threads), Quimb 1.15.1.dev66+ge927f06e1, Stim 1.16.0. Circuit SHA256:
`18c6f2bbc9ad59bbf2e7d805c07b21e3c8be3ef974ada74ab26f25c850067149`.

## Artifacts / follow-up

Temporary driver `/tmp/helix_layout_benchmark.py`; diagnostic driver
`/tmp/helix_layout_benchmark_debug.py`. Case JSONs and logs use the prefix
`/tmp/helix-layout-`; compact comparison is `/tmp/helix_layout_results.json`
and `/tmp/helix_layout_results.csv`. Scripts were run under activated genpy.
Package tests were not rerun because this task changes no package code.

The requested experiment is complete, including the failed cases. A separate
numerical investigation should isolate the deterministic fixed-frame projector
failure and the RCM localizer SVD failure before treating these layouts as
reliable no-shortcut configurations. The native shortcut remains unchanged.
