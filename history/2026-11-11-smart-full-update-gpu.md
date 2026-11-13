# 2026-11-11 — smart full-update default and GPU boundary probes

Full update now defaults to smart gate ordering, with explicit ordering options
unchanged. Generic Pepsy callers must submit separate runs for any source or
depth barriers. The Eli runner retains source-slice barriers and now uses smart
by default. Its full-update pair ALS cap remains 20 by default.

On a Torch complex128 RTX A5000, a 2,500-action 10×10 D20 χ4 prefix with
boundary cap 32 took 48.40 s with direct boundaries and explicit gesvd,
94.87 s with DMRG and gesvd, and 65.48 s with direct boundaries and automatic
SVD. It did not saturate the boundary cap in sampled cuts. A synthetic
full-rank χ4 probe did reach rank 32: direct took 3.65/3.68 s and DMRG
11.66/11.75 s for three updates. These are bounded tests, not full-time
costs or final trace fidelities. Fusion gave no measurable prefix speedup;
an isolated Torch ALS prototype gave only a small one-trial improvement, so
neither was promoted as a default.

The example repository's `docs/2026-11-11-fu-chi4-b32-gpu-prefix.md`
records the configurations, saved artifacts, numerical comparison and limits.
See the separate CUDA scheduling note for implementation details and the
earlier source-slice scheduling comparison.

The smart-order and sweep-default tests reported 50 passes and two failures
for single-site `dagger`/`transpose` options unsupported by the installed
Quimb. Both failures reproduce unchanged on an isolated HEAD checkout. The
separately recorded full-suite fermionic-readout failure also remains outside
this change.
