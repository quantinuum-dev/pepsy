# 2026-11-11 — FU CUDA synchronization and source ordering

On `develop` at `1d0a894`, retain the earlier uncommitted 20-iteration
default and reduce Torch CUDA host reads by batching rank/PSD diagnostics,
reusing local loss/fidelity scalars, and batching exact fixed-gate
classification. Restore the optional SU warm start on Quimb 1.15 using the
public single-index diagonal operation. No source ordering, compression or
stopping policy was changed. No commit or push was requested.

See [the CUDA and scheduling note](../docs/development/notes/2026-11-11-fu-cuda-scheduling.md)
for the 10×10 first-source-slice smart/column cache comparison, microprofile,
upstream audit, validation and limitations. The full suite stopped after 354
passes on the previously recorded fermionic exact-readout failure; a complete
T=6 GPU comparison remains unrun.
