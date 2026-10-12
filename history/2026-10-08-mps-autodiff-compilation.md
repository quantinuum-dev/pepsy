# Direct MPS compilation support — 2026-10-08

Implemented fixed-rank dense direct gate MPOs, traceable canonical initialization,
opt-in adaptive JAX QR registration, runtime JAX zero-norm validation and
Torch metadata/scalar compatibility fixes for Gaugy MPSEngine.

Read the [implementation and validation record](../docs/development/notes/2026-10-08-mps-autodiff-compilation.md)
for dependency versions, numerical limits and test scope. Existing native QR
defaults and native Symmray replay remain unchanged. Full Pepsy tests and GPU
validation were not run. Changes are uncommitted and unpublished; concurrent
PEPS work is unrelated.
