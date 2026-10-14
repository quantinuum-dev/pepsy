# Torch/JAX fermionic VMC maturity — 2026-10-13

**Measured:** both backends have native flat-Z2 batched fermionic amplitudes
and working VMC ingredients. Their compilation coverage differs; complete
production dynamics have not been certified.

The user-linked [Symmray Torch example](https://symmray.readthedocs.io/en/latest/examples/batch-gpu-fermionic-amplitudes-torch.html)
uses flat equal-sector Z2 PEPS, Quimb packing, HOTRG at cutoff zero, scaled
amplitudes and GPU `torch.vmap`. It also wraps parameters in a trainable module.
It does not demonstrate `torch.compile`, a VMC loop or numerical gradient
validation. ReadTheDocs was inaccessible to the web reader; a read-only HTTP
fetch retrieved the complete page. No installed libraries were edited.

## Current implementation

| Ingredient | Torch | JAX / NetKet |
| --- | --- | --- |
| Flat fermionic Z2 and native phases | Parameter packing/unpacking, vmap and CUDA | Packing/unpacking, occupation/Fock-phase mapping, jit/vmap and CUDA |
| Exact compiled amplitudes and derivatives | Export/vmap/compile, with explicit parameter inputs | JIT amplitudes and reverse derivatives |
| Boundary-MPS amplitudes | Compiled inference with fixed chi and cutoff zero | JIT with fixed chi and cutoff zero |
| Boundary derivatives | Stabilized original contractions; may fall back to serial batching | Stabilized reverse derivatives work under JIT; compressed forward-mode QGTs rejected |
| HOTRG / CTMRG | Eager/vmap; ordinary amplitude export API rejects these modes | JIT routes exist at cutoff zero |
| Sampling | Stateful chains, log acceptance, symmetry-aware proposals; optional compiled bookkeeping | Native NetKet compiled sampler kernels |
| Connections/local energies | Native graded operator lowering, batched targets, caches and optional compiled boundary reuse | Native NetKet operators/local-estimator kernels |
| SR / tVMC | Direct/minSR and real-coordinate integration; Python orchestration with tensor kernels | Native SR/TDVP and QGT kernels; Python orchestration |
| Sparse U1/U1U1/Z2Z2 | Eager block paths; separate opt-in exact graded projector | Non-JIT amplitude validation; full MCState JIT requires compatible flat storage |

Static **operator compilation** translates terms to connections; it does not
mean the entire Torch local-energy or optimization loop is a `torch.compile`
graph. Likewise NetKet's Python driver is not one fused JIT trajectory.
The APIs and their limits are documented in [VMC](../../api/vmc.md).

## Fresh validation

Installed: Torch 2.14.1+cu126, Symmray 0.4.1.dev18+g44fc5ae53,
Quimb 1.15.1.dev91+gb3bf45a50, NetKet 3.22.4, JAX 0.11.2.

- **19 CPU checks passed**, 92.28 s: Torch compiled amplitudes/proposals,
  boundary reuse, flat-Z2 SR/tVMC, transition plans, fermionic JAX scaling and
  an independent Torch/NetKet Fock-sector operator/local-energy comparison.
  The SR/tVMC flat-Z2 tests use bosonic Abelian tensors; they do not prove a
  complete fermionic CUDA trajectory.
- **Native fermionic CUDA Inductor**, 2x3, D=4, batch four, complex128:
  exact amplitudes match eager exactly; compiled-gradient directional error
  is 5.79e-10 relative. Updated parameters reach the same compiled graph.
- **Genuinely truncated native fermionic boundary CUDA Inductor**, chi=2:
  amplitude error is 4.77e-14 absolute; phase/log outputs agree with eager.
  Boundary-versus-exact amplitude discrepancy is 0.2344 relative, confirming
  this exercises actual compression, not an accidentally exact cap.
  Original stabilized gradients pass finite differences to 1.07e-8 relative;
  this probe selected serial derivative batching. Compiled gradients are
  intentionally rejected. Native flat fermionic tensor types are preserved.
- **Native fermionic JAX CUDA**, existing 2x3/D=4/chi=2 example:
  JIT amplitude error zero; JIT real-loss gradient directional error 2.48e-9
  relative. An earlier sandbox invocation fell back to CPU after CUDA device
  discovery failed; that CPU result is not counted as GPU evidence. The
  explicit CUDA invocation ran with GPU access and clean library resolution.

Torch emitted its complex-Inductor code-generation warning. These are
correctness checks, not speedup measurements. No new physical trajectory,
large-system mixing benchmark or complete fused VMC compilation was run.

## Remaining maturity gaps

1. Preserve and validate stabilized compressed VJPs through Torch compilation.
   Existing export can remove custom pullbacks, so compiled forward agreement
   does not suffice. Keep independent complex finite-difference checks.
2. Add an integrated native fermionic CUDA regression combining compiled
   sampling, local energies, SR/tVMC and parameter/cache updates. Existing
   component tests and bosonic flat-Z2 integration cover different scopes.
3. Normalize zero-amplitude log-output behavior. A diagnostic batch containing
   rows outside native flat-Z2 parity support returned raw amplitude zero,
   phase zero and NaN log magnitude in both eager and compiled boundary Torch.
   The sampler's nonzero mask rejects these rows, but the public log output
   remains inconsistent with a clean zero representation. This was observed,
   not fixed in this audit. Raw physical indices use parity-grouped storage;
   semantic `empty/down/up/double` labels must use the resolved encoding.
4. Broaden compiler/contraction and long-time accuracy evidence. Both paths
   retain full Jacobians and lack the paper's small-o storage. Native Symmray
   PEPO fitting/handoff and native symmetry analytic gauge projection remain
   separate gaps.

No implementation changes were made for this audit. Earlier working-tree
TDVP guards/diagnostics remain intact and uncommitted. See the
[handoff](../../../history/2026-10-13-torch-jax-fermionic-maturity.md).
