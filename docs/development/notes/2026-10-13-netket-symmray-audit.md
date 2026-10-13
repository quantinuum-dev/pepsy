# 2026-10-13 — NetKet/Symmray audit before larger PEPS runs

The implemented changes fix sampler controls and derivative selection before
scaling lattice size, D and χ. This audit does not establish large-system
convergence or the paper's gauge-projected/minSR tVMC implementation.

| Upstream finding | Decision |
| --- | --- |
| Latest released NetKet is 3.22.4; main is `5e4511b7883d89d83b0aba534b37c3c672a15a42`, 19 commits ahead | **Prototype** isolated main with JAX 0.11.1; retain released dependency bound |
| Main fixes JAX 0.11 annotation imports and caps JAX below 0.11.2 for Flax compatibility | **Prototype** isolated main; user independently supplied patched release/JAX 0.11.2/Flax 0.12.10 |
| Main fixes zero-default call-time burn-in; 3.22.4 checks the state default | **Compatibility shim** temporarily apply/restore requested burn-in through public MCState properties |
| VMC_SR automatically selects forward-mode NTK; compressed PEPS have custom VJPs | **Adopt** reverse Jacobians and sample-space SR with `on_the_fly=False` |
| VMC_SR real mode discards imaginary derivatives | **Adopt** native output detection; explicit real mode remains available |
| QGTJacobianDense stores the Jacobian but applies the metric lazily | **Adopt** opt-in public JAX CG; retain native dense default and guard allocation |
| Latest released Symmray is 0.4.0; installed main is `44fc5ae53b91b7c4a9a4b251aeb12bb48fc0c741` | Already current; no installed-library changes |
| New packed Symmray storage uses NumPy/Numba | **Defer** as a JAX substitute; retain native flat-Z2 JIT path |
| Quimb/Symmray bond balancing still needs unsupported two-array einsum | **Defer** native symmetry balancing; do not densify tensors to hide this |

Sources: [NetKet release-to-main comparison](https://github.com/netket/netket/compare/v3.22.4...5e4511b7883d89d83b0aba534b37c3c672a15a42),
[NetKet VMC_SR](https://netket.readthedocs.io/en/stable/api/_generated/driver/netket.driver.VMC_SR.html),
[Symmray changelog](https://symmray.readthedocs.io/en/latest/changelog.html),
[Symmray JAX batching example](https://symmray.readthedocs.io/en/latest/examples/batch-gpu-fermionic-amplitudes-jax.html),
[Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html).

## Tested profiles and numerical evidence

The user subsequently installed NetKet 3.22.4 with JAX/JAXlib 0.11.2 and
CUDA 12 in their active environment, applying the upstream import annotation
fix locally and retaining backups. That is a locally patched release, distinct
from both the unmodified release and isolated upstream main below. Pepsy's
portable extra retains its import-compatible bound for unpatched releases.
Installed versions and CUDA device were verified after sourcing the user's
shell profile and activating their requested environment.

- Released CPU profile: NetKet 3.22.4, JAX/JAXlib 0.8.3, Flax 0.12.2,
  Optax 0.2.6, Equinox 0.13.2, Chex 0.1.91.
- Development profile: pinned NetKet main above (archive version label
  `3.22.5.dev0`), JAX/JAXlib/CUDA plugin/PJRT 0.11.1, Flax 0.12.10,
  Optax 0.2.8, Equinox 0.13.8. CPU imports and numerical tests pass.
- Both use Symmray `0.4.1.dev18+g44fc5ae53`, Quimb
  `1.15.1.dev91+gb3bf45a50`, Autoray `0.11.1.dev14+g014a3f69a`,
  Cotengra `0.8.3.dev8+g8954240f2`, Cotengrust 0.2.1.
- Dense and native flat-Z2 JIT directional derivatives pass with actual
  truncation, including GPU. Native CG TDVP matches full-sum 2×2 D=3
  Schrödinger evolution in complex and real/imaginary coordinates, including
  GPU; the regression forbids metric materialization during CG.
- A bounded RTX A5000 GPU probe used 4×4 D=4, χ=4, cutoff=0, 32 samples,
  16 chains, chunk=8, shift=1e-4 and CG tolerance 1e-10. One Euler step
  dt=1e-4 finished in 44.4 s including compilation. All parameters were
  finite; relative regularized solve residual was `4.0424e-11`.
  The stored reverse Jacobian shape was `(32, 2, 6400)`.
  This short resource probe does not certify χ, sampling or trajectory accuracy.
- Repeating that probe on the user's locally patched NetKet 3.22.4 with
  JAX 0.11.2 and clean pip CUDA library resolution completed in 40.0 s,
  with relative regularized residual `8.7605e-11` and finite parameters.
- With inherited system CUDA library paths, both the development GPU stack
  and locally patched release/JAX 0.11.2 segfaulted inside the real-coordinate
  dense TDVP solve. The process also loaded cuDNN 9.8 despite installed pip
  cuDNN 9.10.2. Clearing `LD_LIBRARY_PATH` for a fresh process resolved the
  installed-stack failure: all four complex/real-coordinate dense/CG
  exact-evolution cases passed on GPU in 23.12 s. This isolates a library
  resolution issue; package dependency checks alone do not catch it.
  The shared shell profile and installed libraries were not modified.
- `balance_bonds_()` preserved a dense 2×2 D=4 vector to relative error
  `2.68e-16`; both native flat and sparse Z2 failed with unsupported
  `einsum('abc,abc->a')`. Preserve native symmetry; skip this operation.

For pip CUDA wheels, use a clean library environment for the PEPS process:

```bash
env -u LD_LIBRARY_PATH python examples/tfim_netket_tdvp.py --solver cg
```

The [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
specifies that `LD_LIBRARY_PATH` can override its pip NVIDIA libraries.
This scoped launch preserves the shell's settings for other software.

Focused CPU checks: the isolated development stack passed 120 VMC/TDVP and
public API/package checks; the released CPU stack passed 78 VMC/TDVP/native
symmetry checks. Both additionally passed three targeted reverse-SR tests
after the derivative-selection fixes. The initial failures were corrected
test construction (`SamplingConfig` replacement and native VMC `.run` API)
and empty-mesh restoration on older JAX. These are focused suites, not a
complete repository run. GPU comparisons passed four installed-stack TDVP
cases and two isolated-development CG cases.
Final installed-stack validation passed 135 CPU VMC/TDVP/native symmetry and
public API/package checks in 123.71 s, four GPU exact-evolution cases in
23.12 s and four GPU dense/flat-Z2 exact/truncated directional derivative
cases in 40.24 s. Ruff and diff whitespace checks passed.

## Scaling constraints

For complex dense spin PEPS with real/imaginary coordinates, one float64
parameter metric alone occupies the following memory; factorization and
Jacobian workspace add to it.

| Geometry / D | Real coordinates | One metric (GiB) |
| --- | ---: | ---: |
| 4×4 / 3 | 2304 | 0.040 |
| 4×4 / 4 | 6400 | 0.305 |
| 6×6 / 4 | 20736 | 3.204 |
| 6×6 / 5 | 48400 | 17.453 |

Increase one of geometry, D, χ or sample count at a time. Check frozen-state
amplitudes/derivatives across χ, regularized solve residuals, sampling mixing,
dt and regularizer sensitivity, and physical observables. CG saves metric
storage but retains the sample Jacobian. Scalable paper-level tVMC still
needs validated gauge/tangent conditioning and a robust sample-space solve
for dynamics, plus longer D/χ/sample convergence evidence and restartable
sampler/integrator checkpoints.
