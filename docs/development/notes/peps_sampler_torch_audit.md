# 2026-09-26 — PepsSampler Torch and Autoray study

Status: study only. No sampler implementation, dependency, or running-job
changes. The prior roughening gauge-default change is a separate task.

## Conclusion

PepsSampler contracts and compresses Torch tensors on their original device
on the tested exact and positive-marginal boundary paths. It is not an
end-to-end Torch sampler: local density matrices move to NumPy; probability
checks, random draws, prefix grouping, probability bookkeeping, and returned
results use host Python/NumPy. There is also a reproducible mixed-backend bug
in the identity future cap used by marginal_chi=None/0 in boundary mode.

## Environment and evidence

- Pepsy develop / 80f451a; existing working-tree changes preserved.
- Autoray 0.11.1.dev3+g1b476b305; Quimb 1.15.1.dev66+ge927f06e1;
  Torch 2.6.0+cu124; RTX A5000 on cuda:0.
- Reused the unchanged numerical dependency audit from the earlier
  [PEPS implementation](../../../history/2026-09-26-peps-simple-update-implementation.md).
- Source: [PepsSampler](../../../src/pepsy/sampling/samplers.py), lines
  4306–5597 at this revision; [public sampler guide](../../api/sampling/samplers.md).
- Bounded probes: /tmp/peps_sampler_torch_audit.py and its JSON/log;
  /tmp/peps_sampler_torch_followup.py and its JSON/log.
- Probes use a fixed random, unnormalized 2x3 dense PEPS, D=2, seed 83,
  four draws with seed 11, sample_chi=4, marginal_chi=8, one FIT sweep,
  and greedy contraction planning. One follow-up uses 40 draws to exercise
  the reference-prefix path. These are compatibility and transfer probes,
  not production speed or truncation-accuracy benchmarks.

## Stage-by-stage ownership

| Stage | Actual behavior with Torch input |
| --- | --- |
| Private ket/norm and row copies | Torch arrays; source remains unchanged |
| Future boundary, marginal_chi>0 | Quimb MPS or Pepsy BdyMPS/CompBdy; inspected outputs preserve dtype/device |
| Future cap, marginal_chi=None/0 | Hard-coded np.eye(..., dtype=complex), hence CPU NumPy complex128 |
| Row transfers, local-rho contraction | Torch contractions until explicit to_numpy |
| Local-rho readout | _local_rho and _row_local_rho transfer each matrix to NumPy |
| Conditional probabilities/diagnostics | NumPy diagonal, trace, norm, clipping, normalization, Python scalars |
| Random choices | np.random.default_rng(seed).choice |
| Prefix grouping and selected indices | Python lists/dicts plus NumPy arrays/unique; isel decisions are host integers |
| Projected-row application and ket compression | Torch MPO/MPS operations and Quimb SVD or Pepsy FIT; scalar extraction remains |
| Proposal log probabilities | Python float and math.log10 |
| Final amplitudes | Original Torch ket contracted, then _scalar transfers to host |
| Returned configs/probabilities/amplitudes | Lists of Python ints/floats/complex numbers, including exponent pairs |

The control flow is shared by serial and grouped sampling. Batches share
equal prefixes, not native tensor batch axes. Large-batch reference routing
does not remove the host probability pipeline. With more than nine sites,
sample_batch uses the reference-prefix route above four shots; serial boundary
sampling with positive marginal_chi also uses reference-center contractions.

## Reproduced bug and precision concern

_identity_future constructs NumPy identities while the neighboring tensors
remain Torch. With marginal_chi=0 this failed on Torch CPU complex128,
CUDA complex128, and CUDA complex64:

    TypeError: tensordot(): argument 'other' (position 2) must be Tensor, not numpy.ndarray

The captured first center contained four Torch tensors and two NumPy tensors.
None and 0 take the same identity-cap branch; 0 was exercised explicitly.
The positive-marginal path and exact path avoid this cap.

A temporary subclass under /tmp replaced only identity creation with
ar.get_namespace(template).eye(size). The CUDA complex128 no-future sample
then completed reproducibly and kept its conditioned boundary on cuda:0.
This is a prototype result, not an installed fix.

The conditional validator also uses np.finfo(float).eps after casting the
diagonal to Python/NumPy float precision. Thus complex64 inputs receive a
float64-based tolerance. The complex64 probes passed, but this precision
policy needs explicit review before tolerating roundoff: meaningful negative
probabilities from truncated environments must still be rejected.

## Measured CUDA crossings

TorchDispatchMode counted explicit device-to-host copies and scalar reads.
These counts exclude final dense-reference validation and concern only these
small examples; they are not wall-time measurements.

| Route | Draws | Density matrices to CPU | Amplitudes to CPU | Scalar reads during sampling | Scalar reads during setup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quimb future + Quimb ket compression | 4 | 17 | 4 | 7 | 4 |
| DMRG future + FIT ket compression | 4 | 17 | 4 | 7 | 9 |
| Quimb reference-prefix | 40 | 38 | 40 | 14 | 4 |

The extra sampling scalar reads arose through boundary compression. Setup
also reads scalar convergence/normalization information. Replacing only the
random-number generator will not remove these synchronizations.

The twelve main probes comprise nine successful exact/positive-marginal cases
and three reproduced identity-cap failures. Successful cases preserved the
source tensors. Returned amplitudes versus independent dense contractions
had maximum absolute discrepancies 1.78e-14 (CUDA complex128) and 4.27e-6
(CUDA complex64), for the unnormalized random state.

## What Autoray helps with

The installed version already has the useful facilities:

- get_namespace(template) infers backend, dtype and device and caches dispatch.
  eye/ones/asarray can create tensors directly on the correct device.
- A namespace built from the real probability vector supplies
  random.default_rng(seed).choice(..., p=probabilities). The Torch wrapper
  uses a device-specific torch.Generator and torch.multinomial.
  CPU and CUDA probes produced device-local int64 draws reproducibly.
- do('eye', ..., like=template) also preserved complex64 and cuda:0 in a probe.
- Autoray 0.9 added to/from_numpy/to_device and fixed RNG device inheritance;
  0.10 improved native random dtype/seed handling and external Torch generators;
  0.11 added namespace-aware compose and stable namespaces across registration.
  These are already present in this environment. compose can package shared
  probability kernels; it does not make Python grouping device-resident.

Official sources: [release notes](https://github.com/jcmgray/autoray/blob/f8d1366496/docs/changelog.md),
[namespace API](https://autoray.readthedocs.io/en/latest/autoapi/autoray/index.html#autoray.get_namespace),
[revision comparison](https://github.com/jcmgray/autoray/compare/1b476b305...f8d1366496).
The rendered changelog and commit pages failed through browsing; GitHub's
official API and raw source succeeded, with responses retained under
/tmp/autoray-audit-* and /tmp/autoray-current-changelog.md.

As checked on 2026-09-26, upstream main f8d1366496 is three commits ahead of
the installed 1b476b305. Those commits change compiler and lazy.Function
thread safety, not eager sampling/backend conversion. Updating Autoray alone
would not change the sampler's explicit NumPy operations. No update was made.

## Proposed implementation order

1. **Adopt:** backend/device/dtype-aware identity creation; add a Torch CPU
   and optional CUDA regression for the no-future path. Preserve the exact
   and NumPy proposal references and source-state isolation.
2. **Prototype:** keep rho/probability validation and normalization on the
   tensor backend, with dtype-aware tolerances and backend-local seeded RNG.
   Transfer only grouping decisions and batched diagnostic summaries. Do not
   hide invalid traces or substantial negative probabilities.
3. **Prototype:** keep amplitude/proposal scaling on device until the requested
   output conversion. In sample_batch, contract an amplitude once per final
   identical-prefix group rather than once per shot in that group.
4. **Defer until profiled:** reduce boundary scalar synchronization, native
   tensor batching, and compilation of fixed-shape kernels. A shared Quimb
   index is not a sample axis. Removing all host decisions requires an
   algorithmic batching change, not an Autoray import replacement.

Existing MpsSampler native helpers demonstrate device-local probabilities
and draws; its prefix sampler still transfers integer decisions for Python
grouping. Reuse appropriate numerical helpers rather than coupling the PEPS
sampler to private MPS state conventions. The package allows autoray>=0.9;
newer 0.10/0.11-only features would need capability handling or a separately
justified dependency policy change. Seeded reproducibility should be guaranteed
within each supported backend; matching the old NumPy bitstream is a separate
compatibility decision.

No package tests were changed, no dependency upgrades or production sampler
runs were performed, and no speedup or full-suite result is claimed.
