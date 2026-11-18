# All-L protection, global objective, and TFIM observables

The requested goals are to protect every chosen L density through each
truncation and reduce the remaining global or physically relevant error. The
current implementation can express the full-density objective for small
systems, but explicit maps remain exponential in region size. It does not yet
make all-L protection scalable on 4×5.

## Small-system global objective

On 2×2 and 2×3 open PEPOs, construct the physical parity density

$$
{\color{blue}\rho}=(I^{\otimes N}+0.2Z^{\otimes N})/2^N.
$$

All three-site L densities are protected; a full-lattice weighted region gives
the exact global Frobenius objective, without an approximate environment. A
single bond is reduced from 2 to 1. The ordinary feasible SVD seed changes the
global density, with Frobenius error 0.05 on 2×2 and 0.025 on 2×3.

An optional mixed SVD seed combines one discarded singular direction with the
retained factor. The resulting cross terms vanish in this particular loopy
environment. With the pair-growth guard explicitly disabled for this test,
the optimizer accepts the rank-one pair, preserves every L density, and
reconstructs the **whole density** to below 1e-10 in Frobenius norm on both
sizes. See `test_all_l_constraints_with_exact_global_density_objective`.
This is a controlled structure, not a proof that an arbitrary PEPO has such a
seed. The gauge-dependent pair norm increases, illustrating why the default
pair-growth safeguard can reject a physically good candidate.

On 2×3, one 13-gate TFIM timestep with all eight L regions, PEPO chi 4, direct
boundary cap 8 and four-million-entry map limit matches all exact state-vector
L densities to about 1e-15. Pair errors are roundoff, so this timestep is not
a test of lossy preservation. During a second timestep, the first lossy gate
failed the strict solve at chi 4 with residual 1.73e-6 and a 100-evaluation
budget; this does not prove infeasibility.

## Energy and spatial symmetries

[`examples/pepo_dmt_ising_4x5.py`](../../../examples/pepo_dmt_ising_4x5.py)
now supports other rectangular sizes up to 20 spins and independently measures
the real and imaginary parts of Ising energy, the maximum $|\langle Z_i\rangle|$
(odd under global spin flip), and maximum spatial-reflection defect in
$\langle X_i\rangle$. Reference values come from the same state-vector gate
circuit; they do not measure continuous-time Trotter error.

For two 3×3 steps, chi 4, boundary chi 8, and **one hard L**:

| Soft objective | Worst L error | Energy error | max $|Z_i|$ | max X reflection defect |
| --- | ---: | ---: | ---: | ---: |
| none, feasible seed | 0.0166 | 0.00774 | 0.00120 | 0.0138 |
| two selected pairs | 0.0742 | 0.149 | 0.000581 | 0.000523 |
| all one-site and nearest-neighbor densities | 0.0129 | 0.000217 | 0.000476 | 0.000292 |

The symmetric all-local objective uses equal weights for all single-site and
nearest-neighbor reduced matrices; it is not an exact symmetry constraint. It
improved these measured values in this one short run but did not guarantee
positivity: the minimum eigenvalue of the Hermitian part of an L panel was
about -9.7e-4. Runtime and quality need systematic chi and timestep sweeps.

For two 4×5 steps, also with one hard L, the feasible seed had worst L error
0.0136, energy error 0.00199, max $|Z_i|$ 0.000782 and X reflection defect
0.000758. The previously tested two-soft-pair objective gave 0.0446, 0.00259,
0.00864 and 0.00125 respectively. These values demonstrate that choosing a few
soft regions can bias other observables; they do not compare all-L protection.

The 4×5 all-48-L evolution still exceeds the explicit-map guard at gate 18.
Frozen boundary maps only certify their approximate constraints, and full
global maps are intractable at 20 qubits. Shared matrix-free environments,
robust feasibility and convergence with chi remain necessary before claiming
a reliable all-L globally optimized PEPO algorithm.

Example commands:

```bash
OPENBLAS_NUM_THREADS=1 python examples/pepo_dmt_ising_4x5.py --lx 2 --ly 3 --mpo-chi 32 --all-l --map-limit 4000000 --no-refine
OPENBLAS_NUM_THREADS=1 python examples/pepo_dmt_ising_4x5.py --lx 3 --ly 3 --steps 2 --all-local-objective --refine-method als
```
