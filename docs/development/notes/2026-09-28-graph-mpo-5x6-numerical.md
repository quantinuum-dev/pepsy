# 2026-09-28 — 5×6 order-four graph MPO numerical construction

## Scope and setup

This closes the numerical-measurement gap in the earlier
[recursive assembly record](2026-09-28-mpo-recursive-assembly.md). The model is
5×6 open square, snake MPO order, uniform `0.2 sum X + 0.4 sum ZZ`, step
`0.03`, connected-cluster order `p=4`, spatial reuse on,
`assembly="recursive"`, `assembly_batch_size=4`, and
`assembly_state_budget=40_000`. The source plan has **492 connected clusters**,
**33,514 shared remaining-site states**, **173,679 DAG branches** and
**249,479,463,512 nonempty compatible collections**. The run is NumPy
complex128 on CPU with one BLAS/OpenMP thread; a 12 GiB process virtual-memory
limit and a wall-time guard applied to each evaluation. The χ=2 run exceeded
its initial 20-minute timeout: the monitor was paused before expiry and
resumed after the Python child completed and wrote its result. The timeout
wrapper then returned 124 due to its pending alarm; the child had completed
successfully. No global dense 30-site operator was allocated. Other user
workloads continued on the shared machine, so wall times are scoped
observations.

An independent scalar reference evaluates `⟨b|U_p|b⟩` for two computational
product states. It starts from the raw connected residual matrices, chooses
the first unoccupied site, and memoizes the exact sum over every compatible
remaining cluster. It does not use assembled MPO cores or numerical bond
compression. The reference's recurrence was checked on a 2×3 p=2 example
against exact dense MPO entries: errors were zero and `6.66e-16`. Thus the
reported errors isolate **numerical assembly at the selected p**, not the
cluster-order error relative to `exp(step H)`. Two diagonal elements are not
a global operator/Frobenius error certificate.

## Finite NumPy SVD fallback

The first chi-1 batch-four attempt stopped after **15,725 compressions** when
NumPy's thin SVD did not converge. Its 164×53 complex128 input was finite,
had maximum entry magnitude `2.32e4`, and had singular-value ratio about
`9.86e26`. Replaying the saved matrix reproduced NumPy failure. SciPy's
public `svd(..., full_matrices=False, lapack_driver="gesvd")` reconstructed
that same matrix with relative Frobenius error `2.99e-18`.

`mpo_semantic._fixed_rank_svd` now retries with that driver only when a
**two-dimensional finite NumPy** input raises `LinAlgError`. It imports SciPy
lazily; without SciPy the original failure is re-raised. Torch, JAX, other
backends, nonfinite data, requested bond caps and rank-selection rules remain
on their existing paths. A deterministic test simulates default-driver
nonconvergence and checks reconstruction, driver choice and nonfinite refusal.
This is a narrow **compatibility shim**, not a guarantee that all finite
matrices or assembly policies converge. The installed environment has NumPy,
SciPy 1.17.1, Quimb 1.15.1.dev66, Autoray 0.11.1.dev3, Cotengra 0.8.3.dev7,
Symmray 0.4.1.dev7 and Torch 2.6.0. The prior
[upstream audit](2026-09-28-mpo-recursive-assembly.md#upstream-audit--classification)
remains applicable. Official [SciPy SVD documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html)
confirms the two LAPACK drivers and the thin-output option.

## Measured numerical result

| Policy | Compile | Residuals | Assembly | Compressions | Peak RSS | Peak cached states | Peak precompression bond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chi=1, batch=4 | 0.412 s | 0.290 s | 486.314 s | 42,671 | 0.810 GiB | 7,761 | 65 |
| chi=2, batch=4 | 0.461 s | 0.288 s | 1237.625 s | 42,671 | 1.017 GiB | 7,761 | 130 |

| Computational basis diagonal | Exact p=4 collection sum | chi=1 MPO | chi=1 relative error | chi=2 MPO | chi=2 relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| all zeros | 1.801293022880322 | 1.004076528403267 | 0.44258 | 1.556793296450943 | 0.13574 |
| alternating bits in snake order | 0.555730615903449 | 1.004076528403389 | 0.80677 | 0.794081737917892 | 0.42890 |

The chi-1 MPO has bond dimension one at every cut; chi-2 has bond dimension
two at every cut. Both retain every collection branch structurally, but both
are too inaccurate for the two measured entries. Increasing the cap from one
to two improves both entries, yet neither this pair of tests nor two rank
settings establish convergence. The error is from numerical bond truncation.
Increasing `assembly_state_budget` or `p` does not fix this rank error;
`assembly_chi` convergence must be checked independently. A larger-cap
5×6 benchmark was not run, so a usable accuracy setting and global operator
error for this case remain unverified. The measured 1.017 GiB is process
peak RSS, not a hard memory bound implied by the state budget.
