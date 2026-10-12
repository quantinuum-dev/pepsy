# 2026-09-28 — Differentiable construction without parameter-dependent SVD

## Status

Read-only production-code review plus an isolated CPU prototype; the design
below is **proposed**, not an integrated construction API or JIT implementation.
Branch `develop`, baseline `4e398e4`, with the earlier uncommitted cluster work.
No production behavior changed during this review; no commits or publication.

## Current implementation, verified against source

- `PauliPEPOBasis` uses fixed Pauli channels for the usual homogeneous orders
  1–4 and fixed history trees for uncapped localized residuals. Its four-site
  path and plaquette helpers use coefficient contractions and selectors rather
  than parameter-dependent SVDs. These paths already have backend autodiff.
- Homogeneous generic PEPO orders 5–9 call
  `_tree_factorize_operator_backend`, which still uses SVD even without a
  rank cap. Rank-capped localized trees also use this routine.
- MPO `_operator_schmidt` has an exact identity-factorization branch for JAX
  with cutoff=None/0 and max_bond=None. NumPy/Torch still use SVD. This local
  decomposition is separate from recursive assembly compression: setting
  assembly_chi=None alone does not make a Torch MPO construction SVD-free.
- `compile_exp()` caches geometry, embeddings, schedules and other structural
  work. It is not a promise of one captured Torch/JAX machine-code graph.
- Current MPO `_is_zero_operator` detaches Torch values and copies to NumPy.
  It is called before the exact JAX branch. A direct probe of
  `_operator_schmidt(theta * (Z tensor Z), 2, 2, cutoff=0)` at theta=0 returned
  cores with no derivative connection, although dR/dtheta=Z tensor Z.
  Existing nonzero finite-gradient tests do not cover this zero-crossing
  derivative. The numerical-zero shortcut must not govern trainable topology.

## Proposed architecture

1. Compile lattice, p, ordered term supports, coefficient bindings, verified
   spatial equivalences, embeddings, inclusion-exclusion schedules, fixed
   channels and contraction indices once. Cache backend constants by dtype
   and device. Include policy and geometry in cache keys; never cache current
   parameter values or an old autograd tape.
2. Evaluate a pure tensor kernel: theta/time -> coefficient arrays -> batched
   local Hamiltonians -> ordered matrix exponentials -> residual subtraction
   -> fixed coefficient channels -> MPO/active PEPO or a scalar contraction.
   Retain all structurally possible channels even at numerical zeros.
3. Generalize exact fixed index splits across backends for MPOs, and provide
   a corresponding exact fixed representation for generic PEPO trees. Use
   static reshapes/permutations/identity routing, with no data-dependent rank
   selection. Keep approximation policies explicit.
4. Separate tensor outputs from Python active-block dictionaries/reports.
   Batch representative clusters by size and static operation pattern. Test
   capture of these bounded kernels before attempting to compile a large
   lattice-wide graph. Backend machine compilation should be a separate,
   observable capability from structural preparation.
5. Where the caller needs a scalar loss, prepare its contraction with the
   fixed layout and consume coefficient blocks directly when possible.
   Preserve the selected expansion and contraction accuracy; avoid forcing
   dense PEPO materialization only to contract it immediately.

For a local split M of shape (a,b), use M=I_a M when a<=b or M=M I_b when b<a.
The identity and index maps are static. The parameter-dependent factor is
linear in M, so there is no singular-vector gauge derivative. Exact generic
p-site qubit MPO bonds are at most 4^min(k,p-k): for p=4, (4,16,4).
Global lattice MPO bonds and fixed PEPO history sectors can still be much
larger. Removing SVD does not bound global bond growth.

For approximate small bonds, a fixed reduced basis/projector is possible but
introduces projection error. A basis prepared once at a reference parameter
is only exact if it spans all needed residuals across the parameter family.
Refreshing it during optimization changes the approximate objective. A fixed
rank alone does not remove the SVD or resolve repeated-singular-value issues.
Detaching a parameter-dependent factorization is not an exact gradient of the
original construction.

## Prototype evidence

Temporary script `/tmp/pepsy_fixed_channel_probe.py` implemented the existing
JAX identity-split idea using Torch reshapes and identities, without modifying
Pepsy. CPU float64, one thread, deterministic random Hermitian local H, with
R(theta)=exp(theta H)-I and theta=0 or 0.17. All six 2/3/4-site dense
reconstructions and weighted scalar gradients matched the direct matrix-exp
reference at 1e-12 tolerance. This includes zero residuals with nonzero
parameter derivatives.

200 warmed forward factorization calls per size (microseconds per call):

| Sites | Existing Torch local SVD | Fixed index prototype | Exact local bond dimensions |
| --- | ---: | ---: | --- |
| 2 | 219.7 | 27.1 | 4 |
| 3 | 423.7 | 40.2 | 4, 4 |
| 4 | 664.8 | 59.4 | 4, 16, 4 |

These are scoped local CPU timings, including the existing path's zero check
and dispatch overhead. They exclude matrix exponentiation, full assembly,
contraction, backward timing, GPU and machine compilation. No end-to-end
speedup or general fastest-method claim follows.

## References and acceptance criteria

[PyTorch SVD](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)
documents singular-vector derivative restrictions at repeated/zero singular
values. [Torch full-graph guidance](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html)
recommends explicit graph-break checks. These current docs do not prove
capabilities of the installed Torch 2.6.0 environment; an implementation must
probe that environment and benchmark capture separately.

An integrated change should check actual absence of SVD calls, full operator
reconstruction, coefficient/time gradients at generic and zero values,
repeated calls with unchanged shapes, complete disjoint cluster collections,
ordered-factor semantics, and forward/backward time and peak memory. Compare
exact fixed-basis and explicitly approximate reduced-basis policies separately.
The earlier 270-pass MPO gate predates this review; no new production suite
was run because no production behavior changed.
