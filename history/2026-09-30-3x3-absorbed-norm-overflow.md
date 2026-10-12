# 2026-09-30 — Exact gauge-absorbed PEPS norms before SU overflow

- Scope: user requested full Cotengra contraction of the gauge-absorbed 3x3
  state to establish its actual norm and explain the SU crash.
- Baseline: Pepsy develop / 1b4ca92, examples main / 40cadd7.
- Commit status: this note is uncommitted; no staging or publishing.
- Follow-up to [the BP/SU test record](2026-09-30-3x3-peps-bp-validation.md).
  Preserved unrelated concurrent package-example removals and documentation edits.
- No production numerical code or tests changed in this follow-up.

## Exact measurement

Replayed the same open 3x3 diagonal-wall runs, J=-1, hx=1, hz=0,
delta_theta=0, snake mapping, dt=0.25, D=2 and D=4, cutoff=1e-12,
NumPy complex128, renorm=False, no periodic gauge equilibration.

At every completed timestep:

```python
optimizer = pepsy.tensors.build_optimizer(
    parallel=False, max_repeats=32, max_time=1.0
)
physical = engine.core.copy(deep=True)
physical.gauge_simple_insert(engine.gauges)
norm2 = physical.make_norm().contract(all, optimize=optimizer)
norm = np.sqrt(norm2.real)
```

`build_optimizer` returned Cotengra's ReusableHyperOptimizer. These are full
18-tensor bra/ket contractions with no boundary approximation, BP, or bond
truncation in the measurement. Every result also agreed with the squared
norm of all 512 independently contracted ket amplitudes (rtol=1e-10,
atol=1e-13). Core tensors and gauges were checked unchanged after measurement.

| t | D=2 norm | D=4 norm |
| --- | ---: | ---: |
| 0 | 1 | 1 |
| 0.5 | 0.999985398949 | 1 |
| 1 | 0.994824075840 | 1.000016412390 |
| 1.5 | 0.894096972249 | 1.002437697740 |
| 2 | 0.719979844421 | 0.982133036478 |
| 2.5 | 0.568975985796 | 0.860344882997 |
| 3 | 0.403357441693 | 0.723183176717 |
| 3.5 | evolution already failed | 0.470078381353 |
| 3.75 | evolution already failed | 0.420790949673 |

Immediately before failure:

- D=2, t=3: norm squared 0.1626972257692198; largest core entry
  2.9824906154924196e158; smallest gauge entry 1.5399751748664551e-120.
  Next step toward t=3.25 fails.
- D=4, t=3.75: norm squared 0.1770650233270634; largest core entry
  about 1.531e178; smallest gauge entry about 1.566e-135.
  Next step toward t=4 fails.

The physical norm remains finite; the external-gauge representation develops
extreme mutually compensating scales. Norm drift under local SU truncation
is distinct from overflow of the stored representation.

## Failing operation and scale-only control

Replayed each final step one gate at a time with RuntimeWarning promoted to
an exception. Both failures are in Quimb's `gauge_simple_remove`, on exit
from `gauge_simple_temp`, at `t.multiply_index_diagonal_(ix, g**-1)`.
Multiplying inverse outer gauges back into the core overflows.

- D=2: zero-based gate 17, edge (1,1)-(2,1).
- D=4: zero-based gate 16, edge (1,1)-(1,2).

On separate private diagnostic copies, rescaled each external gauge by its
maximum entry s and multiplied each endpoint core tensor by sqrt(s):
lambda -> lambda/s, A -> sqrt(s) A, B -> sqrt(s) B. This is scalar
redistribution only: no BP, no full gauge equilibration, and no physical
normalization or alteration of renorm=False.

- D=2 state-vector relative change: 4.35e-16; norm squared after redistribution
  0.16269722576921974. The previously failing next step completes, yielding
  norm squared 0.1442549330794483.
- D=4 state-vector relative change: 7.15e-16; norm squared after redistribution
  0.17706502332706336. The previously failing next step completes, yielding
  norm squared 0.17046475131736216.

This isolates internal scale conditioning as the immediate crash mechanism.
Only one continuation step was tested after this diagnostic redistribution;
it is not a production fix or a long-run stability claim.

## Reproducible artifacts

All completed timesteps, exact squared norms and norms, core/gauge scales,
exceptions, and the executable scripts are saved under:

`/tmp/pepsy_3x3_absorbed_norm_20260930_n9noui_5/`

- `probe.py` / `results.json`: full contraction at each step for D=2 and D=4.
- `gauge_scale_diagnostic.py` / `gauge_scale_diagnostic.json`: failing gates,
  complete tracebacks, scalar redistribution and one-step continuation.

The scripts used the shared py312 environment and local Pepsy source, with
CUDA hidden and two CPU threads per numerical library. Production GPU jobs
were untouched. No new repository suite was required for these standalone
diagnostic scripts; `git diff --check` passed.
