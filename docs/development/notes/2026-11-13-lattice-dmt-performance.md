# Lattice Pauli DMT: normalized environment reuse

The original geometry path contracted a full identity string independently
for every protected region before and after each split. It also rebuilt the
identity parts of each protected Pauli-factor environment separately. The
optimized path reuses log-scaled identity prefixes and suffixes within each
region check and the workspace's invalidated identity environments when
mapping protected factors. Region spans and all per-split checks remain.
The physical-index contraction uses a matrix multiply followed by a reshape
and transpose. The extra per-check identity environments require linear
storage in chain length and bond dimension; there is no cross-write cache of
region panels or evolved tensors.

## Measured comparison

One CPU thread, NumPy, three runs per implementation in one process; medians
are shown. Both use rank-96 seeded positive product mixtures, strength 0.5,
snake-row-major ordering, plaquettes, and `cutoff="auto"`. The complex case
also uses seeded nontrivial virtual phases. Baseline functions were loaded
from `cd71159d`; both versions used the same process and input.

| Lattice | Dtype | Cap | Baseline | Optimized | Ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 5×5 | float64 | 72 | 1.221 s | 0.552 s | 2.21× |
| 6×6 | complex128 | 88 | 23.168 s | 2.408 s | 9.62× |

The two runs in each case had identical recorded plaquette matrices, trace,
bond dimensions, and discarded weights. These timings are limited CPU
measurements, not a general performance or accuracy guarantee. The algorithm
still checks full `4**k` coefficient panels for each protected `k`-site
region, and finite-precision error and conservative bond-budget limitations
remain. The original, unchanged environment and Quimb/Autoray upstream audit
are recorded in the [lattice DMT handoff](../../../history/2026-11-13-lattice-pauli-dmt.md).

A separate 8×8 complex128 run, rank 128 and cap 112 with random virtual
phases, took 11.2 seconds for compression. Against an independent product
mixture, the worst of 49 plaquette Frobenius errors was `6.72e-15`, trace
error `5.85e-15`, and Ising-energy error `6.81e-15`; the residual was nonzero.
This is one run, not a controlled comparison with the previous 8×8 timing,
which included setup and competing work.
