# Pepsy PEPS sampler implementation reference

## Source map

Use the checkout at `/Users/rezah/Documents/pepsy`.

- Direct sampler: `src/pepsy/sampling/samplers.py`, class `PepsSampler`.
- Existing BP sampler: the neighboring class `PepsBpSampler` in the same file.
- Public sampling exports: `src/pepsy/sampling/__init__.py` and
  `src/pepsy/__init__.py`.
- Tagged ket/norm construction: `src/pepsy/boundary/metrics.py`,
  `build_bra_ket`.
- Quimb boundary cache adapter: `src/pepsy/optimizers/sweep/environments.py`,
  `QuimbMpsBoundaryStore`.
- Design plan: `docs/development/plans/peps_sampler.md`.
- Reference tests: `tests/test_peps_sampler.py`.

Before Python, tests, or package commands, run:

```bash
source /Users/rezah/envs/genpy/bin/activate
```

## Phase 1: current serial reference path

`PepsSampler` makes a private copy of the PEPS, then calls
`build_bra_ket(ket=...)`. The resulting norm network has KET and BRA layers
sharing each unmeasured physical index. For a current site `(x, y)`:

1. Copy the conditioned norm network for the local calculation.
2. Select the current BRA tensor using the site tag and `BRA` tag.
3. Reindex its physical index to a private bra label.
4. Contract everything except the ket and bra physical labels with Quimb,
   using `output_inds=(ket_ind, bra_ind)`.
5. Set

   ```text
   p(s | prefix) = real(rho[s, s]) / real(trace(rho)).
   ```

6. Sample `s` and call `working.isel_({ket_ind: s})`. Since ket and bra
   initially share that physical index, this fixes both layers.

After all sites are fixed, project the private ket using the selected physical
indices and contract it to obtain `Psi(S)`. The result stores proposal and
amplitude values as mantissa/exponent pairs, matching the existing PEPS
sampling vocabulary.

This path is expensive because it recontracts the full conditioned network at
each site. That is deliberate: it is the oracle against which boundary-MPS
and batching implementations should be compared.

## Phase 2: conditioned boundary MPS — implemented

`PepsSampler` keeps a mutable single-layer boundary ket `phi` for the sampled
prefix. It does not replace it with Quimb's unconditioned `ymin` double-layer
environment. After completing a row, the projected row is applied to `phi`
and compressed to `sample_chi` with either Quimb MPS compression
(`ket_compression="quimb"`) or Pepsy `FIT`
(`ket_compression="fit"`).

For later rows the projected row is represented as a Quimb MPO and applied
with the native `MatrixProductOperator.apply(phi, contract=True)` operation.
This lets Quimb contract and preserve the MPS topology; custom sitewise bond
fusing is not needed in the normal path.

The lower boundary is shot-dependent. Two shots have the same boundary MPS
only when their sampled prefixes are equal. Otherwise they must be separate
states, even though their tensor topology and maximum bond dimensions may
match.

## Phase 3: future marginal environment — implemented

The sampler caches only unmeasured rows. The Quimb engine uses the
`("ymax", y)` entries from `QuimbMpsBoundaryStore`; the DMRG engine constructs
`BdyMPS` and runs `CompBdy.move_bdy(direction="y_right")`, then attaches its
right-side `mps_b[f"Y{Ly - 2 - y}_r"]` future boundaries. `marginal_chi`
controls this future double-layer approximation, while `sample_chi` controls
the conditioned single-layer boundary. `None`/`0` selects an identity future
cap.

The current numerical policy validates positive finite traces and clips only
tiny negative diagonal probabilities. Full Hermiticity diagnostics and
correction reporting remain follow-up work for more aggressive truncation.

## Phase 4: DMRG/FIT provider — implemented

The dense DMRG/FIT future provider is separate from the conditioned ket. FIT
is available for the latter through `ket_compression="fit"`; it is not used
as a substitute for the shot-conditioned `phi` state.

## Phase 5: prefix-grouped batching — implemented

`PepsSampler.sample_batch` groups shots by their sampled prefix. Each group
gets one local `rho` contraction, splits only when its sampled values diverge,
and carries its own conditioned boundary MPS. This gives safe prefix sharing
without pretending that Quimb can apply different `isel` values to one shared
network. `batch_stats` reports the peak and final group counts.

Native tensor-axis batching is still deferred. For a future native extension,
an MPS-like batch boundary could have a shape such as
`boundary[shot, bond]` or one shot axis on every corresponding boundary-MPS
tensor. The PEPS tensors remain unbatched and shared.

## Phase 6: row suffix transfer cache — implemented adaptively

For compact boundary centers, the sampler contracts each current-row column
into a Quimb ket/bra transfer tensor, traces the physical index to form the
unmeasured transfer, and contracts all right suffixes once from right to left.
At each site it contracts the current transfer with the cached suffix and the
conditioned left prefix, samples immediately, and absorbs the fixed transfer
into that prefix. The suffix contractions do not apply a second cutoff because
the attached future environment has already been controlled by
`marginal_chi`. This path is checked against the full-center reference on
small Quimb and DMRG examples.

The transfer tensors can become denser than the original local center when a
large PEPS already has a collapsed future MPS. In that case, and for highly
fragmented large batches, the implementation adaptively uses the reference
prefix/local-center path. This keeps the optimization from becoming a
performance or memory regression. `row_cache_stats` and `batch_stats` expose
which path was used.

Quimb preserves an index that occurs once globally as a dangling output. But
a normal Quimb contraction does not understand per-shot physical selection.
If shot `n` selects `s_n`, the update is conceptually

```text
new_boundary[n] = boundary[n] * PEPS[..., s_n, ...].
```

Ordinary `isel_` applies one `s` to the whole network. Therefore implement
batching in this order:

1. Group shots with identical sampled prefixes.
2. Contract one Quimb local network per prefix group.
3. Draw choices for the group and split it when choices diverge.
4. Use native NumPy, Torch, or CuPy operations for batched boundary updates
   when group sharing is worthwhile.
5. Fall back to serial groups when the number of prefixes becomes too large.

Never give the same `batch` label to multiple PEPS tensors expecting it to be
ignored. That label is then an ordinary shared tensor-network bond.

## Validation recipe

For a small dense complex PEPS, enumerate every physical configuration and
compare:

```text
product of sequential conditionals
    == |direct projected amplitude|**2 / full PEPS norm.
```

Also check that repeated seeded calls return identical configurations and that
the source PEPS tags, indices, and tensor data are unchanged. Run:

```bash
pytest -o addopts='' -q tests/test_peps_sampler.py
pytest -o addopts='' -q tests/test_sampler.py
pytest -o addopts='' -q tests/test_public_api.py
ruff check src/pepsy/sampling/samplers.py tests/test_peps_sampler.py
pytest -q
```
