# 2026-09-29 — Second review of the shared cluster API

- Baseline: `develop` at `a233a40`, plus the uncommitted API alignment
  recorded in [the preceding handoff](2026-09-29-cluster-api-alignment.md).
- Scope: user-requested review, reproduced API defects and narrow fixes.
- Publication: Pepsy changes remain local and uncommitted. No dependency,
  numerical factorization or contraction implementation changes.

## Findings and fixes

1. Adding `cluster_size` as a second dense-plan dataclass field broke
   `dataclasses.replace(plan, order=...)`: it copied the old alias value and
   triggered the new conflict check. Keep `order` as the sole stored field,
   resolve the keyword alias at construction and expose a read-only property.
   Preserve the generated positional signature, default order three, and
   explicit alias conflict checks. Use `order` for dataclass replacement.
2. MPO product vectors rejected `None` factors, while PEPO/Gaugy already use
   them to retain default term coefficients. Resolve those defaults freshly
   in the MPO binding path. Required parameters/callables still reject missing
   input; independent structural term identities and gradients are preserved.
3. PEPO checked conflicting runtime containers inside the factor loop, after
   resolving callbacks, and accepted `parameters` with all-`None` coefficient
   vectors. Validate containers once before callbacks or trace-plan work,
   matching the documented exclusive-input contract and the other families.

`tests/test_cluster_api_review.py` checks replacements against a three-site
noncommuting dense exponential, keyword introspection, partial defaults,
missing required bindings and rejection before callback evaluation. Gaugy's
cross-package consistency test now exercises the same binding edge cases.

## Validation

- Before fixes: the new review selection reproduced **8 failures / 4 passes**.
- After fixes: review plus initial API tests **33 passed** in 4.54 s.
- Broad cluster/API selection: **318 passed**, five existing warnings,
  125.67 s. Same 15 files listed in the preceding handoff, plus
  `tests/test_cluster_api_review.py`; includes dense/ordered-product checks,
  fixed/recursive/compressed MPO paths, spatial reuse, Torch/JAX gradients,
  public API and package layout.
- Gaugy focused cluster/public/API selection: **197 passed**, six existing
  warnings, 46.17 s. Gaugy production source was unchanged during this review.
- Pepsy `ruff check src tests`, Gaugy changed-test Ruff and both repositories'
  `git diff --check` pass. Shared Python 3.12, CPU-only settings as in the
  preceding handoff. No full repository suite or notebook rerun.
- Native Z2 coefficient-vector smoke probe retained `MPOPhysicalSpace` metadata
  and evaluated both the MPO and trace. This was a smoke check, not a native
  symmetry or backend algorithm change.

No additional correctness issue was identified in the reviewed common call
paths. Representation defaults and Gaugy connected-log versus residual
partition trace semantics remain deliberately distinct.
