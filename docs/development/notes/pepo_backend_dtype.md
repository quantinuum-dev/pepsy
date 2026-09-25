# Fixed-channel PEPO backend dtype audit

Audit date: 2026-09-15.

The fixed-channel `PauliPEPOBasis` fuses coefficient slots into static real
onsite and edge maps with `autoray.do("tensordot", ...)`.  Complex Torch slot
values exposed two missing dtype-alignment steps: converting the NumPy maps
to Torch preserved `float64`, while `torch.tensordot` requires both inputs to
have the same dtype; and a Python complex product prefactor could be cast to
the first real Torch term slot's dtype.  The compatibility fix converts maps
to the coefficient batch's dtype and uses the promoted result dtype for
product scalars.  It does not alter coefficient values, channel topology,
factor order, or NumPy behavior.

The active environment was inspected with these versions and call surfaces:

- Quimb `1.15.1.dev55+gd0591eb70`: current tensor/gating changes do not alter
  this pre-contraction coefficient fusion; classification: **defer**.
- Autoray `0.11.1.dev3+g1b476b305`: `do(fn, *args, like=None, **kwargs)` routes
  `tensordot` directly to the Torch implementation, so Pepsy performs the
  narrow dtype alignment; classification: **compatibility shim**.
- Cotengra `0.8.3.dev7+g1d7fd333f`: contraction planning is downstream of the
  failing coefficient fusion; classification: **defer**.
- Symmray `0.3.2.dev8+g6c6dd34b5`: native block conversion is downstream and
  was not involved in this dense fixed-channel failure; classification:
  **defer**.
- Torch `2.9.1`: `torch.tensordot(a, b, dims=2, out=None)` requires matching
  input dtypes.  A focused complex128 value-and-gradient regression now fixes
  that invariant in Pepsy.

The upstream audit covered the Quimb and Cotengra changelogs plus the Autoray
and Symmray repositories/documentation.  No upstream API should be adopted for
this issue: explicit dtype preservation at Pepsy's static-map boundary is the
smallest backend-neutral policy.

Audit sources: [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray repository](https://github.com/jcmgray/autoray),
[Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html),
[Symmray arrays](https://symmray.readthedocs.io/en/latest/abelian_arrays.html),
and [Symmray repository](https://github.com/jcmgray/symmray).
