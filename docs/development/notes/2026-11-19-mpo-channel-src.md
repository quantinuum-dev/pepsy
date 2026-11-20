# MPO channel FIT warm starts

The dense MPO gate path and the [MPS FIT path](../../api/optimizers/mps.md)
use a capped compression result to initialize the optimizer and retain an
independent exact layered target. The MPO two-site channel path now uses
that same separation. It compresses the tagged interval of a disposable
target copy with the requested guess method and rejoins the exterior across
the original bonds. The native sector path retains its existing initialization.

Quimb 1.15.0 accepts multi-tensor sites and forwards explicit site tags to
SRC. Its compression code inserts an identity chain for a nonlocal bond on
the working copy. Use a generic active tensor network so nested compressors
can operate on interior tags and exposed virtual indices. The existing
sub-MPO projector policy avoids pseudo-canonicalization of zero channel
gauges. Interior `fit-projector`, `fit-zipup`, `zipup-first`, and
`zipup-oversample` were checked with a nonlocal depolarizing channel.

With Autoray 0.8.4 and Cotengra 0.8.4.dev8, the old trace contraction on
the first one-site noise target after SRC-initialized gates attempted a
640 GiB outer product. Tracing each dense physical pair on its own tensor
first keeps that diagonal reduction out of the virtual contraction planner.
Symmray 0.5.1.dev2 retains its native trace route. Reviewed the upstream
[Quimb](https://quimb.readthedocs.io/en/latest/changelog.html),
[Autoray](https://github.com/jcmgray/autoray),
[Cotengra](https://cotengra.readthedocs.io/en/latest/changelog.html), and
[Symmray](https://github.com/jcmgray/symmray) sources and the installed
Quimb implementations. The upstream Symmray array documentation was unavailable.
