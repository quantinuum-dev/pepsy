# 2026-10-27 — Bounded Torch CPU exact replay

## Scope and installed capabilities

The requested 6×6, two-corner-removed, `complex64` exact-batch run has 34
active qubits. One dense state uses 128 GiB. The session used the existing
`py312` environment: Torch 2.14.1+cu126, Quimb 1.15.1.dev96,
Autoray 0.11.1.dev14, Cotengra 0.8.3.dev8, Symmray 0.4.1.dev20, and
Numba 0.67.0. Installed Quimb exposes
`TensorNetwork.copy(self, virtual=False, deep=False)`, so the default is a
shallow array copy. Torch permits a writable NumPy view of a contiguous CPU tensor
without gradients or conjugate/negative view bits. Official Quimb,
Autoray, Cotengra, and Symmray release/source documentation was checked;
the Symmray array documentation page was unavailable. No installed library
was modified.

Classification: **adopt**, behind `exact_batch_bounded_cpu=True`. Compatible
one-qubit and structured phase/parity gates use an in-place Numba kernel on
the owned Torch array. Unsupported blocks fail explicitly. Dense input is
copied when the optimizer is constructed with `inplace=False`; after the first
replay, old references to `opt.p` share the live data and change on later
replays. The ordinary exact-batch policy is unchanged.

Final Z sampling has a separate optional bounded path. It scans chunk masses
and builds a CDF only for selected chunks. This avoids a state-sized
probability/CDF allocation. Sampling can be omitted from the 6×6 feasibility
decision because evolution and direct readout dominate memory.

## Measured evidence

The 5×6, 28-active-qubit, four-step CPU pilot used 16 threads, final direct
readout, and 8192 final Z shots. One state uses 2 GiB. With a deep copy on
each replay, peak RSS reached 8.74 GiB by step three. Reusing the live array
held peak RSS at 6.74 GiB on step one and 4.75 GiB on steps two through four.
Final direct observables between the two implementations differed by at most
3.2×10⁻⁹, excluding timing fields. At 16 Numba workers, the first 5×6 step
took 6.66 seconds instead of about 9.8 seconds with four workers, with the
same 6.73 GiB peak RSS. Its direct energy differed from the four-worker run
by 1.6×10⁻¹⁰. The smaller matched ordinary exact-batch
reference agreed at approximately 6×10⁻⁸ in energy and magnetization.
These are focused checks, not a full numerical suite or a 6×6 production
validation.

A matched 5×6 one-step comparison at 16 Torch threads measured 6.66 seconds
and 6.73 GiB peak RSS for bounded replay versus 8.98 seconds and 8.73 GiB
for ordinary exact-batch. Final direct observables agreed within 5.7×10⁻⁸.
This supported gate stream gained speed as well as memory headroom at that
size; the result does not predict every gate stream.

The 6×6, one-step pilot completed 124 evolution gates in a 287-second engine
stage (177 seconds in the gate progress display) with 16 Numba workers.
Peak RSS was 384.73 GiB under a 440 GiB process virtual
memory limit; RSS after evolution was 256.74 GiB. A 64-shot final Z sample
also completed with no material memory increase. The optional direct readout
was interrupted because its chunked scan over 2³⁴ amplitudes is much slower
than evolution; the pilot therefore has no finalized observable archive.
This establishes one-step evolution feasibility, not full-depth runtime or
direct-readout feasibility. A full 60-step, seven-angle sweep was not launched.
If later steps repeat the measured 177-second gate time, a 60-step angle would
spend roughly three hours in evolution; that is an estimate, not a timed run.

The benchmark's opt-in direct CPU readout was subsequently changed from
chunked Torch operations to a 16-worker, read-only Numba scan. On the same
5×6 one-step case, it reduced final readout from 23.29 to 4.61 seconds;
direct observables agreed within 1.3×10⁻⁸. A new 6×6 one-step pilot then
completed with a finalized observable archive: evolution took 243 seconds,
direct readout 326 seconds, and peak RSS was 384.73 GiB. The final energy per
site was −1.24255948, with ⟨Mx⟩ = −0.25226115 and ⟨Mz⟩ = −0.17039650.
This verifies one final direct readout on this machine. Repeating it at every
time step would add substantial cost; the final-only schedule avoids that.

## Sources

- [Quimb changelog](https://quimb.readthedocs.io/en/latest/changelog.html)
- [Autoray repository](https://github.com/jcmgray/autoray)
- [Cotengra documentation](https://cotengra.readthedocs.io/en/latest/)
- [Cotengra changelog](https://cotengra.readthedocs.io/en/latest/changelog.html)
- [Symmray repository](https://github.com/jcmgray/symmray)
