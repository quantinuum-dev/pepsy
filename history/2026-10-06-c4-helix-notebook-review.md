# 2026-10-06 — C4 Helix notebook review

- Scope: explain `Helix-pFT/C4_helix_STAR.ipynb` and remaining work relevant to
  `StabilizerMpsSimulator`; review only.
- Branch / baseline: Pepsy `develop`, `d41ea9f`.
- Commit status: this handoff is uncommitted; no implementation or notebook
  edits, commit or push. Earlier native-measurement review handoff remains.

## Current workflow and findings

Read all 32 main notebook cells, its `helper.py`, archive note, relevant Pepsy
Stim compiler/layout code and the four related examples notebooks. The main
notebook has 17 code cells with no executed counts or saved outputs. Its default
is archived X-memory replay, not a noisy STAR experiment. Counts verified from
current input/compiled analysis: 52 qubits, 3,818 replay entries, 2,738 Clifford
dense matrix entries, 550 visible measurements, 530 resets, no stochastic
events, detectors or logical observables. Configured error rates and round
count apply only to builders; they do not modify an exported input file.

The original `qecc2026Stim` and `circuit_utils` builders are unavailable through
the current import path; `tesseract_decoder` is available. Decoder setup only
compiles a decoder and does not connect reference-relative detection events,
correction, postselection or logical-error aggregation. STAR explicitly raises
NotImplementedError. It needs a concrete preparation/acceptance/readout protocol;
terminal memory readout cannot serve as pre-readout state-fidelity evidence.

The current unsampled stream translator supports independent Pauli noise;
correlated/heralded noise needs the existing Stim shot runners. Measurement
readout-error arguments, inverted measurement targets and MPAD remain rejected
at compilation. Determine requirements from the actual noisy export before
expanding support. The notebook's local feedback guard also restricts forms
more than the translator, and its layout helper rejects feedback outright.

For eligible Clifford/Pauli trajectories, native collapse already keeps
coefficient bond one. Layout search/compression ablations are not justified
as MPS-compression improvements for that path; measure setup and replay
separately. The notebook does not currently report collapse-routing counts or
stage timings. Norm one alone is not physical fidelity. The related correctness
notebook compares independent samples; per-shot Hamming differences are not
an agreement criterion, and 16 shots provide only weak marginal evidence.

## Fresh verification

Activated local genpy interpreter and executed the main notebook's input,
settings, compilation, replay/result and disabled-decoder cells in an isolated
namespace. Set two shots and disabled layout search; did not run SVG preview,
layout optimization or change/write the source notebook.

- Automatic strategy selected coalesced replay, two leaves with counts [1, 1].
- Both leaves: 550 visible records, normalized norm 1, coefficient bond 1.
- 2,160 count-weighted collapse events: all `stim`, no fallback.
- Detector and observable arrays both have shape (2, 0), as expected.
- End-to-end probe took 12.75 seconds, including imports/setup/compilation;
  this is not isolated replay throughput or the default ten-shot/layout run.

## Remaining work, in practical order

1. Obtain the complete noisy circuit/builders and its readout annotations.
2. Validate actual instruction support and compare Stim/Pepsy detection and
   logical statistics with shot multiplicities and meaningful uncertainty.
3. Connect reference-relative detector events, decoder correction, acceptance
   and count-weighted logical failure estimates.
4. Define and implement the STAR protocol and its non-Clifford preparation.
5. Profile the actual workload; only then choose simulator optimizations.

The user previously chose to retain the reviewed native-measurement
implementation. This review does not reactivate deferred compile-time routing
or other unrequested implementation changes.
