# Fixed-norm full-update and MPS comparison launches

User authorized CUDA:0 MPS chi=2046 and CUDA:1 MPS chi=1024, both 6×6,
matching the existing CUDA:3 5×6 chi4096 DMRG settings. Launched CuPy
complex128, dt=.1, depth60, twelve delta-theta angles, 8192 Z shots only at
t=6, no entropy/energy/XX readout, no explicit thread overrides. Parents
1389744/1389745, workers 1390045/1390046. Verified live manifests and GPU
allocations. Exact argv/env/output roots are in
`/tmp/mps_6x6_launches_20261008.json` and each root's launch_record.json.

User additionally authorized CUDA:2 PEPS gate-by-gate two-site full update
D4, chi_norm=4D²=64, no independent overlaps. Lattice explicitly clarified
as 5×6. The earlier 6×6 preparation was marked superseded and never launched.
Uses Torch complex128, DMRG/eff boundary fitting, fixed cap64 for reduced norm
and output normalization, column ordering, shared Quimb ALS (auto, max50),
dt=.1 depth60 and twelve angles. Both adaptive norm/overlap calibration and
independent pre/post overlap checks are disabled; no measurements/samples.
Internal local positive-environment ALS objectives remain active.

Launch exposed missing fixed-chi handling: use boundary_chi rather than the
overlap-evaluation cap for the reduced N, and retain a checked boundary object
across gates when calibration is off. Environment construction refreshes its
source network, validates lattice/backend/device layout, and retunes rank.
Regression checks forbid global overlap contractions and compare output norm.
Pepsy focused tests: 34 passed. Examples gained update-style/gate-order CLI
wiring and ALS budget mapping; 33 optimizer tests passed, plus 8 convergence
tests after synchronizing their expected reuse_environments=True default.
MPS entrypoint checks: 2 passed. All production commands passed dry runs.

CUDA:2 parent 1395401, worker 1395542. Root:
`/tmp/pepsy_examples_runs/peps_full_update_5x6_D4_normchi64_gpu2_dt01_depth60_dtheta12_20261008_003135`.
Launch record: `/tmp/peps_full_update_gpu2_launch.json`. First live checkpoint
confirmed mode peps-full-update, norm64, overlap checks false, samples0. At
inspection 29 of 49 first-step pair updates were saved, using Quimb ALS;
boundary cache reported 158 hits/111 rebuilds. No production restart of
CUDA:3. Its MPS worker 89592 remains running.

Changes remain uncommitted in both repositories; no pull, commit or push was
requested. This launch authorization supersedes the previous stop-only status.
