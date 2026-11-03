# 2026-10-29 — Gibbs overlay missing external results

- Scope: diagnose and repair the sibling Gibbs/QMC notebook's execution error.
- Baselines: Pepsy `develop` at `e658bed`; examples `main` at `58e3f8d`.
- Commit status: working-tree edits only; nothing staged or published.

Reproduced `FileNotFoundError` in the completed-results cell of
[gibbs_qmc_overlay.ipynb](../../pepsy_examples/experiments/mps_magnetization/benchmark/plots/gibbs_qmc_overlay.ipynb).
Its original `/tmp/gibbs_itf/` result directories are absent. Searches of
the local workspaces, temporary storage, and likely archive locations found
no matching result JSON or exported point CSV. The original MPS data have
not been recovered; the user was asked for their current location.

The notebook now renders the supplied QMC reference when Gibbs files are
absent, explicitly labeling the missing MPS data in its status and every
figure. `PEPSY_GIBBS_ROOT` selects the common parent of restored scan folders;
existing per-run overrides remain available. `PEPSY_GIBBS_PLOT_PREFIX`
controls exports. Restored valid JSON resumes MPS loading; mismatched model
or run-policy records still fail. No measured values were fabricated and
no simulation was launched. Updated the plotting guide and saved notebook
outputs; unrelated edits were preserved.

Validation: all notebook cells executed in a temporary copy, producing three
QMC-only figures; combined and temperature views were visually checked.
The four focused overlay tests passed, including missing/empty directories,
restored results, rejected mismatched records, and measured-point exports.
Changed-file Ruff and `git diff --check` passed. Broader benchmark Ruff still
reports six existing errors in `magnetization/effective/correlations.py`,
`tests/test_roughening_eff_correlations.py`, and
`tests/test_roughening_eff_theta.py`; those files were not changed.
No full numerical suite was run because the numerical algorithms are unchanged.

Remaining: restore the original Gibbs result files, set the data root, and
rerun the notebook to recover the actual MPS comparison. Successful QMC-only
execution does not validate or reconstruct the missing finite-χ curves.

## Recovery after the user reported a cluster restart

The installed tmpfiles policy contains `D /tmp`; the startup service invokes
`systemd-tmpfiles --create --remove --boot`. The installed manual states that
`D` removes directory contents with `--remove`. This is consistent with the
reported restart and absent Gibbs directory. No Gibbs files were deleted or
moved by this repair. The roughening archives under `benchmark/store/` remain
present. No original Gibbs JSON files were found in accessible local storage,
Trash, or notebook attachment snapshots; the boot journal was unavailable.

Recovered from the October 26–28 local run-session records:

- 28 saved plot images, including the October 27 complete overview with
  26 temperatures at each of χ=32,64,128,256 (104 plotted MPS points).
- 46 distinct direct-MPS energy values: χ=32: 15; χ=64: 5; χ=128: 15;
  χ=256: 11. Most were logged to ten decimal places; printed precision and
  original source excerpts are retained. Seventeen repeated observations
  agree within their recorded precision. The selected χ=256 critical value
  is the 20-thread result, kept distinct from the earlier 12-thread run.

Recovery artifacts are preserved in ignored
`pepsy_examples/experiments/mps_magnetization/benchmark/store/gibbs_itf/recovery_20261029/`:
`recovered_complete_overview.png`, `recovered_direct_energy_points.csv`,
`figure_provenance.json`, `numerical_provenance.json`, and `figures/`.
These are a saved visual comparison and partial numeric recovery, not a
reconstruction of the original runner JSON schema or all original precision.

The notebook now embeds the recovered overview as an explicitly dated
historical image. Freshly generated figures still label the absent original
Gibbs data. Changed the default data/export root to persistent
`benchmark/store/gibbs_itf/`, retaining explicit environment overrides, and
updated the plotting guide. No production job was launched.

New validation: four focused tests and changed-file Ruff passed. All notebook
cells executed, showing the recovered overview plus three explicitly labeled
QMC-only regenerated figures. CSV uniqueness, reciprocal temperature/beta,
logged energy values, notebook schema, ignored recovery paths, and whitespace
checks passed. The original full numerical dataset remains unrecovered.

## Follow-up: VS Code renderer failure

The user clarified the visible failure as `Error loading renderer
jupyter-notebook-renderer`. The notebook still contains four saved PNG
outputs and no saved Python errors. The installed remote renderer extension
is `ms-toolsai.jupyter-renderers` 1.3.0; its renderer entrypoint, preload,
and extension entrypoint files exist and are readable. Remote extension logs
show renderer activation but do not establish why the client failed to load
it. This is separate from the earlier missing-results exception.

Opened the recovered overview directly with `code --reuse-window`; the
command succeeded. Notebook display in the user's editor remains unverified.
The next UI troubleshooting step is `Developer: Reload Window`, followed by
selecting the built-in image renderer or disabling Jupyter Notebook Renderers
for the workspace if needed. No extension was removed, no server or kernel
was restarted, and no notebook/data code was changed for this display issue.
Nothing was committed or published.

## Notebook-level renderer workaround

After the user requested a notebook fix, replaced all four `Image` displays
with `display_plot`, which embeds the PNG bytes in `text/html` output. This
uses the built-in HTML renderer instead of `jupyter-notebook-renderer`.
Executed a temporary notebook copy and refreshed the four saved figure
outputs in the source notebook. Each embedded PNG is byte-identical to its
previous saved image; no result data, numerical code, or plot style changed.
Updated the plotting guide and extended the existing focused checks to verify
embedded images against exported files, recovered-image availability, and
the saved HTML output format.

Validation: all notebook cells executed successfully; four focused tests,
changed-file Ruff, notebook schema validation, and `git diff --check` passed.
Broader benchmark Ruff still reports the same six pre-existing errors listed
above. Opened the updated notebook using `code --reuse-window`. Actual display
in the user's editor is unverified; a local headless Firefox check could not
start because its snap binary requires newer host GLIBC/GLIBCXX libraries.
No editor extension or system library was changed. No full numerical suite
was run. All edits remain uncommitted and unpublished.
