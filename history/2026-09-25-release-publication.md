# 2026-09-25 — Tag 0.5.0 and attempt registry publication

- Scope: user authorized tagging, release validation, TestPyPI publication and
  installation checks, then PyPI publication.
- Branch / release baseline: `develop` and `main` at `e5bb212`.
- Commit status: this handoff accompanies a documentation-only follow-up;
  the release tag remains on `e5bb212`.

## Completed

- Verified release-commit CI passed on both
  [main](https://github.com/quantinuum-dev/pepsy/actions/runs/36165760058) and
  [develop](https://github.com/quantinuum-dev/pepsy/actions/runs/36165760763).
- Created and pushed annotated tag `v0.5.0` at
  `e5bb2120ec9f7993219c9f25fdf34a8141c62baf`.
- The [tag release workflow](https://github.com/quantinuum-dev/pepsy/actions/runs/36172465035)
  passed: built distributions, validated metadata, checked wheel import and
  version, and uploaded `pepsy-dist-0.5.0` (artifact ID 10880857304).
- Dispatched the release workflow at `v0.5.0` with
  `publish_to_testpypi=true`, `publish_to_pypi=false`.

## Publication blocker

The [TestPyPI workflow](https://github.com/quantinuum-dev/pepsy/actions/runs/36172595919)
built and validated the package successfully, but its publish job failed:
`invalid-publisher`: no Trusted Publisher matched the GitHub identity.
The claims match the intended repository, workflow, tag, and `testpypi`
environment. This requires registry-account configuration, not source changes.

Both registries' JSON APIs returned HTTP 404 for the `pepsy` project and for
version 0.5.0 at the time of this attempt. No package was published; registry
installation checks and the PyPI dispatch remain pending. No connected browser
was available to configure the accounts. The user was asked to add pending
publishers, following the official
[new-project Trusted Publisher instructions](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

| Field | TestPyPI | PyPI |
| --- | --- | --- |
| Project name | `pepsy` | `pepsy` |
| GitHub owner | `quantinuum-dev` | `quantinuum-dev` |
| Repository | `pepsy` | `pepsy` |
| Workflow filename | `release.yml` | `release.yml` |
| Environment | `testpypi` | `pypi` |

## Resume

1. After configuration, rerun the failed TestPyPI job or dispatch the workflow
   at the existing `v0.5.0` tag with only TestPyPI enabled.
2. Download/install 0.5.0 from TestPyPI into a temporary target, confirm
   distribution/runtime version and numerical smoke behavior. Keep the
   selected shared Python environment unchanged.
3. Dispatch the same tag with only PyPI enabled, then verify registry metadata
   and installation. Do not move or recreate the existing release tag.

The GitHub CLI API connection reset repeatedly. A temporary helper successfully
used the CLI's existing credentials with curl over HTTP/1.1; credentials were
not printed or stored in the helper. Git SSH operations worked normally.
This handoff changes no release artifact or implementation; validation is
`git diff --check` and remote-ref verification.
