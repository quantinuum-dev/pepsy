# Development documentation

Use this section to understand the implementation, past decisions, and proposed
work. For application code, start with the [API guides](../api/index.md).

## Find a document

| Need | Read |
| --- | --- |
| Upgrade existing code | [API migration](api-migration.md) |
| Find source ownership and imports | [Package layout](package_layout.md) and [module maps](modules/README.md) |
| Understand compatibility aliases | [Public API surface](api-surface.md) |
| Read design rationale and measurements | [Research notes](notes/README.md) |
| Review proposed or historical work | [Plans](plans/README.md) |
| Find papers and background | [Technical references](references/README.md) |
| Inspect import costs | [Import measurements](import-weight.md) |
| Check fermionic MPO conventions | [Fermi-Hubbard notes](fermi_hubbard_u1u1_mpo_notes.md) |
| Prepare plots | [Plotting policy](plot_policy.md) |

Plans and dated notes describe their recorded state, not current guarantees.
Check the implementation and tests before treating a proposal as supported.
Session handoffs live in the repository's
[history directory](https://github.com/quantinuum-dev/pepsy/tree/develop/history).

## Writing Markdown

- Start with the page's purpose and the action the reader can take.
- Keep installation commands in the [installation guide](../installation.md)
  and test commands in [CONTRIBUTING](https://github.com/quantinuum-dev/pepsy/blob/develop/CONTRIBUTING.md).
  Link to them instead of copying long instructions.
- Put public behavior and examples in `docs/api/`, complete walkthroughs in
  `docs/tutorials/`, and specific tasks in `docs/howto/`.
- Put implementation maps in `modules/`, evidence and rationale in `notes/`,
  proposals in `plans/`, and session records in `history/`.
- Use short paragraphs, descriptive headings, and concrete examples. Add
  section links to long pages; keep numerical assumptions and limitations.
- Label proposals, measurements, and historical results clearly. Preserve
  completed session records and add a dated correction when decisions change.
- Add new pages to the nearest index and Sphinx toctree. Check links and run
  the [strict documentation build](../installation.md#build-the-documentation).
  Generated `docs/api/reference/` pages are build output, not editing targets.

## Release distribution

Pepsy is distributed through GitHub. Version `0.5.0` is tagged there;
publication to PyPI and TestPyPI was cancelled by the maintainer. The earlier
registry setup handoff is historical and does not authorize a retry.
The release workflow builds and validates GitHub artifacts on version tags
or manual dispatch. It has no registry-publishing jobs.
These are GitHub Actions artifacts; the workflow does not create a GitHub
Release entry. Commits after a version tag are development changes until a
new version is selected and tagged. Do not move an existing release tag to
include subsequent cleanup.

```{toctree}
:hidden:

plans/README
notes/README
modules/README
references/README
package_layout
fermi_hubbard_u1u1_mpo_notes
plot_policy
api-migration
api-surface
import-weight
```
