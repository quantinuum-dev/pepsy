# Development documentation

Use this section for implementation ownership, contributor workflow and dated
design evidence. Application behavior belongs in the [API guides](../api/index.md).

<a id="find-a-document"></a>

## Document ownership

| Need | Owning document |
| --- | --- |
| Supported behavior and public examples | [API guides](../api/index.md), [tutorials](../tutorials/index.md) and [how-to guides](../howto/index.md) |
| Source ownership, imports and compatibility | [Package layout](package_layout.md), [module maps](modules/README.md), [API migration](api-migration.md) and [public surface](api-surface.md) |
| Contributor workflow | [CONTRIBUTING](../../CONTRIBUTING.md) |
| Test commands, backend coverage and CI | [Test profiles and CI](testing.md) |
| Code rationale and behavior-change evidence | [Code-change policy](code_policy.md) |
| Markdown style | [Concise mathematical Markdown](markdown_policy.md) |
| Proposals and historical workstreams | [Planning index](plans/README.md) |
| Cluster/downstream capabilities and validation | [Current status](cluster_optimization_status.md) |
| Measurements and scientific rationale | [Research notes](notes/README.md), [references](references/README.md), [import costs](import-weight.md) and [fermionic MPO notes](fermi_hubbard_u1u1_mpo_notes.md) |
| Plot conventions | [Plotting policy](plot_policy.md) |
| Reusable agent instructions | [Skill catalog](../../.github/skills/README.md) |
| Dated decisions and publication | [Session journal](../../history/README.md) |

## Writing Markdown

Follow the [writing policy](markdown_policy.md) and edit the owning page.
Link shared setup, contracts and validation commands rather than copying them.
Add each new page to its nearest index and check relative links, anchors and
filename case. Plans and historical records retain their stated baselines;
they do not establish current implementation or test coverage.

## Release distribution

Pepsy is distributed through GitHub. The current source version is `0.6.0`;
`v0.5.0` remains the latest release tag.
The release workflow validates GitHub Actions artifacts and has no registry
publishing or GitHub Release-creation jobs. PyPI/TestPyPI publication remains
cancelled. Later commits are development changes until a new version is tagged;
preserve existing tags. See [installation](../installation.md) and
[contributor checks](testing.md) for the relevant workflow.
