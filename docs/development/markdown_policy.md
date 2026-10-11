# Concise mathematical Markdown

Applies to new or revised Markdown, skill references and notebook Markdown.
Keep existing scientific records and saved notebook outputs intact.

## Writing rules

- Lead with the purpose or result in one sentence. Keep paragraphs and cells
  short; remove repeated context and link the owning explanation.
- Prefer a precise equation and brief symbol definitions for objectives,
  symmetries, parameter constraints and approximation errors.
- Retain assumptions, support, normalization, product order and validation scope
  when they affect the result. Concision must preserve the mathematical contract.
- Put important formulas in display math with `$$ ... $$`;
  use inline math for short symbols. Keep formulas outside code fences.
- Color important formula terms with scoped `{\color{blue} ...}`
  groups. Use two or three named colors consistently within a page or workflow;
  retain its existing palette when editing.
- Highlight a few key labels with **bold** or matching colored
  `\text{...}` inside math. Define the roles in words; colors reinforce them.
- Give one useful formula or example before adding derivations. Link longer
  proofs, API details and measured results instead of repeating them.

## Plans and skills

Keep a plan's status, scope, intended result, next steps, validation and owning
references easy to scan. Distinguish implemented, proposed, deferred and measured
work. Preserve superseded designs in labeled archives; record results in dated
handoffs. A plan describes work without authorizing it.

Keep skill entry points short and route to existing domain guidance. Load only
relevant references. Put public usage in its API guide and avoid copying plans
or numerical manuals into skills.

## Formula pattern

For nonzero states, blue identifies the reference and purple the candidate:

$$
F(\psi,\phi)=
\frac{\left|\left\langle {\color{blue}\psi}
\mid {\color{purple}\phi}\right\rangle\right|^2}
{\langle\psi\mid\psi\rangle\,\langle\phi\mid\phi\rangle},
\qquad \mathcal L=1-F.
$$

Define the states and normalization once, then explain the decision the formula
supports in one short sentence.
