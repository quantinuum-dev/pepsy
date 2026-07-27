#!/usr/bin/env python3
"""Validate Pepsy's local skill catalog without third-party dependencies."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SKILLS = ROOT / ".github" / "skills"


def _frontmatter(text: str, path: Path) -> tuple[str, str]:
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: missing YAML frontmatter")
    try:
        end = text.index("\n---\n", 4)
    except ValueError as exc:
        raise ValueError(f"{path}: unterminated YAML frontmatter") from exc
    fields: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError(f"{path}: malformed frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip("'\"")
    if set(fields) != {"name", "description"}:
        raise ValueError(
            f"{path}: frontmatter must contain only name and description; "
            f"found {sorted(fields)}"
        )
    if not fields["name"] or not fields["description"]:
        raise ValueError(f"{path}: name and description must be non-empty")
    if "TODO" in fields["description"]:
        raise ValueError(f"{path}: description still contains TODO")
    return fields["name"], text[end + 5 :]


def _check_links(skill_dir: Path, text: str, errors: list[str]) -> None:
    for target in re.findall(r"\]\(([^)]+)\)", text):
        target = target.split("#", 1)[0]
        if not target or target.startswith(("http:", "https:", "mailto:")):
            continue
        resolved = (skill_dir / target).resolve()
        if not resolved.exists():
            errors.append(f"{skill_dir / 'SKILL.md'}: broken link {target}")


def main() -> int:
    errors: list[str] = []
    skill_dirs = sorted(
        path
        for path in SKILLS.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )
    if not skill_dirs:
        errors.append(f"no skills found under {SKILLS}")

    names: set[str] = set()
    for skill_dir in skill_dirs:
        skill_path = skill_dir / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        try:
            name, body = _frontmatter(text, skill_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if name != skill_dir.name:
            errors.append(f"{skill_path}: name {name!r} != directory {skill_dir.name!r}")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            errors.append(f"{skill_path}: name is not lowercase hyphen-case")
        if name in names:
            errors.append(f"duplicate skill name: {name}")
        names.add(name)
        if len(text.splitlines()) > 500:
            errors.append(f"{skill_path}: SKILL.md exceeds 500 lines")
        if not (skill_dir / "agents" / "openai.yaml").is_file():
            errors.append(f"{skill_dir}: missing agents/openai.yaml")
        _check_links(skill_dir, body, errors)

    readme = SKILLS / "README.md"
    if not readme.is_file():
        errors.append(f"missing catalog: {readme}")
    else:
        catalog_targets = set(
            re.findall(r"\]\(([^)]+/SKILL\.md)\)", readme.read_text())
        )
        expected_targets = {f"{name}/SKILL.md" for name in names}
        for target in sorted(expected_targets - catalog_targets):
            errors.append(f"{readme}: skill missing from catalog: {target}")
        for target in sorted(catalog_targets):
            if not (SKILLS / target).is_file():
                errors.append(f"{readme}: missing catalog link target {target}")

    manifest = SKILLS / "agent-bundle.yaml"
    if not manifest.is_file():
        errors.append(f"missing upload manifest: {manifest}")
    else:
        manifest_text = manifest.read_text()
        manifest_paths = {
            relative.strip()
            for relative in re.findall(
                r"^\s+path:\s+(.+)$", manifest_text, re.MULTILINE
            )
        }
        expected_paths = {f".github/skills/{name}" for name in names}
        for path in sorted(expected_paths - manifest_paths):
            errors.append(f"{manifest}: skill missing from upload map: {path}")
        for relative in sorted(manifest_paths):
            if not (ROOT / relative.strip()).is_dir():
                errors.append(f"{manifest}: missing skill path {relative.strip()}")
        references = re.search(
            r"^references:\n(?P<body>.*?)(?=^skills:\n)",
            manifest_text,
            re.MULTILINE | re.DOTALL,
        )
        if references:
            for relative in re.findall(r"^\s+-\s+([^#].*?)\s*$", references["body"], re.MULTILINE):
                if not (ROOT / relative.strip()).is_file():
                    errors.append(
                        f"{manifest}: missing reference path {relative.strip()}"
                    )

    if errors:
        print("Pepsy skill catalog validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Pepsy skill catalog OK: {len(skill_dirs)} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
