"""Shared text/ASCII rendering helpers."""

from __future__ import annotations

import re
import sys


def resolve_color_mode(color, *, stream=None):
    """Resolve color mode from bool or ``'auto'``."""
    if color == "auto":
        stream = sys.stdout if stream is None else stream
        return bool(getattr(stream, "isatty", lambda: False)())
    if isinstance(color, bool):
        return color
    raise TypeError("color must be bool or 'auto'")


def ansi_wrap(text, code, enabled):
    """Wrap text with ANSI style code when enabled."""
    if not enabled:
        return text
    return f"\033[{code}m{text}\033[0m"


def colorize_symbols(line, enabled, *, grid_arrows=False):
    """Apply ANSI colors to structural symbols used in ASCII previews."""
    if not enabled:
        return line

    replacements = [
        ("◆", "31"),
        ("●", "31"),
        ("▶", "31"),
        ("◀", "31"),
        ("▼", "1;35"),
        ("▲", "1;35"),
        ("○", "36"),
        ("o", "36"),
    ]
    if grid_arrows:
        replacements.extend(((">", "31"), ("<", "31")))
    else:
        replacements.extend(((">", "1;32"), ("<", "1;34")))

    out = line
    for symbol, code in replacements:
        out = out.replace(symbol, ansi_wrap(symbol, code, enabled))
    return out


def style_show_line(line, *, color_enabled, fancy):
    """Render one boundary-show line with optional symbols and ANSI colors."""
    is_grid_title = line.startswith("grid cut=")
    is_grid_row = re.match(r"^Y\d+\s+", line) is not None
    is_x_axis = re.match(r"^\s+X\d+", line) is not None
    stripped = line.strip()
    is_conn_row = line.startswith("    ") and bool(stripped) and set(stripped) <= {
        "|",
        "v",
        "^",
        " ",
    }

    out = line
    if fancy and (is_grid_row or is_conn_row):
        out = out.replace("--", "──").replace(">>", "> ").replace("<<", " <")
        out = out.replace("|", "│").replace("v", "▼").replace("^", "▲")
        out = out.replace("o", "○")

    if color_enabled:
        if is_grid_title:
            return ansi_wrap(out, "1;36", True)
        if is_grid_row:
            match = re.match(r"^(Y\d+\s+)(.*)$", out)
            if match:
                prefix, body = match.groups()
                return ansi_wrap(prefix, "1;33", True) + colorize_symbols(
                    body,
                    True,
                    grid_arrows=True,
                )
        if is_x_axis:
            return ansi_wrap(out, "1;33", True)
        return colorize_symbols(out, True)

    return out


def style_show_lines(lines, *, color=True, fancy=True):
    """Apply styling to boundary-show output lines."""
    color_enabled = resolve_color_mode(color)
    return [style_show_line(line, color_enabled=color_enabled, fancy=fancy) for line in lines]

