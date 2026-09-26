"""Private MPS report formatting and FIT timing summaries.

These helpers consume existing records only: they do not import numerical
backends, read clocks, contract tensors, or change optimizer state. Replay
timing collection and numerical diagnostics remain on ``MpsOptimizer``.
"""

__all__ = []

# Compatibility totals overlap with their named subsets; do not add every
# field together to estimate elapsed FIT time.
_FIT_TIMING_PHASES = (
    "canonicalization_seconds",
    "sweep_preparation_canonicalization_seconds",
    "moving_canonicalization_seconds",
    "fixed_environment_seconds",
    "effective_seconds",
    "svd_seconds",
    "writeback_seconds",
    "environment_seconds",
    "moving_environment_seconds",
    "non_site_elapsed_seconds",
    "sweep_overhead_seconds",
)


def _summarize_fit_timing(records):
    """Summarize detailed FIT sweep timing without discarding raw records."""
    records = tuple(records)
    fit_indices = {
        int(record["fit_index"])
        for record in records
        if "fit_index" in record
    }
    return {
        "calls": len(fit_indices),
        "sweeps": len(records),
        "site_updates": sum(
            int(record.get("site_count", len(record.get("site_timings", ()))))
            for record in records
        ),
        "elapsed_seconds": sum(
            float(record.get("elapsed_seconds", 0.0)) for record in records
        ),
        **{
            phase: sum(float(record.get(phase, 0.0)) for record in records)
            for phase in _FIT_TIMING_PHASES
        },
    }


def _format_layout_value(value):
    """Format one layout diagnostic value compactly."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.6g}"


def _format_layout_reduction(before, after, *, format_value=_format_layout_value):
    """Format ``before -> after`` with a percent decrease when meaningful."""
    before = float(before or 0.0)
    after = float(after or 0.0)
    text = f"{format_value(before)} -> {format_value(after)}"
    if before > 0.0:
        reduction = 100.0 * (before - after) / before
        text += f" ({reduction:.1f}% lower)"
    return text


def _layout_report_text(
    plan,
    *,
    format_value=_format_layout_value,
    format_reduction=_format_layout_reduction,
):
    """Return a concise human-readable layout improvement report."""
    stats = plan.get("stats", {})
    input_stats = plan.get("input_stats", {})
    if not input_stats:
        return None
    selected = plan.get("selected_order", "<unknown>")
    site_order = plan.get("site_order", plan.get("qubit_inds", ()))
    weight_mode = plan.get("weight_mode", "count")
    objective = plan.get("objective", "locality")
    score_before = input_stats.get("loss", input_stats.get("score", 0.0))
    score_after = stats.get("loss", stats.get("score", 0.0))
    score_label = "score"
    if objective == "replay":
        # Replay selection replaces the primary score with a large,
        # lexicographically scalarized bond objective. Keep this report
        # line about the comparable static graph proxy instead.
        score_before = input_stats.get(
            "loss", input_stats.get("score", 0.0)
        )
        score_after = stats.get(
            "static_loss", stats.get("path_loss", stats.get("loss", 0.0))
        )
        score_label = "graph proxy score"
    lines = [
        (
            "MpsOptimizer layout finder: "
            f"order={selected}, sites={len(site_order)}, "
            f"events={stats.get('num_events', input_stats.get('num_events', 0))}, "
            f"weight_mode={weight_mode}, objective={objective}"
        ),
        (
            "  long-range events: "
            + format_reduction(
                input_stats.get("long_range_events", 0),
                stats.get("long_range_events", 0),
            )
            + " | weighted: "
            + format_reduction(
                input_stats.get("weighted_long_range_events", 0.0),
                stats.get("weighted_long_range_events", 0.0),
            )
        ),
        (
            "  event span max/mean: "
            + format_value(input_stats.get("max_event_span", 0))
            + "/"
            + format_value(input_stats.get("weighted_mean_event_span", 0.0))
            + " -> "
            + format_value(stats.get("max_event_span", 0))
            + "/"
            + format_value(stats.get("weighted_mean_event_span", 0.0))
        ),
        (
            f"  {score_label}: "
            + format_reduction(
                score_before,
                score_after,
            )
            + " | graph span: "
            + format_reduction(
                input_stats.get("weighted_total_span", input_stats.get("total_span", 0.0)),
                stats.get("weighted_total_span", stats.get("total_span", 0.0)),
            )
            + " | cut L2: "
            + format_reduction(
                input_stats.get("weighted_cut_congestion_l2", 0.0),
                stats.get("weighted_cut_congestion_l2", 0.0),
            )
        ),
    ]
    if objective == "compression":
        lines.append(
            "  operator cut load max/total: "
            + format_value(
                stats.get("max_operator_cut_load", 0.0)
            )
            + "/"
            + format_value(
                stats.get("total_operator_cut_load", 0.0)
            )
            + " | bounded cut probes: "
            + format_value(stats.get("rank_bounded_cuts", 0))
        )
    elif objective == "replay":
        replay = stats.get("replay", {})
        if replay.get("status") == "ok":
            lines.append(
                "  replay peak bond/log2: "
                + format_value(replay.get("peak_bond", 0))
                + "/"
                + format_value(replay.get("peak_log2_bond", 0.0))
                + " | profiled events: "
                + format_value(len(replay.get("profile", ())))
            )
    return "\n".join(lines)
