from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.cme_method import _load_iltcme_table  # noqa: E402
from methods.cme_method import solve as solve_cme  # noqa: E402
from methods.dual_bounds_method import solve as solve_dual_bounds  # noqa: E402
from methods.erlang_method import solve as solve_erlang  # noqa: E402
from methods.erlang_solvers import StationarySolverSettings  # noqa: E402
from methods.mpmath_stationary import precision_tolerance as _precision_tolerance  # noqa: E402


@dataclass(frozen=True)
class SvgSeriesStyle:
    color: str
    dash: str | None
    marker: str


# ==========================================================================
# Edit this block when running from PyCharm
# ==========================================================================
RUN_ERLANG = 1
RUN_CME = 1
RUN_DUAL_BOUNDS = 1

SERVICE_TIME = 1.0
CAPACITY = 10
LAMBDA_VALUES = [0.1]
OUTPUT_DIR = ROOT / "results" / "arithmetic_accuracy"

MIN_DIGITS = 5
MAX_DIGITS = 30
DIGIT_STEP = 1

ERLANG_PHASE_COUNTS = [10]
ERLANG_STATIONARY_METHODS = ["sparse_lu_refinement_direct_q"]
ERLANG_MAX_ITERATIONS = 50_000
ERLANG_RELAXATION = None
ERLANG_SOLVER_SETTINGS = StationarySolverSettings()

CME_TARGETS = [
    {"name": "CME 10", "phases": 10},
]
CME_INTEGRAL_METHODS = ["real_schur"]
CME_LINEAR_METHODS = ["sparse_lu_refinement"]

DUAL_BOUNDS_VARIANTS = [
    "faster 10",
    "slower 10",
]

DUAL_BOUNDS_INTEGRAL_METHODS = ["poisson_real_schur"]
DUAL_BOUNDS_LINEAR_METHODS = ["sparse_lu_refinement"]
DUAL_BOUNDS_MAX_ITERATIONS = 50_000
X_AXIS_LABEL = "Residual precision parameter (d)"
# ==========================================================================


@dataclass(frozen=True)
class RunSpec:
    method: str
    solver: str
    integral_solver: str | None = None

    @property
    def label(self) -> str:
        if self.integral_solver:
            return f"{self.method} / {self.integral_solver} + {self.solver}"
        return f"{self.method} / {self.solver}"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()

    specs = list(_run_specs(args))
    total_jobs = len(specs) * len(args.digit_values)
    job_index = 0

    for digits in args.digit_values:
        for spec in specs:
            job_index += 1
            print(f"[{job_index}/{total_jobs}] digits={digits}: {spec.label}")
            _run_spec(spec, digits, args, samples, failures)
            _write_outputs(args.output_dir, args, samples, failures, started, write_plots=False)

    _write_outputs(
        args.output_dir,
        args,
        samples,
        failures,
        started,
        write_plots=not args.skip_plots,
    )
    print(f"wrote arithmetic accuracy outputs to {args.output_dir}")


def _run_spec(
    spec: RunSpec,
    digits: int,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    for arrival_rate in args.lambda_values:
        try:
            rows = _solve_one(spec, digits, float(arrival_rate), args)
            for row in rows:
                samples.append(_sample_row(spec, digits, float(arrival_rate), row))
        except Exception as exc:  # noqa: BLE001 - keep the sweep alive.
            failures.append(_failure_row(spec, digits, float(arrival_rate), type(exc).__name__, str(exc), exc))


def _solve_one(
    spec: RunSpec,
    digits: int,
    arrival_rate: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    tolerance = _precision_tolerance(digits)
    if spec.method == "erlang":
        return solve_erlang(
            arrival_rate=arrival_rate,
            service_time=args.service_time,
            capacity=args.capacity,
            phase_counts=args.erlang_phase_counts,
            stationary_method=spec.solver,
            tolerance=tolerance,
            max_iterations=args.erlang_max_iterations,
            relaxation=args.erlang_relaxation,
            solver_settings=ERLANG_SOLVER_SETTINGS,
            precision_digits=digits,
        )
    if spec.method == "cme":
        return solve_cme(
            arrival_rate=arrival_rate,
            service_time=args.service_time,
            capacity=args.capacity,
            targets=_resolved_cme_targets(args.cme_targets),
            integral_method=str(spec.integral_solver),
            linear_method=spec.solver,
            tolerance=tolerance,
            precision_digits=digits,
        )
    if spec.method == "dual_bounds":
        return solve_dual_bounds(
            arrival_rate=arrival_rate,
            service_time=args.service_time,
            capacity=args.capacity,
            variants=args.dual_bounds_variants,
            integral_method=str(spec.integral_solver),
            linear_method=spec.solver,
            tolerance=tolerance,
            max_iterations=args.dual_bounds_max_iterations,
            precision_digits=digits,
        )
    raise ValueError(f"Unsupported method: {spec.method}")


def _sample_row(
    spec: RunSpec,
    digits: int,
    arrival_rate: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "method": spec.method,
        "approach": result.get("approach"),
        "variant": _variant_label(spec, result),
        "phase_target": _phase_target_label(spec, result),
        "arrival_rate": arrival_rate,
        "precision_digits": digits,
        "tolerance": _precision_tolerance(digits),
        "p_idle": float(result["p_idle"]),
        "p_full": float(result["p_full"]),
        "runtime_seconds": float(result.get("timing_total", math.nan)),
        "timing_build": result.get("timing_build"),
        "timing_solve": result.get("timing_solve"),
        "timing_metrics": result.get("timing_metrics"),
        "solver": spec.solver,
        "solver_label": result.get("solver"),
        "integral_solver": spec.integral_solver,
        "linear_solver": result.get("linear_solver"),
        "solver_converged": result.get("solver_converged"),
        "solver_iterations": result.get("solver_iterations"),
        "stationary_residual_l1": result.get("stationary_residual_l1"),
        "state_count": result.get("state_count"),
        "generator_nnz": result.get("generator_nnz"),
        "memory_bytes": result.get("memory_bytes"),
        "memory_mb": _bytes_to_mb(result.get("memory_bytes")),
        "series_label": _series_label(spec, result),
    }


def _run_specs(args: argparse.Namespace) -> Iterable[RunSpec]:
    selected = set(args.methods)
    if "erlang" in selected:
        for solver in args.erlang_methods:
            yield RunSpec("erlang", solver)
    if "cme" in selected:
        for integral_solver in args.cme_integral_methods:
            for solver in args.cme_linear_methods:
                yield RunSpec("cme", solver, integral_solver)
    if "dual_bounds" in selected:
        for integral_solver in args.dual_bounds_integral_methods:
            for solver in args.dual_bounds_linear_methods:
                yield RunSpec("dual_bounds", solver, integral_solver)


def _resolved_cme_targets(targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = _available_cme_phases()
    resolved = []
    for target in targets:
        phases = int(target["phases"])
        if phases not in available:
            raise ValueError(f"CME phase {phases} is not available in methods/iltcme.json.")
        resolved.append({"name": str(target["name"]), "phases": phases})
    return resolved


def _available_cme_phases() -> set[int]:
    return set(_load_iltcme_table())


def _write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    started: float,
    *,
    write_plots: bool,
) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": _config_payload(args),
        "samples": samples,
        "failures": failures,
    }
    (output_dir / "arithmetic_accuracy_results.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "arithmetic_accuracy_samples.csv", samples)
    _write_csv(output_dir / "arithmetic_accuracy_failures.csv", failures)
    if write_plots:
        _write_plots(output_dir, args, samples)


def _config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "methods": args.methods,
        "lambda_values": args.lambda_values,
        "service_time": args.service_time,
        "capacity": args.capacity,
        "min_digits": args.min_digits,
        "max_digits": args.max_digits,
        "digit_step": args.digit_step,
        "digit_values": args.digit_values,
        "tolerance_rule": "10^(-precision_digits)",
        "erlang_phase_counts": args.erlang_phase_counts,
        "erlang_methods": args.erlang_methods,
        "erlang_max_iterations": args.erlang_max_iterations,
        "erlang_relaxation": args.erlang_relaxation,
        "cme_targets": args.cme_targets,
        "cme_integral_methods": args.cme_integral_methods,
        "cme_linear_methods": args.cme_linear_methods,
        "dual_bounds_variants": args.dual_bounds_variants,
        "dual_bounds_integral_methods": args.dual_bounds_integral_methods,
        "dual_bounds_linear_methods": args.dual_bounds_linear_methods,
        "dual_bounds_max_iterations": args.dual_bounds_max_iterations,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_plots(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
) -> None:
    if not samples:
        return

    for metric, ylabel, filename_stem in (
        ("p_full", "P(Full) (Log Scale)", "p_full_vs_digits"),
        ("p_idle", "P(Idle) (Log Scale)", "p_idle_vs_digits"),
        ("runtime_seconds", "Runtime (S) (Log Scale)", "runtime_vs_digits"),
    ):
        legend_location = _plot_legend_location(metric)
        (output_dir / f"{filename_stem}.svg").write_text(
            _arithmetic_accuracy_svg(metric, ylabel, args, samples, legend_location),
            encoding="utf-8",
        )
        (output_dir / f"{filename_stem}.pdf").write_bytes(
            _arithmetic_accuracy_pdf(metric, ylabel, args, samples, legend_location),
        )


def _arithmetic_accuracy_svg(
    metric: str,
    ylabel: str,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    legend_location: str | None = None,
) -> str:
    legend_location = legend_location or _plot_legend_location(metric)
    width = 1200
    panel_height = 1200
    lambda_values = [float(value) for value in args.lambda_values]
    height = panel_height * max(1, len(lambda_values))

    plot_left = 145.0
    plot_top_offset = 90.0
    plot_size = 860.0
    tick_length = 9.0
    text_color = "#222222"

    digit_values = [int(value) for value in getattr(args, "digit_values", [])]
    if not digit_values:
        digit_values = [
            int(row["precision_digits"])
            for row in samples
            if row.get("precision_digits") is not None
        ]
    x_lower, x_upper = _svg_axis_limits(digit_values)
    x_ticks = _integer_ticks(x_lower, x_upper, max_ticks=8)

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_svg_escape(ylabel)} versus {_svg_escape(X_AXIS_LABEL)}">'
        ),
        "<title>"
        + _svg_escape(f"{ylabel} versus {X_AXIS_LABEL}")
        + "</title>",
        "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>",
        "<style>"
        "text{font-family:Arial,Helvetica,sans-serif;fill:#222}"
        ".tick{font-size:20px}"
        ".axis-label{font-size:26px}"
        ".legend-label{font-size:18px}"
        "</style>",
    ]

    for panel_index, arrival_rate in enumerate(lambda_values):
        panel_top = panel_index * panel_height
        plot_top = panel_top + plot_top_offset
        series_rows = _plot_series_rows(samples, arrival_rate, metric)
        y_values = [value for _, points in series_rows for _, value in points]
        y_log_lower, y_log_upper, y_ticks, y_minor_ticks = _log_ticks(y_values, max_ticks=9)
        y_lower = 10**y_log_lower
        y_upper = 10**y_log_upper

        def x_to_px(value: float) -> float:
            return plot_left + ((value - x_lower) / (x_upper - x_lower)) * plot_size

        def y_to_px(value: float) -> float:
            return plot_top + ((y_log_upper - math.log10(value)) / (y_log_upper - y_log_lower)) * plot_size

        lines.extend(
            _svg_panel_axes(
                plot_left=plot_left,
                plot_top=plot_top,
                plot_size=plot_size,
                x_ticks=x_ticks,
                y_ticks=y_ticks,
                x_to_px=x_to_px,
                y_to_px=y_to_px,
                x_lower=x_lower,
                x_upper=x_upper,
                y_lower=y_lower,
                y_upper=y_upper,
                y_minor_ticks=y_minor_ticks,
                show_x_labels=panel_index == len(lambda_values) - 1,
                ylabel=ylabel,
                xlabel=X_AXIS_LABEL if panel_index == len(lambda_values) - 1 else None,
                tick_length=tick_length,
                text_color=text_color,
            )
        )

        for series_index, (series_label, points) in enumerate(series_rows):
            if not points:
                continue
            style = _svg_series_style(series_index, series_label)
            pixel_points = [(x_to_px(x), y_to_px(y)) for x, y in points]
            lines.append(_svg_polyline(pixel_points, style))
            for x_px, y_px in pixel_points:
                lines.append(_svg_marker(x_px, y_px, style, size=7.0))

        if series_rows:
            lines.extend(
                _svg_legend(
                    labels=[label for label, _ in series_rows],
                    plot_left=plot_left,
                    plot_top=plot_top,
                    plot_size=plot_size,
                    location=legend_location,
                )
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _arithmetic_accuracy_pdf(
    metric: str,
    ylabel: str,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    legend_location: str | None = None,
) -> bytes:
    legend_location = legend_location or _plot_legend_location(metric)
    width = 1200
    panel_height = 1200
    lambda_values = [float(value) for value in args.lambda_values]
    height = panel_height * max(1, len(lambda_values))

    plot_left = 145.0
    plot_top_offset = 90.0
    plot_size = 860.0
    tick_length = 9.0

    digit_values = [int(value) for value in getattr(args, "digit_values", [])]
    if not digit_values:
        digit_values = [
            int(row["precision_digits"])
            for row in samples
            if row.get("precision_digits") is not None
        ]
    x_lower, x_upper = _svg_axis_limits(digit_values)
    x_ticks = _integer_ticks(x_lower, x_upper, max_ticks=8)

    canvas = PdfCanvas(width, height)
    canvas.rect(0.0, 0.0, float(width), float(height), fill="#ffffff")

    for panel_index, arrival_rate in enumerate(lambda_values):
        panel_top = panel_index * panel_height
        plot_top = panel_top + plot_top_offset
        series_rows = _plot_series_rows(samples, arrival_rate, metric)
        y_values = [value for _, points in series_rows for _, value in points]
        y_log_lower, y_log_upper, y_ticks, y_minor_ticks = _log_ticks(y_values, max_ticks=9)
        y_lower = 10**y_log_lower
        y_upper = 10**y_log_upper

        def x_to_px(value: float) -> float:
            return plot_left + ((value - x_lower) / (x_upper - x_lower)) * plot_size

        def y_to_px(value: float) -> float:
            return plot_top + ((y_log_upper - math.log10(value)) / (y_log_upper - y_log_lower)) * plot_size

        _pdf_panel_axes(
            canvas=canvas,
            plot_left=plot_left,
            plot_top=plot_top,
            plot_size=plot_size,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            y_minor_ticks=y_minor_ticks,
            x_to_px=x_to_px,
            y_to_px=y_to_px,
            x_lower=x_lower,
            x_upper=x_upper,
            y_lower=y_lower,
            y_upper=y_upper,
            show_x_labels=panel_index == len(lambda_values) - 1,
            ylabel=ylabel,
            xlabel=X_AXIS_LABEL if panel_index == len(lambda_values) - 1 else None,
            tick_length=tick_length,
        )

        for series_index, (series_label, points) in enumerate(series_rows):
            if not points:
                continue
            style = _svg_series_style(series_index, series_label)
            pixel_points = [(x_to_px(x), y_to_px(y)) for x, y in points]
            canvas.polyline(pixel_points, stroke=style.color, width=2.4, dash=style.dash)
            for x_px, y_px in pixel_points:
                canvas.marker(x_px, y_px, style, size=7.0)

        if series_rows:
            _pdf_legend(
                canvas=canvas,
                labels=[label for label, _ in series_rows],
                plot_left=plot_left,
                plot_top=plot_top,
                plot_size=plot_size,
                location=legend_location,
            )

    return _pdf_document_bytes(width, height, canvas.content())


def _svg_panel_axes(
    *,
    plot_left: float,
    plot_top: float,
    plot_size: float,
    x_ticks: list[int],
    y_ticks: list[float],
    x_to_px: Callable[[float], float],
    y_to_px: Callable[[float], float],
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    y_minor_ticks: list[float],
    show_x_labels: bool,
    ylabel: str,
    xlabel: str | None,
    tick_length: float,
    text_color: str,
) -> list[str]:
    lines = [
        (
            f'<rect x="{_svg_num(plot_left)}" y="{_svg_num(plot_top)}" '
            f'width="{_svg_num(plot_size)}" height="{_svg_num(plot_size)}" '
            'fill="white"/>'
        )
    ]
    axis_bottom = plot_top + plot_size
    axis_right = plot_left + plot_size

    for tick in x_ticks:
        if tick < x_lower or tick > x_upper:
            continue
        x_px = x_to_px(tick)
        lines.append(
            f'<line x1="{_svg_num(x_px)}" y1="{_svg_num(axis_bottom)}" '
            f'x2="{_svg_num(x_px)}" y2="{_svg_num(axis_bottom + tick_length)}" '
            'stroke="#222222" stroke-width="1.4"/>'
        )
        if show_x_labels:
            lines.append(
                f'<text class="tick" x="{_svg_num(x_px)}" y="{_svg_num(axis_bottom + 34)}" '
                'text-anchor="middle">'
                f'{tick}</text>'
            )

    for tick in y_ticks:
        if tick < y_lower or tick > y_upper:
            continue
        y_px = y_to_px(tick)
        lines.append(
            f'<line x1="{_svg_num(plot_left - tick_length)}" y1="{_svg_num(y_px)}" '
            f'x2="{_svg_num(plot_left)}" y2="{_svg_num(y_px)}" '
            'stroke="#222222" stroke-width="1.4"/>'
        )
        lines.append(
            f'<text class="tick" x="{_svg_num(plot_left - tick_length - 8)}" '
            f'y="{_svg_num(y_px + 6)}" text-anchor="end">'
            f'{_svg_tick_label(tick)}</text>'
        )

    for tick in y_minor_ticks:
        if tick <= y_lower or tick >= y_upper:
            continue
        y_px = y_to_px(tick)
        lines.append(
            f'<line x1="{_svg_num(plot_left - tick_length * 0.55)}" y1="{_svg_num(y_px)}" '
            f'x2="{_svg_num(plot_left)}" y2="{_svg_num(y_px)}" '
            'stroke="#222222" stroke-width="1.0"/>'
        )

    lines.append(
        f'<text class="axis-label" x="{_svg_num(plot_left + plot_size / 2)}" '
        f'y="{_svg_num(axis_bottom + 78)}" text-anchor="middle">'
        f'{_svg_escape(xlabel)}</text>'
        if xlabel
        else ""
    )
    lines.append(
        f'<text class="axis-label" transform="translate({_svg_num(plot_left - 95)} '
        f'{_svg_num(plot_top + plot_size / 2)}) rotate(-90)" text-anchor="middle" '
        f'fill="{text_color}">{_svg_escape(ylabel)}</text>'
    )
    lines.append(
        f'<line x1="{_svg_num(plot_left)}" y1="{_svg_num(plot_top)}" '
        f'x2="{_svg_num(plot_left)}" y2="{_svg_num(axis_bottom)}" '
        'stroke="#222222" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{_svg_num(plot_left)}" y1="{_svg_num(axis_bottom)}" '
        f'x2="{_svg_num(axis_right)}" y2="{_svg_num(axis_bottom)}" '
        'stroke="#222222" stroke-width="1.5"/>'
    )
    return [line for line in lines if line]


def _svg_legend(
    *,
    labels: list[str],
    plot_left: float,
    plot_top: float,
    plot_size: float,
    location: str,
) -> list[str]:
    font_size = 18.0
    row_height = 25.0
    padding = 13.0
    width = max(150.0, max(8.2 * len(label) for label in labels) + 78.0)
    height = padding * 2 + row_height * len(labels)
    legend_left = plot_left + plot_size - width - 18.0
    legend_top = (
        plot_top + plot_size - height - 18.0
        if location == "lower_right"
        else plot_top + 18.0
    )

    lines = [
        f'<rect x="{_svg_num(legend_left)}" y="{_svg_num(legend_top)}" '
        f'width="{_svg_num(width)}" height="{_svg_num(height)}" '
        'fill="white" opacity="0.88" stroke="#bbbbbb" stroke-width="1"/>'
    ]
    for index, label in enumerate(labels):
        style = _svg_series_style(index, label)
        y = legend_top + padding + row_height * index + row_height / 2
        x1 = legend_left + padding
        x2 = x1 + 36.0
        marker_x = 0.5 * (x1 + x2)
        lines.append(
            f'<line x1="{_svg_num(x1)}" y1="{_svg_num(y)}" '
            f'x2="{_svg_num(x2)}" y2="{_svg_num(y)}" stroke="{style.color}" '
            f'stroke-width="2.4" stroke-linecap="round"'
            f'{_svg_dash_attr(style)}/>'
        )
        lines.append(_svg_marker(marker_x, y, style, size=6.5))
        lines.append(
            f'<text class="legend-label" font-size="{_svg_num(font_size)}" '
            f'x="{_svg_num(x2 + 12)}" y="{_svg_num(y + 5)}">'
            f'{_svg_escape(label)}</text>'
        )
    return lines


def _pdf_panel_axes(
    *,
    canvas: PdfCanvas,
    plot_left: float,
    plot_top: float,
    plot_size: float,
    x_ticks: list[int],
    y_ticks: list[float],
    y_minor_ticks: list[float],
    x_to_px: Callable[[float], float],
    y_to_px: Callable[[float], float],
    x_lower: float,
    x_upper: float,
    y_lower: float,
    y_upper: float,
    show_x_labels: bool,
    ylabel: str,
    xlabel: str | None,
    tick_length: float,
) -> None:
    canvas.rect(
        plot_left,
        plot_top,
        plot_size,
        plot_size,
        fill="#ffffff",
    )
    axis_bottom = plot_top + plot_size
    axis_right = plot_left + plot_size

    for tick in x_ticks:
        if tick < x_lower or tick > x_upper:
            continue
        x_px = x_to_px(tick)
        canvas.line(x_px, axis_bottom, x_px, axis_bottom + tick_length, width=1.4)
        if show_x_labels:
            canvas.text(str(tick), x_px, axis_bottom + 34.0, size=20.0, anchor="middle")

    for tick in y_ticks:
        if tick < y_lower or tick > y_upper:
            continue
        y_px = y_to_px(tick)
        canvas.line(plot_left - tick_length, y_px, plot_left, y_px, width=1.4)
        canvas.text(
            _svg_tick_label(tick),
            plot_left - tick_length - 8.0,
            y_px + 6.0,
            size=20.0,
            anchor="end",
        )

    for tick in y_minor_ticks:
        if tick <= y_lower or tick >= y_upper:
            continue
        y_px = y_to_px(tick)
        canvas.line(plot_left - tick_length * 0.55, y_px, plot_left, y_px, width=1.0)

    if xlabel:
        canvas.text(
            xlabel,
            plot_left + plot_size / 2,
            axis_bottom + 78.0,
            size=26.0,
            anchor="middle",
        )
    canvas.text(
        ylabel,
        plot_left - 95.0,
        plot_top + plot_size / 2,
        size=26.0,
        anchor="middle",
        rotate=-90,
    )
    canvas.line(plot_left, plot_top, plot_left, axis_bottom, width=1.5)
    canvas.line(plot_left, axis_bottom, axis_right, axis_bottom, width=1.5)


def _pdf_legend(
    *,
    canvas: PdfCanvas,
    labels: list[str],
    plot_left: float,
    plot_top: float,
    plot_size: float,
    location: str,
) -> None:
    row_height = 25.0
    padding = 13.0
    width = max(150.0, max(8.2 * len(label) for label in labels) + 78.0)
    height = padding * 2 + row_height * len(labels)
    legend_left = plot_left + plot_size - width - 18.0
    legend_top = (
        plot_top + plot_size - height - 18.0
        if location == "lower_right"
        else plot_top + 18.0
    )

    canvas.rect(
        legend_left,
        legend_top,
        width,
        height,
        fill="#ffffff",
        stroke="#bbbbbb",
        stroke_width=1.0,
    )
    for index, label in enumerate(labels):
        style = _svg_series_style(index, label)
        y = legend_top + padding + row_height * index + row_height / 2
        x1 = legend_left + padding
        x2 = x1 + 36.0
        marker_x = 0.5 * (x1 + x2)
        canvas.line(x1, y, x2, y, stroke=style.color, width=2.4, dash=style.dash)
        canvas.marker(marker_x, y, style, size=6.5)
        canvas.text(label, x2 + 12.0, y + 5.0, size=18.0)


def _plot_legend_location(metric: str) -> str:
    if metric == "runtime_seconds":
        return "lower_right"
    return "upper_right"


def _svg_polyline(points: list[tuple[float, float]], style: SvgSeriesStyle) -> str:
    path = " ".join(
        f'{"M" if index == 0 else "L"} {_svg_num(x)} {_svg_num(y)}'
        for index, (x, y) in enumerate(points)
    )
    return (
        f'<path d="{path}" fill="none" stroke="{style.color}" stroke-width="2.4" '
        f'stroke-linecap="round" stroke-linejoin="round"{_svg_dash_attr(style)}/>'
    )


def _svg_marker(x: float, y: float, style: SvgSeriesStyle, *, size: float) -> str:
    color = style.color
    if style.marker == "circle":
        return (
            f'<circle cx="{_svg_num(x)}" cy="{_svg_num(y)}" r="{_svg_num(size)}" '
            f'fill="white" stroke="{color}" stroke-width="2.1"/>'
        )
    if style.marker == "square":
        side = size * 1.7
        return (
            f'<rect x="{_svg_num(x - side / 2)}" y="{_svg_num(y - side / 2)}" '
            f'width="{_svg_num(side)}" height="{_svg_num(side)}" fill="white" '
            f'stroke="{color}" stroke-width="2.1"/>'
        )
    if style.marker == "triangle-up":
        points = [(x, y - size), (x + size, y + size * 0.85), (x - size, y + size * 0.85)]
        return _svg_polygon(points, color)
    if style.marker == "diamond":
        points = [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
        return _svg_polygon(points, color)
    if style.marker == "triangle-down":
        points = [(x - size, y - size * 0.85), (x + size, y - size * 0.85), (x, y + size)]
        return _svg_polygon(points, color)
    if style.marker == "plus":
        return (
            f'<path d="M {_svg_num(x - size)} {_svg_num(y)} L {_svg_num(x + size)} {_svg_num(y)} '
            f'M {_svg_num(x)} {_svg_num(y - size)} L {_svg_num(x)} {_svg_num(y + size)}" '
            f'fill="none" stroke="{color}" stroke-width="2.3" stroke-linecap="round"/>'
        )
    if style.marker == "x":
        return (
            f'<path d="M {_svg_num(x - size)} {_svg_num(y - size)} L {_svg_num(x + size)} {_svg_num(y + size)} '
            f'M {_svg_num(x + size)} {_svg_num(y - size)} L {_svg_num(x - size)} {_svg_num(y + size)}" '
            f'fill="none" stroke="{color}" stroke-width="2.3" stroke-linecap="round"/>'
        )
    if style.marker == "star":
        return _svg_polygon(_star_points(x, y, size * 1.12, size * 0.45), color)
    if style.marker == "triangle-left":
        points = [(x - size, y), (x + size * 0.85, y - size), (x + size * 0.85, y + size)]
        return _svg_polygon(points, color)
    if style.marker == "triangle-right":
        points = [(x + size, y), (x - size * 0.85, y - size), (x - size * 0.85, y + size)]
        return _svg_polygon(points, color)
    return (
        f'<circle cx="{_svg_num(x)}" cy="{_svg_num(y)}" r="{_svg_num(size)}" '
        f'fill="white" stroke="{color}" stroke-width="2.1"/>'
    )


def _svg_polygon(points: list[tuple[float, float]], color: str) -> str:
    points_text = " ".join(f"{_svg_num(x)},{_svg_num(y)}" for x, y in points)
    return f'<polygon points="{points_text}" fill="white" stroke="{color}" stroke-width="2.1"/>'


def _star_points(x: float, y: float, outer: float, inner: float) -> list[tuple[float, float]]:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        radius = outer if index % 2 == 0 else inner
        points.append((x + math.cos(angle) * radius, y + math.sin(angle) * radius))
    return points


def _svg_series_style(index: int, label: str | None = None) -> SvgSeriesStyle:
    normalized_label = str(label or "").replace("_", " ").strip().lower()
    first_word = normalized_label.split(maxsplit=1)[0] if normalized_label else ""
    if first_word in {"slow", "slower"}:
        return SvgSeriesStyle(color="#ff0000", dash="10 5", marker="triangle-down")
    if first_word in {"fast", "faster"}:
        return SvgSeriesStyle(color="#0000ff", dash="3 3", marker="triangle-up")
    if first_word == "cme":
        return SvgSeriesStyle(color="#008000", dash="8 4 2 4", marker="square")
    if first_word == "erlang":
        return SvgSeriesStyle(color="#808080", dash="1 4", marker="circle")

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    dashes = [
        None,
        "9 5",
        "2 4",
        "9 4 2 4",
        "13 5 3 5",
        "5 4",
        "2 3 8 3",
        "1 4",
        "11 3 2 3 2 3",
        "7 3 2 3",
    ]
    markers = [
        "circle",
        "square",
        "triangle-up",
        "diamond",
        "triangle-down",
        "plus",
        "x",
        "star",
        "triangle-left",
        "triangle-right",
    ]
    return SvgSeriesStyle(
        color=colors[index % len(colors)],
        dash=dashes[index % len(dashes)],
        marker=markers[index % len(markers)],
    )


def _svg_dash_attr(style: SvgSeriesStyle) -> str:
    if style.dash is None:
        return ""
    return f' stroke-dasharray="{style.dash}"'


class PdfCanvas:
    def __init__(self, width: int, height: int) -> None:
        self.width = float(width)
        self.height = float(height)
        self._commands: list[str] = []

    def content(self) -> str:
        return "\n".join(self._commands) + "\n"

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str = "#222222",
        width: float = 1.0,
        dash: str | None = None,
    ) -> None:
        self._commands.append(
            "q\n"
            f"{_pdf_stroke_state(stroke, width, dash)}"
            f"{_pdf_num(x1)} {_pdf_num(self._pdf_y(y1))} m "
            f"{_pdf_num(x2)} {_pdf_num(self._pdf_y(y2))} l S\n"
            "Q"
        )

    def rect(
        self,
        x: float,
        y: float,
        rect_width: float,
        rect_height: float,
        *,
        fill: str | None = None,
        stroke: str | None = None,
        stroke_width: float = 1.0,
    ) -> None:
        operator = "B" if fill and stroke else "f" if fill else "S"
        state = ""
        if fill:
            state += _pdf_fill_color(fill)
        if stroke:
            state += _pdf_stroke_state(stroke, stroke_width, None)
        self._commands.append(
            "q\n"
            f"{state}"
            f"{_pdf_num(x)} {_pdf_num(self._pdf_y(y + rect_height))} "
            f"{_pdf_num(rect_width)} {_pdf_num(rect_height)} re {operator}\n"
            "Q"
        )

    def polyline(
        self,
        points: list[tuple[float, float]],
        *,
        stroke: str,
        width: float,
        dash: str | None,
    ) -> None:
        if len(points) < 2:
            return
        path_parts = [
            f"{_pdf_num(points[0][0])} {_pdf_num(self._pdf_y(points[0][1]))} m"
        ]
        path_parts.extend(
            f"{_pdf_num(x)} {_pdf_num(self._pdf_y(y))} l"
            for x, y in points[1:]
        )
        self._commands.append(
            "q\n"
            f"{_pdf_stroke_state(stroke, width, dash)}"
            f"{' '.join(path_parts)} S\n"
            "Q"
        )

    def polygon(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        stroke: str,
        width: float,
    ) -> None:
        if not points:
            return
        path_parts = [
            f"{_pdf_num(points[0][0])} {_pdf_num(self._pdf_y(points[0][1]))} m"
        ]
        path_parts.extend(
            f"{_pdf_num(x)} {_pdf_num(self._pdf_y(y))} l"
            for x, y in points[1:]
        )
        self._commands.append(
            "q\n"
            f"{_pdf_fill_color(fill)}{_pdf_stroke_state(stroke, width, None)}"
            f"{' '.join(path_parts)} h B\n"
            "Q"
        )

    def circle(
        self,
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str,
        width: float,
    ) -> None:
        y_pdf = self._pdf_y(y)
        kappa = 0.5522847498307936
        control = radius * kappa
        path = (
            f"{_pdf_num(x + radius)} {_pdf_num(y_pdf)} m "
            f"{_pdf_num(x + radius)} {_pdf_num(y_pdf + control)} "
            f"{_pdf_num(x + control)} {_pdf_num(y_pdf + radius)} "
            f"{_pdf_num(x)} {_pdf_num(y_pdf + radius)} c "
            f"{_pdf_num(x - control)} {_pdf_num(y_pdf + radius)} "
            f"{_pdf_num(x - radius)} {_pdf_num(y_pdf + control)} "
            f"{_pdf_num(x - radius)} {_pdf_num(y_pdf)} c "
            f"{_pdf_num(x - radius)} {_pdf_num(y_pdf - control)} "
            f"{_pdf_num(x - control)} {_pdf_num(y_pdf - radius)} "
            f"{_pdf_num(x)} {_pdf_num(y_pdf - radius)} c "
            f"{_pdf_num(x + control)} {_pdf_num(y_pdf - radius)} "
            f"{_pdf_num(x + radius)} {_pdf_num(y_pdf - control)} "
            f"{_pdf_num(x + radius)} {_pdf_num(y_pdf)} c"
        )
        self._commands.append(
            "q\n"
            f"{_pdf_fill_color(fill)}{_pdf_stroke_state(stroke, width, None)}"
            f"{path} B\n"
            "Q"
        )

    def marker(self, x: float, y: float, style: SvgSeriesStyle, *, size: float) -> None:
        color = style.color
        if style.marker == "circle":
            self.circle(x, y, size, fill="#ffffff", stroke=color, width=2.1)
        elif style.marker == "square":
            side = size * 1.7
            self.rect(
                x - side / 2,
                y - side / 2,
                side,
                side,
                fill="#ffffff",
                stroke=color,
                stroke_width=2.1,
            )
        elif style.marker == "triangle-up":
            self.polygon(
                [(x, y - size), (x + size, y + size * 0.85), (x - size, y + size * 0.85)],
                fill="#ffffff",
                stroke=color,
                width=2.1,
            )
        elif style.marker == "diamond":
            self.polygon(
                [(x, y - size), (x + size, y), (x, y + size), (x - size, y)],
                fill="#ffffff",
                stroke=color,
                width=2.1,
            )
        elif style.marker == "triangle-down":
            self.polygon(
                [(x - size, y - size * 0.85), (x + size, y - size * 0.85), (x, y + size)],
                fill="#ffffff",
                stroke=color,
                width=2.1,
            )
        elif style.marker == "plus":
            self.line(x - size, y, x + size, y, stroke=color, width=2.3)
            self.line(x, y - size, x, y + size, stroke=color, width=2.3)
        elif style.marker == "x":
            self.line(x - size, y - size, x + size, y + size, stroke=color, width=2.3)
            self.line(x + size, y - size, x - size, y + size, stroke=color, width=2.3)
        elif style.marker == "star":
            self.polygon(_star_points(x, y, size * 1.12, size * 0.45), fill="#ffffff", stroke=color, width=2.1)
        elif style.marker == "triangle-left":
            self.polygon(
                [(x - size, y), (x + size * 0.85, y - size), (x + size * 0.85, y + size)],
                fill="#ffffff",
                stroke=color,
                width=2.1,
            )
        elif style.marker == "triangle-right":
            self.polygon(
                [(x + size, y), (x - size * 0.85, y - size), (x - size * 0.85, y + size)],
                fill="#ffffff",
                stroke=color,
                width=2.1,
            )
        else:
            self.circle(x, y, size, fill="#ffffff", stroke=color, width=2.1)

    def text(
        self,
        value: str,
        x: float,
        y: float,
        *,
        size: float,
        anchor: str = "start",
        rotate: int | None = None,
        fill: str = "#222222",
    ) -> None:
        escaped = _pdf_escape_text(value)
        width = _pdf_text_width(value, size)
        if rotate == -90:
            text_y = self._pdf_y(y)
            if anchor == "middle":
                text_y -= width / 2
            elif anchor == "end":
                text_y -= width
            matrix = f"0 1 -1 0 {_pdf_num(x)} {_pdf_num(text_y)} Tm"
        else:
            text_x = x
            if anchor == "middle":
                text_x -= width / 2
            elif anchor == "end":
                text_x -= width
            matrix = f"1 0 0 1 {_pdf_num(text_x)} {_pdf_num(self._pdf_y(y))} Tm"
        self._commands.append(
            "q\n"
            f"{_pdf_fill_color(fill)}"
            "BT\n"
            f"/F1 {_pdf_num(size)} Tf\n"
            f"{matrix}\n"
            f"({escaped}) Tj\n"
            "ET\n"
            "Q"
        )

    def _pdf_y(self, y: float) -> float:
        return self.height - y


def _svg_axis_limits(values: list[int]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lower = float(min(values))
    upper = float(max(values))
    if lower == upper:
        return lower - 0.5, upper + 0.5
    padding = max(0.5, 0.04 * (upper - lower))
    return lower - padding, upper + padding


def _integer_ticks(lower: float, upper: float, *, max_ticks: int) -> list[int]:
    start = math.ceil(lower)
    end = math.floor(upper)
    if end < start:
        return [round(0.5 * (lower + upper))]
    count = end - start + 1
    step = max(1, math.ceil(count / max_ticks))
    ticks = list(range(start, end + 1, step))
    if ticks[-1] != end:
        ticks.append(end)
    return ticks


def _log_ticks(values: list[float], *, max_ticks: int) -> tuple[float, float, list[float], list[float]]:
    positive_values = [value for value in values if math.isfinite(value) and value > 0.0]
    if not positive_values:
        return -1.0, 1.0, [0.1, 1.0, 10.0], [value / 10 for value in range(2, 10)]

    log_values = [math.log10(value) for value in positive_values]
    log_lower = min(log_values)
    log_upper = max(log_values)
    if log_lower == log_upper:
        log_lower -= 0.5
        log_upper += 0.5
    else:
        padding = 0.05 * (log_upper - log_lower)
        log_lower -= padding
        log_upper += padding

    major_ticks = _log_major_ticks(log_lower, log_upper, max_ticks=max_ticks)

    minor_ticks: list[float] = []
    if log_upper - log_lower <= 12:
        for exponent in range(math.floor(log_lower), math.ceil(log_upper) + 1):
            decade = 10.0**exponent
            for multiplier in range(2, 10):
                tick = multiplier * decade
                log_tick = math.log10(tick)
                if log_lower <= log_tick <= log_upper and tick not in major_ticks:
                    minor_ticks.append(tick)
    return log_lower, log_upper, major_ticks, minor_ticks


def _log_major_ticks(log_lower: float, log_upper: float, *, max_ticks: int) -> list[float]:
    span = log_upper - log_lower
    multipliers = [1] if span > 4 else [1, 2, 5]
    candidates: list[float] = []
    for exponent in range(math.floor(log_lower), math.ceil(log_upper) + 1):
        decade = 10.0**exponent
        for multiplier in multipliers:
            tick = multiplier * decade
            log_tick = math.log10(tick)
            if log_lower <= log_tick <= log_upper:
                candidates.append(tick)

    candidates = sorted(set(candidates))
    if not candidates:
        tick_count = min(max_ticks, 5)
        if tick_count <= 1:
            return [10.0 ** (0.5 * (log_lower + log_upper))]
        step = span / (tick_count - 1)
        return [10.0 ** (log_lower + step * index) for index in range(tick_count)]

    if len(candidates) <= max_ticks:
        return candidates

    step = math.ceil(len(candidates) / max_ticks)
    ticks = candidates[::step]
    if ticks[-1] != candidates[-1]:
        ticks.append(candidates[-1])
    return ticks


def _svg_tick_label(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10000 or magnitude < 0.001:
        return f"{value:.2e}"
    return f"{value:.6g}"


def _svg_num(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pdf_document_bytes(width: int, height: int, content: str) -> bytes:
    content_bytes = content.encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(content_bytes)} >>\nstream\n".encode("ascii")
            + content_bytes
            + b"endstream"
        ),
    ]

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _pdf_stroke_state(color: str, width: float, dash: str | None) -> str:
    dash_text = f"[{dash}] 0 d\n" if dash else "[] 0 d\n"
    return f"{_pdf_stroke_color(color)}{_pdf_num(width)} w\n1 J\n1 j\n{dash_text}"


def _pdf_stroke_color(color: str) -> str:
    red, green, blue = _pdf_rgb(color)
    return f"{red} {green} {blue} RG\n"


def _pdf_fill_color(color: str) -> str:
    red, green, blue = _pdf_rgb(color)
    return f"{red} {green} {blue} rg\n"


def _pdf_rgb(color: str) -> tuple[str, str, str]:
    text = color.lstrip("#")
    if len(text) != 6:
        return "0", "0", "0"
    values = [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    return tuple(_pdf_num(value) for value in values)  # type: ignore[return-value]


def _pdf_escape_text(value: Any) -> str:
    text = str(value).encode("latin-1", errors="replace").decode("latin-1")
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text_width(value: Any, font_size: float) -> float:
    total = 0.0
    for char in str(value):
        if char in "ilI1.,:;|":
            total += 0.28
        elif char in "MW@":
            total += 0.9
        elif char == " ":
            total += 0.32
        else:
            total += 0.56
    return total * font_size


def _pdf_num(value: float) -> str:
    return _svg_num(value)


def _plot_series_rows(
    samples: list[dict[str, Any]],
    arrival_rate: float,
    metric: str,
) -> list[tuple[str, list[tuple[int, float]]]]:
    series_rows: list[tuple[str, list[tuple[int, float]]]] = []
    for label, rows in _plot_groups(samples, arrival_rate).items():
        points: list[tuple[int, float]] = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            y_value = float(value)
            if not math.isfinite(y_value) or y_value <= 0.0:
                continue
            points.append((int(row["precision_digits"]), y_value))
        series_rows.append((label, points))
    return series_rows


def _plot_groups(
    samples: list[dict[str, Any]],
    arrival_rate: float,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        if float(row["arrival_rate"]) != float(arrival_rate):
            continue
        groups.setdefault(_plot_label(row), []).append(row)

    for rows in groups.values():
        rows.sort(key=lambda item: int(item["precision_digits"]))
    return dict(sorted(groups.items()))


def _plot_label(row: dict[str, Any]) -> str:
    for key in ("approach", "phase_target", "variant", "method"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return _display_method_label(str(value))
    return "Series"


def _series_label(spec: RunSpec, result: dict[str, Any]) -> str:
    approach = result.get("approach")
    if approach:
        return _display_method_label(str(approach))
    return _display_method_label(spec.method)


def _display_method_label(label: str) -> str:
    words = str(label).replace("_", " ").split()
    if not words:
        return str(label)

    first = words[0].lower()
    if first == "faster":
        words[0] = "Fast"
    elif first == "slower":
        words[0] = "Slow"
    else:
        words[0] = words[0].title()

    return " ".join([words[0], *words[1:]])


def _variant_label(spec: RunSpec, result: dict[str, Any]) -> str | None:
    if spec.method == "dual_bounds":
        return str(result.get("approach")) if result.get("approach") is not None else None
    return None


def _phase_target_label(spec: RunSpec, result: dict[str, Any]) -> str | None:
    if spec.method in {"erlang", "cme"}:
        return str(result.get("approach")) if result.get("approach") is not None else None
    return None


def _failure_row(
    spec: RunSpec,
    digits: int,
    arrival_rate: float,
    error_type: str,
    message: str,
    exc: BaseException | None,
) -> dict[str, Any]:
    row = {
        "method": spec.method,
        "solver": spec.solver,
        "integral_solver": spec.integral_solver,
        "precision_digits": digits,
        "tolerance": _precision_tolerance(digits),
        "arrival_rate": arrival_rate,
        "error_type": error_type,
        "message": message,
    }
    if exc is not None:
        row["traceback"] = traceback.format_exc()
    return row


def _digit_values(min_digits: int, max_digits: int, step: int) -> list[int]:
    values = list(range(min_digits, max_digits + 1, step))
    if values[-1] != max_digits:
        values.append(max_digits)
    return values


def _bytes_to_mb(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / (1024.0**2)
    except (TypeError, ValueError):
        return math.nan


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, list | tuple | dict):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.12g}"


def _parse_targets(text: str) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in _split_csv(text):
        if ":" in item:
            name, phase_text = item.split(":", 1)
            phases = int(phase_text)
            targets.append({"name": name.strip(), "phases": phases})
        else:
            phases = int(item)
            targets.append({"name": f"CME {phases}", "phases": phases})
    return targets


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep residual precision parameter d for CME, Erlang, and dual-bounds "
            "methods and record p_idle, p_full, and runtime."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("erlang", "cme", "dual_bounds"),
        default=[
            method
            for method, enabled in (
                ("erlang", RUN_ERLANG),
                ("cme", RUN_CME),
                ("dual_bounds", RUN_DUAL_BOUNDS),
            )
            if enabled
        ],
    )
    parser.add_argument("--lambda-values", type=float, nargs="+", default=LAMBDA_VALUES)
    parser.add_argument("--service-time", type=float, default=SERVICE_TIME)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--min-digits", type=int, default=MIN_DIGITS)
    parser.add_argument("--max-digits", type=int, default=MAX_DIGITS)
    parser.add_argument("--digit-step", type=int, default=DIGIT_STEP)

    parser.add_argument("--erlang-phase-counts", type=int, nargs="+", default=ERLANG_PHASE_COUNTS)
    parser.add_argument("--erlang-methods", nargs="+", default=ERLANG_STATIONARY_METHODS)
    parser.add_argument("--erlang-max-iterations", type=int, default=ERLANG_MAX_ITERATIONS)
    parser.add_argument("--erlang-relaxation", type=float, default=ERLANG_RELAXATION)

    parser.add_argument(
        "--cme-targets",
        type=_parse_targets,
        default=CME_TARGETS,
        help="Comma-separated CME phases, optionally name:phase. Example: 5,10,CME 15:15",
    )
    parser.add_argument("--cme-integral-methods", nargs="+", default=CME_INTEGRAL_METHODS)
    parser.add_argument("--cme-linear-methods", nargs="+", default=CME_LINEAR_METHODS)

    parser.add_argument("--dual-bounds-variants", nargs="+", default=DUAL_BOUNDS_VARIANTS)
    parser.add_argument("--dual-bounds-integral-methods", nargs="+", default=DUAL_BOUNDS_INTEGRAL_METHODS)
    parser.add_argument("--dual-bounds-linear-methods", nargs="+", default=DUAL_BOUNDS_LINEAR_METHODS)
    parser.add_argument("--dual-bounds-max-iterations", type=int, default=DUAL_BOUNDS_MAX_ITERATIONS)

    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if not args.methods:
        raise ValueError("At least one method must be selected.")
    if args.min_digits <= 0:
        raise ValueError("--min-digits must be positive.")
    if args.max_digits < args.min_digits:
        raise ValueError("--max-digits must be greater than or equal to --min-digits.")
    if args.digit_step <= 0:
        raise ValueError("--digit-step must be positive.")
    if args.capacity < 0:
        raise ValueError("--capacity must be non-negative.")
    if args.service_time <= 0.0:
        raise ValueError("--service-time must be positive.")
    if any(float(value) < 0.0 for value in args.lambda_values):
        raise ValueError("--lambda-values must be non-negative.")

    args.digit_values = _digit_values(args.min_digits, args.max_digits, args.digit_step)
    return args


if __name__ == "__main__":
    main()
