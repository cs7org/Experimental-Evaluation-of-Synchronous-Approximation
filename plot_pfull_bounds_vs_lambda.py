from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp
import numpy as np
import plot_distribution_accuracy as base
import scipy.sparse as sp
from methods.embedded_one_server import solve_erlang as solve_one_server_erlang
from methods import pure_erlang as pure_erlang_method
from methods.fast_slow import solve as solve_fast_slow


SERVICE_TIME = 1.0
CAPACITY = 10
ERLANG10_HIGH_PRECISION_DIGITS = 30
ERLANG10_HIGH_PRECISION_MAX_LAMBDA = 0.10
ERLANG10_POSITIVE_ITERATIONS = 5_000
LAMBDA_VALUES = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5]
LAMBDA_VALUES_0_1_TO_1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDA_VALUES_0_01_TO_2 = [0.01] + [round(0.1 * index, 1) for index in range(1, 21)]
LAMBDA_VALUES_0_01_TO_0_3 = [round(0.01 * index, 2) for index in range(1, 31)]
LAMBDA_VALUES_0_2_TO_0_3 = [round(0.01 * index, 2) for index in range(20, 31)]
ZOOM_LAMBDA_VALUES = [0.2, 0.2125, 0.225, 0.2375, 0.25, 0.2625, 0.275, 0.2875, 0.3]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

VARIANTS = ["faster", "slower", "faster 10", "slower 10"]
FULL_NO_BAND_LABELS = [
    "Exponential",
    "Erlang 10",
    "Erlang 10/D",
    "CME 10",
    "faster w/o",
    "average w/o",
    "slower w/o",
    "faster 10",
    "slower 10",
]
COMPACT_NO_BAND_LABELS = [
    "Exponential",
    "Erlang 10/D",
    "faster w/o",
    "slower w/o",
    "faster 10",
    "slower 10",
]
FIG3_FINAL_LABELS = [
    "Exponential",
    "Erlang 10",
    "CME 10",
    "faster w/o",
    "average w/o",
    "slower w/o",
    "faster 10",
    "slower 10",
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_plot_set(
        output_stem="pfull_vs_lambda_bounds",
        lambda_values=LAMBDA_VALUES,
        title=f"P(full) bounds by Lambda for K={CAPACITY}",
    )
    _write_plot_set(
        output_stem="pfull_vs_lambda_bounds_0_1_1",
        lambda_values=LAMBDA_VALUES_0_1_TO_1,
        title=f"P(full) bounds by Lambda for K={CAPACITY} (0.1 to 1)",
    )
    _write_plot_set(
        output_stem="pfull_vs_lambda_bounds_zoom_0_2_0_3",
        lambda_values=ZOOM_LAMBDA_VALUES,
        title=f"P(full) bounds by Lambda for K={CAPACITY} (zoomed)",
    )
    _write_no_band_plot_set(
        output_stem="pfull_vs_lambda_0_01_2_no_band",
        lambda_values=LAMBDA_VALUES_0_01_TO_2,
        title=f"P(full) by Lambda for K={CAPACITY} (0.01 to 2)",
        labels=FIG3_FINAL_LABELS,
        show_bands=True,
        show_inner_band=False,
        simple_outer_band=True,
        simple_outer_band_opacity=0.34,
        show_band_legend=False,
        tick_font_size=19,
        axis_font_size=26,
        legend_font_size=22,
        legend_row_gap=42,
    )
    _write_no_band_plot_set(
        output_stem="pfull_vs_lambda_0_01_2_no_band_embedded",
        lambda_values=LAMBDA_VALUES_0_01_TO_2,
        title=f"P(full) by Lambda for K={CAPACITY} (embedded focus)",
        labels=COMPACT_NO_BAND_LABELS,
    )
    _write_no_band_plot_set(
        output_stem="pfull_vs_lambda_0_01_0_3_no_band_logx",
        lambda_values=LAMBDA_VALUES_0_2_TO_0_3,
        title=f"P(full) by Lambda for K={CAPACITY} (0.2 to 0.3)",
        x_log_scale=True,
        high_precision_erlang10=True,
        labels=FULL_NO_BAND_LABELS,
        show_bands=True,
        show_inner_band=False,
        simple_outer_band=True,
        simple_outer_band_opacity=0.34,
        show_band_legend=False,
        tick_font_size=19,
        axis_font_size=26,
        legend_font_size=22,
        legend_row_gap=42,
    )


def _write_plot_set(*, output_stem: str, lambda_values: list[float], title: str) -> None:
    series = _compute_series(lambda_values)
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    summary_path = OUTPUT_DIR / f"{output_stem}_summary.txt"

    _write_svg(svg_path, series, title=title)
    csv_path.write_text(_build_csv(series), encoding="utf-8")
    summary_path.write_text(_build_summary(series), encoding="utf-8")

    print(f"Saved {svg_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


def _write_no_band_plot_set(
    *,
    output_stem: str,
    lambda_values: list[float],
    title: str,
    labels: list[str],
    x_log_scale: bool = False,
    high_precision_erlang10: bool = False,
    show_bands: bool = False,
    show_inner_band: bool = True,
    simple_outer_band: bool = False,
    simple_outer_band_opacity: float = 0.20,
    show_band_legend: bool = True,
    tick_font_size: int = 13,
    axis_font_size: int = 17,
    legend_font_size: float = 13.5,
    legend_row_gap: int = 32,
) -> None:
    all_series = _compute_series(
        lambda_values,
        include_exponential=True,
        include_erlang2=False,
        include_erlang10_d=True,
        high_precision_erlang10=high_precision_erlang10,
    )
    series = {label: all_series[label] for label in labels}
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    summary_path = OUTPUT_DIR / f"{output_stem}_summary.txt"

    _write_svg(
        svg_path,
        series,
        title=title,
        labels=labels,
        show_bands=show_bands,
        show_inner_band=show_inner_band,
        simple_outer_band=simple_outer_band,
        simple_outer_band_opacity=simple_outer_band_opacity,
        show_band_legend=show_band_legend,
        tick_font_size=tick_font_size,
        axis_font_size=axis_font_size,
        legend_font_size=legend_font_size,
        legend_row_gap=legend_row_gap,
        x_log_scale=x_log_scale,
        subtitle=(
            "Exponential, Erlang 10, CME 10, faster/slower without phase expansion, "
            "and faster/slower 10"
            if show_bands
            else "Exponential, Erlang 10, Erlang 10/D, CME 10, faster/slower without phase expansion, and faster/slower 10"
        ),
    )
    csv_path.write_text(_build_csv(series), encoding="utf-8")
    summary_path.write_text(_build_table_summary(series, "P(full) plot by lambda without band visualization"), encoding="utf-8")

    print(f"Saved {svg_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {summary_path}")


def _compute_series(
    lambda_values: list[float],
    *,
    include_exponential: bool = False,
    include_erlang2: bool = True,
    include_erlang10_d: bool = False,
    high_precision_erlang10: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    series = {
        "Erlang 10": [],
        "Erlang 10/D": [],
        "faster w/o": [],
        "slower w/o": [],
        "faster 10": [],
        "slower 10": [],
        "average w/o": [],
        "CME 10": [],
    }
    if include_exponential:
        series = {"Exponential": [], **series}
    if include_erlang2:
        series = {"Erlang 2": [], **series}

    for lam in lambda_values:
        if include_exponential:
            exp_dist = base._exponential_distribution(lam)
            series["Exponential"].append((lam, float(exp_dist[-1])))

        if include_erlang2:
            erlang2_dist = base._erlang_distribution(lam, 2)
            series["Erlang 2"].append((lam, float(erlang2_dist[-1])))

        if high_precision_erlang10 and lam <= ERLANG10_HIGH_PRECISION_MAX_LAMBDA:
            erlang_dist = _positive_erlang10_distribution(lam)
        else:
            erlang_dist = base._erlang_distribution(lam, 10)
        series["Erlang 10"].append((lam, float(erlang_dist[-1])))

        if include_erlang10_d:
            erlang10d_solver = "positive" if lam <= ERLANG10_HIGH_PRECISION_MAX_LAMBDA else "fast"
            erlang10d_digits = ERLANG10_HIGH_PRECISION_DIGITS if erlang10d_solver == "positive" else 80
            erlang10d_row = solve_one_server_erlang(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                phase_count=10,
                precision_digits=erlang10d_digits,
                solver=erlang10d_solver,
            )[0]
            series["Erlang 10/D"].append((lam, float(erlang10d_row["p_full"])))

        without_solver = "mp_direct" if lam <= 0.10 else "direct"
        without_digits = 30 if lam <= 0.10 else 80
        fast_slow_rows = solve_fast_slow(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            variants=["faster", "slower"],
            precision_digits=without_digits,
            solver=without_solver,
        )
        fast_slow_rows.extend(solve_fast_slow(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            variants=["faster 10", "slower 10"],
            precision_digits=80,
        ))
        for row in fast_slow_rows:
            series[str(row["approach"])].append((lam, float(row["p_full"])))

        fast_slow_by_name = {str(row["approach"]): row for row in fast_slow_rows}
        average_value = 0.5 * (
            float(fast_slow_by_name["faster w/o"]["p_full"])
            + float(fast_slow_by_name["slower w/o"]["p_full"])
        )
        series["average w/o"].append((lam, average_value))

        cme_dist = base._cme_distribution(10, lam)
        series["CME 10"].append((lam, float(cme_dist[-1])))

    return series


def _positive_erlang10_distribution(arrival_rate: float) -> np.ndarray:
    phase_count = 10
    generator = pure_erlang_method._build_generator(
        arrival_rate,
        SERVICE_TIME,
        CAPACITY,
        phase_count,
    ).tocsr()
    probability = _solve_stationary_positive_iteration(
        generator,
        iterations=ERLANG10_POSITIVE_ITERATIONS,
    )
    return pure_erlang_method._aggregate(probability, CAPACITY, phase_count)


def _solve_stationary_positive_iteration(
    generator: sp.spmatrix,
    *,
    iterations: int,
) -> np.ndarray:
    generator = generator.tocsr()
    n = generator.shape[0]
    uniformization_rate = float((-generator.diagonal()).max())
    transition = (sp.eye(n, format="csr") + generator / uniformization_rate).tocsr()

    probability = np.zeros(n, dtype=float)
    probability[0] = 1.0
    for _ in range(iterations):
        probability = probability @ transition

    probability = np.real_if_close(probability, tol=1000)
    probability = np.asarray(probability, dtype=float)
    probability[probability < 0.0] = 0.0
    total = float(probability.sum())
    if total <= 0.0:
        raise RuntimeError("Positive Erlang 10 iteration produced an invalid stationary distribution.")
    return probability / total


def _high_precision_erlang10_distribution(
    arrival_rate: float,
    precision_digits: int = ERLANG10_HIGH_PRECISION_DIGITS,
) -> np.ndarray:
    phase_count = 10
    generator = pure_erlang_method._build_generator(
        arrival_rate,
        SERVICE_TIME,
        CAPACITY,
        phase_count,
    )
    probability = _solve_stationary_high_precision(generator, precision_digits)
    return pure_erlang_method._aggregate(probability, CAPACITY, phase_count)


def _solve_stationary_high_precision(generator, precision_digits: int) -> np.ndarray:
    dense = generator.transpose().toarray()
    n = dense.shape[0]

    with mp.workdps(precision_digits):
        system = mp.matrix(n, n)
        rhs = mp.matrix(n, 1)
        for row in range(n):
            rhs[row] = mp.mpf("0")
            for col in range(n):
                system[row, col] = mp.mpf(str(dense[row, col]))

        for col in range(n):
            system[n - 1, col] = mp.mpf("1")
        rhs[n - 1] = mp.mpf("1")

        solution = mp.lu_solve(system, rhs)
        total = mp.fsum(solution[row] for row in range(n))
        if total <= 0:
            raise RuntimeError("High-precision Erlang 10 solve produced an invalid distribution.")
        probability = np.array(
            [
                float(max(solution[row] / total, mp.mpf("0")))
                for row in range(n)
            ],
            dtype=float,
        )
    return probability


def _build_csv(series: dict[str, list[tuple[float, float]]]) -> str:
    lines = ["lambda,series,p_full"]
    for label, points in series.items():
        for lam, value in points:
            lines.append(f"{lam},{label},{value:.18e}")
    return "\n".join(lines) + "\n"


def _build_summary(series: dict[str, list[tuple[float, float]]]) -> str:
    value_map = {
        label: {lam: value for lam, value in points}
        for label, points in series.items()
    }
    lambdas = sorted(value_map["CME 10"])

    lines = [
        "P(full) boundary plot by lambda",
        "Outer band: faster w/o to slower w/o",
        "Inner band: faster 10 to slower 10",
        "",
        "lambda,Erlang 2,Erlang 10,outer_lower,outer_upper,inner_lower,inner_upper,CME 10",
    ]
    for lam in lambdas:
        outer_values = [value_map["faster w/o"][lam], value_map["slower w/o"][lam]]
        inner_values = [value_map["faster 10"][lam], value_map["slower 10"][lam]]
        lines.append(
            ",".join(
                [
                    f"{lam:g}",
                    f"{value_map['Erlang 2'][lam]:.18e}",
                    f"{value_map['Erlang 10'][lam]:.18e}",
                    f"{min(outer_values):.18e}",
                    f"{max(outer_values):.18e}",
                    f"{min(inner_values):.18e}",
                    f"{max(inner_values):.18e}",
                    f"{value_map['CME 10'][lam]:.18e}",
                ],
            ),
        )
    return "\n".join(lines) + "\n"


def _build_table_summary(series: dict[str, list[tuple[float, float]]], heading: str) -> str:
    value_map = {
        label: {lam: value for lam, value in points}
        for label, points in series.items()
    }
    lambdas = sorted(next(iter(value_map.values())))
    labels = list(series)

    lines = [
        heading,
        "",
        ",".join(["lambda", *labels]),
    ]
    for lam in lambdas:
        lines.append(
            ",".join(
                [
                    f"{lam:g}",
                    *(f"{value_map[label][lam]:.18e}" for label in labels),
                ],
            ),
        )
    return "\n".join(lines) + "\n"


def _write_svg(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    *,
    title: str,
    labels: list[str] | None = None,
    show_bands: bool = True,
    show_inner_band: bool = True,
    simple_outer_band: bool = False,
    simple_outer_band_opacity: float = 0.20,
    show_band_legend: bool = True,
    tick_font_size: int = 13,
    axis_font_size: int = 17,
    legend_font_size: float = 13.5,
    legend_row_gap: int = 32,
    x_log_scale: bool = False,
    subtitle: str | None = None,
) -> None:
    if labels is None:
        labels = ["Erlang 2", "Erlang 10", "faster w/o", "faster 10", "CME 10", "slower 10", "slower w/o"]

    width = 1180
    height = 760
    left = 100
    right = 300
    top = 45
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [(x, y) for points in series.values() for x, y in points if y > 0.0]
    x_values = sorted({x for x, _ in all_points})
    y_values = [y for _, y in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    def x_to_px(value: float) -> float:
        if x_log_scale:
            frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        else:
            frac = (value - x_min) / (x_max - x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    style = {
        "Exponential": {"color": "#5c6b73", "dash": "", "width": "2.1", "marker": "circle"},
        "Erlang 2": {"color": "#c7a500", "dash": "", "width": "1.8", "marker": "square"},
        "Erlang 10": {"color": "#1d4ed8", "dash": "10,4", "width": "1.8", "marker": "square"},
        "Erlang 10/D": {"color": "#2563eb", "dash": "4,2", "width": "1.8", "marker": "square"},
        "faster w/o": {"color": "#f4a261", "dash": "8,4", "width": "1.9", "marker": "triangle"},
        "slower w/o": {"color": "#f4a261", "dash": "2,4", "width": "1.9", "marker": "triangle-down"},
        "faster 10": {"color": "#c1121f", "dash": "8,4", "width": "1.9", "marker": "triangle"},
        "slower 10": {"color": "#c1121f", "dash": "4,2", "width": "2.0", "marker": "triangle-down"},
        "average w/o": {"color": "#7c3aed", "dash": "", "width": "2.6", "marker": "circle"},
        "CME 10": {"color": "#0c7c59", "dash": "", "width": "2.6", "marker": "diamond"},
    }

    band_style = {
        "outer": {"fill": "#e5e7eb", "opacity": 0.28, "stroke": "#9ca3af", "stroke_width": 1.0},
        "inner": {"fill": "#9ca3af", "opacity": 0.32, "stroke": "#374151", "stroke_width": 1.25},
    }

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append("<defs>")
    parts.append('  <pattern id="outerMarks" width="5" height="5" patternUnits="userSpaceOnUse">')
    parts.append('    <path d="M2.5 1.55 V3.45 M1.55 2.5 H3.45" stroke="#111827" stroke-width="0.55" stroke-linecap="round" opacity="0.44"/>')
    parts.append("  </pattern>")
    parts.append('  <pattern id="innerHatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">')
    parts.append('    <line x1="0" y1="0" x2="0" y2="6" stroke="#111827" stroke-width="0.55" opacity="0.36"/>')
    parts.append("  </pattern>")
    parts.append("</defs>")
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for power in range(y_min_power, y_max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="{tick_font_size}" fill="#111111">1e{power:+d}</text>')

    if x_log_scale:
        x_tick_values = _log_x_tick_values(x_min, x_max)
    else:
        x_tick_values = x_values

    for lam in x_tick_values:
        x = x_to_px(lam)
        label = f"{lam:.2f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 32}" text-anchor="middle" font-family="Consolas, monospace" font-size="{tick_font_size}" fill="#111111">{label}</text>')

    if show_bands:
        if simple_outer_band:
            outer_fill = "#94a3b8"
            outer_opacity = simple_outer_band_opacity
            outer_stroke = "#64748b"
            outer_stroke_width = 1.2
            outer_pattern = None
        else:
            outer_fill = str(band_style["outer"]["fill"])
            outer_opacity = float(band_style["outer"]["opacity"])
            outer_stroke = str(band_style["outer"]["stroke"])
            outer_stroke_width = float(band_style["outer"]["stroke_width"])
            outer_pattern = "outerMarks"
        parts.append(
            _band_polygon(
                series["faster w/o"],
                series["slower w/o"],
                x_to_px,
                y_to_px,
                outer_fill,
                outer_opacity,
                outer_stroke,
                outer_stroke_width,
                outer_pattern,
                0.55,
            ),
        )
        if show_inner_band:
            parts.append(
                _band_polygon(
                    series["faster 10"],
                    series["slower 10"],
                    x_to_px,
                    y_to_px,
                    band_style["inner"]["fill"],
                    band_style["inner"]["opacity"],
                    band_style["inner"]["stroke"],
                    band_style["inner"]["stroke_width"],
                    "innerHatch",
                    0.42,
                ),
            )

    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    for label in labels:
        color = style[label]["color"]
        dash = style[label]["dash"]
        stroke_width = style[label]["width"]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = series[label]
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points if y > 0.0)
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash_attr} points="{polyline}"/>')
        for x, y in points:
            if y > 0.0:
                parts.extend(_marker_parts(x_to_px(x), y_to_px(y), style[label]["marker"], color, "#ffffff", 4.8))

    x_axis_label = "Arrival rate \u03bb (log scale)" if x_log_scale else "Arrival rate \u03bb"
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="{axis_font_size}" fill="#111111">{x_axis_label}</text>')
    parts.append(f'<text transform="translate(30 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="{axis_font_size}" fill="#111111">P(full) (log scale)</text>')
    parts.extend(
        _legend(
            left + plot_width + 40,
            top + 25,
            style,
            band_style,
            labels=labels,
            show_bands=show_bands,
            show_inner_band=show_inner_band,
            show_band_legend=show_band_legend,
            font_size=legend_font_size,
            row_gap=legend_row_gap,
        ),
    )
    parts.append("</svg>")

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _log_x_tick_values(x_min: float, x_max: float) -> list[float]:
    ticks: list[float] = []
    min_power = math.floor(math.log10(x_min))
    max_power = math.ceil(math.log10(x_max))
    for power in range(min_power, max_power + 1):
        for multiplier in [1, 2, 3, 5]:
            value = multiplier * (10.0 ** power)
            if x_min <= value <= x_max and value not in ticks:
                ticks.append(value)
    if x_max not in ticks:
        ticks.append(x_max)
    return sorted(ticks)


def _band_polygon(
    lower_points: list[tuple[float, float]],
    upper_points: list[tuple[float, float]],
    x_to_px,
    y_to_px,
    color: str,
    opacity: float,
    stroke: str,
    stroke_width: float,
    pattern_id: str | None = None,
    pattern_opacity: float = 1.0,
) -> str:
    lower_map = {x: y for x, y in lower_points}
    upper_map = {x: y for x, y in upper_points}
    lambdas = sorted(set(lower_map) & set(upper_map))
    lower = [(x, min(lower_map[x], upper_map[x])) for x in lambdas]
    upper = [(x, max(lower_map[x], upper_map[x])) for x in lambdas]
    points = [
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in lower),
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in reversed(upper)),
    ]
    points_text = " ".join(points)
    polygon = f'<polygon points="{points_text}" fill="{color}" opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>'
    if pattern_id is None:
        return polygon
    dots = f'<polygon points="{points_text}" fill="url(#{pattern_id})" opacity="{pattern_opacity:.2f}" stroke="none"/>'
    return "\n".join([polygon, dots])


def _legend(
    legend_x: int,
    legend_y: int,
    style: dict[str, dict[str, str]],
    band_style: dict[str, dict[str, object]],
    *,
    labels: list[str],
    show_bands: bool,
    show_inner_band: bool,
    show_band_legend: bool,
    font_size: float,
    row_gap: int,
) -> list[str]:
    display_label = {
        "faster w/o": "Faster w/o",
        "slower w/o": "Slower w/o",
        "faster 10": "Faster 10",
        "slower 10": "Slower 10",
        "average w/o": "Average w/o",
    }
    items: list[tuple[str, str, str, str]] = []
    if show_bands and show_band_legend:
        items.append(("outer band: faster/slower w/o", str(band_style["outer"]["fill"]), "", "band"))
        if show_inner_band:
            items.append(("inner band: faster/slower 10", str(band_style["inner"]["fill"]), "", "band"))
    items.extend((label, style[label]["color"], style[label]["dash"], "line") for label in labels)

    parts = [f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="260" height="{row_gap * len(items) + 22}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>']
    for index, (label, color, dash, kind) in enumerate(items):
        y = legend_y + index * row_gap
        if kind == "band":
            band_key = "outer" if "w/o" in label else "inner"
            stroke = str(band_style[band_key]["stroke"])
            stroke_width = float(band_style[band_key]["stroke_width"])
            opacity = float(band_style[band_key]["opacity"])
            parts.append(f'<rect x="{legend_x}" y="{y - 8}" width="32" height="16" fill="{color}" opacity="{opacity:.2f}" stroke="{stroke}" stroke-width="{stroke_width:.2f}"/>')
            if band_key == "outer":
                parts.append(f'<rect x="{legend_x}" y="{y - 8}" width="32" height="16" fill="url(#outerMarks)" opacity="0.55" stroke="none"/>')
            else:
                parts.append(f'<rect x="{legend_x}" y="{y - 8}" width="32" height="16" fill="url(#innerHatch)" opacity="0.42" stroke="none"/>')
        else:
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            stroke_width = style[label]["width"]
            marker = style[label]["marker"]
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="{stroke_width}"{dash_attr}/>')
            parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 4.8))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 6}" font-family="Segoe UI, Arial, sans-serif" font-size="{font_size}" fill="#111111">{display_label.get(label, label)}</text>')
    return parts


def _marker_parts(
    cx: float,
    cy: float,
    marker: str,
    color: str,
    stroke: str,
    size: float,
) -> list[str]:
    if marker == "square":
        return [f'<rect x="{cx - size:.2f}" y="{cy - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>']
    if marker == "diamond":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy:.2f} {cx:.2f},{cy + size + 1:.2f} {cx - size - 1:.2f},{cy:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>']
    if marker == "triangle":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy + size:.2f} {cx - size - 1:.2f},{cy + size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>']
    if marker == "triangle-down":
        return [f'<polygon points="{cx - size - 1:.2f},{cy - size:.2f} {cx + size + 1:.2f},{cy - size:.2f} {cx:.2f},{cy + size + 1:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>']
    return [f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.5"/>']


if __name__ == "__main__":
    main()
