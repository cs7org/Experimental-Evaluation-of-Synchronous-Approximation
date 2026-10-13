from __future__ import annotations

import csv
import html
import math
from pathlib import Path

from methods.reliability import (
    solve_cme_ground_truth,
    solve_markovian,
    solve_synchronized,
    solve_synchronized_expanded,
)


FAILURE_RATE_REFERENCE = 0.1
REPAIR_TIME = 1.0
MANIFESTATION_RATE = 1.0
LAMBDA_VALUES = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4]
ZOOM_LAMBDA_VALUES = [round(0.09 + 0.0025 * index, 4) for index in range(9)]
CME_ORDERS = [20]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
CME_REFERENCE_LABEL = "CME 20 ground truth"
AVERAGE_LABEL = "synchronized average"


PLOT_ORDER = [
    "exponential repair",
    CME_REFERENCE_LABEL,
    "synchronized faster",
    AVERAGE_LABEL,
    "synchronized slower",
    "synchronized faster 10",
    "synchronized slower 10",
]
DRAW_ORDER = [
    "exponential repair",
    CME_REFERENCE_LABEL,
    "synchronized slower",
    "synchronized slower 10",
    "synchronized faster 10",
    "synchronized faster",
    AVERAGE_LABEL,
]
MARKER_ORDER = [
    "exponential repair",
    CME_REFERENCE_LABEL,
    "synchronized slower",
    "synchronized slower 10",
    "synchronized faster 10",
    "synchronized faster",
    AVERAGE_LABEL,
]
CME_PLOT_ORDER = [
    CME_REFERENCE_LABEL,
]

DISPLAY_LABELS = {
    "exponential repair": "Exponential",
    "synchronized faster": "Faster w/o",
    "synchronized faster 10": "Faster 10",
    "synchronized slower": "Slower w/o",
    "synchronized slower 10": "Slower 10",
    AVERAGE_LABEL: "Average w/o",
    CME_REFERENCE_LABEL: "CME",
}

STYLE = {
    "exponential repair": {"color": "#5c6b73", "dash": "", "width": 2.6, "marker": "circle"},
    CME_REFERENCE_LABEL: {"color": "#0c7c59", "dash": "", "width": 2.6, "marker": "diamond"},
    "synchronized faster": {"color": "#f4a261", "dash": "6,4", "width": 2.6, "marker": "triangle"},
    "synchronized faster 10": {"color": "#c1121f", "dash": "6,4", "width": 2.6, "marker": "triangle"},
    "synchronized slower": {"color": "#f4a261", "dash": "2,4", "width": 2.6, "marker": "triangle-down"},
    "synchronized slower 10": {"color": "#c1121f", "dash": "2,4", "width": 2.6, "marker": "triangle-down"},
    AVERAGE_LABEL: {"color": "#7c3aed", "dash": "", "width": 2.6, "marker": "circle"},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _compute_rows()
    zoom_rows = _compute_rows(ZOOM_LAMBDA_VALUES)

    csv_path = OUTPUT_DIR / "reliability_model_results.csv"
    zoom_csv_path = OUTPUT_DIR / "reliability_model_results_zoom_0_09_0_11.csv"
    p_down_svg = OUTPUT_DIR / "reliability_p_down_bounds.svg"
    p_severe_svg = OUTPUT_DIR / "reliability_p_F_bounds.svg"
    p_severe_zoom_svg = OUTPUT_DIR / "reliability_p_F_bounds_zoom_0_09_0_11.svg"
    summary_path = OUTPUT_DIR / "reliability_model_summary.txt"

    _write_csv(csv_path, rows)
    _write_csv(zoom_csv_path, zoom_rows)
    _write_svg(
        p_down_svg,
        rows,
        metric="p_down",
        title="Reliability model: probability both components are failed",
        subtitle="faster/average/slower w/o and faster/slower 10 approximations with independent-repair CME 20 reference and Exponential",
        y_label="P(11 or F)",
    )
    _write_svg(
        p_severe_svg,
        rows,
        metric="p_F",
        title="Reliability model: manifested severe failure probability",
        subtitle="faster/average/slower w/o and faster/slower 10 approximations with independent-repair CME 20 reference and Exponential",
        y_label="P(F)",
        include_band=False,
    )
    _write_svg(
        p_severe_zoom_svg,
        zoom_rows,
        metric="p_F",
        title="Reliability model: manifested severe failure probability, focused range",
        subtitle="lambda from 0.09 to 0.11 with faster/average/slower w/o, expanded 10, CME, and Exponential",
        y_label="P(F)",
        include_band=False,
    )
    summary_path.write_text(_build_summary(rows), encoding="utf-8")

    print(_build_reference_table(rows))
    print(f"Saved {csv_path}")
    print(f"Saved {zoom_csv_path}")
    print(f"Saved {p_down_svg}")
    print(f"Saved {p_severe_svg}")
    print(f"Saved {p_severe_zoom_svg}")
    print(f"Saved {summary_path}")


def _compute_rows(lambda_values: list[float] | None = None) -> list[dict[str, float | int | str]]:
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES

    rows: list[dict[str, float | int | str]] = []
    for lam in lambda_values:
        base = {
            "lambda": lam,
            "tau": REPAIR_TIME,
            "rho": MANIFESTATION_RATE,
        }

        markovian = solve_markovian(
            failure_rate=lam,
            repair_rate=1.0 / REPAIR_TIME,
            manifestation_rate=MANIFESTATION_RATE,
        )
        rows.append({**base, **markovian})

        synchronized_rows = solve_synchronized(
            failure_rate=lam,
            repair_time=REPAIR_TIME,
            manifestation_rate=MANIFESTATION_RATE,
            variants=("faster", "slower"),
        )
        for row in synchronized_rows:
            rows.append({**base, **row})

        synchronized_by_name = {str(row["approach"]): row for row in synchronized_rows}
        average = _average_synchronized_rows(
            synchronized_by_name["synchronized faster"],
            synchronized_by_name["synchronized slower"],
        )
        rows.append({**base, **average})

        for row in solve_synchronized_expanded(
            failure_rate=lam,
            repair_time=REPAIR_TIME,
            manifestation_rate=MANIFESTATION_RATE,
            phase_counts=(10,),
            variants=("faster", "slower"),
        ):
            rows.append({**base, **row})

        for row in solve_cme_ground_truth(
            failure_rate=lam,
            repair_time=REPAIR_TIME,
            manifestation_rate=MANIFESTATION_RATE,
            orders=CME_ORDERS,
        ):
            rows.append({**base, **row})

    return rows


def _average_synchronized_rows(
    faster: dict[str, float | int | str],
    slower: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    average: dict[str, float | int | str] = {
        "approach": AVERAGE_LABEL,
        "family": "Synchronized",
    }
    for key, faster_value in faster.items():
        if key in {"approach", "family"} or key not in slower:
            continue
        slower_value = slower[key]
        if isinstance(faster_value, (int, float)) and isinstance(slower_value, (int, float)):
            average[key] = 0.5 * (float(faster_value) + float(slower_value))
    return average


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "lambda",
        "tau",
        "rho",
        "approach",
        "family",
        "order",
        "timer_phases",
        "linear_states",
        "p_00",
        "p_10",
        "p_01",
        "p_11",
        "p_F",
        "p_available",
        "p_down",
        "p_severe",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "Reliability model summary",
        f"tau={REPAIR_TIME:g}, rho={MANIFESTATION_RATE:g}",
        "",
        "lambda,CME 20 P(11 or F),faster lower,average,slower upper,CME 20 P(F),faster P(F),average P(F),slower P(F)",
    ]
    for lam in LAMBDA_VALUES:
        by_name = _rows_for_lambda(rows, lam)
        faster = by_name["synchronized faster"]
        average = by_name[AVERAGE_LABEL]
        slower = by_name["synchronized slower"]
        cme = by_name[CME_REFERENCE_LABEL]
        lines.append(
            ",".join(
                [
                    f"{lam:g}",
                    _fmt(float(cme["p_down"])),
                    _fmt(float(faster["p_down"])),
                    _fmt(float(average["p_down"])),
                    _fmt(float(slower["p_down"])),
                    _fmt(float(cme["p_F"])),
                    _fmt(float(faster["p_F"])),
                    _fmt(float(average["p_F"])),
                    _fmt(float(slower["p_F"])),
                ],
            ),
        )
    return "\n".join(lines) + "\n"


def _build_reference_table(rows: list[dict[str, float | int | str]]) -> str:
    by_name = _rows_for_lambda(rows, FAILURE_RATE_REFERENCE)
    names = [
        "exponential repair",
        CME_REFERENCE_LABEL,
        "synchronized faster",
        "synchronized faster 10",
        "synchronized slower 10",
        "synchronized slower",
        AVERAGE_LABEL,
    ]
    lines = [
        f"Reference case: lambda={FAILURE_RATE_REFERENCE:g}, tau={REPAIR_TIME:g}, rho={MANIFESTATION_RATE:g}",
        "approach,p_00,p_10,p_01,p_11,p_F,p_down",
    ]
    for name in names:
        row = by_name[name]
        lines.append(
            ",".join(
                [
                    name,
                    _fmt(float(row["p_00"])),
                    _fmt(float(row["p_10"])),
                    _fmt(float(row["p_01"])),
                    _fmt(float(row["p_11"])),
                    _fmt(float(row["p_F"])),
                    _fmt(float(row["p_down"])),
                ],
            ),
        )
    return "\n".join(lines)


def _rows_for_lambda(
    rows: list[dict[str, float | int | str]],
    lam: float,
) -> dict[str, dict[str, float | int | str]]:
    selected = [row for row in rows if math.isclose(float(row["lambda"]), lam, rel_tol=0.0, abs_tol=1e-12)]
    return {str(row["approach"]): row for row in selected}


def _write_svg(
    path: Path,
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    title: str,
    subtitle: str,
    y_label: str,
    include_band: bool = True,
) -> None:
    series = _series(rows, metric)
    width = 1180
    height = 760
    left = 105
    right = 300
    top = 45
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom

    points = [
        (x, y)
        for label in PLOT_ORDER
        for x, y in series[label]
        if y > 0.0 and math.isfinite(y)
    ]
    x_values = sorted({x for x, _ in points})
    y_values = [y for _, y in points]
    x_min = min(x_values)
    x_max = max(x_values)
    x_tick_decimals = 2
    if len(x_values) > 1:
        min_spacing = min(right - left for left, right in zip(x_values, x_values[1:]))
        if min_spacing < 0.01:
            x_tick_decimals = 4
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))
    if y_min_power == y_max_power:
        y_max_power += 1

    def x_to_px(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for power in range(y_min_power, y_max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">1e{power:+d}</text>')

    for lam in x_values:
        x = x_to_px(lam)
        label = f"{lam:.{x_tick_decimals}f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 30}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{label}</text>')

    parts.append(_band(series["synchronized faster"], series["synchronized slower"], x_to_px, y_to_px))
    parts.append(_band(series["synchronized faster 10"], series["synchronized slower 10"], x_to_px, y_to_px, opacity="0.14"))
    for label in DRAW_ORDER:
        parts.append(_polyline(series[label], x_to_px, y_to_px, label))
    for label in MARKER_ORDER:
        parts.extend(_markers(series[label], x_to_px, y_to_px, label))

    axis_color = "#111111"
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">arrival rate \u03bb</text>')
    parts.append(f'<text x="30" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 30 {top + plot_height / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">{html.escape(y_label)}</text>')
    parts.extend(_legend(left + plot_width + 40, top + 25, include_band=include_band))
    parts.append("</svg>")

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _write_cme_zoom_svg(
    path: Path,
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    title: str,
    subtitle: str,
    y_label: str,
) -> None:
    series = _series(rows, metric, labels=CME_PLOT_ORDER)
    width = 1180
    height = 760
    left = 105
    right = 300
    top = 45
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom

    points = [
        (x, y)
        for label in CME_PLOT_ORDER
        for x, y in series[label]
        if y > 0.0 and math.isfinite(y)
    ]
    x_values = sorted({x for x, _ in points})
    y_values = [y for _, y in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    padding = max((y_max - y_min) * 0.08, y_max * 0.01)
    y_min = max(0.0, y_min - padding)
    y_max += padding

    def x_to_px(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_to_px(value: float) -> float:
        frac = (value - y_min) / (y_max - y_min)
        return top + plot_height - frac * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5.0
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">{value:.5f}</text>')

    for lam in x_values:
        x = x_to_px(lam)
        label = f"{lam:.4f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 30}" text-anchor="middle" font-family="Consolas, monospace" font-size="12.5" fill="#111111">{label}</text>')

    for label in CME_PLOT_ORDER:
        parts.append(_polyline(series[label], x_to_px, y_to_px, label))
        parts.extend(_markers(series[label], x_to_px, y_to_px, label))

    axis_color = "#111111"
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">arrival rate \u03bb</text>')
    parts.append(f'<text x="30" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 30 {top + plot_height / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">{html.escape(y_label)}</text>')
    parts.extend(_legend(left + plot_width + 40, top + 25, labels=CME_PLOT_ORDER, include_band=False))
    parts.append("</svg>")

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _series(
    rows: list[dict[str, float | int | str]],
    metric: str,
    *,
    labels: list[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    if labels is None:
        labels = PLOT_ORDER

    result = {label: [] for label in labels}
    for row in rows:
        label = str(row["approach"])
        if label in result:
            result[label].append((float(row["lambda"]), float(row[metric])))

    for label in result:
        result[label].sort(key=lambda item: item[0])
    return result


def _band(
    lower_points: list[tuple[float, float]],
    upper_points: list[tuple[float, float]],
    x_to_px,
    y_to_px,
    *,
    opacity: str = "0.20",
) -> str:
    lower = [(x, min(y1, y2)) for (x, y1), (_, y2) in zip(lower_points, upper_points)]
    upper = [(x, max(y1, y2)) for (x, y1), (_, y2) in zip(lower_points, upper_points)]
    coords = [(x_to_px(x), y_to_px(y)) for x, y in lower]
    coords += [(x_to_px(x), y_to_px(y)) for x, y in reversed(upper)]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return f'<polygon points="{points}" fill="#94a3b8" opacity="{opacity}" stroke="#64748b" stroke-width="1.2"/>'


def _polyline(points: list[tuple[float, float]], x_to_px, y_to_px, label: str) -> str:
    style = STYLE[label]
    coords = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points if y > 0.0)
    dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
    return f'<polyline points="{coords}" fill="none" stroke="{style["color"]}" stroke-width="{style["width"]}" stroke-linecap="round" stroke-linejoin="round"{dash}/>'


def _markers(points: list[tuple[float, float]], x_to_px, y_to_px, label: str) -> list[str]:
    parts: list[str] = []
    marker_dx = {
        "synchronized faster": -8.0,
        "synchronized faster 10": -16.0,
        "synchronized slower": 8.0,
        "synchronized slower 10": 16.0,
    }.get(label, 0.0)
    for x_value, y_value in points:
        if y_value <= 0.0:
            continue
        x = x_to_px(x_value) + marker_dx
        y = y_to_px(y_value)
        parts.extend(_marker_parts(x, y, label, 5.4))
    return parts


def _legend(
    x: float,
    y: float,
    *,
    labels: list[str] | None = None,
    include_band: bool = True,
) -> list[str]:
    if labels is None:
        labels = PLOT_ORDER

    parts = []
    box_height = len(labels) * 34 + (74 if include_band else 18)
    parts.append(f'<rect x="{x - 14}" y="{y - 24}" width="230" height="{box_height}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')
    for index, label in enumerate(labels):
        style = STYLE[label]
        yy = y + index * 34
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 34}" y2="{yy}" stroke="{style["color"]}" stroke-width="{style["width"]}" stroke-linecap="round"{dash}/>')
        parts.extend(_legend_marker(x + 17, yy, label))
        parts.append(f'<text x="{x + 46}" y="{yy + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">{html.escape(_display_label(label))}</text>')
    if include_band:
        parts.append(f'<rect x="{x}" y="{y + len(labels) * 34 + 10}" width="34" height="12" fill="#94a3b8" opacity="0.20" stroke="#64748b" stroke-width="1.1"/>')
        parts.append(f'<text x="{x + 46}" y="{y + len(labels) * 34 + 22}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">faster/slower w/o band</text>')
        parts.append(f'<rect x="{x}" y="{y + len(labels) * 34 + 32}" width="34" height="12" fill="#94a3b8" opacity="0.14" stroke="#64748b" stroke-width="1.1"/>')
        parts.append(f'<text x="{x + 46}" y="{y + len(labels) * 34 + 44}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">faster/slower 10 band</text>')
    return parts


def _display_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label.replace(" ground truth", ""))


def _legend_marker(x: float, y: float, label: str) -> list[str]:
    return _marker_parts(x, y, label, 5.2)


def _marker_parts(x: float, y: float, label: str, size: float) -> list[str]:
    style = STYLE[label]
    color = str(style["color"])
    marker = str(style["marker"])
    if marker == "circle":
        return [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>']
    if marker == "diamond":
        return [f'<polygon points="{x:.2f},{y - size - 1:.2f} {x + size + 1:.2f},{y:.2f} {x:.2f},{y + size + 1:.2f} {x - size - 1:.2f},{y:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>']
    if marker == "square":
        return [f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>']
    if marker == "triangle":
        return [f'<polygon points="{x:.2f},{y - size - 1:.2f} {x + size + 1:.2f},{y + size:.2f} {x - size - 1:.2f},{y + size:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>']
    if marker == "triangle-down":
        return [f'<polygon points="{x - size - 1:.2f},{y - size:.2f} {x + size + 1:.2f},{y - size:.2f} {x:.2f},{y + size + 1:.2f}" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>']
    return []


def _fmt(value: float) -> str:
    return f"{value:.15g}"


if __name__ == "__main__":
    main()
