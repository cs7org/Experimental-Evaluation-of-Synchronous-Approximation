from __future__ import annotations

import math
from pathlib import Path

from methods.cme import solve as solve_cme
from methods.embedded_one_server import (
    solve_erlang as solve_one_server_erlang,
)
from methods.erlang import solve as solve_erlang_dtmc
from methods.fast_slow import solve as solve_fast_slow
from methods.pure_erlang import solve as solve_erlang


SERVICE_TIME = 1.0
CAPACITY = 10
LAMBDA_VALUES = [0.1]
LAMBDA_DASH = {
    0.1: "",
    0.5: "8,4",
    0.9: "2,4",
}
METHOD_MARKERS = {
    "Exponential": "circle",
    "Exponential/D": "circle",
    "Erlang": "square",
    "Erlang /D": "square",
    "CME": "diamond",
    "Faster": "triangle",
    "Average": "circle",
    "Slower": "triangle-down",
}
SERIES_STYLES = {
    "Exponential": {"color": "#5c6b73", "dash": ""},
    "Exponential/D": {"color": "#a3a3a3", "dash": ""},
    "Erlang": {"color": "#1d4ed8", "dash": ""},
    "Erlang /D": {"color": "#60a5fa", "dash": "4,2"},
    "CME": {"color": "#0c7c59", "dash": ""},
    "Faster": {"color": "#f97316", "dash": "6,4"},
    "Average": {"color": "#7e22ce", "dash": ""},
    "Slower": {"color": "#ca8a04", "dash": "2,4"},
}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ERLANG_PHASE_COUNTS = [2, 10, 100]

CME_TARGETS = [
    {"name": "CME 5", "phases": 5, "scv": 0.026156884691722396},
    {"name": "CME 10", "phases": 10, "scv": 0.0057367735345383161},
    {"name": "CME 15", "phases": 15, "scv": 0.0024052513649265544},
    {"name": "CME 20", "phases": 20, "scv": 0.0012568106539700918},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = ["lambda,series,phases,p_full"]
    lambda_series: dict[float, list[dict[str, object]]] = {}
    for lam in LAMBDA_VALUES:
        series = _compute_series(lam)
        lambda_series[lam] = series
        csv_lines.extend(_csv_rows(lam, series))
        svg_path = OUTPUT_DIR / f"pfull_accuracy_lambda_{str(lam).replace('.', '_')}.svg"
        _write_accuracy_svg(
            svg_path,
            series,
            title=f"P(full) by Phase Count for K={CAPACITY}, lambda={lam}",
        )
        print(f"Saved {svg_path}")

    combined_svg_path = OUTPUT_DIR / "pfull_accuracy_all_lambdas.svg"
    _write_combined_accuracy_svg(
        combined_svg_path,
        lambda_series,
        title=f"P(full) by Phase Count for K={CAPACITY}",
    )
    print(f"Saved {combined_svg_path}")

    csv_path = OUTPUT_DIR / "pfull_accuracy.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")


def _compute_series(lam: float) -> list[dict[str, object]]:
    exponential = solve_erlang_dtmc(
        arrival_rate=lam,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        phase_counts=[1],
        stationary_method="power",
        tolerance=1e-12,
        max_iterations=50_000,
        relaxation=None,
    )[0]

    erlang_rows = solve_erlang(
        arrival_rate=lam,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        phase_counts=ERLANG_PHASE_COUNTS,
    )
    erlang_d_points = []
    exponential_d_row = solve_one_server_erlang(
        arrival_rate=lam,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        phase_count=1,
        precision_digits=80,
        solver="fast_iterative",
    )[0]
    for phase_count in ERLANG_PHASE_COUNTS:
        row = solve_one_server_erlang(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_count=phase_count,
            precision_digits=80,
            # The fast float path loses the tiny P(full) tail at high phase counts.
            solver="fast_iterative",
        )[0]
        erlang_d_points.append((phase_count, float(row["p_full"])))

    cme_points: list[tuple[int, float]] = []
    for target in CME_TARGETS:
        cme_row = _solve_cme_best_effort(lam, target)
        if cme_row is not None:
            phase_count = int(target["phases"])
            p_full = float(cme_row["p_full"])
            cme_points.append((phase_count, p_full))
    fast_rows = solve_fast_slow(
        arrival_rate=lam,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        variants=["faster", "faster 2", "faster 10", "faster 100"],
        precision_digits=80,
    )
    slow_rows = solve_fast_slow(
        arrival_rate=lam,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        variants=["slower", "slower 1", "slower 2", "slower 10", "slower 100"],
        precision_digits=80,
    )

    return [
        {
            "label": "Exponential",
            **SERIES_STYLES["Exponential"],
            "points": [(1, float(exponential["p_full"]))],
            "line_points": [(1, float(exponential["p_full"])), (2, float(erlang_rows[0]["p_full"]))],
        },
        {
            "label": "Exponential/D",
            **SERIES_STYLES["Exponential/D"],
            "points": [(1, float(exponential_d_row["p_full"]))],
            "line_points": [
                (1, float(exponential_d_row["p_full"])),
                (2, float(erlang_d_points[0][1])),
            ],
        },
        {
            "label": "Erlang",
            **SERIES_STYLES["Erlang"],
            "points": [(2, float(erlang_rows[0]["p_full"])), (10, float(erlang_rows[1]["p_full"])), (100, float(erlang_rows[2]["p_full"]))],
        },
        {
            "label": "Erlang /D",
            **SERIES_STYLES["Erlang /D"],
            "points": erlang_d_points,
        },
        {
            "label": "CME",
            **SERIES_STYLES["CME"],
            "points": cme_points,
        },
        {
            "label": "Faster",
            **SERIES_STYLES["Faster"],
            "points": [(1, float(fast_rows[0]["p_full"])), (2, float(fast_rows[1]["p_full"])), (10, float(fast_rows[2]["p_full"])), (100, float(fast_rows[3]["p_full"]))],
        },
        {
            "label": "Average",
            **SERIES_STYLES["Average"],
            "points": [
                (1, 0.5 * (float(fast_rows[0]["p_full"]) + float(slow_rows[0]["p_full"]))),
                (2, 0.5 * (float(fast_rows[1]["p_full"]) + float(slow_rows[2]["p_full"]))),
                (10, 0.5 * (float(fast_rows[2]["p_full"]) + float(slow_rows[3]["p_full"]))),
                (100, 0.5 * (float(fast_rows[3]["p_full"]) + float(slow_rows[4]["p_full"]))),
            ],
        },
        {
            "label": "Slower",
            **SERIES_STYLES["Slower"],
            "points": [(1, float(slow_rows[0]["p_full"])), (2, float(slow_rows[2]["p_full"])), (10, float(slow_rows[3]["p_full"])), (100, float(slow_rows[4]["p_full"]))],
        },
    ]


def _solve_cme_best_effort(lam: float, target: dict[str, object]) -> dict[str, float | str] | None:
    attempts = [
        {"solver": "recursive"},
        {"solver": "recursive_improved"},
        {"solver": "iterative", "max_iterations": 20_000},
    ]
    for kwargs in attempts:
        try:
            return solve_cme(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                targets=[target],
                **kwargs,
            )[0]
        except (RuntimeError, FileNotFoundError, NotImplementedError):
            continue

    print(f"[warning] Skipping {target['name']} at lambda={lam}: all solver modes failed.")
    return None


def _csv_rows(lam: float, series: list[dict[str, object]]) -> list[str]:
    rows: list[str] = []
    for item in series:
        label = str(item["label"])
        for phases, value in item["points"]:  # type: ignore[index]
            rows.append(f"{lam},{label},{phases},{value:.18e}")
    return rows


def _write_accuracy_svg(path: Path, series: list[dict[str, object]], *, title: str) -> None:
    width = 1180
    height = 760
    left = 110
    right = 70
    top = 45
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [(float(x), float(y)) for item in series for x, y in item["points"]]  # type: ignore[index]
    x_values = [x for x, _ in all_points]
    positive_y = [y for _, y in all_points if y > 0.0]

    x_min = min(x_values)
    x_max = max(x_values)
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    y_min_power = math.floor(math.log10(min(positive_y)))
    y_max_power = math.ceil(math.log10(max(positive_y)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

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

    for x_tick in sorted(set(int(v) for v in x_values)):
        x = x_to_px(float(x_tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{x_tick}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    band = _band_polygon(
        _series_points(series, "Faster"),
        _series_points(series, "Slower"),
        x_to_px,
        y_to_px,
    )
    if band:
        parts.append(band)

    for item in series:
        color = str(item["color"])
        dash = str(item["dash"])
        marker = METHOD_MARKERS.get(str(item["label"]), "circle")
        points = [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
        line_source = item.get("line_points", item["points"])
        line_points = [(float(x), float(y)) for x, y in line_source]  # type: ignore[assignment]
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in line_points if y > 0.0)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len([point for point in line_points if point[1] > 0.0]) > 1:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{polyline}"/>')
        for x, y in points:
            if y <= 0.0:
                continue
            px = x_to_px(x)
            py = y_to_px(y)
            parts.extend(_marker_parts(px, py, marker, color, "#ffffff", 5.6))

    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Phase count / CME order (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">P(full) (log scale)</text>')

    legend_width = 220
    legend_height = 36 * len(series) + 20
    legend_rect_x = left + plot_width - legend_width - 14
    legend_rect_y = top + 14
    legend_x = legend_rect_x + 16
    legend_y = legend_rect_y + 24
    parts.append(f'<rect x="{legend_rect_x}" y="{legend_rect_y}" width="{legend_width}" height="{legend_height}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')
    for index, item in enumerate(series):
        y = legend_y + index * 36
        color = str(item["color"])
        dash = str(item["dash"])
        marker = METHOD_MARKERS.get(str(item["label"]), "circle")
        legend_dash = "" if str(item["label"]) == "Erlang" else dash
        dash_attr = f' stroke-dasharray="{legend_dash}"' if legend_dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 5.6))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#111111">{_esc(str(item["label"]))}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _series_points(series: list[dict[str, object]], label: str) -> list[tuple[float, float]]:
    for item in series:
        if str(item["label"]) == label:
            return [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
    return []


def _band_polygon(
    first_points: list[tuple[float, float]],
    second_points: list[tuple[float, float]],
    x_to_px,
    y_to_px,
) -> str:
    first = {x: y for x, y in first_points if y > 0.0}
    second = {x: y for x, y in second_points if y > 0.0}
    phases = sorted(set(first) & set(second))
    if len(phases) < 2:
        return ""
    lower = [(phase, min(first[phase], second[phase])) for phase in phases]
    upper = [(phase, max(first[phase], second[phase])) for phase in phases]
    points = [
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in lower),
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in reversed(upper)),
    ]
    return f'<polygon points="{" ".join(points)}" fill="#d1d5db" opacity="0.32" stroke="none"/>'


def _write_combined_accuracy_svg(
    path: Path,
    lambda_series: dict[float, list[dict[str, object]]],
    *,
    title: str,
) -> None:
    width = 1260
    height = 820
    left = 110
    right = 360
    top = 45
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [
        (float(x), float(y))
        for series in lambda_series.values()
        for item in series
        for x, y in item.get("line_points", item["points"])  # type: ignore[index]
        if float(y) > 0.0
    ]
    x_values = [x for x, _ in all_points]
    positive_y = [y for _, y in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    y_min_power = math.floor(math.log10(min(positive_y)))
    y_max_power = math.ceil(math.log10(max(positive_y)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    first_series = lambda_series[LAMBDA_VALUES[0]]
    method_items = [
        (
            str(item["label"]),
            str(item["color"]),
            METHOD_MARKERS.get(str(item["label"]), "circle"),
        )
        for item in first_series
    ]
    lambda_items = [(f"lambda={lam:g}", LAMBDA_DASH[lam]) for lam in LAMBDA_VALUES]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for power in range(y_min_power, y_max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">1e{power:+d}</text>')

    for x_tick in sorted(set(int(v) for v in x_values)):
        x = x_to_px(float(x_tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{x_tick}</text>')

    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    for lam in LAMBDA_VALUES:
        dash = LAMBDA_DASH[lam]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        for item in lambda_series[lam]:
            label = str(item["label"])
            color = str(item["color"])
            marker = METHOD_MARKERS.get(label, "circle")
            points = [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
            line_source = item.get("line_points", item["points"])
            line_points = [(float(x), float(y)) for x, y in line_source]  # type: ignore[assignment]
            positive_line_points = [(x, y) for x, y in line_points if y > 0.0]
            positive_points = [(x, y) for x, y in points if y > 0.0]
            if len(positive_line_points) > 1:
                polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in positive_line_points)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash_attr} points="{polyline}"/>')
            for x, y in positive_points:
                px = x_to_px(x)
                py = y_to_px(y)
                parts.append(f'<line x1="{px - 12:.2f}" y1="{py:.2f}" x2="{px + 12:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
                parts.extend(_marker_parts(px, py, marker, color, "#ffffff", 5.6))

    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Phase count / CME order (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">P(full) (log scale)</text>')

    legend_x = left + plot_width + 28
    legend_y = top + 20
    legend_count = len(method_items) + len(lambda_items) + 2
    parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="305" height="{34 * legend_count + 44}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')

    parts.append(f'<text x="{legend_x}" y="{legend_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#111111">Methods</text>')
    for index, (label, color, marker) in enumerate(method_items):
        y = legend_y + 24 + index * 34
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.4"/>')
        parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 5.8))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{_esc(label)}</text>')

    lambda_y = legend_y + 24 + len(method_items) * 34 + 24
    parts.append(f'<text x="{legend_x}" y="{lambda_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#111111">Lambda line style</text>')
    for index, (label, dash) in enumerate(lambda_items):
        y = lambda_y + 24 + index * 34
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="#111111" stroke-width="2.4"{dash_attr}/>')
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{_esc(label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _marker_parts(
    cx: float,
    cy: float,
    marker: str,
    color: str,
    stroke: str,
    size: float,
) -> list[str]:
    if marker == "square":
        return [f'<rect x="{cx - size:.2f}" y="{cy - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if marker == "diamond":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy:.2f} {cx:.2f},{cy + size + 1:.2f} {cx - size - 1:.2f},{cy:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if marker == "triangle":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy + size:.2f} {cx - size - 1:.2f},{cy + size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if marker == "triangle-down":
        return [f'<polygon points="{cx - size - 1:.2f},{cy - size:.2f} {cx + size + 1:.2f},{cy - size:.2f} {cx:.2f},{cy + size + 1:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if marker == "cross":
        return [
            f'<line x1="{cx - size:.2f}" y1="{cy:.2f}" x2="{cx + size:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>',
            f'<line x1="{cx:.2f}" y1="{cy - size:.2f}" x2="{cx:.2f}" y2="{cy + size:.2f}" stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>',
        ]
    return [f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
