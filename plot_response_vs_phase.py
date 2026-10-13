from __future__ import annotations

import math
from pathlib import Path

from methods.cme import solve as solve_cme
from methods.embedded_one_server import (
    solve_cme as solve_one_server_cme,
    solve_erlang as solve_one_server_erlang,
)
from methods.erlang import solve as solve_erlang_dtmc
from methods.fast_slow import solve as solve_fast_slow
from methods.pure_erlang import solve as solve_erlang


SERVICE_TIME = 1.0
CAPACITY = 10
LAMBDA_VALUES = [0.1, 0.5, 0.9, 1.0]
LAMBDA_DASH = {
    0.1: "",
    0.5: "8,4",
    0.9: "2,4",
    1.0: "9,3,2,3",
}
METHOD_MARKERS = {
    "Exponential": "circle",
    "Exponential/D": "circle",
    "Erlang": "square",
    "Erlang /D": "square",
    "CME": "diamond",
    "CME /D": "diamond",
    "Faster": "triangle",
    "Average": "circle",
    "Slower": "triangle-down",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

CME_TARGETS = [
    {"name": "CME 5", "phases": 5, "scv": 0.026156884691722396},
    {"name": "CME 10", "phases": 10, "scv": 0.0057367735345383161},
    {"name": "CME 15", "phases": 15, "scv": 0.0024052513649265544},
    {"name": "CME 20", "phases": 20, "scv": 0.0012568106539700918},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = ["lambda,series,phases,response_time"]
    lambda_series: dict[float, list[dict[str, object]]] = {}
    for lam in LAMBDA_VALUES:
        series = _compute_series(lam)
        lambda_series[lam] = series
        csv_lines.extend(_csv_rows(lam, series))
        svg_path = OUTPUT_DIR / f"response_vs_phase_lambda_{_lambda_key(lam)}.svg"
        _write_response_svg(
            svg_path,
            series,
            title=f"Mean Response Time by Phase Count for K={CAPACITY}, lambda={lam}",
        )
        print(f"Saved {svg_path}")

    _write_variant(
        output_stem="response_vs_phase_lambda_1_with_erlang_d",
        lam=1.0,
        include_erlang_d=True,
        include_cme_d=False,
    )
    _write_variant(
        output_stem="response_vs_phase_lambda_1_with_erlang_d_cme_d",
        lam=1.0,
        include_erlang_d=True,
        include_cme_d=True,
    )

    combined_svg_path = OUTPUT_DIR / "response_vs_phase_all_lambdas.svg"
    _write_combined_response_svg(
        combined_svg_path,
        lambda_series,
        title=f"Mean Response Time by Phase Count for K={CAPACITY}",
    )
    print(f"Saved {combined_svg_path}")

    csv_path = OUTPUT_DIR / "response_vs_phase.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")


def _compute_series(
    lam: float,
    *,
    include_erlang_d: bool = False,
    include_cme_d: bool = False,
) -> list[dict[str, object]]:
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
        phase_counts=[2, 10, 100],
    )
    erlang_d_points: list[tuple[int, float]] = []
    exponential_d_row = None
    if include_erlang_d:
        exponential_d_row = solve_one_server_erlang(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_count=1,
            precision_digits=80,
            solver="fast_iterative",
        )[0]
        for phase_count in (2, 10, 100):
            row = solve_one_server_erlang(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                phase_count=phase_count,
                precision_digits=80,
                solver="fast_iterative",
            )[0]
            erlang_d_points.append((phase_count, float(row["response_time"])))

    cme_points: list[tuple[int, float]] = []
    for target in CME_TARGETS:
        cme_row = _solve_cme_best_effort(lam, target)
        if cme_row is not None:
            cme_points.append((int(target["phases"]), float(cme_row["response_time"])))
    cme_d_points: list[tuple[int, float]] = []
    if include_cme_d:
        for target in CME_TARGETS:
            cme_d_row = _solve_cme_d_best_effort(lam, int(target["phases"]))
            if cme_d_row is not None:
                cme_d_points.append((int(target["phases"]), float(cme_d_row["response_time"])))
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
        variants=["slower", "slower 2", "slower 10", "slower 100"],
        precision_digits=80,
    )

    series = [
        {
            "label": "Exponential",
            "color": "#5c6b73",
            "dash": "",
            "points": [(1, float(exponential["response_time"]))],
            "line_points": [
                (1, float(exponential["response_time"])),
                (2, float(erlang_rows[0]["response_time"])),
            ],
        },
        {
            "label": "Erlang",
            "color": "#1d4ed8",
            "dash": "",
            "points": [
                (2, float(erlang_rows[0]["response_time"])),
                (10, float(erlang_rows[1]["response_time"])),
                (100, float(erlang_rows[2]["response_time"])),
            ],
        },
    ]
    if include_erlang_d:
        assert exponential_d_row is not None
        series.append(
            {
                "label": "Exponential/D",
                "color": "#a3a3a3",
                "dash": "",
                "points": [(1, float(exponential_d_row["response_time"]))],
                "line_points": [
                    (1, float(exponential_d_row["response_time"])),
                    (2, float(erlang_d_points[0][1])),
                ],
            },
        )
        series.append(
            {
                "label": "Erlang /D",
                "color": "#60a5fa",
                "dash": "4,2",
                "points": erlang_d_points,
            },
        )
    series.append(
        {
            "label": "CME",
            "color": "#0c7c59",
            "dash": "",
            "points": cme_points,
        },
    )
    if include_cme_d:
        series.append(
            {
                "label": "CME /D",
                "color": "#14b8a6",
                "dash": "4,2",
                "points": cme_d_points,
            },
        )
    series.extend(
        [
        {
            "label": "Faster",
            "color": "#f97316",
            "dash": "6,4",
            "points": [
                (1, float(fast_rows[0]["response_time"])),
                (2, float(fast_rows[1]["response_time"])),
                (10, float(fast_rows[2]["response_time"])),
                (100, float(fast_rows[3]["response_time"])),
            ],
        },
        {
            "label": "Average",
            "color": "#7e22ce",
            "dash": "",
            "points": [
                (1, 0.5 * (float(fast_rows[0]["response_time"]) + float(slow_rows[0]["response_time"]))),
                (2, 0.5 * (float(fast_rows[1]["response_time"]) + float(slow_rows[1]["response_time"]))),
                (10, 0.5 * (float(fast_rows[2]["response_time"]) + float(slow_rows[2]["response_time"]))),
                (100, 0.5 * (float(fast_rows[3]["response_time"]) + float(slow_rows[3]["response_time"]))),
            ],
        },
        {
            "label": "Slower",
            "color": "#ca8a04",
            "dash": "2,4",
            "points": [
                (1, float(slow_rows[0]["response_time"])),
                (2, float(slow_rows[1]["response_time"])),
                (10, float(slow_rows[2]["response_time"])),
                (100, float(slow_rows[3]["response_time"])),
            ],
        },
        ],
    )
    return series


def _write_variant(
    *,
    output_stem: str,
    lam: float,
    include_erlang_d: bool,
    include_cme_d: bool,
) -> None:
    series = _compute_series(
        lam,
        include_erlang_d=include_erlang_d,
        include_cme_d=include_cme_d,
    )
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    csv_path.write_text("\n".join(["lambda,series,phases,response_time", *_csv_rows(lam, series)]) + "\n", encoding="utf-8")
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    _write_response_svg(
        svg_path,
        series,
        title=f"Mean Response Time by Phase Count for K={CAPACITY}, lambda={lam}",
    )
    print(f"Saved {svg_path}")
    print(f"Saved {csv_path}")


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


def _solve_cme_d_best_effort(lam: float, order: int) -> dict[str, float | str] | None:
    for solver in ("signed_fast", "signed"):
        try:
            row = solve_one_server_cme(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                order=order,
                precision_digits=80,
                solver=solver,
            )[0]
            response_time = float(row["response_time"])
            if math.isfinite(response_time) and response_time > 0.0:
                return row
        except (RuntimeError, FileNotFoundError, NotImplementedError):
            continue

    print(f"[warning] Skipping CME {order}/D at lambda={lam}: all solver modes failed.")
    return None


def _csv_rows(lam: float, series: list[dict[str, object]]) -> list[str]:
    rows: list[str] = []
    for item in series:
        label = str(item["label"])
        for phases, value in item["points"]:  # type: ignore[index]
            rows.append(f"{lam},{label},{phases},{value:.18e}")
    return rows


def _write_response_svg(path: Path, series: list[dict[str, object]], *, title: str) -> None:
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
    y_values = [y for _, y in all_points if math.isfinite(y)]

    x_min = min(x_values)
    x_max = max(x_values)
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.12, 0.005)
    y_axis_min = max(0.0, y_min - y_pad)
    y_axis_max = y_max + y_pad

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (value - y_axis_min) / (y_axis_max - y_axis_min)
        return top + plot_height - frac * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick_index in range(6):
        value = y_axis_min + (y_axis_max - y_axis_min) * tick_index / 5
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">{_format_tick(value)}</text>')

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
        label = str(item["label"])
        color = str(item["color"])
        dash = str(item["dash"])
        marker = METHOD_MARKERS.get(label, "circle")
        points = [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
        line_source = item.get("line_points", item["points"])
        line_points = [(float(x), float(y)) for x, y in line_source]  # type: ignore[assignment]
        finite_line_points = [(x, y) for x, y in line_points if math.isfinite(y)]
        finite_points = [(x, y) for x, y in points if math.isfinite(y)]
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in finite_line_points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(finite_line_points) > 1:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{polyline}"/>')
        for x, y in finite_points:
            px = x_to_px(x)
            py = y_to_px(y)
            parts.extend(_marker_parts(px, py, marker, color, "#ffffff", 5.6))

    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">number of phases (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Mean response time R</text>')

    legend_width = 220
    legend_height = 36 * len(series) + 20
    legend_rect_x = left + plot_width - legend_width - 14
    legend_rect_y = top + 14
    legend_x = legend_rect_x + 16
    legend_y = legend_rect_y + 24
    parts.append(f'<rect x="{legend_rect_x}" y="{legend_rect_y}" width="{legend_width}" height="{legend_height}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')
    for index, item in enumerate(series):
        y = legend_y + index * 36
        label = str(item["label"])
        color = str(item["color"])
        dash = str(item["dash"])
        marker = METHOD_MARKERS.get(label, "circle")
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 5.6))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#111111">{_esc(label)}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


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
    first = {x: y for x, y in first_points if math.isfinite(y)}
    second = {x: y for x, y in second_points if math.isfinite(y)}
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


def _write_combined_response_svg(
    path: Path,
    lambda_series: dict[float, list[dict[str, object]]],
    *,
    title: str,
) -> None:
    width = 1260
    height = 820
    left = 110
    right = 340
    top = 45
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [
        (float(x), float(y))
        for series in lambda_series.values()
        for item in series
        for x, y in item.get("line_points", item["points"])  # type: ignore[index]
        if math.isfinite(float(y))
    ]
    x_values = [x for x, _ in all_points]
    y_values = [y for _, y in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    log_x_min = math.log10(x_min)
    log_x_max = math.log10(x_max)

    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.12, 0.005)
    y_axis_min = max(0.0, y_min - y_pad)
    y_axis_max = y_max + y_pad

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (value - y_axis_min) / (y_axis_max - y_axis_min)
        return top + plot_height - frac * plot_height

    first_series = lambda_series[LAMBDA_VALUES[0]]
    method_styles = {
        str(item["label"]): (
            str(item["color"]),
            METHOD_MARKERS.get(str(item["label"]), "circle"),
        )
        for item in first_series
    }

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick_index in range(6):
        value = y_axis_min + (y_axis_max - y_axis_min) * tick_index / 5
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">{_format_tick(value)}</text>')

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
            finite_line_points = [(x, y) for x, y in line_points if math.isfinite(y)]
            finite_points = [(x, y) for x, y in points if math.isfinite(y)]
            if len(finite_line_points) > 1:
                polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in finite_line_points)
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash_attr} points="{polyline}"/>')
            for x, y in finite_points:
                px = x_to_px(x)
                py = y_to_px(y)
                parts.append(f'<line x1="{px - 11:.2f}" y1="{py:.2f}" x2="{px + 11:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
                parts.extend(_marker_parts(px, py, marker, color, "#ffffff", 4.8))

    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">number of phases (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Mean response time R</text>')

    legend_x = left + plot_width + 28
    legend_y = top + 20
    method_items = list(method_styles.items())
    lambda_items = [(f"lambda={lam:g}", LAMBDA_DASH[lam]) for lam in LAMBDA_VALUES]
    legend_count = len(method_items) + len(lambda_items) + 1
    parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="285" height="{34 * legend_count + 44}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')

    parts.append(f'<text x="{legend_x}" y="{legend_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#111111">Methods</text>')
    for index, (label, (color, marker)) in enumerate(method_items):
        y = legend_y + 24 + index * 34
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 5.2))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{_esc(label)}</text>')

    lambda_y = legend_y + 24 + len(method_items) * 34 + 24
    parts.append(f'<text x="{legend_x}" y="{lambda_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#111111">Lambda style</text>')
    for index, (label, dash) in enumerate(lambda_items):
        y = lambda_y + 24 + index * 34
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="#111111" stroke-width="2.5"{dash_attr}/>')
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
    return [f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']


def _format_tick(value: float) -> str:
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _lambda_key(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
