from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

import plot_distribution_accuracy as base


LAMBDA_DASH = {
    0.1: "",
    0.5: "8,4",
    0.9: "2,4",
}
LAMBDA_VALUES = [0.1]
REFERENCE_SPECS = [
    spec
    for spec in base.REFERENCE_SPECS
    if str(spec["key"]) == "cme20"
]


def _max_relative_error(dist: np.ndarray, reference: np.ndarray) -> float:
    relative = np.divide(
        np.abs(dist - reference),
        reference,
        out=np.full_like(reference, np.inf, dtype=float),
        where=reference != 0.0,
    )
    return float(np.max(relative))


def main() -> None:
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base._mean_relative_error = _max_relative_error

    csv_lines = [
        "lambda,reference_key,reference_label,series,label,phase_or_order,runtime_seconds,error_max_relative,reference",
    ]
    combined_runtime_points: dict[str, list[dict[str, object]]] = {
        str(reference_spec["key"]): []
        for reference_spec in REFERENCE_SPECS
    }
    combined_phase_points: dict[str, list[dict[str, object]]] = {
        str(reference_spec["key"]): []
        for reference_spec in REFERENCE_SPECS
    }

    for lam in LAMBDA_VALUES:
        for reference_spec in REFERENCE_SPECS:
            reference_key = str(reference_spec["key"])
            reference = base._normalise(reference_spec["builder"](lam))
            phase_points, runtime_points = base._compute_points(
                lam,
                reference,
                reference_spec["label"],
                include_embedded_variants=True,
            )
            runtime_points = _with_fresh_runtime(lam, reference, runtime_points)
            csv_lines.extend(base._csv_rows(lam, phase_points, runtime_points, reference_spec))
            combined_phase_points[reference_key].extend(
                {**point, "lambda": lam}
                for point in phase_points
            )
            combined_runtime_points[reference_key].extend(
                {**point, "lambda": lam}
                for point in runtime_points
            )

            phase_svg = base.OUTPUT_DIR / (
                f"distribution_error_max_vs_phase_lambda_{str(lam).replace('.', '_')}_ref_{reference_spec['key']}.svg"
            )
            runtime_svg = base.OUTPUT_DIR / (
                f"distribution_error_max_vs_runtime_lambda_{str(lam).replace('.', '_')}_ref_{reference_spec['key']}.svg"
            )

            _write_phase_svg(
                phase_svg,
                phase_points,
                title=f"Distribution Error vs Phase Count for K={base.CAPACITY}, lambda={lam}",
                reference_label=str(reference_spec["label"]),
            )
            _write_runtime_svg(
                runtime_svg,
                runtime_points,
                title=f"Distribution Error vs Runtime for K={base.CAPACITY}, lambda={lam}",
                reference_label=str(reference_spec["label"]),
            )

            print(f"Saved {phase_svg}")
            print(f"Saved {runtime_svg}")

    for reference_spec in REFERENCE_SPECS:
        reference_key = str(reference_spec["key"])
        combined_phase_svg = base.OUTPUT_DIR / (
            f"distribution_error_max_vs_phase_all_lambdas_ref_{reference_key}.svg"
        )
        _write_combined_phase_svg(
            combined_phase_svg,
            combined_phase_points[reference_key],
            title=f"Distribution Error vs Phase Count for K={base.CAPACITY}",
            reference_label=str(reference_spec["label"]),
        )
        print(f"Saved {combined_phase_svg}")

        combined_runtime_svg = base.OUTPUT_DIR / (
            f"distribution_error_max_vs_runtime_all_lambdas_ref_{reference_key}.svg"
        )
        _write_combined_runtime_svg(
            combined_runtime_svg,
            combined_runtime_points[reference_key],
            title=f"Distribution Error vs Runtime for K={base.CAPACITY}",
            reference_label=str(reference_spec["label"]),
        )
        print(f"Saved {combined_runtime_svg}")

    csv_path = base.OUTPUT_DIR / "distribution_error_accuracy_max.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")


def _with_fresh_runtime(
    lam: float,
    reference: np.ndarray,
    points: list[dict[str, object]],
) -> list[dict[str, object]]:
    measured: list[dict[str, object]] = []
    for point in points:
        label = str(point["label"])
        dist, elapsed = _measure_distribution_runtime_in_fresh_process(lam, label)
        error = _max_relative_error(base._normalise(dist), reference)
        measured.append({**point, "runtime": elapsed, "error": error})
        print(f"Fresh runtime for {label}: {elapsed:.6g} s")
    return measured


def _measure_distribution_runtime_in_fresh_process(lam: float, label: str) -> tuple[np.ndarray, float]:
    child_code = r'''
import json
import sys
import time

import plot_distribution_accuracy as base

print("READY", flush=True)
request = json.loads(sys.stdin.readline())
lam = float(request["lambda"])
label = str(request["label"])

if label == "Exponential":
    solver = lambda: base._exponential_distribution(lam)
elif label == "Exponential/D":
    solver = lambda: base._erlang_d_distribution(lam, 1)
elif label.startswith("Erlang /D "):
    solver = lambda: base._erlang_d_distribution(lam, int(label.split()[2]))
elif label.startswith("Erlang "):
    solver = lambda: base._erlang_distribution(lam, int(label.split()[1]))
elif label.startswith("CME "):
    solver = lambda: base._cme_distribution(int(label.split()[1]), lam)
elif label == "faster w/o":
    solver = lambda: base._fast_slow_distribution("faster", lam)
elif label == "slower w/o":
    solver = lambda: base._fast_slow_distribution("slower", lam)
elif label.startswith("faster"):
    solver = lambda: base._fast_slow_distribution(label, lam)
elif label.startswith("slower"):
    solver = lambda: base._fast_slow_distribution(label, lam)
elif label == "average w/o":
    solver = lambda: base._average_fast_slow_distribution("faster", "slower", lam)
elif label.startswith("average"):
    phase = int(label.split()[1])
    solver = lambda: base._average_fast_slow_distribution(f"faster {phase}", f"slower {phase}", lam)
else:
    raise ValueError(f"Unsupported method label: {label}")

start = time.perf_counter()
dist = base._normalise(solver())
elapsed = time.perf_counter() - start
print(json.dumps({"elapsed": elapsed, "dist": [float(value) for value in dist]}), flush=True)
'''
    proc = subprocess.Popen(
        [sys.executable, "-c", child_code],
        cwd=Path(__file__).resolve().parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    ready = proc.stdout.readline().strip()
    if ready != "READY":
        _stdout, stderr = proc.communicate(timeout=5)
        raise RuntimeError(f"{label}: child process did not become ready. stderr={stderr}")

    proc.stdin.write(json.dumps({"lambda": lam, "label": label}) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    _stdout, stderr = proc.communicate(timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: child process failed with code {proc.returncode}. stderr={stderr}")
    if not line:
        raise RuntimeError(f"{label}: child process produced no timing output.")

    payload = json.loads(line)
    return np.array(payload["dist"], dtype=float), float(payload["elapsed"])


def _write_phase_svg(path, points, *, title: str, reference_label: str) -> None:
    plotted = [point for point in points if not bool(point["reference"]) and float(point["error"]) > 0.0]
    width = 1260
    height = 800
    left = 120
    right = 320
    top = 90
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["x_value"]) for point in plotted]
    y_values = [float(point["error"]) for point in plotted]
    log_x_min = math.log10(min(x_values))
    log_x_max = math.log10(max(x_values))
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts = base._svg_shell(width, height, title, f"Max relative error of (P0,...,PK) vs {reference_label} reference")
    parts.extend(base._log_grid(left, top, plot_width, plot_height, y_min_power, y_max_power, vertical=False))
    parts.extend(base._phase_ticks(left, top, plot_height, x_values, x_to_px))
    parts.extend(base._axes(left, top, plot_width, plot_height))
    parts.extend(base._series_polylines(plotted, x_to_px, y_to_px, x_key="x_value"))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Phase count / CME order (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Max relative error over P0..PK (log scale)</text>')
    parts.extend(base._legend(left + plot_width + 28, top + 20, points))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _write_runtime_svg(path, points, *, title: str, reference_label: str) -> None:
    plotted = [
        point
        for point in points
        if (
            not bool(point["reference"])
            and float(point["error"]) > 0.0
            and math.isfinite(float(point["error"]))
            and math.isfinite(float(point["runtime"]))
        )
    ]
    width = 1260
    height = 800
    left = 120
    right = 70
    top = 45
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [max(float(point["runtime"]), 1e-9) for point in plotted]
    y_values = [float(point["error"]) for point in plotted]
    x_min_power = math.floor(math.log10(min(x_values)))
    x_max_power = math.ceil(math.log10(max(x_values)))
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - x_min_power) / (x_max_power - x_min_power)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    parts.extend(_log_x_ticks(left, top, plot_width, plot_height, x_min_power, x_max_power, x_to_px))
    parts.extend(_log_y_ticks(left, top, plot_height, y_min_power, y_max_power, y_to_px))
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.extend(_runtime_error_lines(plotted, x_to_px, y_to_px))
    parts.extend(_runtime_error_markers(plotted, x_to_px, y_to_px))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Runtime (seconds, log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Max relative error over P0..PK (log scale)</text>')
    legend_width = 280
    legend_rect_x = left + plot_width - legend_width - 14
    legend_rect_y = top + 14
    parts.extend(_runtime_error_legend(legend_rect_x + 16, legend_rect_y + 24, plotted))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _log_x_ticks(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    min_power: int,
    max_power: int,
    x_to_px,
) -> list[str]:
    parts: list[str] = []
    for power in range(min_power, max_power + 1):
        value = 10.0 ** power
        x = x_to_px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">1e{power:+d}</text>')
    return parts


def _log_y_ticks(
    left: int,
    top: int,
    plot_height: int,
    min_power: int,
    max_power: int,
    y_to_px,
) -> list[str]:
    parts: list[str] = []
    for power in range(min_power, max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">1e{power:+d}</text>')
    return parts


def _runtime_error_lines(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for point in points:
        label = str(point["label"])
        series = str(point["series"])
        if label == "Exponential":
            group_key = "Exponential"
        elif series == "Exponential/D":
            group_key = "Exponential/D"
        elif series in {"Erlang", "Erlang Iterative"}:
            group_key = "Erlang"
        elif series == "Erlang /D":
            group_key = series
        elif series == "CME":
            group_key = "CME"
        elif series == "Faster":
            group_key = "Faster"
        elif series == "Slower":
            group_key = "Slower"
        elif series == "Average":
            group_key = "Average"
        else:
            group_key = label
        grouped.setdefault(group_key, []).append(point)

    line_styles = {
        "Exponential": ("#5c6b73", ""),
        "Exponential/D": ("#a3a3a3", ""),
        "Erlang": ("#1d4ed8", ""),
        "Erlang /D": ("#60a5fa", "4,2"),
        "CME": ("#0c7c59", ""),
        "faster w/o": ("#f4a261", "6,4"),
        "Faster": ("#d17b0f", "6,4"),
        "Average": ("#7c3aed", ""),
        "slower w/o": ("#c1121f", "2,4"),
        "Slower": ("#9b2226", "2,4"),
    }
    parts: list[str] = []
    for group_key, group in grouped.items():
        group.sort(key=lambda point: float(point["x_value"]))
        if len(group) < 2:
            continue
        color, dash = line_styles.get(group_key, (str(group[0]["color"]), str(group[0]["dash"])))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        polyline = " ".join(
            f"{x_to_px(max(float(point['runtime']), 1e-9)):.2f},{y_to_px(float(point['error'])):.2f}"
            for point in group
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{polyline}"/>')
    for source_key, target_key, bridge_color in (
        ("Exponential", "Erlang", "#5c6b73"),
        ("Exponential/D", "Erlang /D", "#a3a3a3"),
    ):
        source = grouped.get(source_key, [])
        target = grouped.get(target_key, [])
        if source and target:
            target.sort(key=lambda point: float(point["x_value"]))
            bridge = [source[0], target[0]]
            polyline = " ".join(
                f"{x_to_px(max(float(point['runtime']), 1e-9)):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in bridge
            )
            parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="2.6" points="{polyline}"/>')
    return parts


def _runtime_error_markers(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    parts: list[str] = []
    for point in points:
        px = x_to_px(max(float(point["runtime"]), 1e-9))
        py = y_to_px(float(point["error"]))
        parts.extend(_marker_parts(px, py, str(point["symbol"]), str(point["color"]), "#ffffff", 5.4))
    return parts


def _runtime_error_legend(legend_x: int, legend_y: int, points: list[dict[str, object]]) -> list[str]:
    method_items = _combined_method_items(points)
    item_gap = 30
    parts = [
        f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="280" height="{item_gap * len(method_items) + 24}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>',
    ]
    for index, (label, color, symbol, dash) in enumerate(method_items):
        y = legend_y + index * item_gap
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(_marker_parts(legend_x + 16, y, symbol, color, "#ffffff", 5.2))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{base._esc(label)}</text>')
    return parts


def _write_combined_phase_svg(path, points, *, title: str, reference_label: str) -> None:
    plotted = _phase_plotted(points)
    width = 1320
    height = 840
    left = 120
    right = 360
    top = 90
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["x_value"]) for point in plotted]
    y_values = [float(point["error"]) for point in plotted]
    log_x_min = math.log10(min(x_values))
    log_x_max = math.log10(max(x_values))
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - log_x_min) / (log_x_max - log_x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts = base._svg_shell(
        width,
        height,
        title,
        f"Max relative error of (P0,...,PK) vs {reference_label} reference; lambda is line style",
    )
    parts.extend(base._log_grid(left, top, plot_width, plot_height, y_min_power, y_max_power, vertical=False))
    parts.extend(base._phase_ticks(left, top, plot_height, x_values, x_to_px))
    parts.extend(base._axes(left, top, plot_width, plot_height))
    parts.extend(_combined_phase_lines(plotted, x_to_px, y_to_px))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Phase count / CME order (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Max relative error over P0..PK (log scale)</text>')
    parts.extend(_combined_runtime_legend(left + plot_width + 28, top + 20, plotted))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _write_combined_runtime_svg(path, points, *, title: str, reference_label: str) -> None:
    plotted = _runtime_plotted(points)
    width = 1320
    height = 840
    left = 120
    right = 360
    top = 90
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [max(float(point["runtime"]), 1e-9) for point in plotted]
    y_values = [float(point["error"]) for point in plotted]
    x_min_power = math.floor(math.log10(min(x_values)))
    x_max_power = math.ceil(math.log10(max(x_values)))
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - x_min_power) / (x_max_power - x_min_power)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts = base._svg_shell(
        width,
        height,
        title,
        f"Max relative error of (P0,...,PK) vs {reference_label} reference; lambda is line style",
    )
    parts.extend(base._log_grid(left, top, plot_width, plot_height, y_min_power, y_max_power, vertical=False))
    parts.extend(base._runtime_ticks(left, top, plot_width, plot_height, x_min_power, x_max_power, x_to_px))
    parts.extend(base._axes(left, top, plot_width, plot_height))
    parts.extend(_combined_runtime_lines(plotted, x_to_px, y_to_px))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Runtime (seconds, log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Max relative error over P0..PK (log scale)</text>')
    parts.extend(_combined_runtime_legend(left + plot_width + 28, top + 20, plotted))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _runtime_plotted(points: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        point
        for point in points
        if (
            not bool(point["reference"])
            and float(point["error"]) > 0.0
        )
    ]


def _phase_plotted(points: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        point
        for point in points
        if not bool(point["reference"]) and float(point["error"]) > 0.0
    ]


def _combined_phase_lines(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for point in points:
        lam = float(point["lambda"])
        family = _line_family(point)
        grouped.setdefault((lam, family), []).append(point)

    parts: list[str] = []
    for (lam, _family), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group.sort(key=lambda point: float(point["x_value"]))
        color = str(group[0]["color"])
        dash = _lambda_dash(lam)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(group) > 1:
            polyline = " ".join(
                f"{x_to_px(float(point['x_value'])):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in group
            )
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash_attr} points="{polyline}"/>')
        for point in group:
            px = x_to_px(float(point["x_value"]))
            py = y_to_px(float(point["error"]))
            parts.append(f'<line x1="{px - 11:.2f}" y1="{py:.2f}" x2="{px + 11:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
            parts.extend(_marker_parts(px, py, str(point["symbol"]), color, "#fffdf8", 5.2))
    for lam in sorted({float(point["lambda"]) for point in points}):
        for source_key, target_key, bridge_color in (
            ("Exponential", "Erlang", "#5c6b73"),
            ("Exponential/D", "Erlang /D", "#a3a3a3"),
        ):
            source = grouped.get((lam, source_key), [])
            target = grouped.get((lam, target_key), [])
            if source and target:
                target.sort(key=lambda point: float(point["x_value"]))
                bridge = [source[0], target[0]]
                polyline = " ".join(
                    f"{x_to_px(float(point['x_value'])):.2f},{y_to_px(float(point['error'])):.2f}"
                    for point in bridge
                )
                parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="2.4" points="{polyline}"/>')
    return parts


def _combined_runtime_lines(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for point in points:
        lam = float(point["lambda"])
        family = _line_family(point)
        grouped.setdefault((lam, family), []).append(point)

    parts: list[str] = []
    for (lam, _family), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group.sort(key=lambda point: float(point["runtime"]))
        color = str(group[0]["color"])
        dash = _lambda_dash(lam)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(group) > 1:
            polyline = " ".join(
                f"{x_to_px(max(float(point['runtime']), 1e-9)):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in group
            )
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash_attr} points="{polyline}"/>')
        for point in group:
            px = x_to_px(max(float(point["runtime"]), 1e-9))
            py = y_to_px(float(point["error"]))
            parts.append(f'<line x1="{px - 11:.2f}" y1="{py:.2f}" x2="{px + 11:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
            parts.extend(_marker_parts(px, py, str(point["symbol"]), color, "#fffdf8", 5.2))
    for lam in sorted({float(point["lambda"]) for point in points}):
        for source_key, target_key, bridge_color in (
            ("Exponential", "Erlang", "#5c6b73"),
            ("Exponential/D", "Erlang /D", "#a3a3a3"),
        ):
            source = grouped.get((lam, source_key), [])
            target = grouped.get((lam, target_key), [])
            if source and target:
                target.sort(key=lambda point: float(point["runtime"]))
                bridge = [source[0], target[0]]
                polyline = " ".join(
                    f"{x_to_px(max(float(point['runtime']), 1e-9)):.2f},{y_to_px(float(point['error'])):.2f}"
                    for point in bridge
                )
                parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="2.4" points="{polyline}"/>')
    return parts


def _combined_runtime_legend(legend_x: int, legend_y: int, points: list[dict[str, object]]) -> list[str]:
    method_items = _combined_method_items(points)
    lambda_items = [(f"lambda={lam:g}", _lambda_dash(lam)) for lam in sorted({float(point["lambda"]) for point in points})]

    item_gap = 30
    heading_gap = 26
    height = item_gap * (len(method_items) + len(lambda_items)) + heading_gap * 2 + 30
    parts = [
        f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="300" height="{height}" rx="16" fill="#fffaf0" stroke="#e6d5b8"/>',
        f'<text x="{legend_x}" y="{legend_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#52606d">Methods</text>',
    ]

    y_start = legend_y + 24
    for index, (label, color, symbol, dash) in enumerate(method_items):
        y = y_start + index * item_gap
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(_marker_parts(legend_x + 16, y, symbol, color, "#fffaf0", 5.2))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">{base._esc(label)}</text>')

    lambda_y = y_start + len(method_items) * item_gap + heading_gap
    parts.append(f'<text x="{legend_x}" y="{lambda_y - 2}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#52606d">Lambda line style</text>')
    for index, (label, dash) in enumerate(lambda_items):
        y = lambda_y + 24 + index * item_gap
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="#1f2933" stroke-width="2.6"{dash_attr}/>')
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">{base._esc(label)}</text>')
    return parts


def _combined_method_items(points: list[dict[str, object]]) -> list[tuple[str, str, str, str]]:
    groups: dict[str, dict[str, object]] = {}
    for point in sorted(points, key=lambda item: (_method_family(item), float(item["x_value"]))):
        rank, family = _method_family(point)
        group = groups.setdefault(
            family,
            {
                "rank": rank,
                "color": str(point["color"]),
                "symbol": str(point["symbol"]),
                "dash": str(point["dash"]),
                "values": [],
            },
        )
        values = group["values"]
        if isinstance(values, list):
            x_value = float(point["x_value"])
            if x_value not in values:
                values.append(x_value)

    items: list[tuple[int, str, str, str, str]] = []
    for family, group in groups.items():
        values = group["values"]
        if family in {"Exponential", "Exponential/D", "faster w/o", "slower w/o"}:
            label = family
        elif isinstance(values, list):
            phases = ", ".join(
                "w/o" if family == "average" and math.isclose(float(value), 1.0) else _phase_label(float(value))
                for value in sorted(values)
            )
            label = f"{family} {phases}"
        else:
            label = family
        items.append((int(group["rank"]), label, str(group["color"]), str(group["symbol"]), str(group["dash"])))
    items.sort(key=lambda item: item[0])
    return [(label, color, symbol, dash) for _rank, label, color, symbol, dash in items]


def _method_family(point: dict[str, object]) -> tuple[int, str]:
    label = str(point["label"])
    series = str(point["series"])
    if label == "Exponential":
        return (0, "Exponential")
    if series == "Exponential/D":
        return (1, "Exponential/D")
    if series in {"Erlang", "Erlang Iterative"}:
        return (2, "Erlang")
    if series == "Erlang /D":
        return (3, series)
    if series == "CME":
        return (4, "CME")
    if label == "faster w/o":
        return (7, "faster w/o")
    if series == "Faster":
        return (8, "faster")
    if label == "slower w/o":
        return (9, "slower w/o")
    if series == "Slower":
        return (10, "slower")
    if series == "Average":
        return (11, "average")
    return (12, label)


def _line_family(point: dict[str, object]) -> str:
    label = str(point["label"])
    series = str(point["series"])
    if label == "Exponential":
        return "Exponential"
    if series == "Exponential/D":
        return "Exponential/D"
    if series in {"Erlang", "Erlang Iterative"}:
        return "Erlang"
    if series == "Erlang /D":
        return series
    if series == "CME":
        return "CME"
    if series == "Faster":
        return "faster"
    if series == "Slower":
        return "slower"
    if series == "Average":
        return "average"
    return label


def _phase_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _lambda_dash(lam: float) -> str:
    for key, dash in LAMBDA_DASH.items():
        if math.isclose(lam, key):
            return dash
    return "5,3"


def _marker_parts(
    cx: float,
    cy: float,
    symbol: str,
    color: str,
    stroke: str,
    size: float,
) -> list[str]:
    if symbol == "square":
        return [f'<rect x="{cx - size:.2f}" y="{cy - size:.2f}" width="{2 * size:.2f}" height="{2 * size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if symbol == "diamond":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy:.2f} {cx:.2f},{cy + size + 1:.2f} {cx - size - 1:.2f},{cy:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if symbol == "triangle":
        return [f'<polygon points="{cx:.2f},{cy - size - 1:.2f} {cx + size + 1:.2f},{cy + size:.2f} {cx - size - 1:.2f},{cy + size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    if symbol == "triangle-down":
        return [f'<polygon points="{cx - size - 1:.2f},{cy - size:.2f} {cx + size + 1:.2f},{cy - size:.2f} {cx:.2f},{cy + size + 1:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']
    return [f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="1.6"/>']


if __name__ == "__main__":
    main()
