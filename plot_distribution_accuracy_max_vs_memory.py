from __future__ import annotations

import ctypes
import json
import math
import subprocess
import sys
import statistics
import time
from pathlib import Path

import numpy as np

import plot_distribution_accuracy as base
import plot_distribution_accuracy_max as max_accuracy


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
LAMBDA_VALUES = [0.1]
REFERENCE_SPECS = [
    spec
    for spec in base.REFERENCE_SPECS
    if str(spec["key"]) == "cme20"
]

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEMORY_PROXY_LABELS = {
    "faster w/o": "faster 1",
    "slower w/o": "slower 1",
    "average w/o": "average 1",
}
MEMORY_REPEATS = 3


class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
_PSAPI = ctypes.WinDLL("psapi", use_last_error=True)
_OPEN_PROCESS = _KERNEL32.OpenProcess
_OPEN_PROCESS.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
_OPEN_PROCESS.restype = ctypes.c_void_p
_CLOSE_HANDLE = _KERNEL32.CloseHandle
_CLOSE_HANDLE.argtypes = [ctypes.c_void_p]
_GET_PROCESS_MEMORY_INFO = _PSAPI.GetProcessMemoryInfo
_GET_PROCESS_MEMORY_INFO.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
_GET_PROCESS_MEMORY_INFO.restype = ctypes.c_int


def _max_relative_error(dist: np.ndarray, reference: np.ndarray) -> float:
    relative = np.divide(
        np.abs(dist - reference),
        reference,
        out=np.full_like(reference, np.inf, dtype=float),
        where=reference != 0.0,
    )
    return float(np.max(relative))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = [
        "lambda,reference_key,reference_label,series,label,phase_or_order,peak_memory_mib,error_max_relative,reference",
    ]
    combined_memory_points: dict[str, list[dict[str, object]]] = {
        str(reference_spec["key"]): []
        for reference_spec in REFERENCE_SPECS
    }

    for lam in LAMBDA_VALUES:
        for reference_spec in REFERENCE_SPECS:
            reference_key = str(reference_spec["key"])
            reference = base._normalise(reference_spec["builder"](lam))
            memory_points = _compute_points(lam, reference, str(reference_spec["label"]))
            csv_lines.extend(_csv_rows(lam, memory_points, reference_spec))
            combined_memory_points[reference_key].extend(
                {**point, "lambda": lam}
                for point in memory_points
            )

            memory_svg = OUTPUT_DIR / (
                f"distribution_error_max_vs_memory_lambda_{str(lam).replace('.', '_')}_ref_{reference_spec['key']}.svg"
            )

            _write_memory_svg(
                memory_svg,
                memory_points,
                title=f"Distribution Error vs Memory for K={base.CAPACITY}, lambda={lam}",
                reference_label=str(reference_spec["label"]),
            )

            print(f"Saved {memory_svg}")

    for reference_spec in REFERENCE_SPECS:
        reference_key = str(reference_spec["key"])
        combined_memory_svg = OUTPUT_DIR / (
            f"distribution_error_max_vs_memory_all_lambdas_ref_{reference_key}.svg"
        )
        _write_combined_memory_svg(
            combined_memory_svg,
            combined_memory_points[reference_key],
            title=f"Distribution Error vs Memory for K={base.CAPACITY}",
            reference_label=str(reference_spec["label"]),
        )
        print(f"Saved {combined_memory_svg}")

    csv_path = OUTPUT_DIR / "distribution_error_accuracy_max_memory.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")


def _compute_points(lam: float, reference: np.ndarray, reference_label: str) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []

    configs = [
        {
            "series": "Exponential",
            "label": "Exponential",
            "x_value": 1,
            "color": "#5c6b73",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._exponential_distribution(lam),
        },
        {
            "series": "Exponential/D",
            "label": "Exponential/D",
            "x_value": 1,
            "color": "#a3a3a3",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._erlang_d_distribution(lam, 1),
        },
        {
            "series": "Erlang",
            "label": "Erlang 2",
            "x_value": 2,
            "color": "#1d4ed8",
            "dash": "",
            "symbol": "square",
            "solver": lambda: base._erlang_distribution(lam, 2),
        },
        {
            "series": "Erlang",
            "label": "Erlang 10",
            "x_value": 10,
            "color": "#1d4ed8",
            "dash": "",
            "symbol": "square",
            "solver": lambda: base._erlang_distribution(lam, 10),
        },
        {
            "series": "Erlang",
            "label": "Erlang 100",
            "x_value": 100,
            "color": "#1d4ed8",
            "dash": "2,4",
            "symbol": "square",
            "solver": lambda: base._erlang_distribution(lam, 100),
        },
        {
            "series": "Erlang /D",
            "label": "Erlang /D 2",
            "x_value": 2,
            "color": "#60a5fa",
            "dash": "4,2",
            "symbol": "square",
            "solver": lambda: base._erlang_d_distribution(lam, 2),
        },
        {
            "series": "Erlang /D",
            "label": "Erlang /D 10",
            "x_value": 10,
            "color": "#60a5fa",
            "dash": "4,2",
            "symbol": "square",
            "solver": lambda: base._erlang_d_distribution(lam, 10),
        },
        {
            "series": "Erlang /D",
            "label": "Erlang /D 100",
            "x_value": 100,
            "color": "#60a5fa",
            "dash": "4,2",
            "symbol": "square",
            "solver": lambda: base._erlang_d_distribution(lam, 100),
        },
        {
            "series": "CME",
            "label": "CME 5",
            "x_value": 5,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: base._cme_distribution(5, lam),
        },
        {
            "series": "CME",
            "label": "CME 10",
            "x_value": 10,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: base._cme_distribution(10, lam),
        },
        {
            "series": "CME",
            "label": "CME 15",
            "x_value": 15,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: base._cme_distribution(15, lam),
            "reference": reference_label == "CME 15",
        },
        {
            "series": "Faster",
            "label": "faster w/o",
            "x_value": 1,
            "color": "#f4a261",
            "dash": "6,4",
            "symbol": "triangle",
        },
        {
            "series": "Faster",
            "label": "faster 2",
            "x_value": 2,
            "color": "#c1121f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: base._fast_slow_distribution("faster 2", lam),
        },
        {
            "series": "Faster",
            "label": "faster 10",
            "x_value": 10,
            "color": "#c1121f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: base._fast_slow_distribution("faster 10", lam),
        },
        {
            "series": "Faster",
            "label": "faster 100",
            "x_value": 100,
            "color": "#c1121f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: base._fast_slow_distribution("faster 100", lam),
        },
        {
            "series": "Slower",
            "label": "slower w/o",
            "x_value": 1,
            "color": "#f4a261",
            "dash": "2,4",
            "symbol": "triangle-down",
        },
        {
            "series": "Slower",
            "label": "slower 2",
            "x_value": 2,
            "color": "#c1121f",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: base._fast_slow_distribution("slower 2", lam),
        },
        {
            "series": "Slower",
            "label": "slower 10",
            "x_value": 10,
            "color": "#c1121f",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: base._fast_slow_distribution("slower 10", lam),
        },
        {
            "series": "Slower",
            "label": "slower 100",
            "x_value": 100,
            "color": "#c1121f",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: base._fast_slow_distribution("slower 100", lam),
        },
        {
            "series": "Average",
            "label": "average w/o",
            "x_value": 1,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._average_fast_slow_distribution("faster", "slower", lam),
        },
        {
            "series": "Average",
            "label": "average 2",
            "x_value": 2,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._average_fast_slow_distribution("faster 2", "slower 2", lam),
        },
        {
            "series": "Average",
            "label": "average 10",
            "x_value": 10,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._average_fast_slow_distribution("faster 10", "slower 10", lam),
        },
        {
            "series": "Average",
            "label": "average 100",
            "x_value": 100,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: base._average_fast_slow_distribution("faster 100", "slower 100", lam),
        },
    ]

    for config in configs:
        label = str(config["label"])
        distributions: list[np.ndarray] = []
        memory_samples: list[int] = []
        elapsed_samples: list[float] = []
        for _ in range(MEMORY_REPEATS):
            start = time.perf_counter()
            sample_dist, bytes_used = _measure_distribution_in_fresh_process(lam, label)
            proxy_label = MEMORY_PROXY_LABELS.get(label)
            if proxy_label is not None:
                _proxy_dist, bytes_used = _measure_distribution_in_fresh_process(lam, proxy_label)
            elapsed_samples.append(time.perf_counter() - start)
            distributions.append(sample_dist)
            memory_samples.append(bytes_used)

        # The stationary distribution is deterministic; use the first sample
        # for the error and the median for noisy process-level measurements.
        dist = distributions[0]
        elapsed = statistics.median(elapsed_samples)
        bytes_used = int(statistics.median(memory_samples))
        clean = base._normalise(dist)
        error = _max_relative_error(clean, reference)
        points.append(
            {
                "series": config["series"],
                "label": config["label"],
                "x_value": config["x_value"],
                "runtime": elapsed,
                "memory_mib": bytes_used / (1024 * 1024),
                "error": error,
                "color": config["color"],
                "dash": config["dash"],
                "symbol": config["symbol"],
                "reference": bool(config.get("reference", False)),
            }
        )

    return points


def _measure_distribution_in_fresh_process(lam: float, label: str) -> tuple[np.ndarray, int]:
    child_code = r'''
import json
import sys

import plot_distribution_accuracy as base

print("READY", flush=True)
request = json.loads(sys.stdin.readline())
lam = float(request["lambda"])
label = str(request["label"])

if label == "Exponential":
    dist = base._exponential_distribution(lam)
elif label == "Exponential/D":
    dist = base._erlang_d_distribution(lam, 1)
elif label.startswith("Erlang /D "):
    dist = base._erlang_d_distribution(lam, int(label.split()[2]))
elif label.startswith("Erlang "):
    dist = base._erlang_distribution(lam, int(label.split()[1]))
elif label.startswith("CME "):
    dist = base._cme_distribution(int(label.split()[1]), lam)
elif label == "faster w/o":
    dist = base._fast_slow_distribution("faster", lam)
elif label == "slower w/o":
    dist = base._fast_slow_distribution("slower", lam)
elif label.startswith("faster"):
    dist = base._fast_slow_distribution(label, lam)
elif label.startswith("slower"):
    dist = base._fast_slow_distribution(label, lam)
elif label == "average 1" or label == "average w/o":
    dist = base._average_fast_slow_distribution("faster", "slower", lam)
elif label.startswith("average"):
    phase = int(label.split()[1])
    dist = base._average_fast_slow_distribution(f"faster {phase}", f"slower {phase}", lam)
else:
    raise ValueError(f"Unsupported method label: {label}")

dist = base._normalise(dist)
print(json.dumps([float(value) for value in dist]), flush=True)
# Keep the process alive while the parent reads the OS peak counter.
sys.stdin.readline()
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

    baseline_info = _process_memory_info(proc.pid)
    baseline_peak = baseline_info[1] if baseline_info is not None else 0
    try:
        proc.stdin.write(json.dumps({"lambda": lam, "label": label}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        current_info = _process_memory_info(proc.pid)
        peak = current_info[1] if current_info is not None else baseline_peak
        proc.stdin.write("STOP\n")
        proc.stdin.flush()
        _stdout, stderr = proc.communicate(timeout=180)
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.write("STOP\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: child process failed with code {proc.returncode}. stderr={stderr}")
    if not line:
        raise RuntimeError(f"{label}: child process produced no distribution output.")

    return np.array(json.loads(line), dtype=float), max(0, peak - baseline_peak)


def _process_memory_info(pid: int) -> tuple[int, int, int] | None:
    handle = _OPEN_PROCESS(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not handle:
        return None
    try:
        counters = _PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        ok = _GET_PROCESS_MEMORY_INFO(handle, ctypes.byref(counters), counters.cb)
        if not ok:
            return None
        return (
            int(counters.WorkingSetSize),
            int(counters.PeakWorkingSetSize),
            int(counters.PrivateUsage),
        )
    finally:
        _CLOSE_HANDLE(handle)


def _csv_rows(
    lam: float,
    points: list[dict[str, object]],
    reference_spec: dict[str, object],
) -> list[str]:
    rows: list[str] = []
    for point in points:
        rows.append(
            ",".join(
                [
                    str(lam),
                    str(reference_spec["key"]),
                    str(reference_spec["label"]),
                    str(point["series"]),
                    str(point["label"]),
                    str(point["x_value"]),
                    f"{float(point['memory_mib']):.6f}",
                    f"{float(point['error']):.18e}",
                    "Yes" if bool(point["reference"]) else "No",
                ],
            )
        )
    return rows


def _write_memory_svg(path: Path, points: list[dict[str, object]], *, title: str, reference_label: str) -> None:
    plotted = [
        point
        for point in points
        if (
            not bool(point["reference"])
            and float(point["error"]) > 0.0
            and float(point["memory_mib"]) > 0.0
            and math.isfinite(float(point["error"]))
            and math.isfinite(float(point["memory_mib"]))
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

    x_values = [float(point["memory_mib"]) for point in plotted]
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
    parts.extend(_memory_error_lines(plotted, x_to_px, y_to_px))
    parts.extend(_memory_error_markers(plotted, x_to_px, y_to_px))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Peak memory increase (MiB, log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Max relative error over P0..PK (log scale)</text>')
    legend_width = 280
    legend_rect_x = left + plot_width - legend_width - 14
    legend_rect_y = top + 14
    parts.extend(_memory_error_legend(legend_rect_x + 16, legend_rect_y + 24, plotted))
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


def _memory_error_lines(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for point in points:
        label = str(point["label"])
        series = str(point["series"])
        if label == "Exponential" or series == "Exponential":
            group_key = "Exponential"
        elif series == "Exponential/D":
            group_key = "Exponential/D"
        elif series in {"Erlang", "Erlang Iterative"}:
            group_key = "Erlang"
        elif series == "Erlang /D":
            group_key = "Erlang /D"
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
        "Faster": ("#f4a261", "6,4"),
        "Average": ("#7c3aed", ""),
        "slower w/o": ("#f4a261", "2,4"),
        "Slower": ("#f4a261", "2,4"),
    }
    parts: list[str] = []
    for group_key, group in grouped.items():
        group.sort(key=lambda point: float(point["x_value"]))
        if len(group) < 2:
            continue
        color, dash = line_styles.get(group_key, (str(group[0]["color"]), str(group[0]["dash"])))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if group_key in {"Faster", "Slower"}:
            for index in range(len(group) - 1):
                segment_color = "#f4a261" if index == 0 else "#c1121f"
                polyline = " ".join(
                    f"{x_to_px(float(point['memory_mib'])):.2f},{y_to_px(float(point['error'])):.2f}"
                    for point in group[index:index + 2]
                )
                parts.append(f'<polyline fill="none" stroke="{segment_color}" stroke-width="2.6"{dash_attr} points="{polyline}"/>')
        else:
            polyline = " ".join(
                f"{x_to_px(float(point['memory_mib'])):.2f},{y_to_px(float(point['error'])):.2f}"
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
            target.sort(key=lambda point: float(point["memory_mib"]))
            bridge = [source[0], target[0]]
            polyline = " ".join(
                f"{x_to_px(float(point['memory_mib'])):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in bridge
            )
            parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="2.6" points="{polyline}"/>')
    return parts


def _memory_error_markers(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    parts: list[str] = []
    for point in points:
        px = x_to_px(float(point["memory_mib"]))
        py = y_to_px(float(point["error"]))
        parts.extend(max_accuracy._marker_parts(px, py, str(point["symbol"]), str(point["color"]), "#ffffff", 5.4))
    return parts


def _memory_error_legend(legend_x: int, legend_y: int, points: list[dict[str, object]]) -> list[str]:
    method_items = _memory_error_method_items(points)
    item_gap = 30
    parts = [
        f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="280" height="{item_gap * len(method_items) + 24}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>',
    ]
    for index, (label, color, symbol, dash) in enumerate(method_items):
        y = legend_y + index * item_gap
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(max_accuracy._marker_parts(legend_x + 16, y, symbol, color, "#ffffff", 5.2))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{base._esc(label)}</text>')
    return parts


def _memory_error_method_items(points: list[dict[str, object]]) -> list[tuple[str, str, str, str]]:
    raw_items = {
        label: (color, symbol, dash)
        for label, color, symbol, dash in max_accuracy._combined_method_items(points)
    }
    ordered_labels = [
        "Exponential",
        "Exponential/D",
        "Erlang 2, 10, 100",
        "Erlang /D 2, 10, 100",
        "CME 5, 10, 15",
        "faster w/o",
        "average w/o, 2, 10, 100",
        "slower w/o",
        "faster 2, 10, 100",
        "slower 2, 10, 100",
    ]
    display_labels = {
        "Erlang /D 2, 10, 100": "Erlang/D 2, 10, 100",
        "faster w/o": "Faster w/o",
        "average w/o, 2, 10, 100": "Average w/o, 2, 10, 100",
        "slower w/o": "Slower w/o",
        "faster 2, 10, 100": "Faster 2, 10, 100",
        "slower 2, 10, 100": "Slower 2, 10, 100",
    }
    return [
        (display_labels.get(label, label), *raw_items[label])
        for label in ordered_labels
        if label in raw_items
    ]


def _write_combined_memory_svg(path: Path, points: list[dict[str, object]], *, title: str, reference_label: str) -> None:
    plotted = _memory_plotted(points)
    width = 1320
    height = 840
    left = 120
    right = 360
    top = 90
    bottom = 115
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["memory_mib"]) for point in plotted]
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
    parts.extend(_memory_ticks(left, top, plot_width, plot_height, x_min_power, x_max_power, x_to_px))
    parts.extend(base._axes(left, top, plot_width, plot_height))
    parts.extend(_combined_memory_lines(plotted, x_to_px, y_to_px))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Peak memory increase (MiB, log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Max relative error over P0..PK (log scale)</text>')
    parts.extend(max_accuracy._combined_runtime_legend(left + plot_width + 28, top + 20, plotted))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _memory_plotted(points: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        point
        for point in points
        if (
            not bool(point["reference"])
            and float(point["error"]) > 0.0
            and float(point["memory_mib"]) > 0.0
        )
    ]


def _combined_memory_lines(points: list[dict[str, object]], x_to_px, y_to_px) -> list[str]:
    grouped: dict[tuple[float, str], list[dict[str, object]]] = {}
    for point in points:
        lam = float(point["lambda"])
        family = max_accuracy._line_family(point)
        grouped.setdefault((lam, family), []).append(point)

    parts: list[str] = []
    for (lam, _family), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        group.sort(key=lambda point: float(point["memory_mib"]))
        color = str(group[0]["color"])
        dash = max_accuracy._lambda_dash(lam)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        if len(group) > 1:
            polyline = " ".join(
                f"{x_to_px(float(point['memory_mib'])):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in group
            )
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash_attr} points="{polyline}"/>')
        for point in group:
            px = x_to_px(float(point["memory_mib"]))
            py = y_to_px(float(point["error"]))
            parts.append(f'<line x1="{px - 11:.2f}" y1="{py:.2f}" x2="{px + 11:.2f}" y2="{py:.2f}" stroke="{color}" stroke-width="2.4"{dash_attr}/>')
            parts.extend(max_accuracy._marker_parts(px, py, str(point["symbol"]), color, "#fffdf8", 5.2))
    for lam in sorted({float(point["lambda"]) for point in points}):
        for source_key, target_key, bridge_color in (
            ("Exponential", "Erlang", "#5c6b73"),
            ("Exponential/D", "Erlang /D", "#a3a3a3"),
        ):
            source = grouped.get((lam, source_key), [])
            target = grouped.get((lam, target_key), [])
            if source and target:
                target.sort(key=lambda point: float(point["memory_mib"]))
                bridge = [source[0], target[0]]
                polyline = " ".join(
                    f"{x_to_px(float(point['memory_mib'])):.2f},{y_to_px(float(point['error'])):.2f}"
                    for point in bridge
                )
                parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="2.4" points="{polyline}"/>')
    return parts


def _memory_ticks(left: int, top: int, plot_width: int, plot_height: int, min_power: int, max_power: int, x_to_px) -> list[str]:
    parts: list[str] = []
    for power in range(min_power, max_power + 1):
        value = 10.0 ** power
        x = x_to_px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#52606d">1e{power:+d}</text>')
    return parts


if __name__ == "__main__":
    main()
