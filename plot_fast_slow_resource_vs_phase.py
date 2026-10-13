from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path


CAPACITY = 10
LAMBDA = 0.1
SERVICE_TIME = 1.0
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "fast_slow_resource_vs_phase"

METHODS = [
    {
        "series": "Exponential",
        "label": "Exponential",
        "method": "exponential",
        "phase": 1,
        "color": "#5c6b73",
        "dash": "",
        "symbol": "circle",
    },
    {
        "series": "Exponential/D",
        "label": "Exponential/D",
        "method": "erlang_d",
        "phase": 1,
        "color": "#a3a3a3",
        "dash": "",
        "symbol": "circle",
    },
    {
        "series": "Erlang",
        "label": "Erlang 2",
        "method": "erlang",
        "phase": 2,
        "color": "#1d4ed8",
        "dash": "",
        "symbol": "square",
    },
    {
        "series": "Erlang",
        "label": "Erlang 10",
        "method": "erlang",
        "phase": 10,
        "color": "#1d4ed8",
        "dash": "",
        "symbol": "square",
    },
    {
        "series": "Erlang",
        "label": "Erlang 100",
        "method": "erlang",
        "phase": 100,
        "color": "#1d4ed8",
        "dash": "2,4",
        "symbol": "square",
    },
    {
        "series": "Erlang lumped",
        "label": "Erlang lumped 2",
        "method": "erlang_lumped",
        "phase": 2,
        "color": "#2563eb",
        "dash": "1,3",
        "symbol": "square",
    },
    {
        "series": "Erlang lumped",
        "label": "Erlang lumped 10",
        "method": "erlang_lumped",
        "phase": 10,
        "color": "#2563eb",
        "dash": "1,3",
        "symbol": "square",
    },
    {
        "series": "Erlang lumped",
        "label": "Erlang lumped 100",
        "method": "erlang_lumped",
        "phase": 100,
        "color": "#2563eb",
        "dash": "1,3",
        "symbol": "square",
    },
    {
        "series": "Erlang /D",
        "label": "Erlang /D 2",
        "method": "erlang_d",
        "phase": 2,
        "color": "#60a5fa",
        "dash": "4,2",
        "symbol": "square",
    },
    {
        "series": "Erlang /D",
        "label": "Erlang /D 10",
        "method": "erlang_d",
        "phase": 10,
        "color": "#60a5fa",
        "dash": "4,2",
        "symbol": "square",
    },
    {
        "series": "Erlang /D",
        "label": "Erlang /D 100",
        "method": "erlang_d",
        "phase": 100,
        "color": "#60a5fa",
        "dash": "4,2",
        "symbol": "square",
    },
    {
        "series": "CME",
        "label": "CME 5",
        "method": "cme",
        "phase": 5,
        "color": "#0c7c59",
        "dash": "",
        "symbol": "diamond",
    },
    {
        "series": "CME",
        "label": "CME 10",
        "method": "cme",
        "phase": 10,
        "color": "#0c7c59",
        "dash": "",
        "symbol": "diamond",
    },
    {
        "series": "CME",
        "label": "CME 15",
        "method": "cme",
        "phase": 15,
        "color": "#0c7c59",
        "dash": "",
        "symbol": "diamond",
    },
    {
        "series": "CME",
        "label": "CME 20",
        "method": "cme",
        "phase": 20,
        "color": "#0c7c59",
        "dash": "4,3",
        "symbol": "diamond",
    },
    {
        "series": "Faster",
        "label": "faster w/o",
        "method": "fast_slow",
        "variant": "faster",
        "phase": 1,
        "color": "#f97316",
        "dash": "6,4",
        "symbol": "triangle",
    },
    {
        "series": "Faster",
        "label": "faster 2",
        "method": "fast_slow",
        "variant": "faster 2",
        "phase": 2,
        "color": "#f97316",
        "dash": "6,4",
        "symbol": "triangle",
    },
    {
        "series": "Faster",
        "label": "faster 10",
        "method": "fast_slow",
        "variant": "faster 10",
        "phase": 10,
        "color": "#f97316",
        "dash": "6,4",
        "symbol": "triangle",
    },
    {
        "series": "Faster",
        "label": "faster 100",
        "method": "fast_slow",
        "variant": "faster 100",
        "phase": 100,
        "color": "#f97316",
        "dash": "6,4",
        "symbol": "triangle",
    },
    {
        "series": "Slower",
        "label": "slower w/o",
        "method": "fast_slow",
        "variant": "slower",
        "phase": 1,
        "color": "#ca8a04",
        "dash": "2,4",
        "symbol": "triangle-down",
    },
    {
        "series": "Slower",
        "label": "slower 2",
        "method": "fast_slow",
        "variant": "slower 2",
        "phase": 2,
        "color": "#ca8a04",
        "dash": "2,4",
        "symbol": "triangle-down",
    },
    {
        "series": "Slower",
        "label": "slower 10",
        "method": "fast_slow",
        "variant": "slower 10",
        "phase": 10,
        "color": "#ca8a04",
        "dash": "2,4",
        "symbol": "triangle-down",
    },
    {
        "series": "Slower",
        "label": "slower 100",
        "method": "fast_slow",
        "variant": "slower 100",
        "phase": 100,
        "color": "#ca8a04",
        "dash": "2,4",
        "symbol": "triangle-down",
    },
    {
        "series": "Average",
        "label": "average w/o",
        "method": "average_fast_slow",
        "phase": 1,
        "color": "#7e22ce",
        "dash": "",
        "symbol": "circle",
    },
    {
        "series": "Average",
        "label": "average 2",
        "method": "average_fast_slow",
        "phase": 2,
        "color": "#7e22ce",
        "dash": "",
        "symbol": "circle",
    },
    {
        "series": "Average",
        "label": "average 10",
        "method": "average_fast_slow",
        "phase": 10,
        "color": "#7e22ce",
        "dash": "",
        "symbol": "circle",
    },
    {
        "series": "Average",
        "label": "average 100",
        "method": "average_fast_slow",
        "phase": 100,
        "color": "#7e22ce",
        "dash": "",
        "symbol": "circle",
    },
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = _measure_points()

    csv_path = OUTPUT_DIR / "all_methods_resource_vs_phase_lambda_0_1_k_10.csv"
    runtime_svg = OUTPUT_DIR / "all_methods_runtime_vs_phase_lambda_0_1_k_10.svg"
    memory_svg = OUTPUT_DIR / "all_methods_memory_vs_phase_lambda_0_1_k_10.svg"
    state_svg = OUTPUT_DIR / "all_methods_states_vs_phase_lambda_0_1_k_10.svg"

    _write_csv(csv_path, points)
    _write_svg(
        runtime_svg,
        points,
        y_key="runtime_seconds",
        y_label="Runtime (seconds, log scale)",
        title=f"Runtime vs Phase Count for K={CAPACITY}, lambda={LAMBDA}",
    )
    _write_svg(
        memory_svg,
        points,
        y_key="peak_memory_mib",
        y_label="Peak memory increase (MiB, log scale)",
        title=f"Memory vs Phase Count for K={CAPACITY}, lambda={LAMBDA}",
    )
    _write_svg(
        state_svg,
        points,
        y_key="linear_states",
        y_label="Linear state count (log scale)",
        title=f"State Count vs Phase Count for K={CAPACITY}",
    )

    print(f"Saved {csv_path}")
    print(f"Saved {runtime_svg}")
    print(f"Saved {memory_svg}")
    print(f"Saved {state_svg}")


def _measure_points() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for config in METHODS:
        payload = _measure_method_in_fresh_process(config)
        point = {
            **config,
            "runtime_seconds": payload["runtime_seconds"],
            "peak_memory_mib": payload["peak_memory_bytes"] / (1024 * 1024),
            "linear_states": payload["linear_states"],
            "p_full": payload["p_full"],
            "response_time": payload["response_time"],
        }
        points.append(point)
        print(
            f"{point['label']}: "
            f"{float(point['runtime_seconds']):.6g} s, "
            f"{float(point['peak_memory_mib']):.6g} MiB"
        )
    return points


def _measure_method_in_fresh_process(config: dict[str, object]) -> dict[str, float]:
    child_code = r'''
import ctypes
import json
import math
import sys
import threading
import time

import numpy as np

import plot_distribution_accuracy as base
from methods.cme import _cme_representation, solve as solve_cme_lumped
from methods.embedded_one_server import (
    solve_cme as solve_one_server_cme,
    solve_erlang as solve_one_server_erlang,
)
from methods.erlang_lumped import (
    solve as solve_erlang_lumped,
    state_count as erlang_lumped_state_count,
)
from methods.fast_slow import solve as solve_fast_slow


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
_GET_CURRENT_PROCESS = _KERNEL32.GetCurrentProcess
_GET_CURRENT_PROCESS.restype = ctypes.c_void_p
_GET_PROCESS_MEMORY_INFO = _PSAPI.GetProcessMemoryInfo
_GET_PROCESS_MEMORY_INFO.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
_GET_PROCESS_MEMORY_INFO.restype = ctypes.c_int


def _current_working_set_bytes():
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    ok = _GET_PROCESS_MEMORY_INFO(
        _GET_CURRENT_PROCESS(),
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _measure(func):
    baseline = _current_working_set_bytes()
    peak = baseline
    stop = False

    def poller():
        nonlocal peak
        while not stop:
            current = _current_working_set_bytes()
            if current > peak:
                peak = current
            time.sleep(0.001)

    thread = threading.Thread(target=poller, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = func()
    finally:
        elapsed = time.perf_counter() - start
        stop = True
        thread.join(timeout=1.0)
    final = _current_working_set_bytes()
    peak = max(peak, final)
    return result, elapsed, max(0, peak - baseline)


print("READY", flush=True)
request = json.loads(sys.stdin.readline())
method = str(request["method"])
variant = str(request.get("variant", ""))
phase = int(request["phase"])
arrival_rate = float(request["lambda"])
service_time = float(request["service_time"])
capacity = int(request["capacity"])

def _metrics(dist):
    clean = base._normalise(np.asarray(dist, dtype=float))
    p_full = float(clean[-1])
    mean_jobs = sum(index * float(value) for index, value in enumerate(clean))
    response_time = mean_jobs / (arrival_rate * (1.0 - p_full))
    return p_full, response_time


def _linear_states():
    if method == "exponential":
        return 1 + 2 + (capacity - 1)
    if method == "erlang":
        return 1 + 2 * phase + (capacity - 1) * phase * phase
    if method == "erlang_d":
        return 2 + phase + (capacity - 1) * phase
    if method == "erlang_lumped":
        return erlang_lumped_state_count(capacity, phase)
    if method == "cme":
        alpha, _matrix_t = _cme_representation(phase, service_time)
        order = len(alpha)
        return 1 + 2 * order + (capacity - 1) * order * order
    if method == "cme_d":
        alpha, _matrix_t = _cme_representation(phase, service_time)
        order = len(alpha)
        return 2 + order + (capacity - 1) * order
    if method == "cme_lumped":
        alpha, _matrix_t = _cme_representation(phase, service_time)
        order = len(alpha)
        return 1 + 2 * order + (capacity - 1) * order * order
    if method == "fast_slow":
        if phase == 1:
            return capacity + 2
        if variant.startswith("faster"):
            return capacity * phase + 2
        return capacity * (phase + 1) + 1
    if method == "average_fast_slow":
        if phase == 1:
            return 2 * (capacity + 2)
        faster_states = capacity * phase + 2
        slower_states = capacity * (phase + 1) + 1
        return faster_states + slower_states
    raise ValueError(f"Unsupported method: {method}")


def _solve():
    if method == "exponential":
        dist = base._exponential_distribution(arrival_rate)
        p_full, response_time = _metrics(dist)
    elif method == "erlang":
        dist = base._erlang_distribution(arrival_rate, phase)
        p_full, response_time = _metrics(dist)
    elif method == "erlang_d":
        row = solve_one_server_erlang(
            arrival_rate=arrival_rate,
            service_time=service_time,
            capacity=capacity,
            phase_count=phase,
            precision_digits=80,
            solver="fast",
        )[0]
        p_full = float(row["p_full"])
        response_time = float(row["response_time"])
    elif method == "erlang_lumped":
        row = solve_erlang_lumped(
            arrival_rate=arrival_rate,
            service_time=service_time,
            capacity=capacity,
            phase_count=phase,
        )[0]
        p_full = float(row["p_full"])
        response_time = float(row["response_time"])
    elif method == "cme":
        dist = base._cme_distribution(phase, arrival_rate)
        p_full, response_time = _metrics(dist)
    elif method == "cme_d":
        row = None
        for solver in ("signed_fast", "signed"):
            try:
                candidate = solve_one_server_cme(
                    arrival_rate=arrival_rate,
                    service_time=service_time,
                    capacity=capacity,
                    order=phase,
                    precision_digits=80,
                    solver=solver,
                )[0]
                if math.isfinite(float(candidate["response_time"])) and float(candidate["response_time"]) > 0.0:
                    row = candidate
                    break
            except (RuntimeError, FileNotFoundError, NotImplementedError):
                continue
        if row is None:
            raise RuntimeError(f"CME {phase}/D failed for lambda={arrival_rate}.")
        p_full = float(row["p_full"])
        response_time = float(row["response_time"])
    elif method == "cme_lumped":
        row = solve_cme_lumped(
            arrival_rate=arrival_rate,
            service_time=service_time,
            capacity=capacity,
            targets=[{"name": f"CME {phase}", "phases": phase}],
            solver="recursive",
        )[0]
        p_full = float(row["p_full"])
        response_time = float(row["response_time"])
    elif method == "fast_slow":
        result = solve_fast_slow(
            arrival_rate=arrival_rate,
            service_time=service_time,
            capacity=capacity,
            variants=[variant],
        )[0]
        p_full = float(result["p_full"])
        response_time = float(result["response_time"])
    elif method == "average_fast_slow":
        if phase == 1:
            faster_variant = "faster"
            slower_variant = "slower"
        else:
            faster_variant = f"faster {phase}"
            slower_variant = f"slower {phase}"
        faster = base._normalise(base._fast_slow_distribution(faster_variant, arrival_rate))
        slower = base._normalise(base._fast_slow_distribution(slower_variant, arrival_rate))
        dist = base._normalise(0.5 * (faster + slower))
        p_full, response_time = _metrics(dist)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return {
        "p_full": p_full,
        "response_time": response_time,
        "linear_states": _linear_states(),
    }

result, elapsed, peak_bytes = _measure(
    _solve
)
print(
    json.dumps(
        {
            "runtime_seconds": elapsed,
            "peak_memory_bytes": peak_bytes,
            "linear_states": int(result["linear_states"]),
            "p_full": float(result["p_full"]),
            "response_time": float(result["response_time"]),
        }
    ),
    flush=True,
)
'''
    label = str(config["label"])
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

    proc.stdin.write(
        json.dumps(
            {
                "method": config["method"],
                "variant": config.get("variant", ""),
                "phase": config["phase"],
                "lambda": LAMBDA,
                "service_time": SERVICE_TIME,
                "capacity": CAPACITY,
            }
        )
        + "\n"
    )
    proc.stdin.flush()
    line = proc.stdout.readline()
    _stdout, stderr = proc.communicate(timeout=240)
    if proc.returncode != 0:
        raise RuntimeError(f"{label}: child process failed with code {proc.returncode}. stderr={stderr}")
    if not line:
        raise RuntimeError(f"{label}: child process produced no output.")
    return json.loads(line)


def _write_csv(path: Path, points: list[dict[str, object]]) -> None:
    lines = [
        "lambda,capacity,series,label,phase,linear_states,runtime_seconds,peak_memory_mib,p_full,response_time",
    ]
    for point in points:
        lines.append(
            ",".join(
                [
                    f"{LAMBDA:g}",
                    str(CAPACITY),
                    str(point["series"]),
                    str(point["label"]),
                    str(point["phase"]),
                    str(point["linear_states"]),
                    f"{float(point['runtime_seconds']):.9f}",
                    f"{float(point['peak_memory_mib']):.6f}",
                    f"{float(point['p_full']):.18e}",
                    f"{float(point['response_time']):.18e}",
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_svg(
    path: Path,
    points: list[dict[str, object]],
    *,
    y_key: str,
    y_label: str,
    title: str,
) -> None:
    plotted = [point for point in points if float(point[y_key]) > 0.0]
    width = 1320
    height = 720
    left = 110
    right = 350
    top = 55
    bottom = 95
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["phase"]) for point in plotted]
    y_values = [float(point[y_key]) for point in plotted]
    x_min_power = math.floor(math.log10(min(x_values)))
    x_max_power = math.ceil(math.log10(max(x_values)))
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))
    if y_min_power == y_max_power:
        y_max_power += 1

    def x_to_px(value: float) -> float:
        frac = (math.log10(value) - x_min_power) / (x_max_power - x_min_power)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - frac * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    for power in range(x_min_power, x_max_power + 1):
        value = 10.0 ** power
        x = x_to_px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{value:g}</text>')
    for power in range(y_min_power, y_max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">1e{power:+d}</text>')

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    for series in (
        "Erlang",
        "Erlang lumped",
        "Erlang /D",
        "CME",
        "Faster",
        "Slower",
        "Average",
    ):
        series_points = [point for point in plotted if point["series"] == series]
        series_points.sort(key=lambda point: float(point["phase"]))
        parts.append(_polyline(series_points, x_to_px, y_to_px, y_key))
    exponential_points = [point for point in plotted if point["series"] == "Exponential"]
    erlang_points = [point for point in plotted if point["series"] == "Erlang"]
    if exponential_points and erlang_points:
        erlang_points.sort(key=lambda point: float(point["phase"]))
        parts.append(_polyline([exponential_points[0], erlang_points[0]], x_to_px, y_to_px, y_key))
    exponential_d_points = [point for point in plotted if point["series"] == "Exponential/D"]
    erlang_d_points = [point for point in plotted if point["series"] == "Erlang /D"]
    if exponential_d_points and erlang_d_points:
        erlang_d_points.sort(key=lambda point: float(point["phase"]))
        parts.append(_polyline([exponential_d_points[0], erlang_d_points[0]], x_to_px, y_to_px, y_key))
    for point in plotted:
        parts.append(_marker(point, x_to_px(float(point["phase"])), y_to_px(float(point[y_key]))))

    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">Phase count (w/o shown at 1, log scale)</text>')
    parts.append(f'<text x="32" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 32 {top + plot_height / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#111111">{y_label}</text>')
    parts.extend(_legend(left + plot_width + 35, top + 15))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _polyline(points: list[dict[str, object]], x_to_px, y_to_px, y_key: str) -> str:
    if not points:
        return ""
    first = points[0]
    line_color = {
        "Exponential": "#5c6b73",
        "Exponential/D": "#a3a3a3",
        "Erlang": "#1d4ed8",
        "Erlang lumped": "#2563eb",
        "Erlang /D": "#60a5fa",
        "CME": "#0c7c59",
        "Faster": "#f97316",
        "Average": "#7e22ce",
        "Slower": "#ca8a04",
    }.get(str(first["series"]), str(first["color"]))
    coords = " ".join(
        f"{x_to_px(float(point['phase'])):.2f},{y_to_px(float(point[y_key])):.2f}"
        for point in points
    )
    dash = f' stroke-dasharray="{first["dash"]}"' if first["dash"] else ""
    return f'<polyline points="{coords}" fill="none" stroke="{line_color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"{dash}/>'


def _marker(point: dict[str, object], x: float, y: float) -> str:
    color = str(point["color"])
    symbol = str(point["symbol"])
    if symbol == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.8" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'
    if symbol == "square":
        return f'<rect x="{x - 4.8:.2f}" y="{y - 4.8:.2f}" width="9.6" height="9.6" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'
    if symbol == "diamond":
        return f'<path d="M{x:.2f},{y - 5.6:.2f} L{x + 5.6:.2f},{y:.2f} L{x:.2f},{y + 5.6:.2f} L{x - 5.6:.2f},{y:.2f} Z" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'
    if symbol == "triangle":
        return f'<path d="M{x:.2f},{y - 5.4:.2f} L{x + 5.4:.2f},{y + 4.4:.2f} L{x - 5.4:.2f},{y + 4.4:.2f} Z" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'
    return f'<path d="M{x - 5.4:.2f},{y - 4.4:.2f} L{x + 5.4:.2f},{y - 4.4:.2f} L{x:.2f},{y + 5.4:.2f} Z" fill="{color}" stroke="#ffffff" stroke-width="1.6"/>'


def _legend(x: float, y: float) -> list[str]:
    entries = [
        ("Exponential", "#5c6b73", "", "circle"),
        ("Exponential/D", "#a3a3a3", "", "circle"),
        ("Erlang", "#1d4ed8", "", "square"),
        ("Erlang lumped", "#2563eb", "1,3", "square"),
        ("Erlang /D", "#60a5fa", "4,2", "square"),
        ("CME", "#0c7c59", "", "diamond"),
        ("Faster", "#f97316", "6,4", "triangle"),
        ("Slower", "#ca8a04", "2,4", "triangle-down"),
        ("Average", "#7e22ce", "", "circle"),
    ]
    parts = [f'<rect x="{x - 12}" y="{y - 20}" width="275" height="{34 * len(entries) + 16}" fill="#ffffff" opacity="0.94" stroke="#bdbdbd" stroke-width="1"/>']
    for index, (label, color, dash, symbol) in enumerate(entries):
        yy = y + index * 34
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 34}" y2="{yy}" stroke="{color}" stroke-width="2.2" stroke-dasharray="{dash}" stroke-linecap="round"/>')
        parts.append(_marker({"color": color, "symbol": symbol}, x + 17, yy))
        parts.append(f'<text x="{x + 46}" y="{yy + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#111111">{label}</text>')
    return parts


if __name__ == "__main__":
    main()
