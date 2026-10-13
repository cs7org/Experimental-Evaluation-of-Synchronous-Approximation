from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg as scipy_linalg

from methods import cme as cme_method
from methods.embedded_one_server import (
    solve_cme as solve_one_server_cme,
    solve_erlang as solve_one_server_erlang,
)
from methods import erlang as erlang_dtmc_method
from methods import fast_slow as fast_slow_method
from methods import pure_erlang as pure_erlang_method


SERVICE_TIME = 1.0
CAPACITY = 10
LAMBDA_VALUES = [0.1, 0.5, 0.9]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
REFERENCE_SPECS = [
    {
        "key": "cme20",
        "label": "CME 20",
        "builder": lambda lam: _cme_distribution(20, lam),
    },
    {
        "key": "midbound",
        "label": "(faster 100 + slower 100) / 2",
        "builder": lambda lam: _midbound_distribution(lam),
    },
]
TIMING_REPEATS = {
    "Exponential": 7,
    "Erlang 2": 7,
}
FAST_SLOW_TIMING_REPEATS = 5
FAST_SLOW_LARGE_TIMING_REPEATS = 3


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_lines = [
        "lambda,reference_key,reference_label,series,label,phase_or_order,runtime_seconds,error_mean_relative,reference",
    ]
    for lam in LAMBDA_VALUES:
        for reference_spec in REFERENCE_SPECS:
            reference = _normalise(reference_spec["builder"](lam))
            phase_points, runtime_points = _compute_points(lam, reference, reference_spec["label"])
            csv_lines.extend(_csv_rows(lam, phase_points, runtime_points, reference_spec))

            phase_svg = OUTPUT_DIR / (
                f"distribution_error_vs_phase_lambda_{str(lam).replace('.', '_')}_ref_{reference_spec['key']}.svg"
            )
            runtime_svg = OUTPUT_DIR / (
                f"distribution_error_vs_runtime_lambda_{str(lam).replace('.', '_')}_ref_{reference_spec['key']}.svg"
            )

            _write_phase_svg(
                phase_svg,
                phase_points,
                title=f"Distribution Error vs Phase Count for K={CAPACITY}, lambda={lam}",
                reference_label=reference_spec["label"],
            )
            _write_runtime_svg(
                runtime_svg,
                runtime_points,
                title=f"Distribution Error vs Runtime for K={CAPACITY}, lambda={lam}",
                reference_label=reference_spec["label"],
            )

            print(f"Saved {phase_svg}")
            print(f"Saved {runtime_svg}")

    csv_path = OUTPUT_DIR / "distribution_error_accuracy.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")


def _compute_points(
    lam: float,
    reference: np.ndarray,
    reference_label: str,
    *,
    include_embedded_variants: bool = False,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    phase_points: list[dict[str, object]] = []
    runtime_points: list[dict[str, object]] = []

    configs = [
        {
            "series": "Exponential",
            "label": "Exponential",
            "x_value": 1,
            "color": "#5c6b73",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: _exponential_distribution(lam),
        },
        {
            "series": "Erlang",
            "label": "Erlang 2",
            "x_value": 2,
            "color": "#2f6fed",
            "dash": "",
            "symbol": "square",
            "solver": lambda: _erlang_distribution(lam, 2),
        },
        {
            "series": "Erlang",
            "label": "Erlang 10",
            "x_value": 10,
            "color": "#2f6fed",
            "dash": "",
            "symbol": "square",
            "solver": lambda: _erlang_distribution(lam, 10),
        },
        {
            "series": "Erlang",
            "label": "Erlang 100",
            "x_value": 100,
            "color": "#2f6fed",
            "dash": "2,4",
            "symbol": "square",
            "solver": lambda: _erlang_distribution(lam, 100),
        },
        {
            "series": "CME",
            "label": "CME 5",
            "x_value": 5,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: _cme_distribution(5, lam),
        },
        {
            "series": "CME",
            "label": "CME 10",
            "x_value": 10,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: _cme_distribution(10, lam),
        },
        {
            "series": "CME",
            "label": "CME 15",
            "x_value": 15,
            "color": "#0c7c59",
            "dash": "",
            "symbol": "diamond",
            "solver": lambda: _cme_distribution(15, lam),
            "reference": reference_label == "CME 15",
        },
        {
            "series": "Faster",
            "label": "faster w/o",
            "x_value": 1,
            "color": "#f4a261",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: _fast_slow_distribution("faster", lam),
        },
        {
            "series": "Faster",
            "label": "faster 2",
            "x_value": 2,
            "color": "#d17b0f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: _fast_slow_distribution("faster 2", lam),
        },
        {
            "series": "Faster",
            "label": "faster 10",
            "x_value": 10,
            "color": "#d17b0f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: _fast_slow_distribution("faster 10", lam),
        },
        {
            "series": "Faster",
            "label": "faster 100",
            "x_value": 100,
            "color": "#d17b0f",
            "dash": "6,4",
            "symbol": "triangle",
            "solver": lambda: _fast_slow_distribution("faster 100", lam),
        },
        {
            "series": "Slower",
            "label": "slower w/o",
            "x_value": 1,
            "color": "#c1121f",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: _fast_slow_distribution("slower", lam),
        },
        {
            "series": "Slower",
            "label": "slower 2",
            "x_value": 2,
            "color": "#9b2226",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: _fast_slow_distribution("slower 2", lam),
        },
        {
            "series": "Slower",
            "label": "slower 10",
            "x_value": 10,
            "color": "#9b2226",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: _fast_slow_distribution("slower 10", lam),
        },
        {
            "series": "Slower",
            "label": "slower 100",
            "x_value": 100,
            "color": "#9b2226",
            "dash": "2,4",
            "symbol": "triangle-down",
            "solver": lambda: _fast_slow_distribution("slower 100", lam),
        },
        {
            "series": "Average",
            "label": "average w/o",
            "x_value": 1,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: _average_fast_slow_distribution("faster", "slower", lam),
        },
        {
            "series": "Average",
            "label": "average 2",
            "x_value": 2,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: _average_fast_slow_distribution("faster 2", "slower 2", lam),
        },
        {
            "series": "Average",
            "label": "average 10",
            "x_value": 10,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: _average_fast_slow_distribution("faster 10", "slower 10", lam),
        },
        {
            "series": "Average",
            "label": "average 100",
            "x_value": 100,
            "color": "#7c3aed",
            "dash": "",
            "symbol": "circle",
            "solver": lambda: _average_fast_slow_distribution("faster 100", "slower 100", lam),
        },
    ]

    if include_embedded_variants:
        configs[4:4] = [
            {
                "series": "Exponential/D",
                "label": "Exponential/D",
                "x_value": 1,
                "color": "#a3a3a3",
                "dash": "",
                "symbol": "circle",
                "solver": lambda: _erlang_d_distribution(lam, 1),
            },
            {
                "series": "Erlang /D",
                "label": "Erlang /D 2",
                "x_value": 2,
                "color": "#60a5fa",
                "dash": "4,2",
                "symbol": "square",
                "solver": lambda: _erlang_d_distribution(lam, 2),
            },
            {
                "series": "Erlang /D",
                "label": "Erlang /D 10",
                "x_value": 10,
                "color": "#60a5fa",
                "dash": "4,2",
                "symbol": "square",
                "solver": lambda: _erlang_d_distribution(lam, 10),
            },
            {
                "series": "Erlang /D",
                "label": "Erlang /D 100",
                "x_value": 100,
                "color": "#60a5fa",
                "dash": "4,2",
                "symbol": "square",
                "solver": lambda: _erlang_d_distribution(lam, 100),
            },
        ]

    for config in configs:
        dist, elapsed = _timed_distribution(
            config["solver"],  # type: ignore[arg-type]
            repeats=_timing_repeats(config),
        )
        dist = _normalise(dist)
        error = _mean_relative_error(dist, reference)
        point = {
            "series": config["series"],
            "label": config["label"],
            "x_value": config["x_value"],
            "runtime": elapsed,
            "error": error,
            "color": config["color"],
            "dash": config["dash"],
            "symbol": config["symbol"],
            "reference": bool(config.get("reference", False)),
        }
        phase_points.append(point)
        runtime_points.append(point)

    return phase_points, runtime_points


def _timing_repeats(config: dict[str, object]) -> int:
    label = str(config["label"])
    if label in TIMING_REPEATS:
        return TIMING_REPEATS[label]
    if str(config["series"]) in {"Faster", "Slower"}:
        if float(config["x_value"]) >= 100:
            return FAST_SLOW_LARGE_TIMING_REPEATS
        return FAST_SLOW_TIMING_REPEATS
    return 1


def _timed_distribution(solver, *, repeats: int) -> tuple[np.ndarray, float]:
    best_elapsed = float("inf")
    best_dist: np.ndarray | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        dist = solver()
        elapsed = time.perf_counter() - start
        if elapsed < best_elapsed:
            best_elapsed = elapsed
            best_dist = np.asarray(dist, dtype=float)
    if best_dist is None:
        raise RuntimeError("No timing run was completed.")
    return best_dist, best_elapsed


def _exponential_distribution(arrival_rate: float) -> np.ndarray:
    rho = arrival_rate * SERVICE_TIME
    states = np.arange(CAPACITY + 1, dtype=float)
    if math.isclose(rho, 1.0):
        return np.full(CAPACITY + 1, 1.0 / (CAPACITY + 1), dtype=float)
    powers = np.power(rho, states)
    return ((1.0 - rho) / (1.0 - rho ** (CAPACITY + 1))) * powers


def _erlang_distribution(arrival_rate: float, phase_count: int) -> np.ndarray:
    generator = pure_erlang_method._build_generator(arrival_rate, SERVICE_TIME, CAPACITY, phase_count)
    probability = pure_erlang_method._solve_stationary(generator)
    return pure_erlang_method._aggregate(probability, CAPACITY, phase_count)


def _erlang_d_distribution(arrival_rate: float, phase_count: int) -> np.ndarray:
    row = solve_one_server_erlang(
        arrival_rate=arrival_rate,
        service_time=SERVICE_TIME,
        capacity=CAPACITY,
        phase_count=phase_count,
        precision_digits=80,
        solver="fast",
    )[0]
    return _normalise(np.asarray(row["distribution"], dtype=float))


def _erlang_dtmc_distribution(arrival_rate: float, phase_count: int) -> np.ndarray:
    generator = erlang_dtmc_method._build_generator(arrival_rate, SERVICE_TIME, CAPACITY, phase_count)
    probability = erlang_dtmc_method._solve_stationary(
        generator,
        "power",
        1e-12,
        50_000,
        None,
    )
    return erlang_dtmc_method._aggregate(probability, CAPACITY, phase_count)


def _cme_distribution(order: int, arrival_rate: float) -> np.ndarray:
    alpha, matrix_t = cme_method._cme_representation(order, SERVICE_TIME)
    attempts = [
        {},
        {"solver": "recursive_improved"},
        {"solver": "iterative", "max_iterations": 20_000},
    ]
    for kwargs in attempts:
        try:
            return cme_method._compute_queue_distribution(arrival_rate, CAPACITY, alpha, matrix_t, **kwargs)
        except RuntimeError:
            continue
    raise RuntimeError(f"CME {order} failed for lambda={arrival_rate}.")


def _cme_d_distribution(order: int, arrival_rate: float) -> np.ndarray:
    for solver in ("signed_fast", "signed"):
        try:
            row = solve_one_server_cme(
                arrival_rate=arrival_rate,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                order=order,
                precision_digits=80,
                solver=solver,
            )[0]
            return _normalise(np.asarray(row["distribution"], dtype=float))
        except (RuntimeError, FileNotFoundError, NotImplementedError):
            continue
    raise RuntimeError(f"CME {order}/D failed for lambda={arrival_rate}.")


def _midbound_distribution(arrival_rate: float) -> np.ndarray:
    faster = _normalise(_fast_slow_distribution("faster 100", arrival_rate))
    slower = _normalise(_fast_slow_distribution("slower 100", arrival_rate))
    return 0.5 * (faster + slower)


def _average_fast_slow_distribution(
    faster_variant: str,
    slower_variant: str,
    arrival_rate: float,
) -> np.ndarray:
    faster = _normalise(_fast_slow_distribution(faster_variant, arrival_rate))
    slower = _normalise(_fast_slow_distribution(slower_variant, arrival_rate))
    return 0.5 * (faster + slower)


def _fast_slow_distribution(variant: str, arrival_rate: float) -> np.ndarray:
    spec = fast_slow_method._parse_variant(variant)
    if spec.kind in {"faster_noexp", "slower"}:
        generator, deferred, departure, groups = fast_slow_method._build_simple_inputs(
            arrival_rate,
            CAPACITY,
            spec.kind,
        )
        probability = _solve_embedded_float(
            generator,
            deferred,
            departure,
            groups,
            SERVICE_TIME,
        )
        dist = fast_slow_method._aggregate_simple(probability, CAPACITY)
    elif spec.kind == "slower_additional":
        generator, deferred, departure, groups = fast_slow_method._build_slower_additional_inputs(
            arrival_rate,
            CAPACITY,
        )
        probability = fast_slow_method._solve_embedded(
            generator,
            deferred,
            departure,
            groups,
            SERVICE_TIME,
            80,
        )
        dist = fast_slow_method._aggregate_slower_additional(probability, CAPACITY)
    elif spec.kind == "faster_expanded":
        dist = fast_slow_method._solve_expanded(
            topology=fast_slow_method._build_faster_expanded_topology(CAPACITY, int(spec.phases)),
            arrival_rate=arrival_rate,
            service_time=SERVICE_TIME,
            phase_count=int(spec.phases),
            precision_digits=80,
        )
    elif spec.kind == "slower_expanded":
        dist = fast_slow_method._solve_expanded(
            topology=fast_slow_method._build_slower_expanded_topology(CAPACITY, int(spec.phases)),
            arrival_rate=arrival_rate,
            service_time=SERVICE_TIME,
            phase_count=int(spec.phases),
            precision_digits=80,
        )
    else:
        raise ValueError(f"Unsupported fast-slow variant: {variant}")
    return np.asarray(dist, dtype=float)


def _solve_embedded_float(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tau: float,
) -> list[float]:
    dim = generator.shape[0]
    groups_list = [list(group) for group in groups]
    all_states = [state for group in groups_list for state in group]
    exp_states = groups_list[0]
    gen_states = [state for state in all_states if state not in exp_states]

    i_exp = _id_subset_float(dim, exp_states)
    i_gen = _id_subset_float(dim, gen_states)

    q_gen = i_gen @ generator
    q_exp = i_exp @ generator
    exp_block, int_block = _exponential_integrals_float(q_gen, tau)

    omega = i_gen @ exp_block
    psi = i_gen @ int_block

    transition = (
        i_exp
        - (_diag_inv_float(q_exp) @ q_exp)
        + (omega @ departure)
        + (psi @ deferred)
    )
    correction = -_diag_inv_float(q_exp) + psi

    a = (transition - i_exp - i_gen).T
    norm_row = correction @ np.ones(dim, dtype=float)
    a[-1, :] = norm_row

    rhs = np.zeros(dim, dtype=float)
    rhs[-1] = 1.0
    multiplier = np.linalg.solve(a, rhs)
    stat = correction.T @ multiplier
    return _sanitise_float(stat)


def _exponential_integrals_float(q_sub: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    dim = q_sub.shape[0]
    aug = np.zeros((2 * dim, 2 * dim), dtype=float)
    aug[:dim, :dim] = q_sub
    aug[:dim, dim:] = np.eye(dim, dtype=float)
    block = scipy_linalg.expm(aug * tau)
    return block[:dim, :dim], block[:dim, dim:]


def _id_subset_float(dim: int, states_1based: list[int]) -> np.ndarray:
    matrix = np.zeros((dim, dim), dtype=float)
    for state in states_1based:
        matrix[state - 1, state - 1] = 1.0
    return matrix


def _diag_inv_float(matrix: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    inv = np.zeros_like(matrix, dtype=float)
    diag = np.diag(matrix)
    for index, value in enumerate(diag):
        if abs(value) > tol:
            inv[index, index] = 1.0 / value
    return inv


def _sanitise_float(probability: np.ndarray) -> list[float]:
    clean = np.real_if_close(np.asarray(probability, dtype=float))
    clean[clean < 0.0] = 0.0
    total = float(clean.sum())
    if total <= 0.0:
        raise RuntimeError("Fast-slow produced an invalid stationary distribution.")
    return list(clean / total)


def _normalise(dist: np.ndarray) -> np.ndarray:
    clean = np.real_if_close(np.asarray(dist, dtype=float))
    clean[clean < 0.0] = 0.0
    total = float(clean.sum())
    if total <= 0.0:
        raise RuntimeError("Distribution is invalid.")
    return clean / total


def _mean_relative_error(dist: np.ndarray, reference: np.ndarray) -> float:
    relative = np.divide(
        np.abs(dist - reference),
        reference,
        out=np.full_like(reference, np.inf, dtype=float),
        where=reference != 0.0,
    )
    return float(np.mean(relative))


def _csv_rows(
    lam: float,
    phase_points: list[dict[str, object]],
    runtime_points: list[dict[str, object]],
    reference_spec: dict[str, object],
) -> list[str]:
    phase_map = {str(point["label"]): point for point in phase_points}
    rows: list[str] = []
    for point in runtime_points:
        phase_point = phase_map[str(point["label"])]
        rows.append(
            ",".join(
                [
                    str(lam),
                    str(reference_spec["key"]),
                    str(reference_spec["label"]),
                    str(point["series"]),
                    str(point["label"]),
                    str(phase_point["x_value"]),
                    f"{float(point['runtime']):.9f}",
                    f"{float(point['error']):.18e}",
                    "Yes" if bool(point["reference"]) else "No",
                ],
            ),
        )
    return rows


def _write_phase_svg(path: Path, points: list[dict[str, object]], *, title: str, reference_label: str) -> None:
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

    parts = _svg_shell(width, height, title, f"Mean relative error of (P0,...,PK) vs {reference_label} reference")
    parts.extend(_log_grid(left, top, plot_width, plot_height, y_min_power, y_max_power, vertical=False))
    parts.extend(_phase_ticks(left, top, plot_height, x_values, x_to_px))
    parts.extend(_axes(left, top, plot_width, plot_height))
    parts.extend(_series_polylines(plotted, x_to_px, y_to_px, x_key="x_value"))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Phase count / CME order (log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Mean relative error over P0..PK (log scale)</text>')
    parts.extend(_legend(left + plot_width + 28, top + 20, points))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _write_runtime_svg(path: Path, points: list[dict[str, object]], *, title: str, reference_label: str) -> None:
    plotted = [
        point
        for point in points
        if (
            not bool(point["reference"])
            and float(point["error"]) > 0.0
            and str(point["label"]) not in {"faster w/o", "slower w/o"}
        )
    ]
    width = 1260
    height = 800
    left = 120
    right = 320
    top = 90
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

    parts = _svg_shell(width, height, title, f"Mean relative error of (P0,...,PK) vs {reference_label} reference")
    parts.extend(_log_grid(left, top, plot_width, plot_height, y_min_power, y_max_power, vertical=False))
    parts.extend(_runtime_ticks(left, top, plot_width, plot_height, x_min_power, x_max_power, x_to_px))
    parts.extend(_axes(left, top, plot_width, plot_height))
    parts.extend(_series_polylines(plotted, x_to_px, y_to_px, x_key="runtime", connect_wo=False))
    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 28}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Runtime (seconds, log scale)</text>')
    parts.append(f'<text transform="translate(34 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#52606d">Mean relative error over P0..PK (log scale)</text>')
    parts.extend(_point_legend(left + plot_width + 28, top + 20, plotted))
    path.write_text("\n".join(parts + ["</svg>"]) + "\n", encoding="utf-8")


def _svg_shell(width: int, height: int, title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '  <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        '    <stop offset="0%" stop-color="#fffdf8"/>',
        '    <stop offset="100%" stop-color="#f4efe7"/>',
        "  </linearGradient>",
        "</defs>",
        '<rect width="100%" height="100%" fill="url(#bg)"/>',
        f'<text x="{width/2:.1f}" y="42" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#1f2933">{_esc(title)}</text>',
        f'<text x="{width/2:.1f}" y="72" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#52606d">{_esc(subtitle)}</text>',
    ]


def _log_grid(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    min_power: int,
    max_power: int,
    *,
    vertical: bool,
) -> list[str]:
    parts: list[str] = []
    for power in range(min_power, max_power + 1):
        value = 10.0 ** power
        if vertical:
            raise NotImplementedError
        y = top + plot_height - ((math.log10(value) - min_power) / (max_power - min_power)) * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#d9e2ec" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#52606d">1e{power:+d}</text>')
    return parts


def _phase_ticks(left: int, top: int, plot_height: int, x_values: list[float], x_to_px) -> list[str]:
    parts: list[str] = []
    for x_tick in sorted(set(int(v) for v in x_values)):
        x = x_to_px(float(x_tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#52606d">{x_tick}</text>')
    return parts


def _runtime_ticks(left: int, top: int, plot_width: int, plot_height: int, min_power: int, max_power: int, x_to_px) -> list[str]:
    parts: list[str] = []
    for power in range(min_power, max_power + 1):
        value = 10.0 ** power
        x = x_to_px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#eef2f6" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#52606d">1e{power:+d}</text>')
    return parts


def _axes(left: int, top: int, plot_width: int, plot_height: int) -> list[str]:
    return [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334e68" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334e68" stroke-width="2"/>',
    ]


def _series_polylines(points: list[dict[str, object]], x_to_px, y_to_px, *, x_key: str, connect_wo: bool = True) -> list[str]:
    parts: list[str] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for point in points:
        series = str(point["series"])
        label = str(point["label"])
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
            if not connect_wo and label == "faster w/o":
                group_key = "faster w/o"
            else:
                group_key = "Faster"
        elif series == "Slower":
            if not connect_wo and label == "slower w/o":
                group_key = "slower w/o"
            else:
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
        "Faster": ("#d17b0f", "6,4"),
        "Slower": ("#9b2226", "2,4"),
        "faster w/o": ("#f4a261", "6,4"),
        "slower w/o": ("#c1121f", "2,4"),
        "Average": ("#7c3aed", ""),
    }

    for group_key, group in grouped.items():
        group.sort(key=lambda item: float(item[x_key]))
        color, dash = line_styles.get(group_key, (str(group[0]["color"]), str(group[0]["dash"])))
        if len(group) > 1:
            polyline = " ".join(
                f"{x_to_px(max(float(point[x_key]), 1e-9) if x_key == 'runtime' else float(point[x_key])):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in group
            )
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3"{dash_attr} points="{polyline}"/>')
    for source_key, target_key, bridge_color in (
        ("Exponential", "Erlang", "#5c6b73"),
        ("Exponential/D", "Erlang /D", "#a3a3a3"),
    ):
        source = grouped.get(source_key, [])
        target = grouped.get(target_key, [])
        if source and target:
            target.sort(key=lambda item: float(item[x_key]))
            bridge = [source[0], target[0]]
            polyline = " ".join(
                f"{x_to_px(max(float(point[x_key]), 1e-9) if x_key == 'runtime' else float(point[x_key])):.2f},{y_to_px(float(point['error'])):.2f}"
                for point in bridge
            )
            parts.append(f'<polyline fill="none" stroke="{bridge_color}" stroke-width="3" points="{polyline}"/>')
    parts.extend(_scatter_points(points, x_to_px, y_to_px, x_key=x_key))
    return parts


def _scatter_points(points: list[dict[str, object]], x_to_px, y_to_px, *, x_key: str) -> list[str]:
    parts: list[str] = []
    for point in points:
        raw_x = float(point[x_key])
        px = x_to_px(max(raw_x, 1e-9) if x_key == "runtime" else raw_x)
        py = y_to_px(float(point["error"]))
        color = str(point["color"])
        symbol = str(point["symbol"])
        if symbol == "square":
            parts.append(f'<rect x="{px - 6:.2f}" y="{py - 6:.2f}" width="12" height="12" fill="{color}" stroke="#fffdf8" stroke-width="2"/>')
        elif symbol == "diamond":
            parts.append(f'<polygon points="{px:.2f},{py - 8:.2f} {px + 8:.2f},{py:.2f} {px:.2f},{py + 8:.2f} {px - 8:.2f},{py:.2f}" fill="{color}" stroke="#fffdf8" stroke-width="2"/>')
        elif symbol == "triangle":
            parts.append(f'<polygon points="{px:.2f},{py - 8:.2f} {px + 7:.2f},{py + 6:.2f} {px - 7:.2f},{py + 6:.2f}" fill="{color}" stroke="#fffdf8" stroke-width="2"/>')
        elif symbol == "triangle-down":
            parts.append(f'<polygon points="{px - 7:.2f},{py - 6:.2f} {px + 7:.2f},{py - 6:.2f} {px:.2f},{py + 8:.2f}" fill="{color}" stroke="#fffdf8" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6.5" fill="{color}" stroke="#fffdf8" stroke-width="2"/>')
    return parts


def _legend(legend_x: int, legend_y: int, points: list[dict[str, object]]) -> list[str]:
    legend_styles: dict[str, tuple[int, str, str, str]] = {}
    for point in points:
        series = str(point["series"])
        label = str(point["label"])
        if label == "Exponential":
            family = "Exponential"
            symbol = "circle"
            priority = 0
        elif series == "Exponential/D":
            family = "Exponential/D"
            symbol = "circle"
            priority = 0
        elif series == "Erlang":
            family = "Erlang"
            symbol = "square"
            priority = 0
        elif series == "Erlang /D":
            family = series
            symbol = str(point["symbol"])
            priority = 0
        elif series == "CME":
            family = "CME"
            symbol = str(point["symbol"])
            priority = 0
        elif series == "Faster":
            if label == "faster w/o":
                family = "faster w/o"
                priority = 0
            else:
                family = "faster"
                priority = 0
            symbol = str(point["symbol"])
        elif series == "Slower":
            if label == "slower w/o":
                family = "slower w/o"
                priority = 0
            else:
                family = "slower"
                priority = 0 if label != "slower 1" else 1
            symbol = str(point["symbol"])
        elif series == "Average":
            family = "average"
            priority = 0
            symbol = str(point["symbol"])
        else:
            continue
        candidate = (priority, str(point["color"]), str(point["dash"]), symbol)
        current = legend_styles.get(family)
        if current is None or candidate[0] < current[0]:
            legend_styles[family] = candidate

    family_order = [
        "Exponential",
        "Exponential/D",
        "Erlang",
        "Erlang /D",
        "CME",
        "faster w/o",
        "faster",
        "average",
        "slower w/o",
        "slower",
    ]
    legend_items = [
        (family, *legend_styles[family][1:])
        for family in family_order
        if family in legend_styles
    ]

    parts = [f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="260" height="{36 * len(legend_items) + 48}" rx="16" fill="#fffaf0" stroke="#e6d5b8"/>']
    for index, (label, color, dash, symbol) in enumerate(legend_items):
        y = legend_y + index * 36
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        cx = legend_x + 16
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        if symbol == "square":
            parts.append(f'<rect x="{cx - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "diamond":
            parts.append(f'<polygon points="{cx:.2f},{y - 8:.2f} {cx + 8:.2f},{y:.2f} {cx:.2f},{y + 8:.2f} {cx - 8:.2f},{y:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "triangle":
            parts.append(f'<polygon points="{cx:.2f},{y - 8:.2f} {cx + 7:.2f},{y + 6:.2f} {cx - 7:.2f},{y + 6:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "triangle-down":
            parts.append(f'<polygon points="{cx - 7:.2f},{y - 6:.2f} {cx + 7:.2f},{y - 6:.2f} {cx:.2f},{y + 8:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{y}" r="6" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#1f2933">{_esc(label)}</text>')
    return parts


def _point_legend(legend_x: int, legend_y: int, points: list[dict[str, object]]) -> list[str]:
    def family_for(point: dict[str, object]) -> tuple[int, str]:
        label = str(point["label"])
        series = str(point["series"])
        if label == "Exponential":
            return (0, "Exponential")
        if series in {"Erlang", "Erlang Iterative"}:
            return (1, "Erlang")
        if series == "CME":
            return (2, "CME")
        if series == "Faster":
            return (3, "faster")
        if series == "Slower":
            return (4, "slower")
        if series == "Average":
            return (5, "average")
        return (6, label)

    def phase_label(value: float) -> str:
        return str(int(value)) if value.is_integer() else f"{value:g}"

    groups: dict[str, dict[str, object]] = {}
    for point in sorted(points, key=lambda item: (*family_for(item), float(item["x_value"]))):
        _rank, family = family_for(point)
        group = groups.setdefault(
            family,
            {
                "rank": _rank,
                "color": str(point["color"]),
                "dash": str(point["dash"]),
                "symbol": str(point["symbol"]),
                "values": [],
            },
        )
        values = group["values"]
        if isinstance(values, list):
            x_value = float(point["x_value"])
            if x_value not in values:
                values.append(x_value)

    items: list[tuple[str, str, str, str, int]] = []
    for family, group in groups.items():
        values = group["values"]
        if family == "Exponential":
            label = family
        elif isinstance(values, list) and values:
            phases = ", ".join(phase_label(float(value)) for value in sorted(values))
            label = f"{family} {phases}"
        else:
            label = family
        items.append((label, str(group["color"]), str(group["dash"]), str(group["symbol"]), int(group["rank"])))
    items.sort(key=lambda item: item[4])

    item_gap = 30
    parts = [
        f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="260" height="{item_gap * len(items) + 42}" rx="16" fill="#fffaf0" stroke="#e6d5b8"/>'
    ]
    for index, (label, color, dash, symbol, _rank) in enumerate(items):
        y = legend_y + index * item_gap
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        cx = legend_x + 16
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        if symbol == "square":
            parts.append(f'<rect x="{cx - 6:.2f}" y="{y - 6:.2f}" width="12" height="12" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "diamond":
            parts.append(f'<polygon points="{cx:.2f},{y - 8:.2f} {cx + 8:.2f},{y:.2f} {cx:.2f},{y + 8:.2f} {cx - 8:.2f},{y:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "triangle":
            parts.append(f'<polygon points="{cx:.2f},{y - 8:.2f} {cx + 7:.2f},{y + 6:.2f} {cx - 7:.2f},{y + 6:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        elif symbol == "triangle-down":
            parts.append(f'<polygon points="{cx - 7:.2f},{y - 6:.2f} {cx + 7:.2f},{y - 6:.2f} {cx:.2f},{y + 8:.2f}" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{y}" r="6" fill="{color}" stroke="#fffaf0" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#1f2933">{_esc(label)}</text>')
    return parts


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
