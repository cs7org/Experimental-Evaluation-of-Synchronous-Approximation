from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import numpy as np
import scipy.linalg as la

from methods import cme as cme_method
from methods import fast_slow as fast_slow_method


CAPACITY = 10
UNIFORM_LOW = 0.5
UNIFORM_HIGH = 1.5
LAMBDA_VALUES = [round(0.1 * index, 1) for index in range(1, 5)]
BERNSTEIN_FILES = {
    "Bernstein 20": "Bernstein20.csv",
    "Bernstein 50": "Bernstein50.csv",
}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    series, means = _compute_series()

    csv_path = OUTPUT_DIR / "uniform_bernstein_pfull_vs_lambda.csv"
    svg_path = OUTPUT_DIR / "uniform_bernstein_pfull_vs_lambda.svg"
    csv_path.write_text(_build_csv(series, means), encoding="utf-8")
    _write_svg(svg_path, series)

    print(f"Saved {csv_path}")
    print(f"Saved {svg_path}")


def _compute_series() -> tuple[dict[str, list[tuple[float, float]]], dict[str, float]]:
    representations = {
        label: _load_bernstein(_data_path(file_name))
        for label, file_name in BERNSTEIN_FILES.items()
    }
    means = {
        label: _ph_mean(alpha, matrix_t)
        for label, (alpha, matrix_t) in representations.items()
    }
    series: dict[str, list[tuple[float, float]]] = {
        "faster w/o": [],
        "Bernstein 20": [],
        "Bernstein 50": [],
        "slower w/o": [],
    }

    for lam in LAMBDA_VALUES:
        faster_dist = _uniform_fast_slow_distribution(lam, "faster_noexp")
        slower_dist = _uniform_fast_slow_distribution(lam, "slower")
        series["faster w/o"].append((lam, float(faster_dist[-1])))
        series["slower w/o"].append((lam, float(slower_dist[-1])))

        for label, (alpha, matrix_t) in representations.items():
            dist = cme_method._compute_queue_distribution(lam, CAPACITY, alpha, matrix_t)
            probability = np.asarray(dist, dtype=float)
            probability = probability / float(np.sum(probability))
            series[label].append((lam, float(probability[-1])))

    return series, means


def _uniform_fast_slow_distribution(arrival_rate: float, kind: str) -> np.ndarray:
    generator, deferred, departure, groups = fast_slow_method._build_simple_inputs(
        arrival_rate,
        CAPACITY,
        kind,
    )
    probability = _solve_embedded_uniform(
        generator,
        deferred,
        departure,
        groups,
        UNIFORM_LOW,
        UNIFORM_HIGH,
    )
    return fast_slow_method._aggregate_simple(probability, CAPACITY)


def _solve_embedded_uniform(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    low: float,
    high: float,
) -> list[float]:
    dim = generator.shape[0]
    groups_list = [list(group) for group in groups]
    all_states = [state for group in groups_list for state in group]
    exp_states = groups_list[0]
    gen_states = [state for state in all_states if state not in exp_states]

    id_exp = fast_slow_method._id_subset_np(dim, exp_states)
    id_gen = fast_slow_method._id_subset_np(dim, gen_states)
    q_gen = id_gen @ generator
    q_exp = id_exp @ generator

    omega = id_gen @ _uniform_expm_average(q_gen, low, high)
    psi = id_gen @ _uniform_occupation_integral(q_gen, low, high)

    transition = (
        id_exp
        - (fast_slow_method._diag_inv_np(q_exp) @ q_exp)
        + (omega @ departure)
        + (psi @ deferred)
    )
    correction = -fast_slow_method._diag_inv_np(q_exp) + psi

    system = (transition - id_exp - id_gen).T
    rhs = np.zeros(dim, dtype=float)
    system[-1, :] = correction @ np.ones(dim, dtype=float)
    rhs[-1] = 1.0

    multiplier = np.linalg.solve(system, rhs)
    probability = correction.T @ multiplier
    probability = np.real_if_close(probability, tol=1000)
    probability = np.asarray(probability, dtype=float)
    probability[probability < 0.0] = 0.0
    total = float(np.sum(probability))
    if total <= 0.0:
        raise RuntimeError("Uniform embedded solver produced an invalid stationary distribution.")
    return [float(value) for value in probability / total]


def _uniform_expm_average(q_sub: np.ndarray, low: float, high: float) -> np.ndarray:
    def weight(_: float) -> float:
        return 1.0 / (high - low)

    return _gauss_matrix_integral(q_sub, low, high, weight)


def _uniform_occupation_integral(q_sub: np.ndarray, low: float, high: float) -> np.ndarray:
    _, first_part = fast_slow_method._exponential_integrals_np(q_sub, low)

    def survival_weight(time: float) -> float:
        return (high - time) / (high - low)

    return first_part + _gauss_matrix_integral(q_sub, low, high, survival_weight)


def _gauss_matrix_integral(
    q_sub: np.ndarray,
    low: float,
    high: float,
    weight,
    nodes: int = 64,
) -> np.ndarray:
    points, weights = np.polynomial.legendre.leggauss(nodes)
    center = 0.5 * (low + high)
    half_width = 0.5 * (high - low)
    result = np.zeros_like(q_sub, dtype=float)
    for point, quadrature_weight in zip(points, weights):
        time = center + half_width * float(point)
        result += float(quadrature_weight) * weight(time) * la.expm(q_sub * time)
    return half_width * result


def _load_bernstein(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8-sig")))
    if len(rows) != 2:
        raise RuntimeError(f"Expected {path} to contain alpha and matrix rows.")
    alpha = np.array([float(value) for value in rows[0]], dtype=float)
    matrix_t = np.array(
        [
            ast.literal_eval(value.replace("{", "[").replace("}", "]"))
            for value in rows[1]
        ],
        dtype=float,
    )
    return alpha, matrix_t


def _data_path(file_name: str) -> Path:
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / file_name,
        current_dir / "data" / file_name,
        current_dir.parent / file_name,
        current_dir.parent / "data" / file_name,
        Path(r"C:\Users\Dibaei\Desktop\Non Markovian Models\code") / file_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {file_name}. Looked in: "
        + ", ".join(str(path) for path in candidates)
    )


def _ph_mean(alpha: np.ndarray, matrix_t: np.ndarray) -> float:
    return float(alpha @ np.linalg.solve(-matrix_t, np.ones(len(alpha))))


def _build_csv(series: dict[str, list[tuple[float, float]]], means: dict[str, float]) -> str:
    faster = {lam: value for lam, value in series["faster w/o"]}
    slower = {lam: value for lam, value in series["slower w/o"]}
    lines = ["lambda,series,p_full,ph_mean,inside_bounds"]
    for label, points in series.items():
        mean_text = "" if label not in means else f"{means[label]:.12g}"
        for lam, value in points:
            inside = ""
            if label in means:
                lower = min(faster[lam], slower[lam])
                upper = max(faster[lam], slower[lam])
                inside = str(lower <= value <= upper)
            lines.append(f"{lam:g},{label},{value:.18e},{mean_text},{inside}")
    return "\n".join(lines) + "\n"


def _write_svg(path: Path, series: dict[str, list[tuple[float, float]]]) -> None:
    width = 1030
    height = 680
    left = 100
    right = 260
    top = 42
    bottom = 92
    plot_width = width - left - right
    plot_height = height - top - bottom

    points = [(x, y) for values in series.values() for x, y in values if y > 0.0]
    x_values = sorted({x for x, _ in points})
    y_values = [y for _, y in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min_power = math.floor(math.log10(min(y_values)))
    y_max_power = math.ceil(math.log10(max(y_values)))

    def x_to_px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_to_px(value: float) -> float:
        fraction = (math.log10(value) - y_min_power) / (y_max_power - y_min_power)
        return top + plot_height - fraction * plot_height

    style = {
        "faster w/o": {"color": "#f4a261", "dash": "8,4", "width": "2.0", "marker": "triangle"},
        "Bernstein 20": {"color": "#0c7c59", "dash": "", "width": "2.2", "marker": "square"},
        "Bernstein 50": {"color": "#1d4ed8", "dash": "", "width": "2.2", "marker": "diamond"},
        "slower w/o": {"color": "#f4a261", "dash": "2,4", "width": "2.0", "marker": "triangle-down"},
    }
    labels = ["faster w/o", "Bernstein 20", "Bernstein 50", "slower w/o"]

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for power in range(y_min_power, y_max_power + 1):
        value = 10.0 ** power
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="14" fill="#111111">1e{power:+d}</text>')

    for lam in x_values:
        x = x_to_px(lam)
        label = f"{lam:.1f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 29}" text-anchor="middle" font-family="Consolas, monospace" font-size="14" fill="#111111">{label}</text>')

    parts.append(_band(series["faster w/o"], series["slower w/o"], x_to_px, y_to_px))
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    for label in labels:
        item = style[label]
        dash = f' stroke-dasharray="{item["dash"]}"' if item["dash"] else ""
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in series[label] if y > 0.0)
        parts.append(f'<polyline fill="none" stroke="{item["color"]}" stroke-width="{item["width"]}"{dash} points="{polyline}"/>')
        for x, y in series[label]:
            if y > 0.0:
                parts.extend(_marker_parts(x_to_px(x), y_to_px(y), item["marker"], item["color"], "#ffffff", 5.2))

    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 32}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#111111">arrival rate λ</text>')
    parts.append(f'<text transform="translate(31 {top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#111111">P(full) (log scale)</text>')
    parts.extend(_legend(left + plot_width + 42, top + 18, labels, style))
    parts.append("</svg>")

    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _band(
    lower_points: list[tuple[float, float]],
    upper_points: list[tuple[float, float]],
    x_to_px,
    y_to_px,
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
    return f'<polygon points="{" ".join(points)}" fill="#e5e7eb" opacity="0.34" stroke="#9ca3af" stroke-width="1.1"/>'


def _legend(
    x: float,
    y: float,
    labels: list[str],
    style: dict[str, dict[str, str]],
) -> list[str]:
    parts = [f'<rect x="{x - 16:.1f}" y="{y - 24:.1f}" width="214" height="{34 * len(labels) + 20}" fill="#ffffff" opacity="0.94" stroke="#bdbdbd" stroke-width="1"/>']
    for index, label in enumerate(labels):
        item = style[label]
        yy = y + index * 34
        dash = f' stroke-dasharray="{item["dash"]}"' if item["dash"] else ""
        parts.append(f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + 34:.1f}" y2="{yy:.1f}" stroke="{item["color"]}" stroke-width="{item["width"]}"{dash}/>')
        parts.extend(_marker_parts(x + 17, yy, item["marker"], item["color"], "#ffffff", 5.0))
        parts.append(f'<text x="{x + 46:.1f}" y="{yy + 5:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="14.5" fill="#111111">{label}</text>')
    return parts


def _marker_parts(cx: float, cy: float, marker: str, color: str, stroke: str, size: float) -> list[str]:
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
