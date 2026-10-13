from __future__ import annotations

import csv
import html
import math
from pathlib import Path

from methods.message_passing import (
    STATE_LABELS,
    solve_cme_ground_truth,
    solve_markovian,
    solve_synchronized,
)


ARRIVAL_RATE_REFERENCE = 0.1
FORWARD_RATE = 1.0
SERVICE_TIME = 1.0
ARRIVAL_RATE_VALUES = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4]
CME_ORDERS = [15]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


PLOT_ORDER = [
    "exponential timing",
    "CME 15 ground truth",
    "synchronized faster",
    "synchronized average",
    "synchronized slower",
]

DISPLAY_LABELS = {
    "exponential timing": "Exponential",
    "CME 15 ground truth": "CME",
    "synchronized faster": "Faster w/o",
    "synchronized slower": "Slower w/o",
    "synchronized average": "Average w/o",
}

STYLE = {
    "exponential timing": {"color": "#5c6b73", "dash": "", "width": 2.6, "marker": "circle"},
    "CME 15 ground truth": {"color": "#0c7c59", "dash": "", "width": 2.6, "marker": "diamond"},
    "synchronized faster": {"color": "#f4a261", "dash": "6,4", "width": 2.6, "marker": "triangle"},
    "synchronized slower": {"color": "#f4a261", "dash": "2,4", "width": 2.6, "marker": "triangle-down"},
    "synchronized average": {"color": "#7c3aed", "dash": "", "width": 2.6, "marker": "circle"},
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _compute_rows()

    csv_path = OUTPUT_DIR / "message_passing_results.csv"
    s2_svg = OUTPUT_DIR / "message_passing_S2_bounds.svg"
    s3_svg = OUTPUT_DIR / "message_passing_S3_bounds.svg"
    mean_svg = OUTPUT_DIR / "message_passing_mean_messages.svg"
    summary_path = OUTPUT_DIR / "message_passing_summary.txt"

    _write_csv(csv_path, rows)
    _write_svg(
        s2_svg,
        rows,
        metric="S2",
        title="Message passing: forwarding throughput",
        subtitle="faster/average/slower w/o approximations with CME reference",
        y_label="Throughput measured at the exponential transition",
    )
    _write_svg(
        s3_svg,
        rows,
        metric="S3",
        title="Message passing: output throughput",
        subtitle="faster/average/slower w/o approximations with CME 15 reference",
        y_label="S3",
    )
    _write_svg(
        mean_svg,
        rows,
        metric="mean_messages",
        title="Message passing: mean number of messages",
        subtitle="K=2, exponential forwarding, deterministic first and final stations",
        y_label="E[N]",
    )
    summary_path.write_text(_build_summary(rows), encoding="utf-8")

    print(_build_reference_table(rows))
    print(f"Saved {csv_path}")
    print(f"Saved {s2_svg}")
    print(f"Saved {s3_svg}")
    print(f"Saved {mean_svg}")
    print(f"Saved {summary_path}")


def _compute_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for lam in ARRIVAL_RATE_VALUES:
        base = {
            "lambda": lam,
            "mu": FORWARD_RATE,
            "tau": SERVICE_TIME,
        }

        markovian = solve_markovian(
            arrival_rate=lam,
            forward_rate=FORWARD_RATE,
            service_time=SERVICE_TIME,
        )
        markovian["S2_raw"] = markovian["S2"]
        rows.append({**base, **markovian})
        synchronized_rows = [
            _with_s2_bound_metric(row)
            for row in solve_synchronized(
                arrival_rate=lam,
                forward_rate=FORWARD_RATE,
                service_time=SERVICE_TIME,
                variants=("faster", "slower"),
            )
        ]
        for row in synchronized_rows:
            rows.append({**base, **row})

        synchronized_by_name = {str(row["approach"]): row for row in synchronized_rows}
        average = _average_synchronized_rows(
            synchronized_by_name["synchronized faster"],
            synchronized_by_name["synchronized slower"],
        )
        rows.append({**base, **average})

        for row in solve_cme_ground_truth(
            arrival_rate=lam,
            forward_rate=FORWARD_RATE,
            service_time=SERVICE_TIME,
            orders=CME_ORDERS,
        ):
            row["S2_raw"] = row["S2"]
            rows.append({**base, **row})

    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fieldnames = [
        "lambda",
        "mu",
        "tau",
        "approach",
        "family",
        "order",
        "timer_phases",
        "linear_states",
        "S1",
        "S2",
        "S2_raw",
        "S3",
        "mean_stage1",
        "mean_forwarding",
        "mean_stage3",
        "mean_messages",
    ] + [f"p_{label}" for label in STATE_LABELS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _with_s2_bound_metric(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    updated = dict(row)
    raw_s2 = float(updated["S2"])
    s1 = float(updated["S1"])
    s3 = float(updated["S3"])
    updated["S2_raw"] = raw_s2
    approach = str(updated["approach"])
    if approach == "synchronized faster":
        updated["S2"] = min(s1, s3)
    elif approach == "synchronized slower":
        updated["S2"] = max(s1, s3)
    return updated


def _average_synchronized_rows(
    faster: dict[str, float | int | str],
    slower: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    average: dict[str, float | int | str] = {
        "approach": "synchronized average",
        "family": "Synchronized",
    }
    for key, faster_value in faster.items():
        if key in {"approach", "family"} or key not in slower:
            continue
        slower_value = slower[key]
        if isinstance(faster_value, (int, float)) and isinstance(slower_value, (int, float)):
            average[key] = 0.5 * (float(faster_value) + float(slower_value))
    return average


def _build_summary(rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "Message passing summary",
        f"mu={FORWARD_RATE:g}, tau={SERVICE_TIME:g}, K=2",
        "",
        "lambda,CME 15 S1,CME 15 S2,CME 15 S3,faster S2,average S2,slower S2,CME 15 E[N]",
    ]
    for lam in ARRIVAL_RATE_VALUES:
        by_name = _rows_for_lambda(rows, lam)
        cme = by_name["CME 15 ground truth"]
        faster = by_name["synchronized faster"]
        average = by_name["synchronized average"]
        slower = by_name["synchronized slower"]
        lines.append(
            ",".join(
                [
                    f"{lam:g}",
                    _fmt(float(cme["S1"])),
                    _fmt(float(cme["S2"])),
                    _fmt(float(cme["S3"])),
                    _fmt(float(faster["S2"])),
                    _fmt(float(average["S2"])),
                    _fmt(float(slower["S2"])),
                    _fmt(float(cme["mean_messages"])),
                ],
            ),
        )
    return "\n".join(lines) + "\n"


def _build_reference_table(rows: list[dict[str, float | int | str]]) -> str:
    by_name = _rows_for_lambda(rows, ARRIVAL_RATE_REFERENCE)
    lines = [
        f"Reference case: lambda={ARRIVAL_RATE_REFERENCE:g}, mu={FORWARD_RATE:g}, tau={SERVICE_TIME:g}",
        "approach,S1,S2,S3,mean_messages",
    ]
    for name in PLOT_ORDER:
        row = by_name[name]
        lines.append(
            ",".join(
                [
                    name,
                    _fmt(float(row["S1"])),
                    _fmt(float(row["S2"])),
                    _fmt(float(row["S3"])),
                    _fmt(float(row["mean_messages"])),
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


PUBLICATION_WIDTH = 885
PUBLICATION_HEIGHT = 570
PUBLICATION_LEFT = 70
PUBLICATION_RIGHT = 220
PUBLICATION_TOP = 35
PUBLICATION_BOTTOM = 80
PUBLICATION_Y_TICKS = [0.0, 0.06, 0.13, 0.20, 0.26, 0.33]
PUBLICATION_X_TICKS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40]


def _publication_geometry() -> tuple[float, float, float, float]:
    return (
        float(PUBLICATION_LEFT),
        float(PUBLICATION_BOTTOM),
        float(PUBLICATION_WIDTH - PUBLICATION_LEFT - PUBLICATION_RIGHT),
        float(PUBLICATION_HEIGHT - PUBLICATION_TOP - PUBLICATION_BOTTOM),
    )


def _publication_maps():
    left, bottom, plot_width, plot_height = _publication_geometry()

    def x_to_px(value: float) -> float:
        return left + (value - 0.01) / (0.40 - 0.01) * plot_width

    def y_to_px(value: float) -> float:
        return bottom + value / 0.33 * plot_height

    return x_to_px, y_to_px


def _write_publication_svg(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    series = _series(rows, "S2")
    x_to_px, y_to_px = _publication_maps()
    left, bottom, plot_width, plot_height = _publication_geometry()
    top = PUBLICATION_HEIGHT - PUBLICATION_TOP
    right = left + plot_width

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PUBLICATION_WIDTH}" height="{PUBLICATION_HEIGHT}" viewBox="0 0 {PUBLICATION_WIDTH} {PUBLICATION_HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    for value in PUBLICATION_Y_TICKS:
        y = PUBLICATION_HEIGHT - y_to_px(value)
        label = f"{value:.2f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{left - 5:.2f}" y1="{y:.2f}" x2="{left:.2f}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12:.2f}" y="{y + 5:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="14" fill="#111111">{label}</text>')

    for value in PUBLICATION_X_TICKS:
        x = x_to_px(value)
        label = f"{value:.2f}".rstrip("0").rstrip(".")
        y = PUBLICATION_HEIGHT - bottom
        parts.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{x:.2f}" y2="{y + 6:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{y + 28:.2f}" text-anchor="middle" font-family="Consolas, monospace" font-size="14" fill="#111111">{label}</text>')

    lower = series["synchronized faster"]
    upper = series["synchronized slower"]
    band_coords = [
        (x_to_px(x), PUBLICATION_HEIGHT - y_to_px(min(y1, y2)))
        for (x, y1), (_, y2) in zip(lower, upper)
    ]
    band_coords += [
        (x_to_px(x), PUBLICATION_HEIGHT - y_to_px(max(y1, y2)))
        for (x, y1), (_, y2) in reversed(list(zip(lower, upper)))
    ]
    band_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in band_coords)
    parts.append(f'<polygon points="{band_points}" fill="#e5e7eb" opacity="0.42"/>')

    for label in PLOT_ORDER:
        parts.append(_publication_svg_polyline(series[label], x_to_px, y_to_px, label))
        parts.extend(_publication_svg_markers(series[label], x_to_px, y_to_px, label))

    parts.append(f'<line x1="{left:.2f}" y1="{PUBLICATION_HEIGHT - top:.2f}" x2="{left:.2f}" y2="{PUBLICATION_HEIGHT - bottom:.2f}" stroke="#111111" stroke-width="1.6"/>')
    parts.append(f'<line x1="{left:.2f}" y1="{PUBLICATION_HEIGHT - bottom:.2f}" x2="{right:.2f}" y2="{PUBLICATION_HEIGHT - bottom:.2f}" stroke="#111111" stroke-width="1.6"/>')
    parts.append(f'<text x="{left + plot_width / 2:.2f}" y="{PUBLICATION_HEIGHT - 34:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="19" fill="#111111">Arrival rate &#955;</text>')
    parts.append(f'<text x="20" y="{PUBLICATION_HEIGHT / 2:.2f}" text-anchor="middle" transform="rotate(-90 20 {PUBLICATION_HEIGHT / 2:.2f})" font-family="Segoe UI, Arial, sans-serif" font-size="18" fill="#111111">Throughput measured at the exponential transition</text>')
    parts.extend(_publication_svg_legend(510, 315))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _publication_svg_polyline(points, x_to_px, y_to_px, label: str) -> str:
    style = STYLE[label]
    coords = " ".join(
        f"{x_to_px(x):.2f},{PUBLICATION_HEIGHT - y_to_px(y):.2f}"
        for x, y in points
    )
    dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
    return f'<polyline points="{coords}" fill="none" stroke="{style["color"]}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"{dash}/>'


def _publication_svg_markers(points, x_to_px, y_to_px, label: str) -> list[str]:
    return [
        part
        for x, y in points
        for part in _marker_parts(
            x_to_px(x),
            PUBLICATION_HEIGHT - y_to_px(y),
            label,
            5.6,
        )
    ]


def _publication_svg_legend(x: float, y: float) -> list[str]:
    item_gap = 31
    box_height = len(PLOT_ORDER) * item_gap + 30
    parts = [
        f'<rect x="{x - 14:.2f}" y="{y - 22:.2f}" width="190" height="{box_height:.2f}" fill="#ffffff" opacity="0.94" stroke="#bdbdbd" stroke-width="1"/>',
    ]
    for index, label in enumerate(PLOT_ORDER):
        style = STYLE[label]
        yy = y + index * item_gap
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(f'<line x1="{x:.2f}" y1="{yy:.2f}" x2="{x + 34:.2f}" y2="{yy:.2f}" stroke="{style["color"]}" stroke-width="2.4" stroke-linecap="round"{dash}/>')
        parts.extend(_marker_parts(x + 17, yy, label, 5.3))
        parts.append(f'<text x="{x + 46:.2f}" y="{yy + 5:.2f}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#1f2933">{html.escape(_display_label(label))}</text>')
    return parts


def _write_publication_pdf(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    series = _series(rows, "S2")
    x_to_px, y_to_px = _publication_maps()
    left, bottom, plot_width, plot_height = _publication_geometry()
    right = left + plot_width
    commands: list[str] = ["1 1 1 rg", f"0 0 {PUBLICATION_WIDTH} {PUBLICATION_HEIGHT} re f"]

    for value in PUBLICATION_Y_TICKS:
        y = y_to_px(value)
        label = f"{value:.2f}".rstrip("0").rstrip(".")
        commands.append(f"0 0 0 RG 0.8 w {left - 5:.2f} {y:.2f} m {left:.2f} {y:.2f} l S")
        commands.append(_pdf_text("F2", 14, left - 12 - len(label) * 7.2, y - 4, label))

    for value in PUBLICATION_X_TICKS:
        x = x_to_px(value)
        label = f"{value:.2f}".rstrip("0").rstrip(".")
        commands.append(f"0 0 0 RG 0.8 w {x:.2f} {bottom:.2f} m {x:.2f} {bottom - 6:.2f} l S")
        commands.append(_pdf_text("F2", 14, x - len(label) * 4.2, bottom - 29, label))

    lower = series["synchronized faster"]
    upper = series["synchronized slower"]
    band = [
        (x_to_px(x), y_to_px(min(y1, y2)))
        for (x, y1), (_, y2) in zip(lower, upper)
    ]
    band += [
        (x_to_px(x), y_to_px(max(y1, y2)))
        for (x, y1), (_, y2) in reversed(list(zip(lower, upper)))
    ]
    commands.append("0.90 0.91 0.93 rg " + _pdf_polygon(band) + " f")

    for label in PLOT_ORDER:
        commands.extend(_pdf_polyline(series[label], x_to_px, y_to_px, label))
        for x, y in series[label]:
            commands.extend(_pdf_marker(x_to_px(x), y_to_px(y), label, 5.5))

    commands.extend([
        f"0 0 0 RG 1.6 w {left:.2f} {bottom:.2f} m {left:.2f} {bottom + plot_height:.2f} l S",
        f"0 0 0 RG 1.6 w {left:.2f} {bottom:.2f} m {right:.2f} {bottom:.2f} l S",
        _pdf_text("F1", 19, left + plot_width / 2 - 55, 37, "Arrival rate"),
        _pdf_lambda_symbol(left + plot_width / 2 + 43, 37, 17),
        "0 0 0 rg BT /F1 16 Tf 0 1 -1 0 23 52 Tm (Throughput measured at the exponential transition) Tj ET",
    ])
    commands.extend(_pdf_legend(510, 302))
    _write_simple_pdf(path, "\n".join(commands) + "\n")


def _pdf_text(font: str, size: float, x: float, y: float, value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"0 0 0 rg BT /{font} {size:g} Tf 1 0 0 1 {x:.2f} {y:.2f} Tm ({escaped}) Tj ET"


def _pdf_lambda_symbol(x: float, y: float, size: float) -> str:
    return f"0 0 0 RG 1.4 w {x + 2:.2f} {y + 1:.2f} m {x + size * 0.52:.2f} {y + size:.2f} l {x + size:.2f} {y + 1:.2f} l S"


def _pdf_color(color: str) -> str:
    value = color.lstrip("#")
    return " ".join(f"{int(value[index:index + 2], 16) / 255:.4f}" for index in (0, 2, 4))


def _pdf_polyline(points, x_to_px, y_to_px, label: str) -> list[str]:
    style = STYLE[label]
    if not points:
        return []
    path = f"{x_to_px(points[0][0]):.2f} {y_to_px(points[0][1]):.2f} m "
    path += " ".join(f"{x_to_px(x):.2f} {y_to_px(y):.2f} l" for x, y in points[1:])
    dash = f"[{style['dash'].replace(',', ' ')}] 0 d" if style["dash"] else "[] 0 d"
    return [f"{_pdf_color(style['color'])} RG 2.4 w {dash} {path} S", "[] 0 d"]


def _pdf_marker(x: float, y: float, label: str, size: float) -> list[str]:
    style = STYLE[label]
    color = _pdf_color(style["color"])
    marker = str(style["marker"])
    parts = [f"{color} rg 1 1 1 RG 0.8 w"]
    if marker == "circle":
        k = 0.5522848 * size
        parts.append(
            f"{x + size:.2f} {y:.2f} m {x + size:.2f} {y + k:.2f} {x + k:.2f} {y + size:.2f} {x:.2f} {y + size:.2f} c "
            f"{x - k:.2f} {y + size:.2f} {x - size:.2f} {y + k:.2f} {x - size:.2f} {y:.2f} c "
            f"{x - size:.2f} {y - k:.2f} {x - k:.2f} {y - size:.2f} {x:.2f} {y - size:.2f} c "
            f"{x + k:.2f} {y - size:.2f} {x + size:.2f} {y - k:.2f} {x + size:.2f} {y:.2f} c B",
        )
    elif marker == "diamond":
        parts.append(f"{x:.2f} {y + size + 1:.2f} m {x + size + 1:.2f} {y:.2f} l {x:.2f} {y - size - 1:.2f} l {x - size - 1:.2f} {y:.2f} l h B")
    elif marker == "triangle":
        parts.append(f"{x:.2f} {y + size + 1:.2f} m {x + size + 1:.2f} {y - size:.2f} l {x - size - 1:.2f} {y - size:.2f} l h B")
    elif marker == "triangle-down":
        parts.append(f"{x - size - 1:.2f} {y + size:.2f} m {x + size + 1:.2f} {y + size:.2f} l {x:.2f} {y - size - 1:.2f} l h B")
    return parts


def _pdf_polygon(points) -> str:
    if not points:
        return ""
    path = f"{points[0][0]:.2f} {points[0][1]:.2f} m "
    path += " ".join(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
    return path + " h"


def _pdf_legend(x: float, y: float) -> list[str]:
    item_gap = 31
    box_height = len(PLOT_ORDER) * item_gap + 30
    parts = [
        f"0.98 0.98 0.98 rg 0.72 0.72 0.72 RG 0.8 w {x - 14:.2f} {y - box_height + 12:.2f} 190 {box_height:.2f} re B",
    ]
    for index, label in enumerate(PLOT_ORDER):
        style = STYLE[label]
        yy = y - index * item_gap
        dash = f"[{style['dash'].replace(',', ' ')}] 0 d" if style["dash"] else "[] 0 d"
        parts.append(f"{_pdf_color(style['color'])} RG 2.4 w {dash} {x:.2f} {yy:.2f} m {x + 34:.2f} {yy:.2f} l S [] 0 d")
        parts.extend(_pdf_marker(x + 17, yy, label, 5.3))
        parts.append(_pdf_text("F1", 15, x + 46, yy - 5, _display_label(label)))
    return parts


def _write_simple_pdf(path: Path, content: str) -> None:
    content_bytes = content.encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PUBLICATION_WIDTH} {PUBLICATION_HEIGHT}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode("ascii"),
        b"<< /Length " + str(len(content_bytes)).encode("ascii") + b" >>\nstream\n" + content_bytes + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    path.write_bytes(output)


def _write_svg(
    path: Path,
    rows: list[dict[str, float | int | str]],
    *,
    metric: str,
    title: str,
    subtitle: str,
    y_label: str,
) -> None:
    series = _series(rows, metric)
    width = 1180
    height = 760
    left = 95
    right = 300
    top = 45
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom

    points = [(x, y) for label in PLOT_ORDER for x, y in series[label] if math.isfinite(y)]
    x_values = sorted({x for x, _ in points})
    y_values = [y for _, y in points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(0.0, min(y_values))
    y_max = max(y_values) * 1.08
    if y_max <= y_min:
        y_max = y_min + 1.0

    def x_to_px(value: float) -> float:
        return left + ((value - x_min) / (x_max - x_min)) * plot_width

    def y_to_px(value: float) -> float:
        return top + plot_height - ((value - y_min) / (y_max - y_min)) * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5.0
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">{value:.3g}</text>')

    for lam in x_values:
        x = x_to_px(lam)
        label = f"{lam:.2f}".rstrip("0").rstrip(".")
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 30}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{label}</text>')

    parts.append(_band(series["synchronized faster"], series["synchronized slower"], x_to_px, y_to_px))
    for label in PLOT_ORDER:
        parts.append(_polyline(series[label], x_to_px, y_to_px, label))
        parts.extend(_markers(series[label], x_to_px, y_to_px, label))

    axis_color = "#111111"
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">arrival rate \u03bb</text>')
    parts.append(f'<text x="30" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 30 {top + plot_height / 2:.1f})" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">{html.escape(y_label)}</text>')
    parts.extend(_legend(left + plot_width + 40, top + 25))
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _series(
    rows: list[dict[str, float | int | str]],
    metric: str,
) -> dict[str, list[tuple[float, float]]]:
    result = {label: [] for label in PLOT_ORDER}
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
) -> str:
    lower = [(x, min(y1, y2)) for (x, y1), (_, y2) in zip(lower_points, upper_points)]
    upper = [(x, max(y1, y2)) for (x, y1), (_, y2) in zip(lower_points, upper_points)]
    coords = [(x_to_px(x), y_to_px(y)) for x, y in lower]
    coords += [(x_to_px(x), y_to_px(y)) for x, y in reversed(upper)]
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    return f'<polygon points="{points}" fill="#94a3b8" opacity="0.20" stroke="#64748b" stroke-width="1.2"/>'


def _polyline(points: list[tuple[float, float]], x_to_px, y_to_px, label: str) -> str:
    style = STYLE[label]
    coords = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points)
    dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
    return f'<polyline points="{coords}" fill="none" stroke="{style["color"]}" stroke-width="{style["width"]}" stroke-linecap="round" stroke-linejoin="round"{dash}/>'


def _markers(points: list[tuple[float, float]], x_to_px, y_to_px, label: str) -> list[str]:
    parts: list[str] = []
    for x_value, y_value in points:
        x = x_to_px(x_value)
        y = y_to_px(y_value)
        parts.extend(_marker_parts(x, y, label, 5.4))
    return parts


def _legend(x: float, y: float) -> list[str]:
    parts = []
    box_height = len(PLOT_ORDER) * 31 + 20
    parts.append(f'<rect x="{x - 14}" y="{y - 22}" width="230" height="{box_height}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')
    for index, label in enumerate(PLOT_ORDER):
        style = STYLE[label]
        yy = y + index * 31
        dash = f' stroke-dasharray="{style["dash"]}"' if style["dash"] else ""
        parts.append(f'<line x1="{x}" y1="{yy}" x2="{x + 34}" y2="{yy}" stroke="{style["color"]}" stroke-width="{style["width"]}" stroke-linecap="round"{dash}/>')
        parts.extend(_legend_marker(x + 17, yy, label))
        parts.append(f'<text x="{x + 46}" y="{yy + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="13.5" fill="#1f2933">{html.escape(_display_label(label))}</text>')
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
