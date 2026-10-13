from __future__ import annotations

import csv
import math
from pathlib import Path

from methods.cme import solve as solve_cme
from methods.embedded_one_server import solve_erlang as solve_one_server_erlang
from methods.erlang import solve as solve_erlang_dtmc
from methods.fast_slow import solve as solve_fast_slow
from methods.pure_erlang import solve as solve_erlang


SERVICE_TIME = 1.0
CAPACITY = 10
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

LAMBDA_VALUES_0_1_TO_2 = [round(0.1 * index, 1) for index in range(1, 21)]
LAMBDA_VALUES_0_9_TO_1_1 = [round(0.9 + 0.01 * index, 2) for index in range(21)]
LAMBDA_VALUES_1_4_TO_1_6 = [round(1.4 + 0.01 * index, 2) for index in range(21)]
LAMBDA_VALUES_1_49_TO_1_51 = [round(1.49 + 0.01 * index, 2) for index in range(3)]
LAMBDA_VALUES_PHASE_10 = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
LAMBDA_VALUES_PHASE_100 = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
CME_SCV = {
    10: 0.0057367735345383161,
    15: 0.0024052513649265544,
}
SERIES_STYLES = {
    "Exponential": {"color": "#5c6b73", "dash": "", "marker": "circle"},
    "Erlang 10": {"color": "#1d4ed8", "dash": "", "marker": "square"},
    "Erlang 10/D": {"color": "#60a5fa", "dash": "4,2", "marker": "square"},
    "CME 10": {"color": "#0c7c59", "dash": "", "marker": "diamond"},
    "CME 10/D": {"color": "#14b8a6", "dash": "4,2", "marker": "diamond"},
    "CME 15": {"color": "#0c7c59", "dash": "", "marker": "diamond"},
    "Faster w/o": {"color": "#f4a261", "dash": "6,4", "marker": "triangle"},
    "Average w/o": {"color": "#7c3aed", "dash": "", "marker": "circle"},
    "Slower w/o": {"color": "#f4a261", "dash": "2,4", "marker": "triangle-down"},
    "Faster 10": {"color": "#c1121f", "dash": "8,4", "marker": "triangle"},
    "Slower 10": {"color": "#c1121f", "dash": "4,2", "marker": "triangle-down"},
    "Faster 100": {"color": "#d17b0f", "dash": "9,3,2,3", "marker": "triangle"},
    "Slower 100": {"color": "#9b2226", "dash": "2,4", "marker": "triangle-down"},
}

DISPLAY_LABELS = {
    "faster w/o": "Faster w/o",
    "average w/o": "Average w/o",
    "slower w/o": "Slower w/o",
    "faster 10": "Faster 10",
    "slower 10": "Slower 10",
    "faster 100": "Faster 100",
    "slower 100": "Slower 100",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_plot_set(
        output_stem="response_vs_lambda_1",
        lambda_values=LAMBDA_VALUES_PHASE_10,
        cme_order=10,
        faster_variant="faster 10",
        slower_variant="slower 10",
        erlang_phases=10,
        legend_right=250,
        subtitle="Representative family comparison using Erlang 10, CME 10, Faster 10, and Slower 10",
    )
    _write_plot_set(
        output_stem="response_vs_lambda_3",
        lambda_values=LAMBDA_VALUES_PHASE_100,
        cme_order=15,
        faster_variant="faster 100",
        slower_variant="slower 100",
        erlang_phases=None,
        legend_right=300,
        subtitle="Representative family comparison using CME 15, Faster 100, and Slower 100",
    )
    _write_response_0_1_2_set(
        output_stem="response_vs_lambda_0_1_2_without_fast_slow_10",
        lambda_values=LAMBDA_VALUES_0_1_TO_2,
        include_fast_slow_10=False,
    )
    _write_response_0_1_2_set(
        output_stem="response_vs_lambda_0_1_2_with_fast_slow_10",
        lambda_values=LAMBDA_VALUES_0_1_TO_2,
        include_fast_slow_10=True,
        include_average_wo=True,
        show_band=True,
    )
    _write_response_0_1_2_d_set(
        output_stem="response_vs_lambda_0_1_2_with_erlang10d_cme10d",
        lambda_values=LAMBDA_VALUES_0_1_TO_2,
        include_fast_slow_10=True,
        replacement_mode=False,
        include_average_wo=True,
    )
    _write_response_0_1_2_d_set(
        output_stem="response_vs_lambda_0_1_2_fig3_1_d_replacements",
        lambda_values=LAMBDA_VALUES_0_1_TO_2,
        include_fast_slow_10=True,
        replacement_mode=True,
        include_average_wo=True,
    )
    _write_response_0_1_2_d_set(
        output_stem="response_vs_lambda_1_4_1_6_with_erlang10d_cme10d",
        lambda_values=LAMBDA_VALUES_1_4_TO_1_6,
        include_fast_slow_10=True,
        replacement_mode=False,
        include_average_wo=True,
    )
    _write_response_0_1_2_d_set(
        output_stem="response_vs_lambda_1_4_1_6_fig6_replacements",
        lambda_values=LAMBDA_VALUES_1_4_TO_1_6,
        include_fast_slow_10=True,
        replacement_mode=True,
        include_average_wo=True,
    )
    _write_response_0_1_2_set(
        output_stem="response_vs_lambda_1_4_1_6_with_average_wo",
        lambda_values=LAMBDA_VALUES_1_49_TO_1_51,
        include_fast_slow_10=True,
        include_average_wo=True,
        include_erlang10_d=True,
        show_band=True,
        band_labels=("Faster 10", "Slower 10"),
    )
    _write_response_0_1_2_set(
        output_stem="response_vs_lambda_0_9_1_1_without_fast_slow_10",
        lambda_values=LAMBDA_VALUES_0_9_TO_1_1,
        include_fast_slow_10=False,
    )
    _write_response_0_1_2_set(
        output_stem="response_vs_lambda_0_9_1_1_with_fast_slow_10",
        lambda_values=LAMBDA_VALUES_0_9_TO_1_1,
        include_fast_slow_10=True,
    )


def _write_response_0_1_2_set(
    *,
    output_stem: str,
    lambda_values: list[float],
    include_fast_slow_10: bool,
    include_average_wo: bool = False,
    include_erlang10_d: bool = False,
    show_band: bool = False,
    band_labels: tuple[str, str] | None = None,
    average_last: bool = False,
) -> None:
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    if csv_path.exists():
        series = _read_csv_series(csv_path)
        if not include_erlang10_d:
            series = [item for item in series if str(item["label"]) != "Erlang 10/D"]
        has_average = any(str(item["label"]) == "Average w/o" for item in series)
        has_erlang10_d = any(str(item["label"]) == "Erlang 10/D" for item in series)
        existing_lambda_values = sorted(
            {float(x) for item in series for x, _ in item["points"]},  # type: ignore[index]
        )
        expected_lambda_values = sorted(float(value) for value in lambda_values)
        if (
            (include_average_wo and not has_average)
            or (include_erlang10_d and not has_erlang10_d)
            or existing_lambda_values != expected_lambda_values
        ):
            series = _compute_response_0_1_2_series(
                lambda_values=lambda_values,
                include_fast_slow_10=include_fast_slow_10,
                include_average_wo=include_average_wo,
                include_erlang10_d=include_erlang10_d,
            )
            csv_path.write_text(_build_csv(series), encoding="utf-8")
            print(f"Regenerated {csv_path} with Average w/o")
        else:
            csv_path.write_text(_build_csv(series), encoding="utf-8")
            print(f"Updated {csv_path}")
    else:
        series = _compute_response_0_1_2_series(
            lambda_values=lambda_values,
            include_fast_slow_10=include_fast_slow_10,
            include_average_wo=include_average_wo,
            include_erlang10_d=include_erlang10_d,
        )
        csv_path.write_text(_build_csv(series), encoding="utf-8")
        print(f"Saved {csv_path}")

    if average_last:
        average_series = [item for item in series if str(item["label"]) == "Average w/o"]
        series = [item for item in series if str(item["label"]) != "Average w/o"] + average_series

    subtitle = "Exponential, Erlang 10, CME 10, Faster/Slower w/o"
    if include_average_wo:
        subtitle = "Exponential, Erlang 10, CME 10, Faster/Average/Slower w/o"
    if include_erlang10_d:
        subtitle = subtitle.replace("Erlang 10, CME 10", "Erlang 10, Erlang 10/D, CME 10")
    if include_fast_slow_10:
        subtitle += ", and Faster/Slower 10"

    _write_svg(
        svg_path,
        series,
        title=f"Mean Response Time by Lambda for K={CAPACITY}",
        subtitle=subtitle,
        legend_right=360 if include_fast_slow_10 and include_average_wo else 330 if include_fast_slow_10 else 280,
        show_band=show_band,
        band_labels=band_labels,
    )

    print(f"Saved {svg_path}")


def _write_response_0_1_2_d_set(
    *,
    output_stem: str,
    lambda_values: list[float],
    include_fast_slow_10: bool,
    replacement_mode: bool,
    include_average_wo: bool = False,
) -> None:
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    series = _compute_response_0_1_2_d_series(
        lambda_values=lambda_values,
        include_fast_slow_10=include_fast_slow_10,
        replacement_mode=replacement_mode,
        include_average_wo=include_average_wo,
    )
    csv_path.write_text(_build_csv(series), encoding="utf-8")
    print(f"Saved {csv_path}")

    if replacement_mode:
        subtitle = "Exponential, Erlang 10/D, CME 10/D, Faster/Slower w/o"
        if include_average_wo:
            subtitle = "Exponential, Erlang 10/D, CME 10/D, Faster/Average/Slower w/o"
        if include_fast_slow_10:
            subtitle += ", and Faster/Slower 10"
        legend_right = 360 if include_fast_slow_10 and include_average_wo else 330 if include_fast_slow_10 else 280
    else:
        subtitle = "Exponential, Erlang 10, Erlang 10/D, CME 10, CME 10/D, Faster/Slower w/o"
        if include_average_wo:
            subtitle = "Exponential, Erlang 10, Erlang 10/D, CME 10, CME 10/D, Faster/Average/Slower w/o"
        if include_fast_slow_10:
            subtitle += ", and Faster/Slower 10"
        legend_right = 390 if include_fast_slow_10 and include_average_wo else 360 if include_fast_slow_10 else 310

    _write_svg(
        svg_path,
        series,
        title=f"Mean Response Time by Lambda for K={CAPACITY}",
        subtitle=subtitle,
        legend_right=legend_right,
    )

    print(f"Saved {svg_path}")


def _compute_response_0_1_2_d_series(
    *,
    lambda_values: list[float],
    include_fast_slow_10: bool,
    replacement_mode: bool,
    include_average_wo: bool = False,
) -> list[dict[str, object]]:
    exponential_points: list[tuple[float, float]] = []
    erlang10_points: list[tuple[float, float]] = []
    erlang10d_points: list[tuple[float, float]] = []
    cme10_points: list[tuple[float, float]] = []
    cme10d_points: list[tuple[float, float]] = []
    faster_wo_points: list[tuple[float, float]] = []
    average_wo_points: list[tuple[float, float]] = []
    slower_wo_points: list[tuple[float, float]] = []
    faster10_points: list[tuple[float, float]] = []
    slower10_points: list[tuple[float, float]] = []

    variants = ["faster w/o", "slower w/o"]
    if include_fast_slow_10:
        variants.extend(["faster 10", "slower 10"])

    for lam in lambda_values:
        exponential_row = solve_erlang_dtmc(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_counts=[1],
            stationary_method="power",
            tolerance=1e-12,
            max_iterations=50_000,
            relaxation=None,
        )[0]
        exponential_points.append((lam, float(exponential_row["response_time"])))

        erlang10_row = solve_erlang(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_counts=[10],
        )[0]
        erlang10_points.append((lam, float(erlang10_row["response_time"])))

        erlang10d_row = solve_one_server_erlang(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_count=10,
            precision_digits=30,
            solver="positive",
        )[0]
        erlang10d_points.append((lam, float(erlang10d_row["response_time"])))

        cme10_row = _solve_cme_best_effort(lam, 10)
        if cme10_row is not None:
            cme10_points.append((lam, float(cme10_row["response_time"])))
            cme10d_points.append((lam, float(cme10_row["response_time"])))

        fast_slow_rows = solve_fast_slow(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            variants=variants,
            precision_digits=80,
        )
        faster_wo_row: dict[str, float | str] | None = None
        slower_wo_row: dict[str, float | str] | None = None
        for row in fast_slow_rows:
            approach = str(row["approach"])
            value = float(row["response_time"])
            if approach == "faster w/o":
                faster_wo_row = row
                faster_wo_points.append((lam, value))
            elif approach == "slower w/o":
                slower_wo_row = row
                slower_wo_points.append((lam, value))
            elif approach == "faster 10":
                faster10_points.append((lam, value))
            elif approach == "slower 10":
                slower10_points.append((lam, value))
        if include_average_wo and faster_wo_row is not None and slower_wo_row is not None:
            average_wo_points.append((lam, _average_response_time(lam, faster_wo_row, slower_wo_row)))

    series = [
        {
            "label": "Exponential",
            "color": "#5c6b73",
            "dash": "",
            "marker": "circle",
            "points": exponential_points,
        },
    ]
    if not replacement_mode:
        series.append(
            {
                "label": "Erlang 10",
                "points": erlang10_points,
                **SERIES_STYLES["Erlang 10"],
            },
        )
    series.extend(
        [
            {
                "label": "Erlang 10/D",
                "points": erlang10d_points,
                **SERIES_STYLES["Erlang 10/D"],
            },
        ],
    )
    if not replacement_mode:
        series.append(
            {
                "label": "CME 10",
                "points": cme10_points,
                **SERIES_STYLES["CME 10"],
            },
        )
    series.extend(
        [
            {
                "label": "CME 10/D",
                "points": cme10d_points,
                **SERIES_STYLES["CME 10/D"],
            },
            {
                "label": "Faster w/o",
                "points": faster_wo_points,
                **SERIES_STYLES["Faster w/o"],
            },
        ],
    )
    if include_average_wo:
        series.append(
            {
                "label": "Average w/o",
                "points": average_wo_points,
                **SERIES_STYLES["Average w/o"],
            },
        )
    series.append(
        {
            "label": "Slower w/o",
            "points": slower_wo_points,
            **SERIES_STYLES["Slower w/o"],
        },
    )
    if include_fast_slow_10:
        series.extend(
            [
                {
                    "label": "Faster 10",
                    "points": faster10_points,
                    **SERIES_STYLES["Faster 10"],
                },
                {
                    "label": "Slower 10",
                    "points": slower10_points,
                    **SERIES_STYLES["Slower 10"],
                },
            ],
        )
    return series


def _compute_response_0_1_2_series(
    *,
    lambda_values: list[float],
    include_fast_slow_10: bool,
    include_average_wo: bool = False,
    include_erlang10_d: bool = False,
) -> list[dict[str, object]]:
    exponential_points: list[tuple[float, float]] = []
    erlang10_points: list[tuple[float, float]] = []
    erlang10d_points: list[tuple[float, float]] = []
    cme10_points: list[tuple[float, float]] = []
    faster_wo_points: list[tuple[float, float]] = []
    average_wo_points: list[tuple[float, float]] = []
    slower_wo_points: list[tuple[float, float]] = []
    faster10_points: list[tuple[float, float]] = []
    slower10_points: list[tuple[float, float]] = []

    variants = ["faster w/o", "slower w/o"]
    if include_fast_slow_10:
        variants.extend(["faster 10", "slower 10"])

    for lam in lambda_values:
        exponential_row = solve_erlang_dtmc(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_counts=[1],
            stationary_method="power",
            tolerance=1e-12,
            max_iterations=50_000,
            relaxation=None,
        )[0]
        exponential_points.append((lam, float(exponential_row["response_time"])))

        erlang10_row = solve_erlang(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            phase_counts=[10],
        )[0]
        erlang10_points.append((lam, float(erlang10_row["response_time"])))

        if include_erlang10_d:
            erlang10d_row = solve_one_server_erlang(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                phase_count=10,
                precision_digits=30,
                solver="positive",
            )[0]
            erlang10d_points.append((lam, float(erlang10d_row["response_time"])))

        cme10_row = _solve_cme_best_effort(lam, 10)
        if cme10_row is not None:
            cme10_points.append((lam, float(cme10_row["response_time"])))

        fast_slow_rows = solve_fast_slow(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            variants=variants,
            precision_digits=80,
        )
        faster_wo_row: dict[str, float | str] | None = None
        slower_wo_row: dict[str, float | str] | None = None
        for row in fast_slow_rows:
            approach = str(row["approach"])
            value = float(row["response_time"])
            if approach == "faster w/o":
                faster_wo_row = row
                faster_wo_points.append((lam, value))
            elif approach == "slower w/o":
                slower_wo_row = row
                slower_wo_points.append((lam, value))
            elif approach == "faster 10":
                faster10_points.append((lam, value))
            elif approach == "slower 10":
                slower10_points.append((lam, value))
        if include_average_wo and faster_wo_row is not None and slower_wo_row is not None:
            average_wo_points.append((lam, _average_response_time(lam, faster_wo_row, slower_wo_row)))

    series = [
        {
            "label": "Exponential",
            "color": "#5c6b73",
            "dash": "",
            "marker": "circle",
            "points": exponential_points,
        },
        {
            "label": "Erlang 10",
            "points": erlang10_points,
            **SERIES_STYLES["Erlang 10"],
        },
    ]
    if include_erlang10_d:
        series.append(
            {
                "label": "Erlang 10/D",
                "points": erlang10d_points,
                **SERIES_STYLES["Erlang 10/D"],
            },
        )
    series.extend(
        [
            {
                "label": "CME 10",
                "points": cme10_points,
                **SERIES_STYLES["CME 10"],
            },
        {
            "label": "Faster w/o",
            "points": faster_wo_points,
            **SERIES_STYLES["Faster w/o"],
        },
        ]
    )
    if include_average_wo:
        series.append(
            {
                "label": "Average w/o",
                "points": average_wo_points,
                **SERIES_STYLES["Average w/o"],
            },
        )
    series.extend(
        [
        {
            "label": "Slower w/o",
            "points": slower_wo_points,
            **SERIES_STYLES["Slower w/o"],
        },
        ],
    )
    if include_fast_slow_10:
        series.extend(
            [
                {
                    "label": "Faster 10",
                    "points": faster10_points,
                    **SERIES_STYLES["Faster 10"],
                },
                {
                    "label": "Slower 10",
                    "points": slower10_points,
                    **SERIES_STYLES["Slower 10"],
                },
            ],
        )
    return series


def _write_plot_set(
    *,
    output_stem: str,
    lambda_values: list[float],
    cme_order: int,
    faster_variant: str,
    slower_variant: str,
    erlang_phases: int | None,
    legend_right: int,
    subtitle: str,
) -> None:
    svg_path = OUTPUT_DIR / f"{output_stem}.svg"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    if csv_path.exists():
        series = _read_csv_series(csv_path)
        csv_path.write_text(_build_csv(series), encoding="utf-8")
        print(f"Updated {csv_path}")
    else:
        series = _compute_series(
            lambda_values=lambda_values,
            cme_order=cme_order,
            faster_variant=faster_variant,
            slower_variant=slower_variant,
            erlang_phases=erlang_phases,
        )
        csv_path.write_text(_build_csv(series), encoding="utf-8")
        print(f"Saved {csv_path}")

    _write_svg(
        svg_path,
        series,
        title=f"Mean Response Time by Lambda for K={CAPACITY}",
        subtitle=subtitle,
        legend_right=legend_right,
    )

    print(f"Saved {svg_path}")


def _compute_series(
    *,
    lambda_values: list[float],
    cme_order: int,
    faster_variant: str,
    slower_variant: str,
    erlang_phases: int | None,
) -> list[dict[str, object]]:
    erlang_points: list[tuple[float, float]] = []
    cme_points: list[tuple[float, float]] = []
    faster_points: list[tuple[float, float]] = []
    slower_points: list[tuple[float, float]] = []

    for lam in lambda_values:
        if erlang_phases is not None:
            erlang_row = solve_erlang(
                arrival_rate=lam,
                service_time=SERVICE_TIME,
                capacity=CAPACITY,
                phase_counts=[erlang_phases],
            )[0]
            erlang_points.append((lam, float(erlang_row["response_time"])))

        cme_row = _solve_cme_best_effort(lam, cme_order)
        if cme_row is not None:
            cme_points.append((lam, float(cme_row["response_time"])))

        fast_slow_rows = solve_fast_slow(
            arrival_rate=lam,
            service_time=SERVICE_TIME,
            capacity=CAPACITY,
            variants=[faster_variant, slower_variant],
            precision_digits=80,
        )
        for row in fast_slow_rows:
            approach = str(row["approach"])
            value = float(row["response_time"])
            if approach == faster_variant:
                faster_points.append((lam, value))
            elif approach == slower_variant:
                slower_points.append((lam, value))

    series: list[dict[str, object]] = []
    if erlang_phases is not None:
        erlang_label = f"Erlang {erlang_phases}"
        series.append(
            {
                "label": erlang_label,
                "dash": "",
                "points": erlang_points,
                **SERIES_STYLES.get(erlang_label, {"color": "#1d4ed8", "dash": "", "marker": "square"}),
            },
        )
    cme_label = f"CME {cme_order}"
    faster_label = DISPLAY_LABELS[faster_variant]
    slower_label = DISPLAY_LABELS[slower_variant]
    series.extend(
        [
            {
                "label": cme_label,
                "dash": "10,4" if erlang_phases is not None else "",
                "points": cme_points,
                **SERIES_STYLES.get(cme_label, {"color": "#0c7c59", "marker": "diamond"}),
            },
            {
                "label": faster_label,
                "points": faster_points,
                **SERIES_STYLES[faster_label],
            },
            {
                "label": slower_label,
                "points": slower_points,
                **SERIES_STYLES[slower_label],
            },
        ],
    )
    return series


def _solve_cme_best_effort(lam: float, order: int) -> dict[str, float | str] | None:
    target = {"name": f"CME {order}", "phases": order, "scv": CME_SCV[order]}
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
        except RuntimeError:
            continue

    print(f"[warning] Skipping CME {order} at lambda={lam}: all solver modes failed.")
    return None


def _average_response_time(
    lam: float,
    faster_row: dict[str, float | str],
    slower_row: dict[str, float | str],
) -> float:
    faster_p_full = float(faster_row["p_full"])
    slower_p_full = float(slower_row["p_full"])
    faster_mean_n = float(faster_row["response_time"]) * lam * (1.0 - faster_p_full)
    slower_mean_n = float(slower_row["response_time"]) * lam * (1.0 - slower_p_full)
    average_mean_n = 0.5 * (faster_mean_n + slower_mean_n)
    average_p_full = 0.5 * (faster_p_full + slower_p_full)
    throughput = lam * (1.0 - average_p_full)
    return average_mean_n / throughput if throughput > 0.0 else float("inf")


def _build_csv(series: list[dict[str, object]]) -> str:
    lines = ["lambda,series,response_time"]
    for item in series:
        label = str(item["label"])
        for lam, value in item["points"]:  # type: ignore[index]
            lines.append(f"{lam},{label},{value:.18e}")
    return "\n".join(lines) + "\n"


def _read_csv_series(path: Path) -> list[dict[str, object]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_label = str(row["series"])
            label = DISPLAY_LABELS.get(raw_label, raw_label)
            grouped.setdefault(label, []).append((float(row["lambda"]), float(row["response_time"])))

    series: list[dict[str, object]] = []
    for label, points in grouped.items():
        style = _style_for_label(label)
        series.append({"label": label, "points": sorted(points), **style})
    return series


def _style_for_label(label: str) -> dict[str, str]:
    display_label = DISPLAY_LABELS.get(label, label)
    return SERIES_STYLES.get(display_label, {"color": "#111111", "dash": "", "marker": "circle"})


def _write_svg(
    path: Path,
    series: list[dict[str, object]],
    *,
    title: str,
    subtitle: str,
    legend_right: int,
    show_band: bool = False,
    band_labels: tuple[str, str] | None = None,
) -> None:
    width = 1180
    height = 760
    left = 100
    right = legend_right
    top = 45
    bottom = 105
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_points = [
        (float(x), float(y))
        for item in series
        for x, y in item["points"]  # type: ignore[index]
        if float(y) > 0.0 and math.isfinite(float(y))
    ]
    x_values = [x for x, _ in all_points]
    y_values = [y for _, y in all_points]

    x_min = min(x_values)
    x_max = max(x_values)
    y_log_min = math.log10(min(y_values))
    y_log_max = math.log10(max(y_values))
    y_log_pad = max((y_log_max - y_log_min) * 0.12, 0.01)
    y_log_min -= y_log_pad
    y_log_max += y_log_pad

    def x_to_px(value: float) -> float:
        frac = (value - x_min) / (x_max - x_min)
        return left + frac * plot_width

    def y_to_px(value: float) -> float:
        frac = (math.log10(value) - y_log_min) / (y_log_max - y_log_min)
        return top + plot_height - frac * plot_height

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    for tick_index in range(6):
        tick_log = y_log_min + (y_log_max - y_log_min) * tick_index / 5
        value = 10.0 ** tick_log
        y = y_to_px(value)
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" font-family="Consolas, monospace" font-size="13" fill="#111111">{_format_log_tick(value)}</text>')

    for lam in sorted(set(x_values)):
        x = x_to_px(lam)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 6}" stroke="#111111" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Consolas, monospace" font-size="13" fill="#111111">{_format_lambda_tick(lam)}</text>')

    if show_band:
        faster_label, slower_label = band_labels or ("Faster w/o", "Slower w/o")
        faster_points = next(item["points"] for item in series if str(item["label"]) == faster_label)
        slower_points = next(item["points"] for item in series if str(item["label"]) == slower_label)
        parts.append(_response_band(faster_points, slower_points, x_to_px, y_to_px))  # type: ignore[arg-type]

    parts.append(f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111111" stroke-width="2"/>')

    for item in series:
        color = str(item["color"])
        dash = str(item["dash"])
        marker = str(item["marker"])
        points = [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
        points = [(x, y) for x, y in points if y > 0.0 and math.isfinite(y)]
        polyline = " ".join(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.6"{dash_attr} points="{polyline}"/>')
        for x, y in points:
            parts.extend(_marker_parts(x_to_px(x), y_to_px(y), marker, color, "#ffffff", 5.4))

    parts.append(f'<text x="{left + plot_width/2:.1f}" y="{height - 34}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">arrival rate \u03bb</text>')
    parts.append(f'<text transform="translate(30 {top + plot_height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="17" fill="#111111">Mean response time R</text>')

    legend_x = left + plot_width + 40
    legend_y = top + 25
    parts.append(f'<rect x="{legend_x - 16}" y="{legend_y - 24}" width="220" height="{36 * len(series) + 20}" fill="#ffffff" opacity="0.92" stroke="#bdbdbd" stroke-width="1"/>')
    for index, item in enumerate(series):
        y = legend_y + index * 36
        color = str(item["color"])
        dash = str(item["dash"])
        marker = str(item["marker"])
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 32}" y2="{y}" stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.extend(_marker_parts(legend_x + 16, y, marker, color, "#ffffff", 5.2))
        parts.append(f'<text x="{legend_x + 44}" y="{y + 5}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#111111">{_esc(str(item["label"]))}</text>')

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


def _response_band(
    faster_points: list[tuple[float, float]],
    slower_points: list[tuple[float, float]],
    x_to_px,
    y_to_px,
) -> str:
    faster = {x: y for x, y in faster_points}
    slower = {x: y for x, y in slower_points}
    lambdas = sorted(set(faster) & set(slower))
    lower = [(x, min(faster[x], slower[x])) for x in lambdas]
    upper = [(x, max(faster[x], slower[x])) for x in lambdas]
    points = [
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in lower),
        *(f"{x_to_px(x):.2f},{y_to_px(y):.2f}" for x, y in reversed(upper)),
    ]
    return (
        f'<polygon points="{" ".join(points)}" fill="#94a3b8" '
        'opacity="0.34" stroke="#64748b" stroke-width="1.2"/>'
    )


def _format_lambda_tick(value: float) -> str:
    return f"{value:g}"


def _format_log_tick(value: float) -> str:
    if value >= 10.0:
        return f"{value:.0f}"
    if value >= 1.0:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()
