from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from methods.me_queue import (
    aggregate as _aggregate,
    mean_and_scv as _mean_and_scv,
    metrics as _metrics,
)
from methods.mpmath_stationary import precision_tolerance as _precision_tolerance
from methods.cme_solvers import solve_queue as _solve_queue


_ILT_CME_PATH = Path(__file__).with_name("iltcme.json")


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    targets: list[dict],
    integral_method: str = "real_schur",
    linear_method: str = "sparse_lu",
    tolerance: float = 1e-12,
    precision_digits: int | None = None,
) -> list[dict[str, object]]:
    """
    Each target is a dict with keys:
        "name":   str   - label for this CME variant (e.g. "CME 5")
        "phases": int   - total number of phases

    The calculation is selected with two independent options:
        integral_method - how to build the regenerative transition/time matrices
        linear_method   - how to solve the embedded-chain stationary equation
    """
    results: list[dict[str, object]] = []
    effective_tolerance = (
        _precision_tolerance(precision_digits)
        if precision_digits is not None
        else tolerance
    )
    for target in targets:
        phase_count = int(target["phases"])
        name = str(target["name"])

        build_started = time.perf_counter()
        entry = _load_cme_entry(phase_count)
        target_scv = float(entry["cv2"])
        initial, phase_gen = _representation(entry, service_time)
        fit_mean, fit_scv = _mean_and_scv(initial, phase_gen)
        build_time = time.perf_counter() - build_started

        solve_started = time.perf_counter()
        solve_result = _solve_queue(
            arrival_rate=arrival_rate,
            capacity=capacity,
            initial=initial,
            phase_gen=phase_gen,
            integral_method=integral_method,
            linear_method=linear_method,
            tolerance=effective_tolerance,
            precision_digits=precision_digits,
        )
        probability = solve_result.probability
        solve_time = time.perf_counter() - solve_started

        metrics_started = time.perf_counter()
        dist = (
            solve_result.level_distribution
            if solve_result.level_distribution is not None
            else _aggregate(probability, capacity, len(initial))
        )
        p_idle, p_full, response_time = _metrics(dist, arrival_rate)
        metrics_time = time.perf_counter() - metrics_started

        results.append({
            "approach": name,
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
            "service_model": "concentrated ME",
            "model_order": len(initial),
            "target_mean": service_time,
            "target_scv": target_scv,
            "fitted_mean": fit_mean,
            "fitted_scv": fit_scv,
            "timing_build": build_time,
            "timing_solve": solve_time,
            "timing_metrics": metrics_time,
            "timing_total": build_time + solve_time + metrics_time,
            "solver": solve_result.solver_name,
            "solver_key": solve_result.solver_key,
            "integral_solver": solve_result.integral_name,
            "integral_solver_key": solve_result.integral_key,
            "linear_solver": solve_result.linear_name,
            "linear_solver_key": solve_result.linear_key,
            "solver_converged": solve_result.converged,
            "solver_iterations": solve_result.iterations,
            "precision_digits": precision_digits,
            "tolerance": effective_tolerance,
            "stationary_residual_l1": solve_result.stationary_residual_l1,
            "embedded_residual_l1": solve_result.embedded_residual_l1,
            "matrix_memory_bytes": solve_result.matrix_memory_bytes,
            "factor_memory_bytes": solve_result.factor_memory_bytes,
            "memory_bytes": solve_result.memory_bytes,
            "state_count": solve_result.state_count,
            "generator_nnz": solve_result.generator_nnz,
        })
    return results


def _representation(entry: dict[str, object], mean: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(entry["n"])
    omega = float(entry["omega"])
    mu = float(entry["mu1"]) / mean
    c = float(entry["c"])
    a = np.asarray(entry["a"], dtype=float)
    b = np.asarray(entry["b"], dtype=float)

    if len(a) != n or len(b) != n:
        raise ValueError("Concentrated ME table entry is malformed.")

    order = 1 + 2 * n
    initial = np.zeros(order)
    phase_gen = np.zeros((order, order))
    initial[0] = c
    phase_gen[0, 0] = -mu

    idx = 1
    for k, (a_k, b_k) in enumerate(zip(a, b), start=1):
        lam = mu
        nu = mu * k * omega
        coeff = np.array([
            [lam + nu, lam - nu],
            [-(lam - nu), lam + nu],
        ])
        rhs = np.array([mu * a_k, mu * b_k])
        initial[idx: idx + 2] = np.linalg.solve(coeff, rhs)
        phase_gen[idx: idx + 2, idx: idx + 2] = np.array([
            [-lam, -nu],
            [nu, -lam],
        ])
        idx += 2

    return initial, phase_gen


def _load_cme_entry(phase_count: int) -> dict[str, object]:
    table = _load_iltcme_table()
    entry = table.get(phase_count)
    if entry is not None:
        return entry

    raise ValueError(
        f"No concentrated ME entry found for CME {phase_count}. "
        f"Check {_ILT_CME_PATH} for an entry with n={phase_count - 1}."
    )


@lru_cache(maxsize=1)
def _load_iltcme_table() -> dict[int, dict[str, object]]:
    if not _ILT_CME_PATH.exists():
        raise FileNotFoundError(f"ILTCME table not found: {_ILT_CME_PATH}")

    with _ILT_CME_PATH.open("r", encoding="utf-8") as handle:
        raw_entries = json.load(handle)

    table: dict[int, dict[str, object]] = {}
    for raw_entry in raw_entries:
        entry = _normalise_iltcme_entry(raw_entry)
        phase_count = int(entry["n"]) + 1
        current = table.get(phase_count)
        if current is None or float(entry["cv2"]) < float(current["cv2"]):
            table[phase_count] = entry
    return table


def _normalise_iltcme_entry(raw_entry: dict[str, object]) -> dict[str, object]:
    n = int(raw_entry["n"])
    return {
        "n": n,
        "mu1": float(raw_entry["mu1"]),
        "omega": float(raw_entry["omega"]),
        "c": float(raw_entry["c"]),
        "a": _as_float_list(raw_entry["a"], n),
        "b": _as_float_list(raw_entry["b"], n),
        "cv2": float(raw_entry["cv2"]),
    }


def _as_float_list(value: object, length: int) -> list[float]:
    if length == 0:
        return []
    if isinstance(value, int | float):
        values = [value]
    else:
        values = list(value)  # type: ignore[arg-type]
    if len(values) != length:
        raise ValueError("ILTCME table entry is malformed.")
    return [float(v) for v in values]
