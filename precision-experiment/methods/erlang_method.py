from __future__ import annotations

import time

import numpy as np

from methods.ph_queue import (
    aggregate as _aggregate,
    build_generator as _build_generator,
    metrics as _metrics,
)
from methods.mpmath_stationary import precision_tolerance as _precision_tolerance
from methods.erlang_solvers import (
    StationarySolverSettings,
    solve_stationary as _solve_stationary,
)


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_counts: list[int],
    stationary_method: str = "sparse_lu_direct_q",
    tolerance: float = 1e-12,
    max_iterations: int = 50_000,
    relaxation: float | None = None,
    solver_settings: StationarySolverSettings | None = None,
    precision_digits: int | None = None,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    effective_tolerance = (
        _precision_tolerance(precision_digits)
        if precision_digits is not None
        else tolerance
    )
    for phase_count in phase_counts:
        initial, phase_gen = _representation(phase_count, service_time)

        build_started = time.perf_counter()
        generator = _build_generator(arrival_rate, capacity, initial, phase_gen)
        build_time = time.perf_counter() - build_started

        solve_started = time.perf_counter()
        solve_result = _solve_stationary(
            generator,
            stationary_method,
            effective_tolerance,
            max_iterations,
            relaxation,
            settings=solver_settings,
            precision_digits=precision_digits,
        )
        probability = solve_result.probability
        solve_time = time.perf_counter() - solve_started

        metrics_started = time.perf_counter()
        dist = _aggregate(probability, capacity, phase_count)
        p_idle, p_full, response_time = _metrics(dist, arrival_rate)
        metrics_time = time.perf_counter() - metrics_started

        results.append({
            "approach": _approach_name(phase_count),
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
            "timing_build": build_time,
            "timing_solve": solve_time,
            "timing_metrics": metrics_time,
            "timing_total": build_time + solve_time + metrics_time,
            "solver": solve_result.solver_name,
            "solver_key": solve_result.solver_key,
            "solver_converged": solve_result.converged,
            "solver_iterations": solve_result.iterations,
            "precision_digits": precision_digits,
            "tolerance": effective_tolerance,
            "stationary_residual_l1": solve_result.residual_l1,
            "matrix_memory_bytes": solve_result.matrix_memory_bytes,
            "factor_memory_bytes": solve_result.factor_memory_bytes,
            "memory_bytes": solve_result.memory_bytes,
            "state_count": generator.shape[0],
            "generator_nnz": generator.nnz,
        })
    return results


def _approach_name(phase_count: int) -> str:
    if phase_count == 1:
        return "Exponential"
    return f"Erlang {phase_count}"


def _representation(phase_count: int, service_time: float) -> tuple[np.ndarray, np.ndarray]:
    service_rate = phase_count / service_time
    initial = np.zeros(phase_count)
    initial[0] = 1.0
    phase_gen = np.zeros((phase_count, phase_count))

    for i in range(phase_count):
        phase_gen[i, i] = -service_rate
        if i < phase_count - 1:
            phase_gen[i, i + 1] = service_rate

    return initial, phase_gen
