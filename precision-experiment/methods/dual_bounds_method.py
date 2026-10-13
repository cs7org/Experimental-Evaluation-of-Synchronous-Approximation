from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from methods.dual_bounds_solvers import (
    DEFAULT_MAX_ITERATIONS,
    _ExpandedTopology,
    build_expanded_model as _build_expanded_model,
    build_one_timer_model as _build_one_timer_model,
    normalise_integral_method as _normalise_integral_method,
    solve_model as _solve_model,
)
from methods.mpmath_stationary import precision_tolerance as _precision_tolerance


@dataclass(frozen=True)
class _VariantSpec:
    kind: str
    name: str
    phases: int | None = None


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    variants: list[str],
    integral_method: str = "poisson_real_schur",
    linear_method: str = "sparse_lu",
    tolerance: float = 1e-12,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    precision_digits: int | None = None,
) -> list[dict[str, object]]:
    if precision_digits is not None and precision_digits <= 0:
        raise ValueError("precision_digits must be positive when provided.")

    effective_tolerance = (
        _precision_tolerance(precision_digits)
        if precision_digits is not None
        else tolerance
    )
    integral_key = _normalise_integral_method(integral_method)

    results: list[dict[str, object]] = []
    for variant in variants:
        build_started = time.perf_counter()
        spec = _parse_variant(variant)

        if spec.kind in {"faster_noexp", "slower"}:
            generator, deferred, departure, groups = _build_simple_inputs(
                arrival_rate, capacity, spec.kind,
            )
            model = _build_one_timer_model(
                generator,
                deferred,
                departure,
                groups,
                service_time,
                arrival_rate,
                integral_key,
                precision_digits,
            )
            aggregate = _aggregate_simple
        elif spec.kind == "slower_additional":
            generator, deferred, departure, groups = _build_slower_additional_inputs(
                arrival_rate, capacity,
            )
            model = _build_one_timer_model(
                generator,
                deferred,
                departure,
                groups,
                service_time,
                arrival_rate,
                integral_key,
                precision_digits,
            )
            aggregate = _aggregate_slower_additional
        elif spec.kind == "faster_expanded":
            if spec.phases is None:
                raise ValueError("Expanded faster variant needs a phase count.")
            topology = _build_faster_expanded_topology(capacity, spec.phases)
            model = _build_expanded_model(
                topology=topology,
                arrival_rate=arrival_rate,
                service_time=service_time,
                phase_count=spec.phases,
                integral_key=integral_key,
                precision_digits=precision_digits,
            )
            aggregate = _expanded_aggregate(topology.aggregate_groups)
        elif spec.kind == "slower_expanded":
            if spec.phases is None:
                raise ValueError("Expanded slower variant needs a phase count.")
            topology = _build_slower_expanded_topology(capacity, spec.phases)
            model = _build_expanded_model(
                topology=topology,
                arrival_rate=arrival_rate,
                service_time=service_time,
                phase_count=spec.phases,
                integral_key=integral_key,
                precision_digits=precision_digits,
            )
            aggregate = _expanded_aggregate(topology.aggregate_groups)
        else:
            raise ValueError(f"Unsupported dual bounds variant kind: {spec.kind}")

        build_time = time.perf_counter() - build_started

        solve_started = time.perf_counter()
        solve_result = _solve_model(
            model,
            linear_method=linear_method,
            tolerance=effective_tolerance,
            max_iterations=max_iterations,
            precision_digits=precision_digits,
        )
        probability = solve_result.probability
        solve_time = time.perf_counter() - solve_started

        metrics_started = time.perf_counter()
        dist = aggregate(probability, capacity)
        p_idle, p_full, response_time = _metrics(
            dist, arrival_rate, normalize_probabilities=spec.kind != "slower_expanded",
        )
        metrics_time = time.perf_counter() - metrics_started

        results.append({
            "approach": spec.name,
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
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
            "solver_recovered": solve_result.recovered,
            "stationary_residual_l1": solve_result.stationary_residual_l1,
            "embedded_residual_l1": solve_result.embedded_residual_l1,
            "matrix_memory_bytes": solve_result.matrix_memory_bytes,
            "factor_memory_bytes": solve_result.factor_memory_bytes,
            "memory_bytes": solve_result.memory_bytes,
            "state_count": solve_result.state_count,
            "generator_nnz": solve_result.transition_nnz,
            "nnz_label": "P_nnz",
        })

    return results


def _parse_variant(variant: str) -> _VariantSpec:
    text = str(variant).strip()
    lower = text.lower()

    if lower in {"faster", "faster w/o"}:
        return _VariantSpec("faster_noexp", "faster w/o")
    if lower in {"slower", "slower w/o"}:
        return _VariantSpec("slower", "slower w/o")
    if lower == "slower 1":
        return _VariantSpec("slower_additional", "slower 1", 1)

    faster_match = re.fullmatch(r"faster\s+(\d+)", lower)
    if faster_match:
        phase_count = int(faster_match.group(1))
        return _VariantSpec("faster_expanded", f"faster {phase_count}", phase_count)

    slower_match = re.fullmatch(r"slower\s+(\d+)", lower)
    if slower_match:
        phase_count = int(slower_match.group(1))
        if phase_count == 1:
            return _VariantSpec("slower_additional", "slower 1", 1)
        return _VariantSpec("slower_expanded", f"slower {phase_count}", phase_count)

    raise ValueError(f"Unsupported dual bounds variant: {variant}")


def _build_simple_inputs(
    arrival_rate: float,
    capacity: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    dim = capacity + 2
    groups = ((1,), tuple(range(2, dim + 1)))
    q = np.zeros((dim, dim))
    qbar = np.zeros((dim, dim))
    departure = np.zeros((dim, dim))

    def s0() -> int:
        return 1

    def s1a() -> int:
        return 2

    def s1b() -> int:
        return 3

    def si(n: int) -> int:
        return n + 2

    q[s0() - 1, s0() - 1] = -arrival_rate
    q[s0() - 1, s1a() - 1] = arrival_rate / 2.0
    q[s0() - 1, s1b() - 1] = arrival_rate / 2.0
    q[s1a() - 1, s1a() - 1] = -arrival_rate
    q[s1b() - 1, s1b() - 1] = -arrival_rate

    departure[s1a() - 1, s0() - 1] = 1.0
    departure[s1b() - 1, s0() - 1] = 1.0
    departure[si(2) - 1, s0() - 1] = 1.0
    departure[si(3) - 1, s1a() - 1] = 0.5
    departure[si(3) - 1, s1b() - 1] = 0.5

    if kind == "faster_noexp":
        q[s1a() - 1, si(2) - 1] = arrival_rate
        q[s1b() - 1, si(2) - 1] = arrival_rate
    elif kind == "slower":
        qbar[s1a() - 1, si(2) - 1] = arrival_rate
        qbar[s1b() - 1, si(2) - 1] = arrival_rate
    else:
        raise ValueError(f"Unsupported non-expanded dual bounds kind: {kind}")

    for n in range(4, capacity + 1):
        departure[si(n) - 1, si(n - 2) - 1] = 1.0

    for n in range(2, capacity):
        q[si(n) - 1, si(n) - 1] += -arrival_rate
        q[si(n) - 1, si(n + 1) - 1] = arrival_rate

    return q, qbar, departure, groups


def _build_slower_additional_inputs(
    arrival_rate: float,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    dim = 2 * capacity + 1
    groups = ((1,), tuple(range(2, dim + 1)))
    q = np.zeros((dim, dim))
    qbar = np.zeros((dim, dim))
    departure = np.zeros((dim, dim))

    def s0() -> int:
        return 1

    def s1a() -> int:
        return 2

    def s1b() -> int:
        return 3

    def s(i: int, phase: int) -> int:
        if phase == -1:
            return (i - 2) * 2 + 4
        if phase == 0:
            return (i - 2) * 2 + 5
        raise ValueError(f"Unsupported slower additional phase: {phase}")

    q[s0() - 1, s0() - 1] = -arrival_rate
    q[s0() - 1, s1a() - 1] = arrival_rate / 2.0
    q[s0() - 1, s1b() - 1] = arrival_rate / 2.0

    for single in (s1a(), s1b()):
        q[single - 1, single - 1] = -arrival_rate
        q[single - 1, s(2, -1) - 1] = arrival_rate
        departure[single - 1, s0() - 1] = 1.0

    departure[s(2, -1) - 1, s(2, 0) - 1] = 1.0
    departure[s(2, 0) - 1, s0() - 1] = 1.0

    departure[s(3, -1) - 1, s(3, 0) - 1] = 1.0
    departure[s(3, 0) - 1, s1a() - 1] = 0.5
    departure[s(3, 0) - 1, s1b() - 1] = 0.5

    for i in range(4, capacity + 1):
        departure[s(i, -1) - 1, s(i, 0) - 1] = 1.0
        departure[s(i, 0) - 1, s(i - 2, 0) - 1] = 1.0

    for i in range(2, capacity):
        for phase in (-1, 0):
            state = s(i, phase) - 1
            q[state, state] += -arrival_rate
            q[state, s(i + 1, phase) - 1] = arrival_rate

    return q, qbar, departure, groups


@lru_cache(maxsize=None)
def _build_faster_expanded_topology(capacity: int, phase_count: int) -> _ExpandedTopology:
    dim = capacity * phase_count + 2
    qbar = sp.lil_matrix((dim, dim), dtype=float)
    delta_a = sp.lil_matrix((dim, dim), dtype=float)
    delta_ba = sp.lil_matrix((dim, dim), dtype=float)
    delta_bhat = sp.lil_matrix((dim, dim), dtype=float)

    def s0() -> int:
        return 0

    def s1a() -> int:
        return 1

    def s1b(phase: int) -> int:
        return 2 + phase

    def s(customers: int, phase: int) -> int:
        return (customers - 1) * phase_count + 2 + phase

    for phase in range(phase_count - 1):
        qbar[s1b(phase), s(2, phase + 1)] = 1.0
        delta_bhat[s1b(phase), s1b(phase + 1)] = 1.0
    qbar[s1b(phase_count - 1), s1a()] = 1.0
    delta_bhat[s1b(phase_count - 1), s0()] = 1.0

    delta_a[s1a(), s0()] = 1.0
    delta_ba[s1a(), s1a()] = 1.0

    for customers in range(2, capacity + 1):
        for phase in range(phase_count - 1):
            delta_ba[s(customers, phase), s(customers, phase + 1)] = 1.0
        if customers == 2:
            delta_ba[s(customers, phase_count - 1), s1a()] = 1.0
        else:
            delta_ba[s(customers, phase_count - 1), s(customers - 1, 0)] = 1.0

        for phase in range(phase_count):
            if customers == 2:
                delta_a[s(customers, phase), s1b(phase)] = 1.0
            else:
                delta_a[s(customers, phase), s(customers - 1, phase)] = 1.0

    sa_rows: list[tuple[int, tuple[int, ...]]] = [
        (s1a(), tuple([s1a()] + [s(customers, 0) for customers in range(2, capacity + 1)])),
    ]
    for phase in range(phase_count):
        for customers in range(2, capacity + 1):
            sa_rows.append((
                s(customers, phase),
                tuple(s(level, phase) for level in range(customers, capacity + 1)),
            ))

    sbhat_states = tuple(s1b(phase) for phase in range(phase_count))

    aggregate_groups: list[tuple[int, ...]] = [
        (s0(),),
        tuple([s1a()] + list(sbhat_states)),
    ]
    for customers in range(2, capacity + 1):
        aggregate_groups.append(tuple(s(customers, phase) for phase in range(phase_count)))

    return _ExpandedTopology(
        dim=dim,
        qbar_pattern=qbar.tocsr(),
        delta_a=delta_a.tocsr(),
        delta_ba=delta_ba.tocsr(),
        delta_bhat=delta_bhat.tocsr(),
        sa_rows=tuple(sa_rows),
        sbhat_states=sbhat_states,
        aggregate_groups=tuple(aggregate_groups),
        first_a=s1a(),
        first_b=s1b(0),
    )


@lru_cache(maxsize=None)
def _build_slower_expanded_topology(capacity: int, phase_count: int) -> _ExpandedTopology:
    dim = capacity * (phase_count + 1) + 1
    qbar = sp.lil_matrix((dim, dim), dtype=float)
    delta_a = sp.lil_matrix((dim, dim), dtype=float)
    delta_ba = sp.lil_matrix((dim, dim), dtype=float)
    delta_bhat = sp.lil_matrix((dim, dim), dtype=float)

    def s0() -> int:
        return 0

    def s1a() -> int:
        return 1

    def s1b(phase: int) -> int:
        return 2 + phase

    def s(customers: int, phase: int) -> int:
        return (customers - 1) * (phase_count + 1) + 2 + phase

    for phase in range(phase_count):
        qbar[s1b(phase), s(2, phase)] = 1.0
    for phase in range(phase_count - 1):
        delta_bhat[s1b(phase), s1b(phase + 1)] = 1.0
    delta_bhat[s1b(phase_count - 1), s0()] = 1.0

    delta_a[s1a(), s0()] = 1.0
    delta_ba[s1a(), s1a()] = 1.0

    for customers in range(2, capacity + 1):
        for phase in range(-1, phase_count - 1):
            delta_ba[s(customers, phase), s(customers, phase + 1)] = 1.0
        if customers == 2:
            delta_ba[s(customers, phase_count - 1), s1a()] = 1.0
        else:
            delta_ba[s(customers, phase_count - 1), s(customers - 1, 0)] = 1.0

    delta_a[s(2, -1), s(2, -1)] = 1.0
    for phase in range(phase_count):
        delta_a[s(2, phase), s1b(phase)] = 1.0

    for customers in range(3, capacity + 1):
        delta_a[s(customers, -1), s(customers, -1)] = 1.0
        for phase in range(phase_count):
            delta_a[s(customers, phase), s(customers - 1, phase)] = 1.0

    sa_rows: list[tuple[int, tuple[int, ...]]] = [
        (s1a(), tuple([s1a()] + [s(customers, -1) for customers in range(2, capacity + 1)])),
    ]
    for phase in range(-1, phase_count):
        for customers in range(2, capacity + 1):
            sa_rows.append((
                s(customers, phase),
                tuple(s(level, phase) for level in range(customers, capacity + 1)),
            ))

    sbhat_states = tuple(s1b(phase) for phase in range(phase_count))

    aggregate_groups: list[tuple[int, ...]] = [
        (s0(),),
        tuple([s1a()] + list(sbhat_states)),
    ]
    for customers in range(2, capacity + 1):
        aggregate_groups.append(tuple(s(customers, phase) for phase in range(0, phase_count)))

    return _ExpandedTopology(
        dim=dim,
        qbar_pattern=qbar.tocsr(),
        delta_a=delta_a.tocsr(),
        delta_ba=delta_ba.tocsr(),
        delta_bhat=delta_bhat.tocsr(),
        sa_rows=tuple(sa_rows),
        sbhat_states=sbhat_states,
        aggregate_groups=tuple(aggregate_groups),
        first_a=s1a(),
        first_b=s1b(0),
    )


def _expanded_aggregate(
    aggregate_groups: tuple[tuple[int, ...], ...],
) -> Callable[[np.ndarray, int], np.ndarray]:
    def aggregate(probability: np.ndarray, _capacity: int) -> np.ndarray:
        dist = np.zeros(len(aggregate_groups), dtype=float)
        for customer_count, states in enumerate(aggregate_groups):
            dist[customer_count] = float(probability[list(states)].sum())
        return dist

    return aggregate


def _aggregate_simple(prob: np.ndarray, capacity: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = float(prob[0])
    agg[1] = float(prob[1] + prob[2])
    for n in range(2, capacity + 1):
        agg[n] = float(prob[n + 1])
    return agg


def _aggregate_slower_additional(prob: np.ndarray, capacity: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = float(prob[0])
    agg[1] = float(prob[1] + prob[2])
    for customers in range(2, capacity + 1):
        base = 2 * (customers - 2) + 3
        agg[customers] = float(prob[base] + prob[base + 1])
    return agg


def _metrics(
    dist: np.ndarray,
    arrival_rate: float,
    *,
    normalize_probabilities: bool = True,
) -> tuple[float, float, float]:
    total = float(dist.sum())
    if total <= 0.0:
        raise RuntimeError("Distribution is invalid.")

    capacity = len(dist) - 1
    if normalize_probabilities:
        p = dist / total
        p_idle = float(p[0])
        p_full = float(p[capacity])
        mean_n = float(np.dot(np.arange(capacity + 1), p))
        throughput = arrival_rate * (1.0 - p_full)
    else:
        p_idle = float(dist[0])
        p_full = float(dist[capacity])
        mean_n = float(np.dot(np.arange(capacity + 1), dist))
        throughput = arrival_rate * float(dist[:-1].sum())

    response_time = mean_n / throughput if throughput > 0 else float("inf")
    return p_idle, p_full, response_time
