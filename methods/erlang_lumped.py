from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from methods import pure_erlang


def state_count(capacity: int, phase_count: int) -> int:
    """Return the exact state count of the server-swap lumped Erlang chain."""
    if capacity < 2:
        raise ValueError("Capacity must be at least 2.")
    if phase_count < 1:
        raise ValueError("Phase count must be positive.")
    pair_count = phase_count * (phase_count + 1) // 2
    return 1 + phase_count + (capacity - 1) * pair_count


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_count: int,
) -> list[dict[str, float | str]]:
    generator = _build_generator(arrival_rate, service_time, capacity, phase_count)
    probability = pure_erlang._solve_stationary(generator)
    distribution = _aggregate(probability, capacity, phase_count)
    p_idle, p_full, response_time = _metrics(distribution, arrival_rate)
    return [{
        "approach": f"Erlang lumped {phase_count}",
        "p_idle": p_idle,
        "p_full": p_full,
        "response_time": response_time,
        "distribution": distribution,
        "linear_states": state_count(capacity, phase_count),
    }]


def _build_generator(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_count: int,
) -> sp.csr_matrix:
    """Build the quotient generator under exchange of the two servers.

    The ordinary model has separate server-A/server-B states and ordered phase
    pairs. Here one-job states are identified by phase, and each busy level
    uses unordered pairs (a, b), so (a, b) and (b, a) are one state.
    """
    if capacity < 2:
        raise ValueError("Capacity must be at least 2.")
    if phase_count < 1:
        raise ValueError("Phase count must be positive.")

    service_rate = phase_count / service_time
    pair_count = phase_count * (phase_count + 1) // 2
    dimension = state_count(capacity, phase_count)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(row: int, column: int, value: float) -> None:
        rows.append(row)
        cols.append(column)
        data.append(value)

    def s0() -> int:
        return 0

    def s1(phase: int) -> int:
        return 1 + phase

    def pair_rank(a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        # Row a of the upper triangular phase-pair table starts after
        # phase_count + (phase_count - 1) + ... + (phase_count - a + 1).
        return a * phase_count - a * (a - 1) // 2 + (b - a)

    def sboth(customers: int, a: int, b: int) -> int:
        return 1 + phase_count + (customers - 2) * pair_count + pair_rank(a, b)

    # Empty system. The two original arrival transitions now enter the same
    # one-job lump and therefore have total rate arrival_rate.
    add(s0(), s0(), -arrival_rate)
    add(s0(), s1(0), arrival_rate)

    # One customer: the two server-specific states are exchange-equivalent.
    for phase in range(phase_count):
        state = s1(phase)
        add(state, state, -arrival_rate)
        add(state, sboth(2, phase, 0), arrival_rate)
        add(state, state, -service_rate)
        add(state, s1(phase + 1) if phase < phase_count - 1 else s0(), service_rate)

    # Two or more customers: only a <= b is needed because phase pairs are
    # unordered in the quotient chain.
    for customers in range(2, capacity + 1):
        for a in range(phase_count):
            for b in range(a, phase_count):
                state = sboth(customers, a, b)

                if customers < capacity:
                    add(state, state, -arrival_rate)
                    add(state, sboth(customers + 1, a, b), arrival_rate)

                add(state, state, -2.0 * service_rate)
                add(
                    state,
                    sboth(customers, a + 1, b)
                    if a < phase_count - 1
                    else (s1(b) if customers == 2 else sboth(customers - 1, 0, b)),
                    service_rate,
                )
                add(
                    state,
                    sboth(customers, a, b + 1)
                    if b < phase_count - 1
                    else (s1(a) if customers == 2 else sboth(customers - 1, a, 0)),
                    service_rate,
                )

    generator = sp.coo_matrix(
        (data, (rows, cols)), shape=(dimension, dimension),
    ).tocsr()
    row_residual = np.asarray(generator.sum(axis=1)).ravel()
    if not np.allclose(row_residual, 0.0, atol=1e-12):
        raise RuntimeError("Erlang lumped generator rows do not sum to zero.")
    return generator


def _aggregate(probability: np.ndarray, capacity: int, phase_count: int) -> np.ndarray:
    distribution = np.zeros(capacity + 1, dtype=float)
    distribution[0] = probability[0]
    distribution[1] = probability[1:1 + phase_count].sum()
    pair_count = phase_count * (phase_count + 1) // 2
    base = 1 + phase_count
    for customers in range(2, capacity + 1):
        start = base + (customers - 2) * pair_count
        distribution[customers] = probability[start:start + pair_count].sum()
    return distribution


def _metrics(distribution: np.ndarray, arrival_rate: float) -> tuple[float, float, float]:
    probability = distribution / distribution.sum()
    capacity = len(probability) - 1
    p_idle = float(probability[0])
    p_full = float(probability[capacity])
    mean_jobs = float(np.dot(np.arange(capacity + 1), probability))
    throughput = arrival_rate * (1.0 - p_full)
    response_time = mean_jobs / throughput if throughput > 0 else float("inf")
    return p_idle, p_full, response_time
