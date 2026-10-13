from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_counts: list[int],
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for phase_count in phase_counts:
        generator = _build_generator(arrival_rate, service_time, capacity, phase_count)
        probability = _solve_stationary(generator)
        dist = _aggregate(probability, capacity, phase_count)
        p_idle, p_full, response_time = _metrics(dist, arrival_rate)

        results.append({
            "approach": f"Erlang {phase_count}",
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
        })
    return results


def _build_generator(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_count: int,
) -> sp.csr_matrix:
    service_rate = phase_count / service_time
    dimension = 1 + 2 * phase_count + (capacity - 1) * phase_count * phase_count
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        data.append(v)

    def s0() -> int:
        return 0

    def s1a(i: int) -> int:
        return 1 + i

    def s1b(i: int) -> int:
        return 1 + phase_count + i

    def sboth(customers: int, a: int, b: int) -> int:
        return 1 + 2 * phase_count + (customers - 2) * phase_count * phase_count + a * phase_count + b

    add(s0(), s1a(0), arrival_rate / 2.0)
    add(s0(), s1b(0), arrival_rate / 2.0)
    add(s0(), s0(), -arrival_rate)

    for i in range(phase_count):
        add(s1a(i), s1a(i), -arrival_rate)
        add(s1a(i), sboth(2, i, 0), arrival_rate)
        add(s1b(i), s1b(i), -arrival_rate)
        add(s1b(i), sboth(2, 0, i), arrival_rate)

        add(s1a(i), s1a(i), -service_rate)
        add(s1a(i), s1a(i + 1) if i < phase_count - 1 else s0(), service_rate)

        add(s1b(i), s1b(i), -service_rate)
        add(s1b(i), s1b(i + 1) if i < phase_count - 1 else s0(), service_rate)

    for customers in range(2, capacity + 1):
        for a in range(phase_count):
            for b in range(phase_count):
                state = sboth(customers, a, b)

                if customers < capacity:
                    add(state, state, -arrival_rate)
                    add(state, sboth(customers + 1, a, b), arrival_rate)

                add(state, state, -2.0 * service_rate)

                add(
                    state,
                    sboth(customers, a + 1, b) if a < phase_count - 1
                    else (s1b(b) if customers == 2 else sboth(customers - 1, 0, b)),
                    service_rate,
                )
                add(
                    state,
                    sboth(customers, a, b + 1) if b < phase_count - 1
                    else (s1a(a) if customers == 2 else sboth(customers - 1, a, 0)),
                    service_rate,
                )

    return sp.coo_matrix((data, (rows, cols)), shape=(dimension, dimension)).tocsr()


def _solve_stationary(generator: sp.spmatrix) -> np.ndarray:
    generator = generator.tocsr()
    n = generator.shape[0]
    system = generator.transpose().tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0

    prob = spla.spsolve(system.tocsr(), rhs)
    prob = np.real(np.asarray(prob, dtype=float))
    prob[prob < 0] = 0.0
    total = prob.sum()
    if total <= 0:
        raise RuntimeError("Linear solve produced an invalid stationary distribution.")
    return prob / total


def _aggregate(prob: np.ndarray, capacity: int, phase_count: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = prob[0]
    agg[1] = prob[1: 1 + 2 * phase_count].sum()
    base = 1 + 2 * phase_count
    block = phase_count * phase_count
    for customers in range(2, capacity + 1):
        start = base + (customers - 2) * block
        agg[customers] = prob[start: start + block].sum()
    return agg


def _metrics(dist: np.ndarray, arrival_rate: float) -> tuple[float, float, float]:
    p = dist / dist.sum()
    capacity = len(p) - 1
    p_idle = float(p[0])
    p_full = float(p[capacity])
    mean_n = float(np.dot(np.arange(capacity + 1), p))
    throughput = arrival_rate * (1.0 - p_full)
    response_time = mean_n / throughput if throughput > 0 else float("inf")
    return p_idle, p_full, response_time
