from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


_NZ_TOL = 1e-15
_LEVEL_NEGATIVE_TOL = 1e-9


def build_generator(
    arrival_rate: float,
    capacity: int,
    initial: np.ndarray,
    phase_gen: np.ndarray,
) -> sp.csr_matrix:
    pc = len(initial)
    completion = -phase_gen @ np.ones(pc)
    alpha_nz = tuple((j, float(v)) for j, v in enumerate(initial) if abs(v) > _NZ_TOL)
    phase_nz = tuple(
        tuple((j, float(v)) for j, v in enumerate(phase_gen[i]) if j != i and abs(v) > _NZ_TOL)
        for i in range(pc)
    )

    dim = 1 + pc + max(0, capacity - 1) * pc * pc
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(r: int, c: int, v: float) -> None:
        if abs(v) > _NZ_TOL:
            rows.append(r)
            cols.append(c)
            data.append(float(v))

    def s0() -> int:
        return 0

    def s1(i: int) -> int:
        return 1 + i

    def sboth(n: int, a: int, b: int) -> int:
        return 1 + pc + (n - 2) * pc * pc + a * pc + b

    if capacity <= 0:
        add(s0(), s0(), 0.0)
        return sp.coo_matrix((data, (rows, cols)), shape=(1, 1)).tocsr()

    add(s0(), s0(), -arrival_rate)
    for j, alpha_j in alpha_nz:
        add(s0(), s1(j), arrival_rate * alpha_j)

    for i in range(pc):
        state = s1(i)

        if capacity >= 2:
            add(state, state, -arrival_rate)
            for j, alpha_j in alpha_nz:
                add(state, sboth(2, i, j), arrival_rate * alpha_j)

        for j, qij in phase_nz[i]:
            add(state, s1(j), qij)

        add(state, state, float(phase_gen[i, i]))
        add(state, s0(), float(completion[i]))

    for n in range(2, capacity + 1):
        for a in range(pc):
            for b in range(pc):
                state = sboth(n, a, b)

                if n < capacity:
                    add(state, state, -arrival_rate)
                    add(state, sboth(n + 1, a, b), arrival_rate)

                for j, qaj in phase_nz[a]:
                    add(state, sboth(n, j, b), qaj)
                for j, qbj in phase_nz[b]:
                    add(state, sboth(n, a, j), qbj)

                add(state, state, float(phase_gen[a, a] + phase_gen[b, b]))

                if n == 2:
                    add(state, s1(b), float(completion[a]))
                    add(state, s1(a), float(completion[b]))
                else:
                    for j, alpha_j in alpha_nz:
                        add(state, sboth(n - 1, j, b), float(completion[a] * alpha_j))
                        add(state, sboth(n - 1, a, j), float(completion[b] * alpha_j))

    return sp.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()


def solve_stationary(generator: sp.spmatrix) -> np.ndarray:
    generator = generator.tocsr()
    n = generator.shape[0]
    system = generator.transpose().tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0

    prob = spla.spsolve(system.tocsr(), rhs)
    prob = np.asarray(np.real_if_close(prob), dtype=float)
    total = float(prob.sum())
    if abs(total) <= _NZ_TOL:
        raise RuntimeError("Linear solve produced an invalid stationary vector.")
    return prob / total


def aggregate(prob: np.ndarray, capacity: int, phase_count: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = prob[0]
    if capacity >= 1:
        agg[1] = prob[1: 1 + phase_count].sum()
    base = 1 + phase_count
    block = phase_count * phase_count
    for n in range(2, capacity + 1):
        start = base + (n - 2) * block
        agg[n] = prob[start: start + block].sum()

    if float(agg.min()) < -_LEVEL_NEGATIVE_TOL:
        raise RuntimeError("ME queue solve produced materially negative level probabilities.")
    agg[agg < 0.0] = 0.0
    return agg


def metrics(dist: np.ndarray, arrival_rate: float) -> tuple[float, float, float]:
    total = float(dist.sum())
    if total <= 0.0:
        raise RuntimeError("Queue-length probabilities are invalid.")
    p = dist / total
    capacity = len(p) - 1
    p_idle = float(p[0])
    p_full = float(p[capacity])
    mean_n = float(np.dot(np.arange(capacity + 1), p))
    throughput = arrival_rate * (1.0 - p_full)
    response_time = mean_n / throughput if throughput > 0 else float("inf")
    return p_idle, p_full, response_time


def mean_and_scv(initial: np.ndarray, phase_gen: np.ndarray) -> tuple[float, float]:
    minus_q = -phase_gen
    ones = np.ones(len(initial))
    first_moment = np.linalg.solve(minus_q, ones)
    second_moment_vector = np.linalg.solve(minus_q, first_moment)

    mean = float(initial @ first_moment)
    second = float(2.0 * (initial @ second_moment_vector))
    variance = max(0.0, second - mean ** 2)
    scv = variance / mean ** 2 if mean > 0 else float("inf")
    return mean, scv
