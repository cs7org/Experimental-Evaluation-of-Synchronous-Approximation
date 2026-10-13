from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_counts: list[int],
    stationary_method: str = "power",
    tolerance: float = 1e-12,
    max_iterations: int = 50_000,
    relaxation: float | None = None,
) -> list[dict[str, float | str]]:
    results: list[dict[str, float | str]] = []
    for phase_count in phase_counts:
        generator = _build_generator(arrival_rate, service_time, capacity, phase_count)
        probability = _solve_stationary(
            generator, stationary_method, tolerance, max_iterations, relaxation,
        )
        dist = _aggregate(probability, capacity, phase_count)
        p_idle, p_full, response_time = _metrics(dist, arrival_rate)

        results.append({
            "approach": _approach_name(phase_count),
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
        })
    return results


def _approach_name(phase_count: int) -> str:
    if phase_count == 1:
        return "Exponential"
    return f"Erlang Iterative {phase_count}"


# ---------------------------------------------------------------------------
# Generator matrix
# ---------------------------------------------------------------------------

def _build_generator(
    arrival_rate: float, service_time: float, capacity: int, phase_count: int,
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

    def sboth(n: int, a: int, b: int) -> int:
        return 1 + 2 * phase_count + (n - 2) * phase_count * phase_count + a * phase_count + b

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

    for n in range(2, capacity + 1):
        for a in range(phase_count):
            for b in range(phase_count):
                state = sboth(n, a, b)

                if n < capacity:
                    add(state, state, -arrival_rate)
                    add(state, sboth(n + 1, a, b), arrival_rate)

                add(state, state, -2.0 * service_rate)

                add(
                    state,
                    sboth(n, a + 1, b) if a < phase_count - 1
                    else (s1b(b) if n == 2 else sboth(n - 1, 0, b)),
                    service_rate,
                )
                add(
                    state,
                    sboth(n, a, b + 1) if b < phase_count - 1
                    else (s1a(a) if n == 2 else sboth(n - 1, a, 0)),
                    service_rate,
                )

    return sp.coo_matrix((data, (rows, cols)), shape=(dimension, dimension)).tocsr()


# ---------------------------------------------------------------------------
# Stationary distribution solvers
# ---------------------------------------------------------------------------

def _solve_stationary(
    generator: sp.spmatrix,
    method: str,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
) -> np.ndarray:
    m = method.lower()
    if m == "power":
        return _power(generator, tolerance, max_iterations)
    if m == "jacobi":
        if relaxation is None:
            raise ValueError("Jacobi requires a relaxation value.")
        return _jacobi(generator, tolerance, max_iterations, relaxation)
    if m == "gauss_seidel":
        if relaxation is None:
            raise ValueError("Gauss-Seidel requires a relaxation value.")
        return _gauss_seidel(generator, tolerance, max_iterations, relaxation)
    raise ValueError(f"Unsupported stationary method: {method}")


def _power(gen: sp.spmatrix, tol: float, max_iter: int) -> np.ndarray:
    gen = gen.tocsr()
    n = gen.shape[0]
    rate = float((-gen.diagonal()).max())
    P = sp.eye(n, format="csr") + gen / rate

    pi = np.zeros(n)
    pi[0] = 1.0
    for _ in range(max_iter):
        new = np.asarray(pi @ P).ravel()
        if np.linalg.norm(new - pi, ord=1) < tol:
            pi = new
            break
        pi = new
    return _normalise(pi)


def _jacobi(gen: sp.spmatrix, tol: float, max_iter: int, w: float) -> np.ndarray:
    gen = gen.tocsr()
    n = gen.shape[0]
    rate = float((-gen.diagonal()).max())
    P = sp.eye(n, format="csr") + gen / rate

    pi = np.zeros(n)
    pi[0] = 1.0
    for _ in range(max_iter):
        new = (1.0 - w) * pi + w * (pi @ P)
        new = np.asarray(new).ravel()
        new = _normalise(new)
        if np.linalg.norm(new - pi, ord=1) < tol:
            return new
        pi = new
    return pi


def _gauss_seidel(gen: sp.spmatrix, tol: float, max_iter: int, w: float) -> np.ndarray:
    gen = gen.tocsr()
    n = gen.shape[0]
    rate = float((-gen.diagonal()).max())
    P = (sp.eye(n, format="csr") + gen / rate).tocsc()

    pi = np.zeros(n)
    pi[0] = 1.0
    for _ in range(max_iter):
        prev = pi.copy()
        for i in range(n):
            start, end = P.indptr[i], P.indptr[i + 1]
            r = P.indices[start:end]
            v = P.data[start:end]
            subtotal = 0.0
            diag = 0.0
            for ri, vi in zip(r, v):
                if ri == i:
                    diag = vi
                elif ri < i:
                    subtotal += vi * pi[ri]
                else:
                    subtotal += vi * prev[ri]
            denom = 1.0 - diag
            cand = prev[i] if abs(denom) < 1e-14 else subtotal / denom
            pi[i] = (1.0 - w) * prev[i] + w * cand

        pi = _normalise(pi)
        if np.linalg.norm(pi - prev, ord=1) < tol:
            return pi
    return pi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _aggregate(prob: np.ndarray, capacity: int, phase_count: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = prob[0]
    agg[1] = prob[1: 1 + 2 * phase_count].sum()
    base = 1 + 2 * phase_count
    block = phase_count * phase_count
    for n in range(2, capacity + 1):
        start = base + (n - 2) * block
        agg[n] = prob[start: start + block].sum()
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


def _normalise(pi: np.ndarray) -> np.ndarray:
    pi = np.real(np.asarray(pi, dtype=float))
    pi[pi < 0] = 0.0
    total = pi.sum()
    if total <= 0:
        raise RuntimeError("Stationary distribution is invalid.")
    return pi / total
