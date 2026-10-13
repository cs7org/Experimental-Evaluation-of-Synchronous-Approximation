from __future__ import annotations

import mpmath as mp
import numpy as np

from methods.cme import _cme_representation_mp
from methods.fast_slow import (
    _solve_embedded,
    _solve_embedded_fast,
    _solve_embedded_fast_iterative,
)


def solve_erlang(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    phase_count: int,
    *,
    precision_digits: int = 80,
    solver: str = "fast",
) -> list[dict[str, float | str]]:
    alpha = np.zeros(phase_count, dtype=float)
    alpha[0] = 1.0
    matrix_t = _build_erlang_subgenerator(phase_count, service_time)
    return _solve(
        arrival_rate=arrival_rate,
        service_time=service_time,
        capacity=capacity,
        alpha=alpha,
        matrix_t=matrix_t,
        label=f"Erlang {phase_count}/D",
        precision_digits=precision_digits,
        solver=solver,
    )


def solve_cme(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    order: int,
    *,
    precision_digits: int = 80,
    solver: str = "signed_fast",
) -> list[dict[str, float | str]]:
    solver_mode = solver.strip().lower()
    if solver_mode in {"signed", "signed_direct"}:
        with mp.workdps(precision_digits):
            alpha, matrix_t = _cme_representation_mp(order, service_time)
    else:
        from methods.cme import _cme_representation

        alpha, matrix_t = _cme_representation(order, service_time)
    return _solve(
        arrival_rate=arrival_rate,
        service_time=service_time,
        capacity=capacity,
        alpha=np.asarray(alpha, dtype=object if solver_mode in {"signed", "signed_direct"} else float),
        matrix_t=np.asarray(matrix_t, dtype=object if solver_mode in {"signed", "signed_direct"} else float),
        label=f"CME {order}/D",
        precision_digits=precision_digits,
        solver=solver,
    )


def _solve(
    *,
    arrival_rate: float,
    service_time: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
    label: str,
    precision_digits: int,
    solver: str,
) -> list[dict[str, float | str]]:
    if capacity < 2:
        raise ValueError("Capacity must be at least 2.")

    use_mp_inputs = matrix_t.dtype == object or alpha.dtype == object
    alpha = np.asarray(alpha, dtype=object if use_mp_inputs else float).ravel()
    matrix_t = np.asarray(matrix_t, dtype=object if use_mp_inputs else float)
    if matrix_t.shape[0] != matrix_t.shape[1]:
        raise ValueError("The phase generator must be square.")
    if matrix_t.shape[0] != alpha.shape[0]:
        raise ValueError("Alpha length must match the phase generator size.")

    phase_count = int(alpha.shape[0])
    dim = 2 + phase_count + (capacity - 1) * phase_count

    q_dtype = object if use_mp_inputs else float
    q = np.zeros((dim, dim), dtype=q_dtype)
    qbar = np.zeros((dim, dim), dtype=q_dtype)
    delta = np.zeros((dim, dim), dtype=q_dtype)
    lam = mp.mpf(str(arrival_rate)) if use_mp_inputs else arrival_rate
    half = mp.mpf("0.5") if use_mp_inputs else 0.5

    def b_state(phase: int) -> int:
        return 2 + phase

    def gen_state(customers: int, phase: int) -> int:
        return 2 + phase_count + (customers - 2) * phase_count + phase

    # Empty system.
    q[0, 0] = -lam
    q[0, 1] = lam * half
    for phase in range(phase_count):
        q[0, b_state(phase)] = lam * half * alpha[phase]

    # One customer, exact server A busy.
    q[1, 1] = -lam
    for phase in range(phase_count):
        q[1, gen_state(2, phase)] = lam * alpha[phase]
    delta[1, 0] = 1.0

    # One customer, expanded server B busy.
    for phase_from in range(phase_count):
        row = b_state(phase_from)
        q[row, row] += -lam
        q[row, gen_state(2, phase_from)] += lam
        for phase_to, rate in enumerate(matrix_t[phase_from]):
            if rate != 0.0:
                q[row, b_state(phase_to)] += rate
        exit_rate = -sum(matrix_t[phase_from, phase_to] for phase_to in range(phase_count))
        if exit_rate != 0.0:
            q[row, 0] += exit_rate

    # Two or more customers in the system.
    for customers in range(2, capacity + 1):
        for phase_from in range(phase_count):
            row = gen_state(customers, phase_from)
            if customers < capacity:
                q[row, row] += -lam
                q[row, gen_state(customers + 1, phase_from)] += lam

            for phase_to, rate in enumerate(matrix_t[phase_from]):
                if rate != 0.0:
                    q[row, gen_state(customers, phase_to)] += rate

            exit_rate = -sum(matrix_t[phase_from, phase_to] for phase_to in range(phase_count))
            if exit_rate != 0.0:
                if customers == 2:
                    q[row, 1] += exit_rate
                else:
                    for phase_to, weight in enumerate(alpha):
                        if weight != 0.0:
                            q[row, gen_state(customers - 1, phase_to)] += exit_rate * weight

            if customers == 2:
                delta[row, b_state(phase_from)] = 1.0
            else:
                delta[row, gen_state(customers - 1, phase_from)] = 1.0

    exp_states = [0] + [b_state(phase) for phase in range(phase_count)]
    deterministic_states = [1] + [
        gen_state(customers, phase)
        for customers in range(2, capacity + 1)
        for phase in range(phase_count)
    ]
    # The embedded solver follows the Mathematica notebook convention and
    # expects one-based state numbers in SlistE/Slistg.
    groups = (
        tuple(state + 1 for state in exp_states),
        tuple(state + 1 for state in deterministic_states),
    )

    solver_mode = solver.strip().lower()
    if solver_mode == "fast":
        probability = _solve_embedded_fast(q, qbar, delta, groups, service_time)
    elif solver_mode == "signed_fast":
        probability = _solve_embedded_fast(
            q,
            qbar,
            delta,
            groups,
            service_time,
            allow_signed=True,
        )
    elif solver_mode in {"fast_iterative", "positive"}:
        probability = _solve_embedded_fast_iterative(q, qbar, delta, groups, service_time)
    else:
        embedded_solver = "direct"
        allow_signed = False
        if solver_mode in {"signed", "signed_direct"}:
            allow_signed = True
        else:
            embedded_solver = solver
        probability = _solve_embedded(
            q,
            qbar,
            delta,
            groups,
            service_time,
            precision_digits,
            solver=embedded_solver,
            allow_signed=allow_signed,
        )

    dist_mp = [mp.mpf("0") for _ in range(capacity + 1)]
    dist_mp[0] = probability[0]
    dist_mp[1] = probability[1] + mp.fsum(probability[2:2 + phase_count])
    base = 2 + phase_count
    for customers in range(2, capacity + 1):
        start = base + (customers - 2) * phase_count
        dist_mp[customers] = mp.fsum(probability[start:start + phase_count])

    if use_mp_inputs and any(value < 0 for value in dist_mp):
        raise RuntimeError("High-precision CME/D solve produced a negative queue probability.")

    dist = np.asarray([float(value) for value in dist_mp], dtype=float)
    distribution = dist / dist.sum()

    p_idle, p_full, response_time = _metrics(dist, arrival_rate)
    return [{
        "approach": label,
        "p_idle": p_idle,
        "p_full": p_full,
        "response_time": response_time,
        "distribution": distribution,
    }]


def _build_erlang_subgenerator(phase_count: int, service_time: float) -> np.ndarray:
    service_rate = phase_count / service_time
    matrix_t = np.zeros((phase_count, phase_count), dtype=float)
    for phase in range(phase_count):
        matrix_t[phase, phase] = -service_rate
        if phase < phase_count - 1:
            matrix_t[phase, phase + 1] = service_rate
    return matrix_t


def _metrics(dist: np.ndarray, arrival_rate: float) -> tuple[float, float, float]:
    p = dist / dist.sum()
    capacity = len(p) - 1
    p_idle = float(p[0])
    p_full = float(p[capacity])
    mean_n = float(np.dot(np.arange(capacity + 1), p))
    throughput = arrival_rate * (1.0 - p_full)
    response_time = mean_n / throughput if throughput > 0.0 else float("inf")
    return p_idle, p_full, response_time
