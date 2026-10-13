from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from methods.cme import _cme_representation
from methods.fast_slow import _solve_embedded


STATE_TUPLES = (
    (0, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (0, 1, 0),
    (1, 1, 0),
    (0, 2, 0),
    (0, 0, 1),
    (1, 0, 1),
    (0, 1, 1),
    (0, 0, 2),
)
STATE_INDEX = {state: index for index, state in enumerate(STATE_TUPLES)}
STATE_LABELS = tuple(f"{state[0]}{state[1]}{state[2]}" for state in STATE_TUPLES)


def solve_markovian(
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
) -> dict[str, float | int | str]:
    """Solve the exponential-timing CTMC from the Mathematica notebook."""
    alpha, matrix_t = _exponential_representation(service_time)
    generator, layout = _build_timer_expanded_generator(
        arrival_rate=arrival_rate,
        forward_rate=forward_rate,
        capacity=2,
        alpha=alpha,
        matrix_t=matrix_t,
    )
    probability = _solve_stationary(generator)
    states = _aggregate(probability, layout)
    return _result_row(
        "exponential timing",
        "Markovian",
        states,
        arrival_rate=arrival_rate,
        forward_rate=forward_rate,
        service_time=service_time,
    )


def solve_synchronized(
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
    variants: Iterable[str] = ("faster", "slower"),
    *,
    precision_digits: int = 80,
) -> list[dict[str, float | int | str]]:
    """Solve the non-expanded synchronized faster/slower approximations."""
    _validate_rates(arrival_rate, forward_rate, service_time)
    results: list[dict[str, float | int | str]] = []

    for variant in variants:
        name = _normalise_variant(variant)
        generator, deferred, completion = _synchronized_matrices(
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            variant=name,
        )
        probability = _solve_embedded(
            generator=generator,
            deferred=deferred,
            departure=completion,
            groups=((1,), tuple(range(2, 11))),
            tau=service_time,
            precision_digits=precision_digits,
        )
        states = np.array([float(value) for value in probability], dtype=float)
        results.append(
            _result_row(
                f"synchronized {name}",
                "Synchronized",
                states,
                arrival_rate=arrival_rate,
                forward_rate=forward_rate,
                service_time=service_time,
            ),
        )

    return results


def solve_cme_ground_truth(
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
    orders: Iterable[int] = (5, 10, 15),
) -> list[dict[str, float | int | str]]:
    """Approximate both deterministic stations with CME timers."""
    _validate_rates(arrival_rate, forward_rate, service_time)
    results: list[dict[str, float | int | str]] = []

    for order in orders:
        cme_order = int(order)
        alpha, matrix_t = _cme_representation(cme_order, service_time)
        generator, layout = _build_timer_expanded_generator(
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            capacity=2,
            alpha=alpha,
            matrix_t=matrix_t,
        )
        probability = _solve_stationary(generator)
        states = _aggregate(probability, layout)
        row = _result_row(
            f"CME {cme_order} ground truth",
            "CME ground truth",
            states,
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            service_time=service_time,
        )
        row["order"] = cme_order
        row["timer_phases"] = len(alpha)
        row["linear_states"] = generator.shape[0]
        results.append(row)

    return results


def solve_erlang_ground_truth(
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
    phase_counts: Iterable[int] = (1, 5, 10, 20),
) -> list[dict[str, float | int | str]]:
    """Positive phase-type approximation, useful as a CME sanity check."""
    _validate_rates(arrival_rate, forward_rate, service_time)
    results: list[dict[str, float | int | str]] = []

    for phase_count in phase_counts:
        phases = int(phase_count)
        if phases < 1:
            raise ValueError("Erlang phase counts must be positive.")
        alpha, matrix_t = _erlang_representation(phases, service_time)
        generator, layout = _build_timer_expanded_generator(
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            capacity=2,
            alpha=alpha,
            matrix_t=matrix_t,
        )
        probability = _solve_stationary(generator)
        states = _aggregate(probability, layout)
        row = _result_row(
            f"Erlang {phases} ground truth",
            "Erlang ground truth",
            states,
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            service_time=service_time,
        )
        row["order"] = phases
        row["timer_phases"] = phases
        row["linear_states"] = generator.shape[0]
        results.append(row)

    return results


def solve_all(
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
    *,
    synchronized_variants: Iterable[str] = ("faster", "slower"),
    cme_orders: Iterable[int] = (5, 10, 15),
) -> list[dict[str, float | int | str]]:
    rows = [solve_markovian(arrival_rate, forward_rate, service_time)]
    rows.extend(
        solve_synchronized(
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            service_time=service_time,
            variants=synchronized_variants,
        ),
    )
    rows.extend(
        solve_cme_ground_truth(
            arrival_rate=arrival_rate,
            forward_rate=forward_rate,
            service_time=service_time,
            orders=cme_orders,
        ),
    )
    return rows


def _synchronized_matrices(
    *,
    arrival_rate: float,
    forward_rate: float,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = float(arrival_rate)
    mu = float(forward_rate)
    generator = np.zeros((10, 10), dtype=float)
    deferred = np.zeros((10, 10), dtype=float)
    completion = np.zeros((10, 10), dtype=float)

    def s(state: tuple[int, int, int]) -> int:
        return STATE_INDEX[state]

    generator[s((0, 0, 0)), s((0, 0, 0))] = -lam
    generator[s((0, 0, 0)), s((1, 0, 0))] = lam

    generator[s((1, 0, 0)), s((1, 0, 0))] = -lam
    generator[s((1, 0, 0)), s((2, 0, 0))] = lam

    generator[s((0, 1, 0)), s((0, 1, 0))] = -(lam + mu)
    generator[s((0, 1, 0)), s((1, 1, 0))] = lam
    generator[s((0, 1, 0)), s((0, 0, 1))] = mu

    generator[s((1, 1, 0)), s((1, 1, 0))] = -mu
    generator[s((0, 2, 0)), s((0, 2, 0))] = -2.0 * mu
    generator[s((0, 2, 0)), s((0, 1, 1))] = 2.0 * mu

    generator[s((0, 0, 1)), s((0, 0, 1))] = -lam
    generator[s((0, 1, 1)), s((0, 1, 1))] = -mu
    generator[s((0, 1, 1)), s((0, 0, 2))] = mu

    if variant == "faster":
        generator[s((1, 1, 0)), s((1, 0, 1))] = mu
        generator[s((0, 0, 1)), s((1, 0, 1))] = lam
    elif variant == "slower":
        deferred[s((1, 1, 0)), s((1, 0, 1))] = mu
        deferred[s((0, 0, 1)), s((1, 0, 1))] = lam
    else:
        raise ValueError(f"Unsupported synchronized message-passing variant: {variant}")

    completion[s((1, 0, 0)), s((0, 1, 0))] = 1.0
    completion[s((2, 0, 0)), s((1, 1, 0))] = 1.0
    completion[s((1, 1, 0)), s((0, 2, 0))] = 1.0
    completion[s((0, 0, 1)), s((0, 0, 0))] = 1.0
    completion[s((1, 0, 1)), s((0, 1, 0))] = 1.0
    completion[s((0, 1, 1)), s((0, 1, 0))] = 1.0
    completion[s((0, 0, 2)), s((0, 0, 1))] = 1.0

    return generator, deferred, completion


def _build_timer_expanded_generator(
    *,
    arrival_rate: float,
    forward_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> tuple[sp.csr_matrix, dict[tuple[int, int | None, int | None], int]]:
    _validate_rates(arrival_rate, forward_rate, 1.0)
    lam = float(arrival_rate)
    mu = float(forward_rate)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    matrix_t = np.asarray(matrix_t, dtype=float)
    if matrix_t.shape != (len(alpha), len(alpha)):
        raise ValueError("matrix_t shape must match alpha length.")

    states = _expanded_states(len(alpha), capacity)
    layout = {state: index for index, state in enumerate(states)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    ones = np.ones(len(alpha), dtype=float)
    exit_rates = -matrix_t @ ones

    def add(row: int, col: int, value: float) -> None:
        if value != 0.0:
            rows.append(row)
            cols.append(col)
            data.append(float(value))

    def add_to_target(
        source: int,
        base: tuple[int, int, int],
        phase1: int | None,
        phase3: int | None,
        value: float,
    ) -> None:
        add(source, layout[(STATE_INDEX[base], phase1, phase3)], value)

    for expanded, row in layout.items():
        base_index, phase1, phase3 = expanded
        x, y, z = STATE_TUPLES[base_index]
        total = x + y + z

        if x > 0:
            assert phase1 is not None
            for target_phase in range(len(alpha)):
                add_to_target(row, (x, y, z), target_phase, phase3, matrix_t[phase1, target_phase])
            next_base = (x - 1, y + 1, z)
            if x - 1 > 0:
                for target_phase in range(len(alpha)):
                    add_to_target(row, next_base, target_phase, phase3, exit_rates[phase1] * alpha[target_phase])
            else:
                add_to_target(row, next_base, None, phase3, exit_rates[phase1])

        if z > 0:
            assert phase3 is not None
            for target_phase in range(len(alpha)):
                add_to_target(row, (x, y, z), phase1, target_phase, matrix_t[phase3, target_phase])
            next_base = (x, y, z - 1)
            if z - 1 > 0:
                for target_phase in range(len(alpha)):
                    add_to_target(row, next_base, phase1, target_phase, exit_rates[phase3] * alpha[target_phase])
            else:
                add_to_target(row, next_base, phase1, None, exit_rates[phase3])

        if total < capacity:
            add(row, row, -lam)
            next_base = (x + 1, y, z)
            if x == 0:
                for target_phase in range(len(alpha)):
                    add_to_target(row, next_base, target_phase, phase3, lam * alpha[target_phase])
            else:
                add_to_target(row, next_base, phase1, phase3, lam)

        if y > 0:
            next_base = (x, y - 1, z + 1)
            rate = y * mu
            add(row, row, -rate)
            if z == 0:
                for target_phase in range(len(alpha)):
                    add_to_target(row, next_base, phase1, target_phase, rate * alpha[target_phase])
            else:
                add_to_target(row, next_base, phase1, phase3, rate)

    generator = sp.coo_matrix((data, (rows, cols)), shape=(len(states), len(states))).tocsr()
    _validate_generator_rows(generator)
    return generator, layout


def _expanded_states(
    phase_count: int,
    capacity: int,
) -> list[tuple[int, int | None, int | None]]:
    states: list[tuple[int, int | None, int | None]] = []
    for base_index, (x, y, z) in enumerate(STATE_TUPLES):
        if x + y + z > capacity:
            continue
        phase1_values: Iterable[int | None] = range(phase_count) if x > 0 else (None,)
        phase3_values: Iterable[int | None] = range(phase_count) if z > 0 else (None,)
        for phase1 in phase1_values:
            for phase3 in phase3_values:
                states.append((base_index, phase1, phase3))
    return states


def _aggregate(
    probability: np.ndarray,
    layout: dict[tuple[int, int | None, int | None], int],
) -> np.ndarray:
    states = np.zeros(len(STATE_TUPLES), dtype=float)
    for (base_index, _phase1, _phase3), index in layout.items():
        states[base_index] += probability[index]

    states[np.abs(states) < 1e-12] = 0.0
    total = float(np.sum(states))
    if abs(total) < 1e-14:
        raise RuntimeError("Aggregated state probabilities have invalid normalization.")
    states /= total
    if np.min(states) < -1e-7:
        raise RuntimeError(f"Aggregated state probabilities contain a negative value: {states}.")
    states[states < 0.0] = 0.0
    return states / float(np.sum(states))


def _result_row(
    approach: str,
    family: str,
    states: np.ndarray,
    *,
    arrival_rate: float,
    forward_rate: float,
    service_time: float,
) -> dict[str, float | int | str]:
    states = np.asarray(states, dtype=float).reshape(len(STATE_TUPLES))
    total = float(np.sum(states))
    if total <= 0.0:
        raise RuntimeError(f"{approach} produced invalid state probabilities.")
    states = states / total

    row: dict[str, float | int | str] = {
        "approach": approach,
        "family": family,
        "S1": float((states[0] + states[1] + states[3] + states[6]) * arrival_rate),
        "S2": float((states[3] + states[4] + 2.0 * states[5] + states[8]) * forward_rate),
        "S3": float((states[6] + states[7] + states[8] + states[9]) / service_time),
        "mean_stage1": float(states[1] + 2.0 * states[2] + states[4] + states[7]),
        "mean_forwarding": float(states[3] + states[4] + 2.0 * states[5] + states[8]),
        "mean_stage3": float(states[6] + states[7] + states[8] + 2.0 * states[9]),
        "mean_messages": float(
            sum(sum(STATE_TUPLES[index]) * states[index] for index in range(len(STATE_TUPLES))),
        ),
    }
    for label, value in zip(STATE_LABELS, states):
        row[f"p_{label}"] = float(value)
    return row


def _exponential_representation(service_time: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([1.0], dtype=float), np.array([[-1.0 / service_time]], dtype=float)


def _erlang_representation(phase_count: int, service_time: float) -> tuple[np.ndarray, np.ndarray]:
    rate = phase_count / service_time
    alpha = np.zeros(phase_count, dtype=float)
    alpha[0] = 1.0
    matrix_t = np.zeros((phase_count, phase_count), dtype=float)
    for i in range(phase_count):
        matrix_t[i, i] = -rate
        if i + 1 < phase_count:
            matrix_t[i, i + 1] = rate
    return alpha, matrix_t


def _solve_stationary(generator: sp.spmatrix) -> np.ndarray:
    generator = generator.tocsr()
    size = generator.shape[0]
    system = generator.transpose().tolil()
    rhs = np.zeros(size, dtype=float)
    system[-1, :] = 1.0
    rhs[-1] = 1.0

    probability = spla.spsolve(system.tocsr(), rhs)
    probability = np.real_if_close(probability, tol=1000)
    if np.iscomplexobj(probability):
        imag_norm = float(np.linalg.norm(np.imag(probability), ord=np.inf))
        if imag_norm > 1e-8:
            raise RuntimeError(f"Stationary solve produced complex residual {imag_norm}.")
        probability = np.real(probability)

    probability = np.asarray(probability, dtype=float)
    total = float(np.sum(probability))
    if not np.isfinite(total) or abs(total) < 1e-14:
        raise RuntimeError("Stationary solve produced an invalid normalization.")
    return probability / total


def _normalise_variant(variant: str) -> str:
    name = str(variant).strip().lower()
    if name in {"faster", "fast"}:
        return "faster"
    if name in {"slower", "slow"}:
        return "slower"
    raise ValueError(f"Unsupported synchronized message-passing variant: {variant}")


def _validate_rates(arrival_rate: float, forward_rate: float, service_time: float) -> None:
    if arrival_rate <= 0.0:
        raise ValueError("arrival_rate must be positive.")
    if forward_rate <= 0.0:
        raise ValueError("forward_rate must be positive.")
    if service_time <= 0.0:
        raise ValueError("service_time must be positive.")


def _validate_generator_rows(generator: sp.spmatrix) -> None:
    row_sums = np.asarray(generator.sum(axis=1)).ravel()
    residual = float(np.linalg.norm(row_sums, ord=np.inf))
    if residual > 1e-8:
        raise RuntimeError(f"Expanded generator rows do not sum to zero; residual={residual}.")
