from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from methods.cme import _cme_representation
from methods.fast_slow import _solve_embedded, _solve_markov_regenerative


STATE_LABELS = ("00", "10", "01", "11", "F")


def solve_synchronized(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    variants: Iterable[str] = ("faster", "slower"),
    *,
    precision_digits: int = 80,
) -> list[dict[str, float | int | str]]:
    """Solve the non-expanded synchronized faster/slower approximations.

    The state order is 00, 10, 01, 11, F, matching the Mathematica notebook's
    1..5 indexing. A single deterministic timer is active outside state 00.
    """
    _validate_rates(failure_rate, repair_time, manifestation_rate)
    results: list[dict[str, float | int | str]] = []

    for variant in variants:
        name = _normalise_variant(variant)
        generator, deferred, repair = _synchronized_matrices(
            failure_rate,
            manifestation_rate,
            name,
        )
        probability = _solve_embedded(
            generator=generator,
            deferred=deferred,
            departure=repair,
            groups=((1,), (2, 3, 4, 5)),
            tau=repair_time,
            precision_digits=precision_digits,
        )
        states = np.array([float(value) for value in probability], dtype=float)
        results.append(_result_row(f"synchronized {name}", "Synchronized", states))

    return results


def solve_synchronized_expanded(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    phase_counts: Iterable[int] = (10,),
    variants: Iterable[str] = ("faster", "slower"),
) -> list[dict[str, float | int | str]]:
    """Solve phase-expanded deterministic faster/slower approximations."""
    _validate_rates(failure_rate, repair_time, manifestation_rate)
    results: list[dict[str, float | int | str]] = []

    for phase_count in phase_counts:
        phases = int(phase_count)
        if phases < 1:
            raise ValueError("Synchronized phase counts must be positive.")
        for variant in variants:
            name = _normalise_variant(variant)
            states = _solve_synchronized_expanded(
                failure_rate=failure_rate,
                repair_time=repair_time,
                manifestation_rate=manifestation_rate,
                phase_count=phases,
                variant=name,
            )
            row = _result_row(f"synchronized {name} {phases}", "Synchronized expanded", states)
            row["order"] = phases
            row["timer_phases"] = phases
            row["linear_states"] = 1 + 2 * phases + 4 * phases * (phases + 1)
            results.append(row)

    return results


def solve_synchronized_erlang(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    phase_counts: Iterable[int] = (10,),
    variants: Iterable[str] = ("faster", "slower"),
) -> list[dict[str, float | int | str]]:
    """Backward-compatible name for the deterministic expanded solver."""
    return solve_synchronized_expanded(
        failure_rate=failure_rate,
        repair_time=repair_time,
        manifestation_rate=manifestation_rate,
        phase_counts=phase_counts,
        variants=variants,
    )


def solve_cme_ground_truth(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    orders: Iterable[int] = (5, 10, 15),
) -> list[dict[str, float | int | str]]:
    """Approximate the original independent-repair model with CME timers."""
    _validate_rates(failure_rate, repair_time, manifestation_rate)
    results: list[dict[str, float | int | str]] = []

    for order in orders:
        cme_order = int(order)
        alpha, matrix_t = _cme_representation(cme_order, repair_time)
        generator = _build_timer_expanded_generator(
            failure_rate=failure_rate,
            manifestation_rate=manifestation_rate,
            alpha=alpha,
            matrix_t=matrix_t,
        )
        probability = _solve_stationary(generator)
        states = _aggregate_timer_expanded(probability, len(alpha))
        row = _result_row(f"CME {cme_order} ground truth", "CME ground truth", states)
        row["order"] = cme_order
        row["timer_phases"] = len(alpha)
        row["linear_states"] = generator.shape[0]
        results.append(row)

    return results


def solve_erlang_ground_truth(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    phase_counts: Iterable[int] = (1, 5, 10, 20),
) -> list[dict[str, float | int | str]]:
    """Approximate the original independent-repair model with Erlang timers."""
    _validate_rates(failure_rate, repair_time, manifestation_rate)
    results: list[dict[str, float | int | str]] = []

    for phase_count in phase_counts:
        phases = int(phase_count)
        if phases < 1:
            raise ValueError("Erlang phase counts must be positive.")
        alpha, matrix_t = _erlang_representation(phases, repair_time)
        generator = _build_timer_expanded_generator(
            failure_rate=failure_rate,
            manifestation_rate=manifestation_rate,
            alpha=alpha,
            matrix_t=matrix_t,
        )
        probability = _solve_stationary(generator)
        states = _aggregate_timer_expanded(probability, phases)
        row = _result_row(f"Erlang {phases} ground truth", "Erlang ground truth", states)
        row["order"] = phases
        row["timer_phases"] = phases
        row["linear_states"] = generator.shape[0]
        results.append(row)

    return results


def solve_markovian(
    failure_rate: float,
    repair_rate: float,
    manifestation_rate: float,
) -> dict[str, float | int | str]:
    """Solve the exponential-repair CTMC shown in the notebook."""
    if failure_rate <= 0.0:
        raise ValueError("failure_rate must be positive.")
    if repair_rate <= 0.0:
        raise ValueError("repair_rate must be positive.")
    if manifestation_rate <= 0.0:
        raise ValueError("manifestation_rate must be positive.")

    lam = float(failure_rate)
    mu = float(repair_rate)
    rho = float(manifestation_rate)
    generator = np.array(
        [
            [-2.0 * lam, lam, lam, 0.0, 0.0],
            [mu, -(lam + mu), 0.0, lam, 0.0],
            [mu, 0.0, -(lam + mu), lam, 0.0],
            [0.0, mu, mu, -(2.0 * mu + rho), rho],
            [0.0, mu, mu, 0.0, -2.0 * mu],
        ],
        dtype=float,
    )
    probability = _solve_stationary(sp.csr_matrix(generator))
    return _result_row("exponential repair", "Markovian", probability)


def solve_all(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    *,
    synchronized_variants: Iterable[str] = ("faster", "slower"),
    cme_orders: Iterable[int] = (5, 10, 15),
) -> list[dict[str, float | int | str]]:
    rows = [
        solve_markovian(
            failure_rate=failure_rate,
            repair_rate=1.0 / repair_time,
            manifestation_rate=manifestation_rate,
        ),
    ]
    rows.extend(
        solve_synchronized(
            failure_rate=failure_rate,
            repair_time=repair_time,
            manifestation_rate=manifestation_rate,
            variants=synchronized_variants,
        ),
    )
    rows.extend(
        solve_cme_ground_truth(
            failure_rate=failure_rate,
            repair_time=repair_time,
            manifestation_rate=manifestation_rate,
            orders=cme_orders,
        ),
    )
    return rows


def _synchronized_matrices(
    failure_rate: float,
    manifestation_rate: float,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = float(failure_rate)
    rho = float(manifestation_rate)

    generator = np.zeros((5, 5), dtype=float)
    deferred = np.zeros((5, 5), dtype=float)
    repair = np.zeros((5, 5), dtype=float)

    generator[0, 0] = -2.0 * lam
    generator[0, 1] = lam
    generator[0, 2] = lam
    generator[1, 1] = -lam
    generator[2, 2] = -lam
    generator[3, 3] = -rho
    generator[3, 4] = rho

    if variant == "faster":
        generator[1, 3] = lam
        generator[2, 3] = lam
        repair[1, 0] = 1.0
        repair[2, 0] = 1.0
        repair[3, 0] = 1.0
        repair[4, 0] = 1.0
    elif variant == "slower":
        deferred[1, 3] = lam
        deferred[2, 3] = lam
        repair[1, 0] = 1.0
        repair[2, 0] = 1.0
        repair[3, 1] = 0.5
        repair[3, 2] = 0.5
        repair[4, 1] = 0.5
        repair[4, 2] = 0.5
    else:
        raise ValueError(f"Unsupported synchronized reliability variant: {variant}")

    return generator, deferred, repair


def _build_timer_expanded_generator(
    *,
    failure_rate: float,
    manifestation_rate: float,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> sp.csr_matrix:
    lam = float(failure_rate)
    rho = float(manifestation_rate)
    alpha = np.asarray(alpha, dtype=float).reshape(-1)
    matrix_t = np.asarray(matrix_t, dtype=float)

    if matrix_t.shape != (len(alpha), len(alpha)):
        raise ValueError("matrix_t shape must match alpha length.")

    phases = len(alpha)
    both_dim = phases * phases

    alpha_row = sp.csr_matrix(alpha.reshape(1, phases))
    exit_col = sp.csr_matrix((-matrix_t @ np.ones(phases)).reshape(phases, 1))
    timer = sp.csr_matrix(matrix_t)
    eye = sp.eye(phases, format="csr")
    eye_both = sp.eye(both_dim, format="csr")

    zero_1_both = sp.csr_matrix((1, both_dim))
    zero_m_m = sp.csr_matrix((phases, phases))
    zero_m_both = sp.csr_matrix((phases, both_dim))
    zero_both_1 = sp.csr_matrix((both_dim, 1))
    zero_both_both = sp.csr_matrix((both_dim, both_dim))

    single_block = timer - lam * eye
    both_timer = sp.kron(timer, eye, format="csr") + sp.kron(eye, timer, format="csr")
    a_to_both = lam * sp.kron(eye, alpha_row, format="csr")
    b_to_both = lam * sp.kron(alpha_row, eye, format="csr")
    both_to_a = sp.kron(eye, exit_col, format="csr")
    both_to_b = sp.kron(exit_col, eye, format="csr")

    generator = sp.bmat(
        [
            [
                sp.csr_matrix([[-2.0 * lam]]),
                lam * alpha_row,
                lam * alpha_row,
                zero_1_both,
                zero_1_both,
            ],
            [
                exit_col,
                single_block,
                zero_m_m,
                a_to_both,
                zero_m_both,
            ],
            [
                exit_col,
                zero_m_m,
                single_block,
                b_to_both,
                zero_m_both,
            ],
            [
                zero_both_1,
                both_to_a,
                both_to_b,
                both_timer - rho * eye_both,
                rho * eye_both,
            ],
            [
                zero_both_1,
                both_to_a,
                both_to_b,
                zero_both_both,
                both_timer,
            ],
        ],
        format="csr",
    )

    _validate_generator_rows(generator)
    return generator


def _build_synchronized_erlang_generator(
    *,
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    phase_count: int,
    variant: str,
) -> sp.csr_matrix:
    raise RuntimeError("Use _solve_synchronized_expanded for synchronized reliability expansions.")


def _solve_synchronized_expanded(
    *,
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
    phase_count: int,
    variant: str,
) -> np.ndarray:
    lam = float(failure_rate)
    rho = float(manifestation_rate)
    phases = int(phase_count)
    delta_time = float(repair_time) / phases
    phase_values = tuple(range(-1, phases))
    phase_index = {phase: offset for offset, phase in enumerate(phase_values)}
    pair_count = phases * len(phase_values)
    size = 1 + 2 * phases + 4 * pair_count

    def s0() -> int:
        return 0

    def s10(phase: int) -> int:
        return 1 + phase

    def s01(phase: int) -> int:
        return 1 + phases + phase

    base = 1 + 2 * phases

    def pair_offset(old_phase: int, new_phase: int) -> int:
        if old_phase < 0 or old_phase >= phases:
            raise ValueError(f"Invalid old repair phase {old_phase}.")
        return old_phase * len(phase_values) + phase_index[new_phase]

    def s11_to01(old_phase: int, new_phase: int) -> int:
        return base + pair_offset(old_phase, new_phase)

    def sf_to01(old_phase: int, new_phase: int) -> int:
        return base + pair_count + pair_offset(old_phase, new_phase)

    def s11_to10(old_phase: int, new_phase: int) -> int:
        return base + 2 * pair_count + pair_offset(old_phase, new_phase)

    def sf_to10(old_phase: int, new_phase: int) -> int:
        return base + 3 * pair_count + pair_offset(old_phase, new_phase)

    def second_repair_start_phase() -> int:
        if variant == "faster":
            return 0
        if variant == "slower":
            return -1
        raise ValueError(f"Unsupported synchronized reliability variant: {variant}")

    q = sp.lil_matrix((size, size), dtype=float)
    boundary = sp.lil_matrix((size, size), dtype=float)

    boundary[s0(), s0()] = 1.0
    for phase in range(phases):
        new_phase = second_repair_start_phase()
        q[s10(phase), s11_to01(phase, new_phase)] = lam
        q[s01(phase), s11_to10(phase, new_phase)] = lam
        q[s10(phase), s10(phase)] = -lam
        q[s01(phase), s01(phase)] = -lam

        next_phase = phase + 1
        boundary[s10(phase), s10(next_phase) if next_phase < phases else s0()] = 1.0
        boundary[s01(phase), s01(next_phase) if next_phase < phases else s0()] = 1.0

    def boundary_after_two_failed(
        old_phase: int,
        new_phase: int,
        *,
        old_remaining: str,
        new_remaining: str,
        manifested: bool,
    ) -> int:
        old_next = old_phase + 1
        new_next = new_phase + 1
        if old_next >= phases and new_next >= phases:
            return s0()
        if old_next >= phases:
            return s01(new_next) if new_remaining == "01" else s10(new_next)
        if new_next >= phases:
            return s10(old_next) if old_remaining == "10" else s01(old_next)
        if old_remaining == "10":
            return sf_to01(old_next, new_next) if manifested else s11_to01(old_next, new_next)
        return sf_to10(old_next, new_next) if manifested else s11_to10(old_next, new_next)

    for old_phase in range(phases):
        for new_phase in phase_values:
            q[s11_to01(old_phase, new_phase), sf_to01(old_phase, new_phase)] = rho
            q[s11_to01(old_phase, new_phase), s11_to01(old_phase, new_phase)] = -rho
            q[s11_to10(old_phase, new_phase), sf_to10(old_phase, new_phase)] = rho
            q[s11_to10(old_phase, new_phase), s11_to10(old_phase, new_phase)] = -rho

            boundary[
                s11_to01(old_phase, new_phase),
                boundary_after_two_failed(
                    old_phase,
                    new_phase,
                    old_remaining="10",
                    new_remaining="01",
                    manifested=False,
                ),
            ] = 1.0
            boundary[
                sf_to01(old_phase, new_phase),
                boundary_after_two_failed(
                    old_phase,
                    new_phase,
                    old_remaining="10",
                    new_remaining="01",
                    manifested=True,
                ),
            ] = 1.0
            boundary[
                s11_to10(old_phase, new_phase),
                boundary_after_two_failed(
                    old_phase,
                    new_phase,
                    old_remaining="01",
                    new_remaining="10",
                    manifested=False,
                ),
            ] = 1.0
            boundary[
                sf_to10(old_phase, new_phase),
                boundary_after_two_failed(
                    old_phase,
                    new_phase,
                    old_remaining="01",
                    new_remaining="10",
                    manifested=True,
                ),
            ] = 1.0

    q_csr = q.tocsr()
    boundary_csr = boundary.tocsr()
    omega, psi = _phase_transition_and_sojourn(q_csr, delta_time)

    p_matrix = (omega @ boundary_csr).tolil()
    c_matrix = psi.tolil()
    p_matrix[s0(), :] = 0.0
    p_matrix[s0(), s10(0)] = 0.5
    p_matrix[s0(), s01(0)] = 0.5
    c_matrix[s0(), :] = 0.0
    c_matrix[s0(), s0()] = 1.0 / (2.0 * lam)

    probability = _solve_markov_regenerative(p_matrix.tocsr(), c_matrix.tocsr())
    states = np.zeros(5, dtype=float)
    states[0] = probability[s0()]
    states[1] = float(np.sum(probability[[s10(phase) for phase in range(phases)]]))
    states[2] = float(np.sum(probability[[s01(phase) for phase in range(phases)]]))
    states[3] = float(np.sum(probability[
        [s11_to01(old_phase, new_phase) for old_phase in range(phases) for new_phase in phase_values]
    ]))
    states[3] += float(np.sum(probability[
        [s11_to10(old_phase, new_phase) for old_phase in range(phases) for new_phase in phase_values]
    ]))
    states[4] = float(np.sum(probability[
        [sf_to01(old_phase, new_phase) for old_phase in range(phases) for new_phase in phase_values]
    ]))
    states[4] += float(np.sum(probability[
        [sf_to10(old_phase, new_phase) for old_phase in range(phases) for new_phase in phase_values]
    ]))
    states[states < 0.0] = 0.0
    return states / float(np.sum(states))


def _phase_transition_and_sojourn(generator: sp.csr_matrix, delta_time: float) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    dense = generator.toarray()
    size = dense.shape[0]
    augmented = np.zeros((2 * size, 2 * size), dtype=float)
    augmented[:size, :size] = dense
    augmented[:size, size:] = np.eye(size, dtype=float)
    block = scipy_linalg.expm(augmented * delta_time)
    return sp.csr_matrix(block[:size, :size]), sp.csr_matrix(block[:size, size:])


def _aggregate_synchronized_erlang(probability: np.ndarray, phase_count: int) -> np.ndarray:
    phases = int(phase_count)
    states = np.zeros(5, dtype=float)
    states[0] = float(probability[0])
    for state in range(1, 5):
        start = 1 + (state - 1) * phases
        states[state] = float(np.sum(probability[start:start + phases]))
    total = float(np.sum(states))
    if total <= 0.0:
        raise RuntimeError("Aggregated synchronized state probabilities have invalid normalization.")
    states[states < 0.0] = 0.0
    return states / float(np.sum(states))


def _erlang_representation(phase_count: int, repair_time: float) -> tuple[np.ndarray, np.ndarray]:
    rate = phase_count / repair_time
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


def _aggregate_timer_expanded(probability: np.ndarray, phase_count: int) -> np.ndarray:
    both_dim = phase_count * phase_count
    offset_00 = 0
    offset_10 = 1
    offset_01 = offset_10 + phase_count
    offset_11 = offset_01 + phase_count
    offset_f = offset_11 + both_dim

    states = np.array(
        [
            probability[offset_00],
            np.sum(probability[offset_10:offset_01]),
            np.sum(probability[offset_01:offset_11]),
            np.sum(probability[offset_11:offset_f]),
            np.sum(probability[offset_f:offset_f + both_dim]),
        ],
        dtype=float,
    )
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
) -> dict[str, float | int | str]:
    states = np.asarray(states, dtype=float).reshape(5)
    total = float(np.sum(states))
    if total <= 0.0:
        raise RuntimeError(f"{approach} produced invalid state probabilities.")
    states = states / total

    return {
        "approach": approach,
        "family": family,
        "p_00": float(states[0]),
        "p_10": float(states[1]),
        "p_01": float(states[2]),
        "p_11": float(states[3]),
        "p_F": float(states[4]),
        "p_available": float(states[0] + states[1] + states[2]),
        "p_down": float(states[3] + states[4]),
        "p_severe": float(states[4]),
    }


def _normalise_variant(variant: str) -> str:
    name = str(variant).strip().lower()
    if name in {"faster", "fast"}:
        return "faster"
    if name in {"slower", "slow"}:
        return "slower"
    raise ValueError(f"Unsupported synchronized reliability variant: {variant}")


def _validate_rates(
    failure_rate: float,
    repair_time: float,
    manifestation_rate: float,
) -> None:
    if failure_rate <= 0.0:
        raise ValueError("failure_rate must be positive.")
    if repair_time <= 0.0:
        raise ValueError("repair_time must be positive.")
    if manifestation_rate <= 0.0:
        raise ValueError("manifestation_rate must be positive.")


def _validate_generator_rows(generator: sp.spmatrix) -> None:
    row_sums = np.asarray(generator.sum(axis=1)).ravel()
    residual = float(np.linalg.norm(row_sums, ord=np.inf))
    if residual > 1e-8:
        raise RuntimeError(f"Expanded generator rows do not sum to zero; residual={residual}.")
