from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class _VariantSpec:
    kind: str
    name: str
    phases: int | None = None


@dataclass(frozen=True)
class _ExpandedTopology:
    dim: int
    qbar_pattern: sp.csr_matrix
    delta_a: sp.csr_matrix
    delta_ba: sp.csr_matrix
    delta_bhat: sp.csr_matrix
    sa_rows: tuple[tuple[int, tuple[int, ...]], ...]
    sbhat_states: tuple[int, ...]
    aggregate_groups: tuple[tuple[int, ...], ...]
    first_a: int
    first_b: int


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    variants: list[str],
    precision_digits: int = 80,
    solver: str = "direct",
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
) -> list[dict[str, float | str]]:
    if precision_digits < 20:
        raise ValueError("Fast-slow precision must be at least 20 digits.")

    results: list[dict[str, float | str]] = []
    for variant in variants:
        spec = _parse_variant(variant)

        if spec.kind in {"faster_noexp", "slower"}:
            generator, deferred, departure, groups = _build_simple_inputs(
                arrival_rate, capacity, spec.kind,
            )
            solver_mode = solver.strip().lower()
            if solver_mode == "direct":
                probability = _solve_embedded_fast(
                    generator,
                    deferred,
                    departure,
                    groups,
                    service_time,
                )
            else:
                embedded_solver = "direct" if solver_mode == "mp_direct" else solver
                probability = _solve_embedded(
                    generator,
                    deferred,
                    departure,
                    groups,
                    service_time,
                    precision_digits,
                    solver=embedded_solver,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            dist = _aggregate_simple(probability, capacity)
        elif spec.kind == "slower_additional":
            generator, deferred, departure, groups = _build_slower_additional_inputs(
                arrival_rate, capacity,
            )
            probability = _solve_embedded(
                generator,
                deferred,
                departure,
                groups,
                service_time,
                precision_digits,
                solver=solver,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            dist = _aggregate_slower_additional(probability, capacity)
        elif spec.kind == "faster_expanded":
            if spec.phases is None:
                raise ValueError("Expanded faster variant needs a phase count.")
            dist = _solve_expanded(
                topology=_build_faster_expanded_topology(capacity, spec.phases),
                arrival_rate=arrival_rate,
                service_time=service_time,
                phase_count=spec.phases,
                precision_digits=precision_digits,
                solver=solver,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        elif spec.kind == "slower_expanded":
            if spec.phases is None:
                raise ValueError("Expanded slower variant needs a phase count.")
            dist = _solve_expanded(
                topology=_build_slower_expanded_topology(capacity, spec.phases),
                arrival_rate=arrival_rate,
                service_time=service_time,
                phase_count=spec.phases,
                precision_digits=precision_digits,
                solver=solver,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        else:
            raise ValueError(f"Unsupported fast-slow variant kind: {spec.kind}")

        # The expanded slower model has physical customer occupancy in both
        # the ordinary phases and the auxiliary phase -1.  Its aggregation
        # therefore represents the complete distribution and should be
        # normalized like every other variant.
        p_idle, p_full, response_time = _metrics(dist, arrival_rate)
        solver_mode = solver.strip().lower()
        results.append({
            "approach": spec.name if solver_mode in {"direct", "mp_direct"} else f"{spec.name} iterative",
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
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

    raise ValueError(f"Unsupported fast-slow variant: {variant}")


# ---------------------------------------------------------------------------
# Non-expanded models
# ---------------------------------------------------------------------------

def _build_simple_inputs(
    arrival_rate: float,
    capacity: int,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    dim = capacity + 2
    groups = ((1,), tuple(range(2, dim + 1)))
    Q = np.zeros((dim, dim))
    Qbar = np.zeros((dim, dim))
    D = np.zeros((dim, dim))

    def s0() -> int:
        return 1

    def s1a() -> int:
        return 2

    def s1b() -> int:
        return 3

    def si(n: int) -> int:
        return n + 2

    Q[s0() - 1, s0() - 1] = -arrival_rate
    Q[s0() - 1, s1a() - 1] = arrival_rate / 2.0
    Q[s0() - 1, s1b() - 1] = arrival_rate / 2.0
    Q[s1a() - 1, s1a() - 1] = -arrival_rate
    Q[s1b() - 1, s1b() - 1] = -arrival_rate

    D[s1a() - 1, s0() - 1] = 1.0
    D[s1b() - 1, s0() - 1] = 1.0
    D[si(2) - 1, s0() - 1] = 1.0
    D[si(3) - 1, s1a() - 1] = 0.5
    D[si(3) - 1, s1b() - 1] = 0.5

    if kind == "faster_noexp":
        Q[s1a() - 1, si(2) - 1] = arrival_rate
        Q[s1b() - 1, si(2) - 1] = arrival_rate
    elif kind == "slower":
        Qbar[s1a() - 1, si(2) - 1] = arrival_rate
        Qbar[s1b() - 1, si(2) - 1] = arrival_rate
    else:
        raise ValueError(f"Unsupported non-expanded fast-slow kind: {kind}")

    for n in range(4, capacity + 1):
        D[si(n) - 1, si(n - 2) - 1] = 1.0

    for n in range(2, capacity):
        Q[si(n) - 1, si(n) - 1] += -arrival_rate
        Q[si(n) - 1, si(n + 1) - 1] = arrival_rate

    return Q, Qbar, D, groups


def _build_slower_additional_inputs(
    arrival_rate: float,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    dim = 2 * capacity + 1
    groups = ((1,), tuple(range(2, dim + 1)))
    Q = np.zeros((dim, dim))
    Qbar = np.zeros((dim, dim))
    D = np.zeros((dim, dim))

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

    Q[s0() - 1, s0() - 1] = -arrival_rate
    Q[s0() - 1, s1a() - 1] = arrival_rate / 2.0
    Q[s0() - 1, s1b() - 1] = arrival_rate / 2.0

    for single in (s1a(), s1b()):
        Q[single - 1, single - 1] = -arrival_rate
        Q[single - 1, s(2, -1) - 1] = arrival_rate
        D[single - 1, s0() - 1] = 1.0

    D[s(2, -1) - 1, s(2, 0) - 1] = 1.0
    D[s(2, 0) - 1, s0() - 1] = 1.0

    D[s(3, -1) - 1, s(3, 0) - 1] = 1.0
    D[s(3, 0) - 1, s1a() - 1] = 0.5
    D[s(3, 0) - 1, s1b() - 1] = 0.5

    for i in range(4, capacity + 1):
        D[s(i, -1) - 1, s(i, 0) - 1] = 1.0
        D[s(i, 0) - 1, s(i - 2, 0) - 1] = 1.0

    for i in range(2, capacity):
        for phase in (-1, 0):
            state = s(i, phase) - 1
            Q[state, state] += -arrival_rate
            Q[state, s(i + 1, phase) - 1] = arrival_rate

    return Q, Qbar, D, groups


# ---------------------------------------------------------------------------
# Expanded models
# ---------------------------------------------------------------------------

def _solve_expanded(
    topology: _ExpandedTopology,
    arrival_rate: float,
    service_time: float,
    phase_count: int,
    precision_digits: int,
    solver: str = "direct",
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
) -> np.ndarray:
    dim = topology.dim
    delta_time = service_time / phase_count

    omega_anth, psi_anth = _build_sa_transients(
        topology.sa_rows, arrival_rate, delta_time, dim, precision_digits,
    )
    omega_bhat, psi_bhat = _build_sbhat_transients(
        topology.sbhat_states, arrival_rate, delta_time, dim, precision_digits,
    )

    M = (omega_anth @ topology.delta_ba).tocsr()
    omega_power = sp.eye(dim, format="csr")
    psi_a = sp.csr_matrix((dim, dim))
    for _ in range(phase_count):
        psi_a = psi_a + (omega_power @ psi_anth)
        omega_power = (omega_power @ M).tocsr()

    omegatilde_a = (omega_power @ topology.delta_a).tocsr()
    omegatilde_bhat = (omega_bhat @ topology.delta_bhat).tocsr()
    omegatilde = (omegatilde_a + omegatilde_bhat).tocsr()
    psi = (psi_a + psi_bhat).tocsr()

    qbar = (topology.qbar_pattern * arrival_rate).tocsr()
    P = (_regen_from_idle(dim, topology.first_a, topology.first_b) + omegatilde + (psi @ qbar)).tocsr()
    C = (_idle_correction(dim, arrival_rate) + psi).tocsr()
    probability = _solve_markov_regenerative(
        P,
        C,
        solver=solver,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )

    dist = np.zeros(len(topology.aggregate_groups), dtype=float)
    for customer_count, states in enumerate(topology.aggregate_groups):
        dist[customer_count] = float(probability[list(states)].sum())
    return dist


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
        aggregate_groups.append(tuple(s(customers, phase) for phase in range(-1, phase_count)))

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


def _build_sa_transients(
    sa_rows: tuple[tuple[int, tuple[int, ...]], ...],
    arrival_rate: float,
    delta_time: float,
    dim: int,
    precision_digits: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    omega = sp.lil_matrix((dim, dim), dtype=float)
    psi = sp.lil_matrix((dim, dim), dtype=float)
    cache: dict[int, tuple[list[float], list[float], float, float]] = {}

    for row, chain in sa_rows:
        probs, coeffs, tail_prob, tail_coeff = _birth_chain_coefficients(
            len(chain) - 1, arrival_rate, delta_time, cache, precision_digits,
        )
        for offset, value in enumerate(probs):
            omega[row, chain[offset]] = value
            psi[row, chain[offset]] = coeffs[offset]
        omega[row, chain[-1]] += tail_prob
        psi[row, chain[-1]] += tail_coeff

    return omega.tocsr(), psi.tocsr()


def _build_sbhat_transients(
    sbhat_states: tuple[int, ...],
    arrival_rate: float,
    delta_time: float,
    dim: int,
    precision_digits: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    omega = sp.lil_matrix((dim, dim), dtype=float)
    psi = sp.lil_matrix((dim, dim), dtype=float)

    with mp.workdps(precision_digits):
        if arrival_rate <= 0.0:
            survival = 1.0
            mean_sojourn = delta_time
        else:
            hazard = mp.mpf(str(arrival_rate)) * mp.mpf(str(delta_time))
            exp_hazard = mp.exp(-hazard)
            survival = float(exp_hazard)
            mean_sojourn = float((mp.mpf("1") - exp_hazard) / mp.mpf(str(arrival_rate)))

    for state in sbhat_states:
        omega[state, state] = survival
        psi[state, state] = mean_sojourn

    return omega.tocsr(), psi.tocsr()


def _birth_chain_coefficients(
    max_jumps: int,
    arrival_rate: float,
    delta_time: float,
    cache: dict[int, tuple[list[float], list[float], float, float]],
    precision_digits: int,
) -> tuple[list[float], list[float], float, float]:
    cached = cache.get(max_jumps)
    if cached is not None:
        return cached

    if max_jumps == 0:
        cached = ([], [], 1.0, delta_time)
        cache[max_jumps] = cached
        return cached

    if arrival_rate <= 0.0:
        probs = [1.0] + [0.0] * (max_jumps - 1)
        coeffs = [delta_time] + [0.0] * (max_jumps - 1)
        cached = (probs, coeffs, 0.0, 0.0)
        cache[max_jumps] = cached
        return cached

    with mp.workdps(precision_digits):
        lam = mp.mpf(str(arrival_rate))
        dt = mp.mpf(str(delta_time))
        hazard = lam * dt
        term = mp.exp(-hazard)
        cumulative_prob = mp.mpf("0")
        probs_mp: list[mp.mpf] = []
        coeffs_mp: list[mp.mpf] = []

        for jumps in range(max_jumps):
            if jumps > 0:
                term *= hazard / jumps
            probs_mp.append(term)
            cumulative_prob += term
            coeff = (mp.mpf("1") - cumulative_prob) / lam
            coeffs_mp.append(coeff if coeff > 0 else mp.mpf("0"))

        tail_prob_mp = mp.mpf("1") - cumulative_prob
        if tail_prob_mp < 0:
            tail_prob_mp = mp.mpf("0")
        tail_coeff_mp = dt - mp.fsum(coeffs_mp)
        if tail_coeff_mp < 0:
            tail_coeff_mp = mp.mpf("0")

    probs = [float(value) for value in probs_mp]
    coeffs = [float(value) for value in coeffs_mp]
    tail_prob = float(tail_prob_mp)
    tail_coeff = float(tail_coeff_mp)
    cached = (probs, coeffs, tail_prob, tail_coeff)
    cache[max_jumps] = cached
    return cached


def _regen_from_idle(dim: int, first_a: int, first_b: int) -> sp.csr_matrix:
    regen = sp.lil_matrix((dim, dim), dtype=float)
    regen[0, first_a] = 0.5
    regen[0, first_b] = 0.5
    return regen.tocsr()


def _idle_correction(dim: int, arrival_rate: float) -> sp.csr_matrix:
    idle = sp.lil_matrix((dim, dim), dtype=float)
    if arrival_rate > 0.0:
        idle[0, 0] = 1.0 / arrival_rate
    return idle.tocsr()


def _solve_markov_regenerative(
    P: sp.csr_matrix,
    C: sp.csr_matrix,
    *,
    solver: str = "direct",
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
) -> np.ndarray:
    dim = P.shape[0]
    ones = np.ones(dim, dtype=float)
    correction_sum = np.asarray(C @ ones).ravel()
    mode = solver.strip().lower()

    if mode == "direct":
        system = (P - sp.eye(dim, format="csr")).transpose().tolil()
        rhs = np.zeros(dim, dtype=float)
        rhs[-1] = 1.0
        system[-1, :] = correction_sum

        # The sparse direct solve is a useful initializer, but the tiny tail
        # probabilities in the expanded models are much more stable after a few
        # positivity-preserving power refinements on the embedded kernel.
        solution = spla.spsolve(system.tocsr(), rhs)
        solution = np.real(np.asarray(solution, dtype=float)).ravel()
        scale = float(solution @ correction_sum)
        if not np.isfinite(scale) or scale <= 0.0:
            solution = np.zeros(dim, dtype=float)
            solution[0] = 1.0
            scale = float(solution @ correction_sum)
        solution = solution / scale

        for _ in range(12):
            solution = np.asarray(solution @ P).ravel()
            scale = float(solution @ correction_sum)
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError("Expanded fast-slow produced an invalid regenerative multiplier.")
            solution = solution / scale
    elif mode == "iterative":
        solution = np.zeros(dim, dtype=float)
        solution[0] = 1.0
        previous = solution.copy()
        for _ in range(max_iterations):
            solution = np.asarray(previous @ P).ravel()
            scale = float(solution @ correction_sum)
            if not np.isfinite(scale) or scale <= 0.0:
                raise RuntimeError("Expanded fast-slow iterative solver produced an invalid regenerative multiplier.")
            solution = solution / scale
            if np.linalg.norm(solution - previous, ord=1) < tolerance:
                break
            previous = solution
        else:
            raise RuntimeError("Expanded fast-slow iterative solver did not converge.")
    else:
        raise ValueError(f"Unsupported fast-slow solver mode: {solver}")

    probability = np.real(np.asarray(C.transpose() @ solution, dtype=float)).ravel()
    probability[probability < 0.0] = 0.0

    total = probability.sum()
    if total <= 0.0:
        raise RuntimeError("Expanded fast-slow produced an invalid stationary distribution.")
    return probability / total


# ---------------------------------------------------------------------------
# Embedded Markov chain solver for one deterministic timer
# ---------------------------------------------------------------------------

def _solve_embedded(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tau: float,
    precision_digits: int,
    *,
    solver: str = "direct",
    tolerance: float = 1e-12,
    max_iterations: int = 20_000,
    allow_signed: bool = False,
) -> list[mp.mpf]:
    prev_dps = mp.mp.dps
    mp.mp.dps = precision_digits
    try:
        dim = generator.shape[0]
        groups_list = [list(g) for g in groups]
        all_states = [s for g in groups_list for s in g]
        exp_states = groups_list[0]
        gen_states = [s for s in all_states if s not in exp_states]

        Q_mp = _np_to_mp(generator)
        Qbar_mp = _np_to_mp(deferred)
        D_mp = _np_to_mp(departure)

        I_exp = _id_subset(dim, exp_states)
        I_gen = _id_subset(dim, gen_states)

        ones = mp.matrix(dim, 1)
        for i in range(dim):
            ones[i] = mp.mpf("1")

        Q_gen = I_gen * Q_mp
        Q_exp = I_exp * Q_mp
        exp_block, int_block = _exponential_integrals(Q_gen, tau)

        omega = I_gen * exp_block
        psi = I_gen * int_block

        transition = (
            I_exp
            - (_diag_inv(Q_exp) * Q_exp)
            + (omega * D_mp)
            + (psi * Qbar_mp)
        )
        correction = -_diag_inv(Q_exp) + psi

        A = (transition - I_exp - I_gen).transpose()
        norm_row = correction * ones
        for c in range(dim):
            A[dim - 1, c] = norm_row[c]

        rhs = mp.matrix(dim, 1)
        for i in range(dim):
            rhs[i] = mp.mpf("0")
        rhs[dim - 1] = mp.mpf("1")

        mode = solver.strip().lower()
        if mode == "direct":
            mult = mp.lu_solve(A, rhs)
        elif mode == "iterative":
            mult = mp.matrix(dim, 1)
            for i in range(dim):
                mult[i] = mp.mpf("0")
            mult[0] = mp.mpf("1")

            transition_t = transition.transpose()
            tol_mp = mp.mpf(str(tolerance))
            for _ in range(max_iterations):
                new_mult = transition_t * mult
                scale = mp.fsum(norm_row[c] * new_mult[c] for c in range(dim))
                if scale <= 0:
                    raise RuntimeError("Fast-slow iterative embedded solver produced an invalid normalization.")
                for i in range(dim):
                    new_mult[i] /= scale
                diff = mp.fsum(mp.fabs(new_mult[i] - mult[i]) for i in range(dim))
                mult = new_mult
                if diff < tol_mp:
                    break
            else:
                raise RuntimeError("Fast-slow iterative embedded solver did not converge.")
        else:
            raise ValueError(f"Unsupported fast-slow solver mode: {solver}")
        stat_col = correction.transpose() * mult
        stat = [stat_col[i] for i in range(dim)]
        if allow_signed:
            total = mp.fsum(stat)
            if mp.fabs(total) <= mp.mpf("1e-80"):
                raise RuntimeError("Signed embedded solver produced an invalid stationary distribution.")
            return [value / total for value in stat]
        return _sanitise(stat)
    finally:
        mp.mp.dps = prev_dps


def _solve_embedded_fast(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tau: float,
    *,
    allow_signed: bool = False,
) -> list[float]:
    dim = generator.shape[0]
    groups_list = [list(group) for group in groups]
    all_states = [state for group in groups_list for state in group]
    exp_states = groups_list[0]
    gen_states = [state for state in all_states if state not in exp_states]

    id_exp = _id_subset_np(dim, exp_states)
    id_gen = _id_subset_np(dim, gen_states)

    q_gen = id_gen @ generator
    q_exp = id_exp @ generator
    exp_block, int_block = _exponential_integrals_np(q_gen, tau)

    omega = id_gen @ exp_block
    psi = id_gen @ int_block
    transition = (
        id_exp
        - (_diag_inv_np(q_exp) @ q_exp)
        + (omega @ departure)
        + (psi @ deferred)
    )
    correction = -_diag_inv_np(q_exp) + psi

    system = (transition - id_exp - id_gen).T
    rhs = np.zeros(dim, dtype=float)
    system[-1, :] = correction @ np.ones(dim, dtype=float)
    rhs[-1] = 1.0

    multiplier = np.linalg.solve(system, rhs)
    probability = correction.T @ multiplier
    probability = np.real_if_close(probability, tol=1000)
    probability = np.asarray(probability, dtype=float)
    if not allow_signed:
        probability[probability < 0.0] = 0.0
    total = float(np.sum(probability))
    if (allow_signed and (not np.isfinite(total) or abs(total) <= 1e-300)) or (
        not allow_signed and total <= 0.0
    ):
        raise RuntimeError("Fast embedded solver produced an invalid stationary distribution.")
    return [float(value) for value in probability / total]


def _solve_embedded_fast_iterative(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tau: float,
    *,
    iterations: int = 2_000,
) -> list[float]:
    dim = generator.shape[0]
    groups_list = [list(group) for group in groups]
    all_states = [state for group in groups_list for state in group]
    exp_states = groups_list[0]
    gen_states = [state for state in all_states if state not in exp_states]

    id_exp = _id_subset_np(dim, exp_states)
    id_gen = _id_subset_np(dim, gen_states)

    q_gen = id_gen @ generator
    q_exp = id_exp @ generator
    exp_block, int_block = _exponential_integrals_np(q_gen, tau)

    omega = id_gen @ exp_block
    psi = id_gen @ int_block
    transition = (
        id_exp
        - (_diag_inv_np(q_exp) @ q_exp)
        + (omega @ departure)
        + (psi @ deferred)
    )
    correction = -_diag_inv_np(q_exp) + psi
    correction_sum = correction @ np.ones(dim, dtype=float)

    multiplier = np.zeros(dim, dtype=float)
    multiplier[0] = 1.0
    transition_t = transition.transpose()
    for _ in range(iterations):
        multiplier = transition_t @ multiplier
        scale = float(correction_sum @ multiplier)
        if not np.isfinite(scale) or scale <= 0.0:
            raise RuntimeError("Fast iterative embedded solver produced an invalid normalization.")
        multiplier = multiplier / scale

    probability = correction.transpose() @ multiplier
    probability = np.real_if_close(probability, tol=1000)
    probability = np.asarray(probability, dtype=float)
    probability[probability < 0.0] = 0.0
    total = float(np.sum(probability))
    if total <= 0.0:
        raise RuntimeError("Fast iterative embedded solver produced an invalid stationary distribution.")
    return [float(value) for value in probability / total]


# ---------------------------------------------------------------------------
# Matrix exponential and integral for the one-timer solver
# ---------------------------------------------------------------------------

def _exponential_integrals(
    Q_sub: mp.matrix,
    tau: float,
) -> tuple[mp.matrix, mp.matrix]:
    dim = Q_sub.rows
    aug = mp.matrix(2 * dim, 2 * dim)
    for r in range(2 * dim):
        for c in range(2 * dim):
            aug[r, c] = mp.mpf("0")

    for r in range(dim):
        for c in range(dim):
            aug[r, c] = Q_sub[r, c]
        aug[r, dim + r] = mp.mpf("1")

    E = mp.expm(aug * mp.mpf(str(tau)))
    B1 = mp.matrix(dim, dim)
    B2 = mp.matrix(dim, dim)
    for r in range(dim):
        for c in range(dim):
            B1[r, c] = E[r, c]
            B2[r, c] = E[r, dim + c]
    return B1, B2


def _exponential_integrals_np(
    q_sub: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    dim = q_sub.shape[0]
    augmented = np.zeros((2 * dim, 2 * dim), dtype=float)
    augmented[:dim, :dim] = q_sub
    augmented[:dim, dim:] = np.eye(dim, dtype=float)
    block = la.expm(augmented * tau)
    return block[:dim, :dim], block[:dim, dim:]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _id_subset(dim: int, states_1based: list[int]) -> mp.matrix:
    M = mp.matrix(dim, dim)
    for r in range(dim):
        for c in range(dim):
            M[r, c] = mp.mpf("0")
    for s in states_1based:
        M[s - 1, s - 1] = mp.mpf("1")
    return M


def _id_subset_np(dim: int, states_1based: list[int]) -> np.ndarray:
    matrix = np.zeros((dim, dim), dtype=float)
    for state in states_1based:
        matrix[state - 1, state - 1] = 1.0
    return matrix


def _diag_inv(M: mp.matrix, tol: mp.mpf | None = None) -> mp.matrix:
    if tol is None:
        tol = mp.mpf("1e-40")
    inv = mp.matrix(M.rows, M.cols)
    for r in range(M.rows):
        for c in range(M.cols):
            inv[r, c] = mp.mpf("0")
    for i in range(M.rows):
        if mp.fabs(M[i, i]) > tol:
            inv[i, i] = 1 / M[i, i]
    return inv


def _diag_inv_np(matrix: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    inv = np.zeros_like(matrix, dtype=float)
    for index in range(matrix.shape[0]):
        value = matrix[index, index]
        if abs(value) > tol:
            inv[index, index] = 1.0 / value
    return inv


def _np_to_mp(arr: np.ndarray) -> mp.matrix:
    rows, cols = arr.shape
    M = mp.matrix(rows, cols)
    for r in range(rows):
        for c in range(cols):
            M[r, c] = mp.mpf(str(arr[r, c]))
    return M


def _sanitise(prob: list[mp.mpf]) -> list[mp.mpf]:
    for i, value in enumerate(prob):
        if value < 0:
            prob[i] = mp.mpf("0")
    total = mp.fsum(prob)
    if total <= 0:
        raise RuntimeError("Fast-slow produced an invalid stationary distribution.")
    return [value / total for value in prob]


def _aggregate_simple(prob: list[mp.mpf], capacity: int) -> np.ndarray:
    agg = np.zeros(capacity + 1)
    agg[0] = float(prob[0])
    agg[1] = float(prob[1] + prob[2])
    for n in range(2, capacity + 1):
        agg[n] = float(prob[n + 1])
    return agg


def _aggregate_slower_additional(prob: list[mp.mpf], capacity: int) -> np.ndarray:
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

