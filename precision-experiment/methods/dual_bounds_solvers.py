from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from methods import mpmath_stationary as mps


_NZ_TOL = 1e-15
_DEFAULT_MAX_ITERATIONS = 50_000
_DEFAULT_KRYLOV_MAX_ITERATIONS = 500
_GMRES_RESTART = 50
_DENSE_EIGEN_LIMIT = 1_000

DEFAULT_MAX_ITERATIONS = _DEFAULT_MAX_ITERATIONS


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

@dataclass(frozen=True)
class _RegenerativeModel:
    transition: sp.csr_matrix
    correction: sp.csr_matrix
    integral_key: str
    integral_name: str
    matrix_memory_bytes: int
    factor_memory_bytes: int

    @property
    def memory_bytes(self) -> int:
        return self.matrix_memory_bytes + self.factor_memory_bytes


@dataclass(frozen=True)
class _DualBoundsSolveResult:
    probability: np.ndarray
    embedded: np.ndarray
    integral_key: str
    integral_name: str
    linear_key: str
    linear_name: str
    converged: bool
    iterations: int | None
    stationary_residual_l1: float
    embedded_residual_l1: float
    recovered: bool
    matrix_memory_bytes: int
    factor_memory_bytes: int
    state_count: int
    transition_nnz: int

    @property
    def solver_key(self) -> str:
        return self.linear_key

    @property
    def solver_name(self) -> str:
        return f"{self.integral_name} + {self.linear_name}"

    @property
    def memory_bytes(self) -> int:
        return self.matrix_memory_bytes + self.factor_memory_bytes


_LinearSolver = Callable[[_RegenerativeModel, float, int], _DualBoundsSolveResult]
_MPMathLinearSolver = Callable[[_RegenerativeModel, float, int, int], _DualBoundsSolveResult]

def normalise_integral_method(method: str) -> str:
    key = _method_key(method)
    _validate_integral_method(key)
    return key


def solve_model(
    model: _RegenerativeModel,
    linear_method: str = "sparse_lu",
    tolerance: float = 1e-12,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    precision_digits: int | None = None,
) -> _DualBoundsSolveResult:
    linear_key = _method_key(linear_method)
    try:
        linear_solver = _LINEAR_SOLVERS[linear_key]
    except KeyError as exc:
        valid = ", ".join(_LINEAR_SOLVERS)
        raise ValueError(f"Unsupported dual bounds linear method: {linear_method}. Valid methods: {valid}") from exc
    if precision_digits is not None:
        digits = mps.validate_precision_digits(precision_digits)
        try:
            mpmath_solver = _MPMATH_LINEAR_SOLVERS[linear_key]
        except KeyError as exc:
            valid = ", ".join(_MPMATH_LINEAR_SOLVERS)
            raise ValueError(
                f"Dual bounds linear method '{linear_key}' does not support precision_digits. "
                f"True mpmath implementations are available for: {valid}."
            ) from exc
        return mpmath_solver(model, mps.precision_tolerance(digits), max_iterations, digits)
    return linear_solver(model, tolerance, max_iterations)


def _build_expanded_model(
    topology: _ExpandedTopology,
    arrival_rate: float,
    service_time: float,
    phase_count: int,
    integral_key: str,
    precision_digits: int | None,
) -> _RegenerativeModel:
    dim = topology.dim
    delta_time = service_time / phase_count

    omega_anth, psi_anth, sa_memory = _build_sa_transients(
        topology.sa_rows, arrival_rate, delta_time, dim, integral_key, precision_digits,
    )
    omega_bhat, psi_bhat, sbhat_memory = _build_sbhat_transients(
        topology.sbhat_states, arrival_rate, delta_time, dim, integral_key,
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

    return _RegenerativeModel(
        transition=P,
        correction=C,
        integral_key=integral_key,
        integral_name=_INTEGRAL_SOLVERS[integral_key],
        matrix_memory_bytes=(
            sa_memory
            + sbhat_memory
            + _sparse_memory_bytes(M)
            + _sparse_memory_bytes(omega_power)
            + _sparse_memory_bytes(psi_a)
            + _sparse_memory_bytes(omegatilde_a)
            + _sparse_memory_bytes(omegatilde_bhat)
            + _sparse_memory_bytes(omegatilde)
            + _sparse_memory_bytes(psi)
            + _sparse_memory_bytes(qbar)
            + _sparse_memory_bytes(P)
            + _sparse_memory_bytes(C)
        ),
        factor_memory_bytes=0,
    )

def _build_sa_transients(
    sa_rows: tuple[tuple[int, tuple[int, ...]], ...],
    arrival_rate: float,
    delta_time: float,
    dim: int,
    integral_key: str,
    precision_digits: int | None,
) -> tuple[sp.csr_matrix, sp.csr_matrix, int]:
    _validate_integral_method(integral_key)
    omega = sp.lil_matrix((dim, dim), dtype=float)
    psi = sp.lil_matrix((dim, dim), dtype=float)
    prob_cache: dict[int, list[float]] = {}
    integral_cache: dict[int, np.ndarray] = {}
    dense_workspace_bytes = 0

    for row, chain in sa_rows:
        max_jumps = len(chain) - 1
        probs = prob_cache.get(max_jumps)
        if probs is None:
            probs = _birth_chain_probabilities(
                max_jumps,
                arrival_rate,
                delta_time,
                precision_digits,
            )
            prob_cache[max_jumps] = probs

        integrals = integral_cache.get(max_jumps)
        if integrals is None:
            integrals, workspace_bytes = _birth_chain_integral_row(max_jumps, arrival_rate, delta_time)
            integral_cache[max_jumps] = integrals
            dense_workspace_bytes += workspace_bytes

        for offset, value in enumerate(probs):
            omega[row, chain[offset]] = value
        for offset, value in enumerate(integrals):
            psi[row, chain[offset]] = value

    omega_csr = omega.tocsr()
    psi_csr = psi.tocsr()
    return (
        omega_csr,
        psi_csr,
        _sparse_memory_bytes(omega_csr) + _sparse_memory_bytes(psi_csr) + dense_workspace_bytes,
    )


def _build_sbhat_transients(
    sbhat_states: tuple[int, ...],
    arrival_rate: float,
    delta_time: float,
    dim: int,
    integral_key: str,
) -> tuple[sp.csr_matrix, sp.csr_matrix, int]:
    _validate_integral_method(integral_key)
    omega = sp.lil_matrix((dim, dim), dtype=float)
    psi = sp.lil_matrix((dim, dim), dtype=float)
    survival, mean_sojourn, workspace_bytes = _killing_transition_and_integral(arrival_rate, delta_time)

    for state in sbhat_states:
        omega[state, state] = survival
        psi[state, state] = mean_sojourn

    omega_csr = omega.tocsr()
    psi_csr = psi.tocsr()
    return (
        omega_csr,
        psi_csr,
        _sparse_memory_bytes(omega_csr) + _sparse_memory_bytes(psi_csr) + workspace_bytes,
    )


def _birth_chain_probabilities(
    max_jumps: int,
    arrival_rate: float,
    delta_time: float,
    precision_digits: int | None,
) -> list[float]:
    if max_jumps == 0:
        return [1.0]

    if arrival_rate <= 0.0:
        return [1.0] + [0.0] * max_jumps

    if precision_digits is None:
        hazard = float(arrival_rate) * float(delta_time)
        term = math.exp(-hazard)
        cumulative_prob = 0.0
        probs: list[float] = []
        for jumps in range(max_jumps):
            if jumps > 0:
                term *= hazard / jumps
            probs.append(term)
            cumulative_prob += term
        tail_prob = max(0.0, 1.0 - cumulative_prob)
        probs.append(tail_prob)
        return probs

    with mp.workdps(precision_digits):
        lam = mp.mpf(str(arrival_rate))
        dt = mp.mpf(str(delta_time))
        hazard = lam * dt
        term = mp.exp(-hazard)
        cumulative_prob = mp.mpf("0")
        probs_mp: list[mp.mpf] = []

        for jumps in range(max_jumps):
            if jumps > 0:
                term *= hazard / jumps
            probs_mp.append(term)
            cumulative_prob += term

        tail_prob_mp = mp.mpf("1") - cumulative_prob
        if tail_prob_mp < 0:
            tail_prob_mp = mp.mpf("0")

    probs = [float(value) for value in probs_mp]
    probs.append(float(tail_prob_mp))
    return probs


def _birth_chain_integral_row(
    max_jumps: int,
    arrival_rate: float,
    delta_time: float,
) -> tuple[np.ndarray, int]:
    size = max_jumps + 1
    generator = np.zeros((size, size), dtype=float)
    if arrival_rate > 0.0:
        for row in range(max_jumps):
            generator[row, row] = -arrival_rate
            generator[row, row + 1] = arrival_rate

    integral, workspace_bytes = _finite_integral_real_schur(generator, delta_time)
    return np.asarray(integral[0, :], dtype=float).ravel(), workspace_bytes


def _killing_transition_and_integral(
    arrival_rate: float,
    delta_time: float,
) -> tuple[float, float, int]:
    if arrival_rate <= 0.0:
        return 1.0, delta_time, int(np.dtype(float).itemsize)

    generator = np.array([[-arrival_rate]], dtype=float)
    integral, workspace_bytes = _finite_integral_real_schur(generator, delta_time)
    survival = math.exp(-arrival_rate * delta_time)
    return survival, float(integral[0, 0]), workspace_bytes


def _finite_integral_real_schur(matrix: np.ndarray, interval: float) -> tuple[np.ndarray, int]:
    n = matrix.shape[0]
    if n == 0:
        return matrix.copy(), 0

    schur_t, schur_z = la.schur(matrix, output="real")
    aug = np.zeros((2 * n, 2 * n), dtype=float)
    aug[:n, :n] = schur_t
    aug[:n, n:] = np.eye(n)
    exp_aug = la.expm(aug * float(interval))
    integral_t = exp_aug[:n, n:]
    integral = np.real_if_close(schur_z @ integral_t @ schur_z.T)
    workspace_bytes = int(
        matrix.nbytes
        + schur_t.nbytes
        + schur_z.nbytes
        + aug.nbytes
        + exp_aug.nbytes
        + integral_t.nbytes
        + integral.nbytes
    )
    return np.asarray(integral, dtype=float), workspace_bytes


def _build_one_timer_model(
    generator: np.ndarray,
    deferred: np.ndarray,
    departure: np.ndarray,
    groups: tuple[tuple[int, ...], ...],
    tau: float,
    arrival_rate: float,
    integral_key: str,
    precision_digits: int | None,
) -> _RegenerativeModel:
    _validate_integral_method(integral_key)

    dim = generator.shape[0]
    groups_list = [list(g) for g in groups]
    all_states = [s - 1 for g in groups_list for s in g]
    exp_states = {s - 1 for s in groups_list[0]}
    gen_states = [s for s in all_states if s not in exp_states]

    immediate = np.zeros((dim, dim), dtype=float)
    correction = np.zeros((dim, dim), dtype=float)
    omega = np.zeros((dim, dim), dtype=float)
    psi = np.zeros((dim, dim), dtype=float)
    dense_workspace_bytes = 0

    for state in exp_states:
        rate = -float(generator[state, state])
        if rate > 0.0:
            for col, value in enumerate(generator[state, :]):
                if col != state and abs(value) > _NZ_TOL:
                    immediate[state, col] = float(value) / rate
            correction[state, state] = 1.0 / rate
        else:
            immediate[state, state] = 1.0

    for state in gen_states:
        chain = _one_timer_birth_chain(generator, state)
        if len(chain) > 1:
            max_jumps = len(chain) - 1
            probs = _birth_chain_probabilities(max_jumps, arrival_rate, tau, precision_digits)
            integrals, workspace_bytes = _birth_chain_integral_row(max_jumps, arrival_rate, tau)
            dense_workspace_bytes += workspace_bytes
            for offset, target in enumerate(chain):
                omega[state, target] = probs[offset]
                psi[state, target] = integrals[offset]
        else:
            rate = -float(generator[state, state])
            survival, integral, workspace_bytes = _killing_transition_and_integral(rate, tau)
            dense_workspace_bytes += workspace_bytes
            omega[state, state] = survival
            psi[state, state] = integral

    transition = immediate + omega @ departure + psi @ deferred
    correction = correction + psi

    transition_sparse = sp.csr_matrix(np.real_if_close(transition))
    correction_sparse = sp.csr_matrix(np.real_if_close(correction))
    dense_bytes = int(
        generator.nbytes
        + deferred.nbytes
        + departure.nbytes
        + immediate.nbytes
        + omega.nbytes
        + psi.nbytes
        + transition.nbytes
        + correction.nbytes
    )

    return _RegenerativeModel(
        transition=transition_sparse,
        correction=correction_sparse,
        integral_key=integral_key,
        integral_name=_INTEGRAL_SOLVERS[integral_key],
        matrix_memory_bytes=dense_bytes + dense_workspace_bytes + _sparse_memory_bytes(transition_sparse) + _sparse_memory_bytes(correction_sparse),
        factor_memory_bytes=0,
    )


def _one_timer_birth_chain(generator: np.ndarray, start: int) -> list[int]:
    chain = [start]
    seen = {start}
    current = start

    while True:
        next_states = [
            col
            for col, value in enumerate(generator[current, :])
            if col != current and value > _NZ_TOL
        ]
        if len(next_states) != 1:
            return chain

        next_state = next_states[0]
        if next_state in seen:
            return chain

        chain.append(next_state)
        seen.add(next_state)
        current = next_state


build_expanded_model = _build_expanded_model
build_one_timer_model = _build_one_timer_model

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


# ---------------------------------------------------------------------------
# Embedded-chain solvers for uP = u, u e = 1
# ---------------------------------------------------------------------------

def _solve_dense_lu(
    model: _RegenerativeModel,
    tolerance: float,
    _max_iterations: int,
) -> _DualBoundsSolveResult:
    transition = model.transition
    system, rhs = _augmented_embedded_stationary_system(transition)
    dense_system = system.toarray()
    lu, pivots = la.lu_factor(dense_system)
    embedded = _normalise_signed_probability(la.lu_solve((lu, pivots), rhs))
    return _finish_regenerative_result(
        model,
        embedded,
        "dense_lu",
        "Dense LU",
        tolerance,
        iterations=None,
        matrix_memory_bytes=int(dense_system.nbytes),
        factor_memory_bytes=int(lu.nbytes + pivots.nbytes),
        converged_hint=True,
    )


def _solve_sparse_lu(
    model: _RegenerativeModel,
    tolerance: float,
    _max_iterations: int,
) -> _DualBoundsSolveResult:
    transition = model.transition
    system, rhs = _augmented_embedded_stationary_system(transition)
    lu = spla.splu(system)
    embedded = _normalise_signed_probability(lu.solve(rhs))
    return _finish_regenerative_result(
        model,
        embedded,
        "sparse_lu",
        "Sparse LU",
        tolerance,
        iterations=None,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=_lu_memory_bytes(lu),
        converged_hint=True,
    )


def _solve_sparse_lu_refinement(
    model: _RegenerativeModel,
    tolerance: float,
    _max_iterations: int,
) -> _DualBoundsSolveResult:
    transition = model.transition
    system, rhs = _augmented_embedded_stationary_system(transition)
    lu = spla.splu(system)
    embedded = _normalise_signed_probability(lu.solve(rhs))
    return _finish_regenerative_result(
        model,
        embedded,
        "sparse_lu_refinement",
        "Sparse LU refinement",
        tolerance,
        iterations=None,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=_lu_memory_bytes(lu),
        converged_hint=True,
    )


def _solve_jacobi(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    embedded, iterations, converged = _power_stationary(model.transition, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "jacobi",
        "Jacobi",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], 2),
        converged_hint=converged,
    )


def _solve_gauss_seide(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    embedded, iterations, converged = _gauss_seidel_stationary(model.transition, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "gauss_seide",
        "Gauss-Seidel",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], 2),
        converged_hint=converged,
    )


def _solve_gmres(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    system, rhs = _augmented_embedded_stationary_system(model.transition)
    embedded, iterations, converged, workspace_bytes = _gmres_system(system.tocsr(), rhs, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "gmres",
        "GMRES",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=workspace_bytes,
        converged_hint=converged,
    )


def _solve_bicgstab(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    system, rhs = _augmented_embedded_stationary_system(model.transition)
    embedded, iterations, converged, workspace_bytes = _bicgstab_system(system.tocsr(), rhs, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "bicgstab",
        "BiCGSTAB",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=workspace_bytes,
        converged_hint=converged,
    )


def _solve_power(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    embedded, iterations, converged = _power_stationary(model.transition, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "power",
        "Power method",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], 2),
        converged_hint=converged,
    )


def _solve_cesaro(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    embedded, iterations, converged = _cesaro_stationary(model.transition, tolerance, max_iterations)
    return _finish_regenerative_result(
        model,
        embedded,
        "cesaro",
        "Cesaro-averaged power",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], 3),
        converged_hint=converged,
    )


def _solve_eigenvector(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
) -> _DualBoundsSolveResult:
    embedded, eigen_gap, workspace_bytes = _eigenvector_for_unit_root(
        model.transition,
        tolerance,
        max_iterations,
    )
    return _finish_regenerative_result(
        model,
        embedded,
        "eigenvector",
        "Eigenvector of P",
        tolerance,
        iterations=None,
        matrix_memory_bytes=workspace_bytes,
        factor_memory_bytes=0,
        converged_hint=bool(eigen_gap <= max(tolerance, 1e-10)),
    )


def _solve_mpmath_dense_lu(
    model: _RegenerativeModel,
    tolerance: float,
    _max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    transition = model.transition
    system, rhs = _augmented_embedded_stationary_system(transition)
    with mp.workdps(precision_digits):
        zero_tolerance = mp.mpf(str(_NZ_TOL))
        embedded_mp, dense_bytes = mps.dense_lu_solve(
            system,
            rhs,
            lambda values: mps.normalise_signed(
                values,
                zero_tolerance,
                "Dual bounds embedded solve produced an invalid stationary vector.",
            ),
        )
        embedded = mps.vector_to_float(embedded_mp)

    return _finish_regenerative_result(
        model,
        embedded,
        "dense_lu",
        "Dense LU (mpmath)",
        tolerance,
        iterations=None,
        matrix_memory_bytes=dense_bytes,
        factor_memory_bytes=dense_bytes,
        converged_hint=True,
        allow_recovery=False,
    )


def _solve_mpmath_jacobi(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    return _solve_mpmath_power_like(
        model,
        tolerance,
        max_iterations,
        precision_digits,
        "jacobi",
        "Jacobi (mpmath)",
        vector_count=2,
        use_cesaro=False,
    )


def _solve_mpmath_sparse_lu_refinement(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    transition = model.transition
    system, rhs = _augmented_embedded_stationary_system(transition)
    lu = spla.splu(system)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    embedded, iterations, converged, residual_mp, _embedded_mp = mps.sparse_lu_refinement(
        system,
        rhs,
        lu.solve,
        precision_digits,
        tolerance,
        max_iterations,
        lambda values: mps.normalise_signed(
            values,
            zero_tolerance,
            "Dual bounds embedded solve produced an invalid stationary vector.",
        ),
    )
    embedded = _normalise_signed_probability(embedded)
    probability = np.asarray(embedded @ model.correction, dtype=float).ravel()
    probability = _normalise_nonnegative_probability(probability)

    return _DualBoundsSolveResult(
        probability=probability,
        embedded=embedded,
        integral_key=model.integral_key,
        integral_name=model.integral_name,
        linear_key="sparse_lu_refinement",
        linear_name="Sparse LU refinement (mpmath residual)",
        converged=converged,
        iterations=iterations,
        stationary_residual_l1=float(residual_mp),
        embedded_residual_l1=float(residual_mp),
        recovered=False,
        matrix_memory_bytes=model.matrix_memory_bytes + _sparse_memory_bytes(system),
        factor_memory_bytes=model.factor_memory_bytes + _lu_memory_bytes(lu) + _vector_workspace_bytes(system.shape[0], 5),
        state_count=transition.shape[0],
        transition_nnz=transition.nnz,
    )


def _solve_mpmath_power(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    return _solve_mpmath_power_like(
        model,
        tolerance,
        max_iterations,
        precision_digits,
        "power",
        "Power method (mpmath)",
        vector_count=2,
        use_cesaro=False,
    )


def _solve_mpmath_cesaro(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    return _solve_mpmath_power_like(
        model,
        tolerance,
        max_iterations,
        precision_digits,
        "cesaro",
        "Cesaro-averaged power (mpmath)",
        vector_count=3,
        use_cesaro=True,
    )


def _solve_mpmath_gauss_seide(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
) -> _DualBoundsSolveResult:
    with mp.workdps(precision_digits):
        embedded_mp, iterations, converged = _mpmath_gauss_seidel_stationary(
            model.transition,
            tolerance,
            max_iterations,
        )
        embedded = mps.vector_to_float(embedded_mp)

    return _finish_regenerative_result(
        model,
        embedded,
        "gauss_seide",
        "Gauss-Seidel (mpmath)",
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], 2),
        converged_hint=converged,
        allow_recovery=False,
    )


def _solve_mpmath_power_like(
    model: _RegenerativeModel,
    tolerance: float,
    max_iterations: int,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
    *,
    vector_count: int,
    use_cesaro: bool,
) -> _DualBoundsSolveResult:
    with mp.workdps(precision_digits):
        if use_cesaro:
            embedded_mp, iterations, converged = _mpmath_cesaro_stationary(
                model.transition,
                tolerance,
                max_iterations,
            )
        else:
            embedded_mp, iterations, converged = _mpmath_power_stationary(
                model.transition,
                tolerance,
                max_iterations,
            )
        embedded = mps.vector_to_float(embedded_mp)

    return _finish_regenerative_result(
        model,
        embedded,
        linear_key,
        linear_name,
        tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(model.transition.shape[0], vector_count),
        converged_hint=converged,
        allow_recovery=False,
    )


def _mpmath_power_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[mps.MPVector, int, bool]:
    n = transition.shape[0]
    rows = mps.sparse_rows(transition)
    probability = mps.initial_guess(n)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    tolerance_mp = mp.mpf(str(tolerance))
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability
        probability = mps.normalise_signed(
            mps.row_vector_times_sparse(previous, rows),
            zero_tolerance,
            "Dual bounds embedded solve produced an invalid stationary vector.",
        )
        if mps.l1_delta(probability, previous) <= tolerance_mp:
            converged = True
            break

    return probability, iterations, converged


def _mpmath_gauss_seidel_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[mps.MPVector, int, bool]:
    columns = mps.sparse_columns(transition)
    n = transition.shape[0]
    probability = mps.initial_guess(n)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    tolerance_mp = mp.mpf(str(tolerance))
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability.copy()
        for col in range(n):
            probability[col] = mp.fsum(value * probability[row] for row, value in columns[col])

        probability = mps.normalise_signed(
            probability,
            zero_tolerance,
            "Dual bounds embedded solve produced an invalid stationary vector.",
        )
        if mps.l1_delta(probability, previous) <= tolerance_mp:
            converged = True
            break

    return probability, iterations, converged


def _mpmath_cesaro_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[mps.MPVector, int, bool]:
    n = transition.shape[0]
    rows = mps.sparse_rows(transition)
    current = mps.initial_guess(n)
    average_sum = mps.zeros(n)
    average = current.copy()
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    tolerance_mp = mp.mpf(str(tolerance))
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        current = mps.normalise_signed(
            mps.row_vector_times_sparse(current, rows),
            zero_tolerance,
            "Dual bounds embedded solve produced an invalid stationary vector.",
        )
        average_sum = [total + value for total, value in zip(average_sum, current)]
        average = mps.normalise_signed(
            average_sum,
            zero_tolerance,
            "Dual bounds embedded solve produced an invalid stationary vector.",
        )
        if mps.embedded_residual_l1(average, rows) <= tolerance_mp:
            converged = True
            break

    return average, iterations, converged


def _finish_regenerative_result(
    model: _RegenerativeModel,
    embedded: np.ndarray,
    linear_key: str,
    linear_name: str,
    tolerance: float,
    iterations: int | None,
    matrix_memory_bytes: int,
    factor_memory_bytes: int,
    converged_hint: bool,
    allow_recovery: bool = True,
) -> _DualBoundsSolveResult:
    transition = model.transition
    correction = model.correction
    embedded = _normalise_signed_probability(embedded)
    embedded_residual_l1 = _embedded_residual_l1(embedded, transition)
    recovered = False
    if (
        allow_recovery
        and linear_key not in {"dense_lu", "sparse_lu"}
        and embedded_residual_l1 > _recovery_residual_tolerance(tolerance)
    ):
        system, rhs = _augmented_embedded_stationary_system(transition)
        lu = spla.splu(system)
        embedded = _normalise_signed_probability(lu.solve(rhs))
        embedded_residual_l1 = _embedded_residual_l1(embedded, transition)
        matrix_memory_bytes += _sparse_memory_bytes(system)
        factor_memory_bytes += _lu_memory_bytes(lu)
        converged_hint = False
        recovered = True

    probability = np.asarray(embedded @ correction, dtype=float).ravel()
    probability = _normalise_nonnegative_probability(probability)

    return _DualBoundsSolveResult(
        probability=probability,
        embedded=embedded,
        integral_key=model.integral_key,
        integral_name=model.integral_name,
        linear_key=linear_key,
        linear_name=linear_name,
        converged=bool(converged_hint and embedded_residual_l1 <= _residual_tolerance(tolerance)),
        iterations=iterations,
        stationary_residual_l1=embedded_residual_l1,
        embedded_residual_l1=embedded_residual_l1,
        recovered=recovered,
        matrix_memory_bytes=model.matrix_memory_bytes + matrix_memory_bytes,
        factor_memory_bytes=model.factor_memory_bytes + factor_memory_bytes,
        state_count=transition.shape[0],
        transition_nnz=transition.nnz,
    )


def _augmented_embedded_stationary_system(transition: sp.csr_matrix) -> tuple[sp.csc_matrix, np.ndarray]:
    n = transition.shape[0]
    system = (transition.transpose() - sp.eye(n, format="csr")).tolil()
    rhs = np.zeros(n, dtype=float)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    return system.tocsc(), rhs


def _gmres_system(
    system: sp.csr_matrix,
    rhs: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool, int]:
    n = system.shape[0]
    preconditioner, preconditioner_bytes = _lu_preconditioner(system)
    iteration_count = 0

    def count_iteration(_: float) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = spla.gmres(
        system,
        rhs,
        x0=_initial_guess(n),
        rtol=tolerance,
        atol=0.0,
        restart=min(_GMRES_RESTART, n),
        maxiter=min(max_iterations, _DEFAULT_KRYLOV_MAX_ITERATIONS),
        M=preconditioner,
        callback=count_iteration,
        callback_type="legacy",
    )
    if info > 0:
        iteration_count = max(iteration_count, info)
    return (
        np.asarray(solution, dtype=float),
        iteration_count,
        bool(info == 0),
        _vector_workspace_bytes(n, min(_GMRES_RESTART, n) + 4) + preconditioner_bytes,
    )


def _bicgstab_system(
    system: sp.csr_matrix,
    rhs: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool, int]:
    n = system.shape[0]
    preconditioner, preconditioner_bytes = _lu_preconditioner(system)
    iteration_count = 0

    def count_iteration(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    solution, info = spla.bicgstab(
        system,
        rhs,
        x0=_initial_guess(n),
        rtol=tolerance,
        atol=0.0,
        maxiter=min(max_iterations, _DEFAULT_KRYLOV_MAX_ITERATIONS),
        M=preconditioner,
        callback=count_iteration,
    )
    if info > 0:
        iteration_count = max(iteration_count, info)
    return (
        np.asarray(solution, dtype=float),
        iteration_count,
        bool(info == 0),
        _vector_workspace_bytes(n, 8) + preconditioner_bytes,
    )


def _power_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    n = transition.shape[0]
    probability = _initial_guess(n)
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability
        probability = _normalise_signed_probability(np.asarray(previous @ transition).ravel())
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            converged = True
            break

    return probability, iterations, converged


def _gauss_seidel_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    transition_t = transition.transpose().tocsr()
    n = transition.shape[0]
    probability = _initial_guess(n)
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability.copy()
        for row in range(n):
            start, end = transition_t.indptr[row], transition_t.indptr[row + 1]
            probability[row] = sum(
                float(value) * probability[col]
                for col, value in zip(transition_t.indices[start:end], transition_t.data[start:end])
            )

        probability = _normalise_signed_probability(probability)
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            converged = True
            break

    return probability, iterations, converged


def _cesaro_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    n = transition.shape[0]
    current = _initial_guess(n)
    average_sum = np.zeros(n, dtype=float)
    average = current.copy()
    converged = False
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        current = _normalise_signed_probability(np.asarray(current @ transition).ravel())
        average_sum += current
        average = _normalise_signed_probability(average_sum)
        if _embedded_residual_l1(average, transition) <= tolerance:
            converged = True
            break

    return average, iterations, converged


def _eigenvector_for_unit_root(
    transition: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, int]:
    n = transition.shape[0]
    if n <= _DENSE_EIGEN_LIMIT:
        dense = transition.transpose().toarray()
        values, vectors = la.eig(dense)
        index = int(np.argmin(np.abs(values - 1.0)))
        vector = np.real_if_close(vectors[:, index])
        eigen_gap = float(abs(values[index] - 1.0))
        return _normalise_signed_probability(vector), eigen_gap, int(dense.nbytes + vectors.nbytes + values.nbytes)

    k = min(3, n - 2)
    values, vectors = spla.eigs(
        transition.transpose(),
        k=k,
        which="LM",
        tol=max(float(tolerance), 1e-10),
        maxiter=max_iterations,
    )
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real_if_close(vectors[:, index])
    eigen_gap = float(abs(values[index] - 1.0))
    return _normalise_signed_probability(vector), eigen_gap, _vector_workspace_bytes(n, 8)

def _validate_integral_method(integral_key: str) -> None:
    if integral_key not in _INTEGRAL_SOLVERS:
        valid = ", ".join(_INTEGRAL_SOLVERS)
        raise ValueError(f"Unsupported dual bounds transition method: {integral_key}. Valid methods: {valid}")


def _method_key(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return _METHOD_ALIASES.get(key, key)


def _initial_guess(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=float)


def _normalise_signed_probability(values: np.ndarray) -> np.ndarray:
    probability = np.real(np.asarray(values, dtype=float)).ravel()
    total = float(probability.sum())
    if abs(total) <= _NZ_TOL:
        raise RuntimeError("Dual bounds embedded solve produced an invalid stationary vector.")
    if total < 0.0:
        probability = -probability
        total = -total
    return probability / total


def _normalise_nonnegative_probability(values: np.ndarray) -> np.ndarray:
    probability = np.real(np.asarray(values, dtype=float)).ravel()
    probability[probability < 0.0] = 0.0
    total = float(probability.sum())
    if total <= _NZ_TOL:
        raise RuntimeError("Dual bounds produced an invalid stationary distribution.")
    return probability / total


def _embedded_residual_l1(embedded: np.ndarray, transition: sp.csr_matrix) -> float:
    residual = np.asarray(embedded @ transition - embedded).ravel()
    return float(np.linalg.norm(residual, ord=1))


def _residual_tolerance(tolerance: float) -> float:
    return max(float(tolerance), 1e-10)


def _recovery_residual_tolerance(tolerance: float) -> float:
    return max(1_000.0 * float(tolerance), 1e-8)


def _lu_preconditioner(system: sp.csr_matrix) -> tuple[spla.LinearOperator | None, int]:
    try:
        lu = spla.splu(system.tocsc())
    except RuntimeError:
        return None, 0
    return spla.LinearOperator(system.shape, matvec=lu.solve), _lu_memory_bytes(lu)


def _vector_workspace_bytes(n: int, vector_count: int) -> int:
    return int(vector_count * n * np.dtype(float).itemsize)


def _sparse_memory_bytes(matrix: sp.spmatrix) -> int:
    if sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix):
        compressed = matrix
    else:
        compressed = matrix.tocsr()
    return int(compressed.data.nbytes + compressed.indices.nbytes + compressed.indptr.nbytes)


def _lu_memory_bytes(lu: spla.SuperLU) -> int:
    return (
        _sparse_memory_bytes(lu.L)
        + _sparse_memory_bytes(lu.U)
        + int(lu.perm_r.nbytes)
        + int(lu.perm_c.nbytes)
    )

_INTEGRAL_SOLVERS: dict[str, str] = {
    "poisson_real_schur": "Poisson birth-chain transitions + Real-Schur integrals",
}

_LINEAR_SOLVERS: dict[str, _LinearSolver] = {
    "dense_lu": _solve_dense_lu,
    "sparse_lu": _solve_sparse_lu,
    "sparse_lu_refinement": _solve_sparse_lu_refinement,
    "jacobi": _solve_jacobi,
    "gauss_seide": _solve_gauss_seide,
    "gmres": _solve_gmres,
    "bicgstab": _solve_bicgstab,
    "power": _solve_power,
    "cesaro": _solve_cesaro,
    "eigenvector": _solve_eigenvector,
}

_MPMATH_LINEAR_SOLVERS: dict[str, _MPMathLinearSolver] = {
    "dense_lu": _solve_mpmath_dense_lu,
    "sparse_lu_refinement": _solve_mpmath_sparse_lu_refinement,
    "jacobi": _solve_mpmath_jacobi,
    "gauss_seide": _solve_mpmath_gauss_seide,
    "power": _solve_mpmath_power,
    "cesaro": _solve_mpmath_cesaro,
}

_METHOD_ALIASES: dict[str, str] = {
    "poisson_birth_chain_real_schur": "poisson_real_schur",
    "poisson_birth_chain_probabilities_real_schur_integrals": "poisson_real_schur",
    "poisson_real_schur_integrals": "poisson_real_schur",
    "real_schur": "poisson_real_schur",
    "realschur": "poisson_real_schur",
    "gauss_seidel": "gauss_seide",
}

