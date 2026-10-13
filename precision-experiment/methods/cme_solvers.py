from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from methods import mpmath_stationary as mps


_NZ_TOL = 1e-15
_REAL_SCHUR_DENSE_LIMIT = 1_000
_MPMATH_GENERATOR_ITERATION_LIMIT = 10_000
_DEFAULT_MAX_ITERATIONS = 50_000
_DEFAULT_KRYLOV_MAX_ITERATIONS = 500
_GMRES_RESTART = 50


@dataclass(frozen=True)
class CMEQueueSolveResult:
    probability: np.ndarray
    integral_key: str
    integral_name: str
    linear_key: str
    linear_name: str
    converged: bool
    iterations: int | None
    stationary_residual_l1: float
    embedded_residual_l1: float | None
    matrix_memory_bytes: int
    factor_memory_bytes: int
    state_count: int
    generator_nnz: int
    level_distribution: np.ndarray | None = None

    @property
    def solver_key(self) -> str:
        return self.linear_key

    @property
    def solver_name(self) -> str:
        return f"{self.integral_name} + {self.linear_name}"

    @property
    def memory_bytes(self) -> int:
        return self.matrix_memory_bytes + self.factor_memory_bytes


@dataclass(frozen=True)
class _GeneratorParts:
    transient: sp.csr_matrix
    completion: sp.csr_matrix
    capacity: int
    phase_count: int

    @property
    def generator(self) -> sp.csr_matrix:
        return (self.transient + self.completion).tocsr()


@dataclass(frozen=True)
class _IntegralResult:
    transition: sp.csr_matrix | None
    correction: sp.csr_matrix | None
    generator: sp.csr_matrix
    transient: sp.csr_matrix | None
    completion: sp.csr_matrix | None
    capacity: int
    phase_count: int
    integral_key: str
    integral_name: str
    matrix_memory_bytes: int
    factor_memory_bytes: int
    used_embedded_chain: bool


_IntegralSolver = Callable[[_GeneratorParts], _IntegralResult]
_LinearSolver = Callable[[_IntegralResult, float], CMEQueueSolveResult]
_MPMathLinearSolver = Callable[[_IntegralResult, float, int], CMEQueueSolveResult]


def solve_queue(
    arrival_rate: float,
    capacity: int,
    initial: np.ndarray,
    phase_gen: np.ndarray,
    integral_method: str = "real_schur",
    linear_method: str = "sparse_lu",
    tolerance: float = 1e-12,
    precision_digits: int | None = None,
) -> CMEQueueSolveResult:
    parts = _build_generator_parts(arrival_rate, capacity, initial, phase_gen)
    try:
        integral_result = _INTEGRAL_SOLVERS[_method_key(integral_method)](parts)
    except KeyError as exc:
        raise ValueError(f"Unsupported CME integral method: {integral_method}") from exc

    linear_key = _method_key(linear_method)
    if precision_digits is not None:
        digits = mps.validate_precision_digits(precision_digits)
        return _solve_mpmath(
            integral_result,
            mps.precision_tolerance(digits),
            linear_key,
            digits,
        )

    try:
        linear_solver = _LINEAR_SOLVERS[linear_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported CME linear method: {linear_method}") from exc

    return linear_solver(integral_result, tolerance)


def _solve_mpmath(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    precision_digits: int,
) -> CMEQueueSolveResult:
    try:
        solver = _MPMATH_LINEAR_SOLVERS[linear_key]
    except KeyError as exc:
        valid = ", ".join(_MPMATH_LINEAR_SOLVERS)
        raise ValueError(
            f"CME linear method '{linear_key}' does not support precision_digits. "
            f"True mpmath implementations are available for: {valid}."
        ) from exc
    return solver(integral_result, tolerance, precision_digits)


def _integrate_real_schur(parts: _GeneratorParts) -> _IntegralResult:
    generator = parts.generator
    dim = parts.transient.shape[0]
    base_matrix_bytes = (
        _sparse_memory_bytes(parts.transient)
        + _sparse_memory_bytes(parts.completion)
        + _sparse_memory_bytes(generator)
    )

    if dim > _REAL_SCHUR_DENSE_LIMIT:
        return _IntegralResult(
            transition=None,
            correction=None,
            generator=generator,
            transient=parts.transient,
            completion=parts.completion,
            capacity=parts.capacity,
            phase_count=parts.phase_count,
            integral_key="real_schur",
            integral_name="Real-Schur integrals (sparse generator equivalent)",
            matrix_memory_bytes=base_matrix_bytes,
            factor_memory_bytes=0,
            used_embedded_chain=False,
        )

    q_dense = parts.transient.toarray()
    delta_dense = parts.completion.toarray()
    schur_t, schur_z = la.schur(q_dense, output="real")
    fundamental = -(schur_z @ la.solve(schur_t, schur_z.T, assume_a="gen"))
    transition = fundamental @ delta_dense

    transition_sparse = sp.csr_matrix(np.real_if_close(transition))
    correction_sparse = sp.csr_matrix(np.real_if_close(fundamental))
    dense_bytes = int(q_dense.nbytes + delta_dense.nbytes + fundamental.nbytes + transition.nbytes)

    return _IntegralResult(
        transition=transition_sparse,
        correction=correction_sparse,
        generator=generator,
        transient=parts.transient,
        completion=parts.completion,
        capacity=parts.capacity,
        phase_count=parts.phase_count,
        integral_key="real_schur",
        integral_name="Real-Schur integrals",
        matrix_memory_bytes=(
            base_matrix_bytes
            + dense_bytes
            + _sparse_memory_bytes(transition_sparse)
            + _sparse_memory_bytes(correction_sparse)
        ),
        factor_memory_bytes=0,
        used_embedded_chain=True,
    )


def _solve_dense_lu(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        return _solve_embedded_dense_lu(integral_result, tolerance, "dense_lu", "Dense LU")
    return _solve_generator_dense_lu(integral_result, tolerance, "dense_lu", "Dense LU")


def _solve_sparse_lu(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        return _solve_embedded_sparse_lu(integral_result, tolerance, "sparse_lu", "Sparse LU")
    return _solve_generator_sparse_lu(integral_result, tolerance, "sparse_lu", "Sparse LU")


def _solve_sparse_lu_refinement(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        return _solve_embedded_sparse_lu(
            integral_result,
            tolerance,
            "sparse_lu_refinement",
            "Sparse LU refinement",
        )
    return _solve_implicit_embedded_power(
        integral_result,
        tolerance,
        "sparse_lu_refinement",
        "Sparse LU refinement (implicit regenerative)",
    )


def _solve_mpmath_dense_lu(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    with mp.workdps(precision_digits):
        zero_tolerance = mp.mpf(str(_NZ_TOL))
        if integral_result.used_embedded_chain:
            transition, _correction = _embedded_matrices(integral_result)
            system, rhs = _augmented_embedded_stationary_system(transition)
            embedded_mp, dense_bytes = mps.dense_lu_solve(
                system,
                rhs,
                lambda values: mps.normalise_by_sum(
                    values,
                    zero_tolerance,
                    "CME embedded solve produced an invalid stationary vector.",
                ),
            )
            embedded = mps.vector_to_float(embedded_mp)
            return _finish_embedded_result(
                integral_result=integral_result,
                embedded=embedded,
                linear_key="dense_lu",
                linear_name="Dense LU (mpmath)",
                tolerance=tolerance,
                iterations=None,
                matrix_memory_bytes=dense_bytes,
                factor_memory_bytes=dense_bytes,
                converged_hint=True,
            )

        generator = integral_result.generator.tocsr()
        system, rhs = _augmented_stationary_system(generator)
        probability_mp, dense_bytes = mps.dense_lu_solve(
            system,
            rhs,
            lambda values: mps.normalise_by_sum(
                values,
                zero_tolerance,
                "CME solve produced an invalid probability vector.",
            ),
        )
        probability = mps.vector_to_float(probability_mp)
        return _finish_generator_result(
            integral_result=integral_result,
            probability=probability,
            linear_key="dense_lu",
            linear_name="Dense LU (mpmath)",
            tolerance=tolerance,
            iterations=None,
            matrix_memory_bytes=dense_bytes,
            factor_memory_bytes=dense_bytes,
            converged_hint=True,
        )


def _solve_mpmath_jacobi(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    if not integral_result.used_embedded_chain:
        return _solve_mpmath_generator_jacobi(
            integral_result,
            tolerance,
            precision_digits,
            "jacobi",
            "Jacobi (mpmath sparse generator)",
        )

    return _solve_mpmath_embedded_power_like(
        integral_result,
        tolerance,
        precision_digits,
        "jacobi",
        "Jacobi (mpmath)",
        vector_count=2,
        use_cesaro=False,
    )


def _solve_mpmath_sparse_lu_refinement(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        return _solve_mpmath_embedded_sparse_lu_refinement(integral_result, tolerance, precision_digits)
    return _solve_implicit_embedded_power(
        integral_result,
        max(float(tolerance), 1e-12),
        "sparse_lu_refinement",
        "Sparse LU refinement (implicit regenerative)",
    )


def _solve_mpmath_power(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    if not integral_result.used_embedded_chain:
        return _solve_implicit_embedded_power(
            integral_result,
            tolerance,
            "power",
            "Power method (implicit sparse LU)",
        )

    return _solve_mpmath_embedded_power_like(
        integral_result,
        tolerance,
        precision_digits,
        "power",
        "Power method (mpmath)",
        vector_count=2,
        use_cesaro=False,
    )


def _solve_mpmath_power_generator(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    # Diagnostic path: exact digit-controlled arithmetic, but very slow for
    # large concentrated ME models because it iterates on the signed generator.
    return _solve_mpmath_generator_power(
        integral_result,
        tolerance,
        precision_digits,
        "power_mpmath_generator",
        "Power method (mpmath sparse uniformized generator)",
    )


def _solve_mpmath_cesaro(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    if not integral_result.used_embedded_chain:
        return _solve_mpmath_generator_cesaro(
            integral_result,
            tolerance,
            precision_digits,
            "cesaro",
            "Cesaro-averaged power (mpmath sparse uniformized generator)",
        )

    return _solve_mpmath_embedded_power_like(
        integral_result,
        tolerance,
        precision_digits,
        "cesaro",
        "Cesaro-averaged power (mpmath)",
        vector_count=3,
        use_cesaro=True,
    )


def _solve_mpmath_gauss_seide(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    if not integral_result.used_embedded_chain:
        return _solve_mpmath_generator_gauss_seidel(
            integral_result,
            tolerance,
            precision_digits,
            "gauss_seide",
            "Gauss-Seidel (mpmath sparse generator)",
        )

    transition = _require_mpmath_embedded_transition(integral_result, "gauss_seide")
    with mp.workdps(precision_digits):
        embedded_mp, iterations, converged = _mpmath_gauss_seidel_stationary(
            transition,
            tolerance,
            _DEFAULT_MAX_ITERATIONS,
        )
        embedded = mps.vector_to_float(embedded_mp)

    return _finish_embedded_result(
        integral_result=integral_result,
        embedded=embedded,
        linear_key="gauss_seide",
        linear_name="Gauss-Seidel (mpmath)",
        tolerance=tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], 2),
        converged_hint=converged,
    )


def _solve_mpmath_embedded_power_like(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
    *,
    vector_count: int,
    use_cesaro: bool,
) -> CMEQueueSolveResult:
    transition = _require_mpmath_embedded_transition(integral_result, linear_key)
    with mp.workdps(precision_digits):
        if use_cesaro:
            embedded_mp, iterations, converged = _mpmath_cesaro_stationary(
                transition,
                tolerance,
                _DEFAULT_MAX_ITERATIONS,
            )
        else:
            embedded_mp, iterations, converged = _mpmath_power_stationary(
                transition,
                tolerance,
                _DEFAULT_MAX_ITERATIONS,
            )
        embedded = mps.vector_to_float(embedded_mp)

    return _finish_embedded_result(
        integral_result=integral_result,
        embedded=embedded,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        matrix_memory_bytes=0,
        factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], vector_count),
        converged_hint=converged,
    )


def _require_mpmath_embedded_transition(
    integral_result: _IntegralResult,
    linear_key: str,
) -> sp.csr_matrix:
    if integral_result.transition is None or integral_result.correction is None:
        raise ValueError(
            f"CME linear method '{linear_key}' with precision_digits requires an "
            f"embedded transition/correction implementation. Integral method "
            f"'{integral_result.integral_key}' produced a generator-only model."
        )
    return integral_result.transition.tocsr()


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
        probability = mps.normalise_by_sum(
            mps.row_vector_times_sparse(previous, rows),
            zero_tolerance,
            "CME embedded solve produced an invalid stationary vector.",
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

        probability = mps.normalise_by_sum(
            probability,
            zero_tolerance,
            "CME embedded solve produced an invalid stationary vector.",
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
        current = mps.normalise_by_sum(
            mps.row_vector_times_sparse(current, rows),
            zero_tolerance,
            "CME embedded solve produced an invalid stationary vector.",
        )
        average_sum = [total + value for total, value in zip(average_sum, current)]
        average = mps.normalise_by_sum(
            average_sum,
            zero_tolerance,
            "CME embedded solve produced an invalid stationary vector.",
        )
        if mps.embedded_residual_l1(average, rows) <= tolerance_mp:
            converged = True
            break

    return average, iterations, converged


def _solve_mpmath_generator_jacobi(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_generator_context(integral_result.generator, tolerance)
        probability = mps.initial_guess(context["n"])
        iterations = 0

        for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
            previous = probability
            probability = [
                mps.stationary_iteration_value(
                    context["columns"],
                    previous,
                    previous,
                    col,
                    "direct_q",
                    context["zero_tolerance"],
                )
                for col in range(context["n"])
            ]
            probability = mps.normalise_by_sum(
                probability,
                context["zero_tolerance"],
                "CME solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return _finish_mpmath_generator_result(
        integral_result=integral_result,
        probability=probability_np,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        residual_mp=residual_mp,
        vector_count=3,
    )


def _solve_mpmath_generator_gauss_seidel(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_generator_context(integral_result.generator, tolerance)
        probability = mps.initial_guess(context["n"])
        iterations = 0

        for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
            previous = probability.copy()
            for col in range(context["n"]):
                probability[col] = mps.stationary_iteration_value(
                    context["columns"],
                    probability,
                    previous,
                    col,
                    "direct_q",
                    context["zero_tolerance"],
                )
            probability = mps.normalise_by_sum(
                probability,
                context["zero_tolerance"],
                "CME solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return _finish_mpmath_generator_result(
        integral_result=integral_result,
        probability=probability_np,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        residual_mp=residual_mp,
        vector_count=2,
    )


def _solve_mpmath_generator_power(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_generator_context(integral_result.generator, tolerance)
        probability = mps.initial_guess(context["n"])
        iterations = 0

        for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
            previous = probability
            generator_step = mps.row_vector_times_sparse(previous, context["generator_rows"])
            probability = [
                previous[index] + generator_step[index] / context["uniformization_rate"]
                for index in range(context["n"])
            ]
            probability = mps.normalise_by_sum(
                probability,
                context["zero_tolerance"],
                "CME solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return _finish_mpmath_generator_result(
        integral_result=integral_result,
        probability=probability_np,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        residual_mp=residual_mp,
        vector_count=2,
    )


def _solve_mpmath_generator_cesaro(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_generator_context(integral_result.generator, tolerance)
        current = mps.initial_guess(context["n"])
        average_sum = mps.zeros(context["n"])
        average = current.copy()
        iterations = 0

        for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
            generator_step = mps.row_vector_times_sparse(current, context["generator_rows"])
            current = [
                current[index] + generator_step[index] / context["uniformization_rate"]
                for index in range(context["n"])
            ]
            current = mps.normalise_by_sum(
                current,
                context["zero_tolerance"],
                "CME solve produced an invalid probability vector.",
            )
            average_sum = [total + value for total, value in zip(average_sum, current)]
            average = mps.normalise_by_sum(
                average_sum,
                context["zero_tolerance"],
                "CME solve produced an invalid probability vector.",
            )
            if mps.stationary_residual_l1(average, context["generator_rows"]) <= context["residual_limit"]:
                break

        residual_mp = mps.stationary_residual_l1(average, context["generator_rows"])
        probability_np = mps.vector_to_float(average)

    return _finish_mpmath_generator_result(
        integral_result=integral_result,
        probability=probability_np,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        residual_mp=residual_mp,
        vector_count=3,
    )


def _mpmath_generator_context(generator: sp.spmatrix, tolerance: float) -> dict[str, object]:
    generator = generator.tocsr()
    if generator.shape[0] > _MPMATH_GENERATOR_ITERATION_LIMIT:
        raise ValueError(
            "True mpmath CME sparse generator iteration is only enabled for "
            f"models with at most {_MPMATH_GENERATOR_ITERATION_LIMIT} states. "
            f"This model has {generator.shape[0]} states; use a smaller CME/capacity "
            "or the existing implicit sparse LU path."
        )
    return {
        "generator": generator,
        "columns": mps.sparse_columns(generator),
        "generator_rows": mps.sparse_rows(generator),
        "n": generator.shape[0],
        "uniformization_rate": mp.mpf(str(_uniformization_rate(generator))),
        "tolerance": mp.mpf(str(tolerance)),
        "residual_limit": mp.mpf(str(tolerance)),
        "zero_tolerance": mp.mpf(str(_NZ_TOL)),
    }


def _finish_mpmath_generator_result(
    *,
    integral_result: _IntegralResult,
    probability: np.ndarray,
    linear_key: str,
    linear_name: str,
    tolerance: float,
    iterations: int | None,
    residual_mp: mp.mpf,
    vector_count: int,
) -> CMEQueueSolveResult:
    generator = integral_result.generator
    residual_limit = mp.mpf(str(tolerance))
    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=f"{integral_result.integral_name} + direct sparse generator iteration",
        linear_key=linear_key,
        linear_name=linear_name,
        converged=bool(residual_mp <= residual_limit),
        iterations=iterations,
        stationary_residual_l1=float(residual_mp),
        embedded_residual_l1=None,
        matrix_memory_bytes=integral_result.matrix_memory_bytes,
        factor_memory_bytes=_vector_workspace_bytes(generator.shape[0], vector_count),
        state_count=generator.shape[0],
        generator_nnz=generator.nnz,
    )


def _solve_jacobi(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        embedded, iterations, converged = _power_stationary(transition, tolerance)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key="jacobi",
            linear_name="Jacobi",
            tolerance=tolerance,
            iterations=iterations,
            matrix_memory_bytes=0,
            factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], 2),
            converged_hint=converged,
        )

    return _recover_generator_result(integral_result, tolerance, "jacobi", "Jacobi", iterations=0)


def _solve_gauss_seide(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        embedded, iterations, converged = _gauss_seidel_stationary(transition, tolerance)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key="gauss_seide",
            linear_name="Gauss-Seidel",
            tolerance=tolerance,
            iterations=iterations,
            matrix_memory_bytes=0,
            factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], 2),
            converged_hint=converged,
        )

    return _recover_generator_result(
        integral_result,
        tolerance,
        "gauss_seide",
        "Gauss-Seidel",
        iterations=0,
    )


def _solve_gmres(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    return _solve_augmented_system_iterative(
        integral_result,
        tolerance,
        "gmres",
        "GMRES",
        _gmres_system,
    )


def _solve_bicgstab(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    return _solve_augmented_system_iterative(
        integral_result,
        tolerance,
        "bicgstab",
        "BiCGSTAB",
        _bicgstab_system,
    )


def _solve_power(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        embedded, iterations, converged = _power_stationary(transition, tolerance)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key="power",
            linear_name="Power method",
            tolerance=tolerance,
            iterations=iterations,
            matrix_memory_bytes=0,
            factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], 2),
            converged_hint=converged,
        )

    return _solve_implicit_embedded_power(
        integral_result,
        tolerance,
        "power",
        "Power method",
    )


def _solve_cesaro(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        embedded, iterations, converged = _cesaro_stationary(transition, tolerance)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key="cesaro",
            linear_name="Cesaro-averaged power",
            tolerance=tolerance,
            iterations=iterations,
            matrix_memory_bytes=0,
            factor_memory_bytes=_vector_workspace_bytes(transition.shape[0], 3),
            converged_hint=converged,
        )

    return _recover_generator_result(
        integral_result,
        tolerance,
        "cesaro",
        "Cesaro-averaged power",
        iterations=0,
    )


def _solve_eigenvector(integral_result: _IntegralResult, tolerance: float) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        embedded, eigen_gap, dense_bytes = _eigenvector_for_unit_root(transition)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key="eigenvector",
            linear_name="Eigenvector of P",
            tolerance=tolerance,
            iterations=None,
            matrix_memory_bytes=dense_bytes,
            factor_memory_bytes=0,
            converged_hint=bool(eigen_gap <= max(tolerance, 1e-10)),
        )

    generator = integral_result.generator.tocsr()
    try:
        probability, eigen_gap, workspace_bytes = _eigenvector_for_zero_root(generator, tolerance)
    except Exception:
        return _recover_generator_result(
            integral_result,
            tolerance,
            "eigenvector",
            "Eigenvector",
            iterations=None,
        )
    return _finish_generator_result(
        integral_result=integral_result,
        probability=probability,
        linear_key="eigenvector",
        linear_name="Eigenvector",
        tolerance=tolerance,
        iterations=None,
        matrix_memory_bytes=workspace_bytes,
        factor_memory_bytes=0,
        converged_hint=bool(eigen_gap <= max(tolerance, 1e-10)),
    )


def _solve_implicit_embedded_power(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    transient, completion = _regenerative_matrices(integral_result)
    lu = spla.splu(transient.transpose().tocsc())
    n = transient.shape[0]
    embedded = _initial_guess(n)
    converged = False
    iterations = 0

    for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
        previous = embedded
        correction = lu.solve(-previous)
        embedded = _normalise_by_sum(np.asarray(correction @ completion).ravel())
        if np.linalg.norm(embedded - previous, ord=1) <= tolerance:
            converged = True
            break

    correction = np.asarray(lu.solve(-embedded), dtype=float).ravel()
    embedded_next = np.asarray(correction @ completion).ravel()
    embedded_residual_l1 = float(np.linalg.norm(embedded_next - embedded, ord=1))
    probability = _normalise(correction)
    stationary_residual_l1 = _stationary_residual_l1(probability, integral_result.generator)
    factor_memory_bytes = _lu_memory_bytes(lu) + _vector_workspace_bytes(n, 3)

    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=f"{integral_result.integral_name} + implicit regenerative action",
        linear_key=linear_key,
        linear_name=linear_name,
        converged=bool(
            converged
            and embedded_residual_l1 <= tolerance
            and stationary_residual_l1 <= _generator_residual_tolerance(tolerance)
        ),
        iterations=iterations,
        stationary_residual_l1=stationary_residual_l1,
        embedded_residual_l1=embedded_residual_l1,
        matrix_memory_bytes=integral_result.matrix_memory_bytes,
        factor_memory_bytes=integral_result.factor_memory_bytes + factor_memory_bytes,
        state_count=integral_result.generator.shape[0],
        generator_nnz=integral_result.generator.nnz,
    )


def _solve_embedded_dense_lu(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    transition, _correction = _embedded_matrices(integral_result)
    system, rhs = _augmented_embedded_stationary_system(transition)
    dense_system = system.toarray()
    lu, pivots = la.lu_factor(dense_system)
    embedded = _normalise_by_sum(la.lu_solve((lu, pivots), rhs))
    return _finish_embedded_result(
        integral_result=integral_result,
        embedded=embedded,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=None,
        matrix_memory_bytes=int(dense_system.nbytes),
        factor_memory_bytes=int(lu.nbytes + pivots.nbytes),
        converged_hint=True,
    )


def _solve_embedded_sparse_lu(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    transition, _correction = _embedded_matrices(integral_result)
    system, rhs = _augmented_embedded_stationary_system(transition)
    system_csc = system.tocsc()
    lu = spla.splu(system_csc)
    embedded = _normalise_by_sum(lu.solve(rhs))
    return _finish_embedded_result(
        integral_result=integral_result,
        embedded=embedded,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=None,
        matrix_memory_bytes=_sparse_memory_bytes(system_csc),
        factor_memory_bytes=_lu_memory_bytes(lu),
        converged_hint=True,
    )


def _solve_generator_dense_lu(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    generator = integral_result.generator.tocsr()
    system, rhs = _augmented_stationary_system(generator)
    dense_system = system.toarray()
    lu, pivots = la.lu_factor(dense_system)
    probability = _normalise(la.lu_solve((lu, pivots), rhs))
    return _finish_generator_result(
        integral_result=integral_result,
        probability=probability,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=None,
        matrix_memory_bytes=int(dense_system.nbytes),
        factor_memory_bytes=int(lu.nbytes + pivots.nbytes),
        converged_hint=True,
    )


def _solve_generator_sparse_lu(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
) -> CMEQueueSolveResult:
    generator = integral_result.generator.tocsr()
    system, rhs = _augmented_stationary_system(generator)

    lu = spla.splu(system)
    probability = _normalise(lu.solve(rhs))
    return _finish_generator_result(
        integral_result=integral_result,
        probability=probability,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=None,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=_lu_memory_bytes(lu),
        converged_hint=True,
    )


def _solve_mpmath_embedded_sparse_lu_refinement(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    transition, correction = _embedded_matrices(integral_result)
    system, rhs = _augmented_embedded_stationary_system(transition)
    lu = spla.splu(system)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    embedded, iterations, converged, residual_mp, _embedded_mp = mps.sparse_lu_refinement(
        system,
        rhs,
        lu.solve,
        precision_digits,
        tolerance,
        _DEFAULT_MAX_ITERATIONS,
        lambda values: mps.normalise_by_sum(
            values,
            zero_tolerance,
            "CME embedded solve produced an invalid stationary vector.",
        ),
    )
    embedded = _normalise_by_sum(embedded)
    probability = _normalise(np.asarray(embedded @ correction, dtype=float).ravel())
    stationary_residual_l1 = _stationary_residual_l1(probability, integral_result.generator)

    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=integral_result.integral_name,
        linear_key="sparse_lu_refinement",
        linear_name="Sparse LU refinement (mpmath residual)",
        converged=bool(
            converged
            and stationary_residual_l1 <= _generator_residual_tolerance(tolerance)
        ),
        iterations=iterations,
        stationary_residual_l1=stationary_residual_l1,
        embedded_residual_l1=float(residual_mp),
        matrix_memory_bytes=integral_result.matrix_memory_bytes + _sparse_memory_bytes(system),
        factor_memory_bytes=integral_result.factor_memory_bytes + _lu_memory_bytes(lu) + _vector_workspace_bytes(system.shape[0], 5),
        state_count=integral_result.generator.shape[0],
        generator_nnz=integral_result.generator.nnz,
        level_distribution=None,
    )


def _solve_mpmath_generator_sparse_lu_refinement(
    integral_result: _IntegralResult,
    tolerance: float,
    precision_digits: int,
) -> CMEQueueSolveResult:
    generator = integral_result.generator.tocsr()
    system, rhs = _augmented_stationary_system(generator)
    lu = spla.splu(system)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    probability, iterations, converged, residual_mp, probability_mp = mps.sparse_lu_refinement(
        system,
        rhs,
        lu.solve,
        precision_digits,
        tolerance,
        _DEFAULT_MAX_ITERATIONS,
        lambda values: mps.normalise_by_sum(
            values,
            zero_tolerance,
            "CME solve produced an invalid probability vector.",
        ),
    )
    probability = _normalise(probability)

    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=integral_result.integral_name,
        linear_key="sparse_lu_refinement",
        linear_name="Sparse LU refinement (mpmath residual)",
        converged=converged,
        iterations=iterations,
        stationary_residual_l1=float(residual_mp),
        embedded_residual_l1=None,
        matrix_memory_bytes=integral_result.matrix_memory_bytes + _sparse_memory_bytes(system),
        factor_memory_bytes=integral_result.factor_memory_bytes + _lu_memory_bytes(lu) + _vector_workspace_bytes(system.shape[0], 5),
        state_count=generator.shape[0],
        generator_nnz=generator.nnz,
        level_distribution=_mpmath_level_distribution(
            probability_mp,
            integral_result.capacity,
            integral_result.phase_count,
        ),
    )


def _solve_augmented_system_iterative(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
    solver: Callable[[sp.csr_matrix, np.ndarray, float], tuple[np.ndarray, int, bool, int]],
) -> CMEQueueSolveResult:
    if integral_result.used_embedded_chain:
        transition, _correction = _embedded_matrices(integral_result)
        system, rhs = _augmented_embedded_stationary_system(transition)
        embedded, iterations, converged, workspace_bytes = solver(system.tocsr(), rhs, tolerance)
        return _finish_embedded_result(
            integral_result=integral_result,
            embedded=embedded,
            linear_key=linear_key,
            linear_name=linear_name,
            tolerance=tolerance,
            iterations=iterations,
            matrix_memory_bytes=_sparse_memory_bytes(system),
            factor_memory_bytes=workspace_bytes,
            converged_hint=converged,
        )

    generator = integral_result.generator.tocsr()
    system, rhs = _augmented_stationary_system(generator)
    probability, iterations, converged, workspace_bytes = solver(system.tocsr(), rhs, tolerance)
    return _finish_generator_result(
        integral_result=integral_result,
        probability=probability,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=workspace_bytes,
        converged_hint=converged,
    )


def _finish_embedded_result(
    integral_result: _IntegralResult,
    embedded: np.ndarray,
    linear_key: str,
    linear_name: str,
    tolerance: float,
    iterations: int | None,
    matrix_memory_bytes: int,
    factor_memory_bytes: int,
    converged_hint: bool,
) -> CMEQueueSolveResult:
    transition, correction = _embedded_matrices(integral_result)
    embedded = _normalise_by_sum(embedded)
    embedded_residual_l1 = _embedded_residual_l1(embedded, transition)

    probability = np.asarray(embedded @ correction, dtype=float).ravel()
    probability = _normalise(probability)
    stationary_residual_l1 = _stationary_residual_l1(probability, integral_result.generator)

    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=integral_result.integral_name,
        linear_key=linear_key,
        linear_name=linear_name,
        converged=bool(
            converged_hint
            and embedded_residual_l1 <= tolerance
            and stationary_residual_l1 <= _generator_residual_tolerance(tolerance)
        ),
        iterations=iterations,
        stationary_residual_l1=stationary_residual_l1,
        embedded_residual_l1=embedded_residual_l1,
        matrix_memory_bytes=integral_result.matrix_memory_bytes + matrix_memory_bytes,
        factor_memory_bytes=integral_result.factor_memory_bytes + factor_memory_bytes,
        state_count=integral_result.generator.shape[0],
        generator_nnz=integral_result.generator.nnz,
    )


def _finish_generator_result(
    integral_result: _IntegralResult,
    probability: np.ndarray,
    linear_key: str,
    linear_name: str,
    tolerance: float,
    iterations: int | None,
    matrix_memory_bytes: int,
    factor_memory_bytes: int,
    converged_hint: bool,
) -> CMEQueueSolveResult:
    generator = integral_result.generator.tocsr()
    probability = _normalise(probability)
    stationary_residual_l1 = _stationary_residual_l1(probability, generator)
    if (
        linear_key not in {"dense_lu", "sparse_lu"}
        and stationary_residual_l1 > _recovery_residual_tolerance(tolerance)
    ):
        system, rhs = _augmented_stationary_system(generator)
        lu = spla.splu(system)
        probability = _normalise(lu.solve(rhs))
        stationary_residual_l1 = _stationary_residual_l1(probability, generator)
        matrix_memory_bytes += _sparse_memory_bytes(system)
        factor_memory_bytes += _lu_memory_bytes(lu)
        converged_hint = False

    return CMEQueueSolveResult(
        probability=probability,
        integral_key=integral_result.integral_key,
        integral_name=integral_result.integral_name,
        linear_key=linear_key,
        linear_name=linear_name,
        converged=bool(
            converged_hint
            and stationary_residual_l1 <= _generator_residual_tolerance(tolerance)
        ),
        iterations=iterations,
        stationary_residual_l1=stationary_residual_l1,
        embedded_residual_l1=None,
        matrix_memory_bytes=integral_result.matrix_memory_bytes + matrix_memory_bytes,
        factor_memory_bytes=integral_result.factor_memory_bytes + factor_memory_bytes,
        state_count=generator.shape[0],
        generator_nnz=generator.nnz,
    )


def _recover_generator_result(
    integral_result: _IntegralResult,
    tolerance: float,
    linear_key: str,
    linear_name: str,
    iterations: int | None,
) -> CMEQueueSolveResult:
    generator = integral_result.generator.tocsr()
    system, rhs = _augmented_stationary_system(generator)
    lu = spla.splu(system)
    probability = _normalise(lu.solve(rhs))
    return _finish_generator_result(
        integral_result=integral_result,
        probability=probability,
        linear_key=linear_key,
        linear_name=linear_name,
        tolerance=tolerance,
        iterations=iterations,
        matrix_memory_bytes=_sparse_memory_bytes(system),
        factor_memory_bytes=_lu_memory_bytes(lu),
        converged_hint=False,
    )


def _build_generator_parts(
    arrival_rate: float,
    capacity: int,
    initial: np.ndarray,
    phase_gen: np.ndarray,
) -> _GeneratorParts:
    pc = len(initial)
    completion = -phase_gen @ np.ones(pc)
    alpha_nz = tuple((j, float(v)) for j, v in enumerate(initial) if abs(v) > _NZ_TOL)
    phase_nz = tuple(
        tuple((j, float(v)) for j, v in enumerate(phase_gen[i]) if j != i and abs(v) > _NZ_TOL)
        for i in range(pc)
    )

    dim = 1 + pc + max(0, capacity - 1) * pc * pc
    q_rows: list[int] = []
    q_cols: list[int] = []
    q_data: list[float] = []
    d_rows: list[int] = []
    d_cols: list[int] = []
    d_data: list[float] = []

    def add_q(r: int, c: int, v: float) -> None:
        if abs(v) > _NZ_TOL:
            q_rows.append(r)
            q_cols.append(c)
            q_data.append(float(v))

    def add_d(r: int, c: int, v: float) -> None:
        if abs(v) > _NZ_TOL:
            d_rows.append(r)
            d_cols.append(c)
            d_data.append(float(v))

    def s0() -> int:
        return 0

    def s1(i: int) -> int:
        return 1 + i

    def sboth(n: int, a: int, b: int) -> int:
        return 1 + pc + (n - 2) * pc * pc + a * pc + b

    if capacity <= 0:
        return _GeneratorParts(
            transient=sp.csr_matrix((1, 1), dtype=float),
            completion=sp.csr_matrix((1, 1), dtype=float),
            capacity=capacity,
            phase_count=pc,
        )

    add_q(s0(), s0(), -arrival_rate)
    for j, alpha_j in alpha_nz:
        add_q(s0(), s1(j), arrival_rate * alpha_j)

    for i in range(pc):
        state = s1(i)

        if capacity >= 2:
            add_q(state, state, -arrival_rate)
            for j, alpha_j in alpha_nz:
                add_q(state, sboth(2, i, j), arrival_rate * alpha_j)

        for j, qij in phase_nz[i]:
            add_q(state, s1(j), qij)

        add_q(state, state, float(phase_gen[i, i]))
        add_d(state, s0(), float(completion[i]))

    for n in range(2, capacity + 1):
        for a in range(pc):
            for b in range(pc):
                state = sboth(n, a, b)

                if n < capacity:
                    add_q(state, state, -arrival_rate)
                    add_q(state, sboth(n + 1, a, b), arrival_rate)

                for j, qaj in phase_nz[a]:
                    add_q(state, sboth(n, j, b), qaj)
                for j, qbj in phase_nz[b]:
                    add_q(state, sboth(n, a, j), qbj)

                add_q(state, state, float(phase_gen[a, a] + phase_gen[b, b]))

                if n == 2:
                    add_d(state, s1(b), float(completion[a]))
                    add_d(state, s1(a), float(completion[b]))
                else:
                    for j, alpha_j in alpha_nz:
                        add_d(state, sboth(n - 1, j, b), float(completion[a] * alpha_j))
                        add_d(state, sboth(n - 1, a, j), float(completion[b] * alpha_j))

    return _GeneratorParts(
        transient=sp.coo_matrix((q_data, (q_rows, q_cols)), shape=(dim, dim)).tocsr(),
        completion=sp.coo_matrix((d_data, (d_rows, d_cols)), shape=(dim, dim)).tocsr(),
        capacity=capacity,
        phase_count=pc,
    )


def _augmented_stationary_system(generator: sp.csr_matrix) -> tuple[sp.csc_matrix, np.ndarray]:
    n = generator.shape[0]
    system = generator.transpose().tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    return system.tocsc(), rhs


def _augmented_embedded_stationary_system(transition: sp.csr_matrix) -> tuple[sp.csc_matrix, np.ndarray]:
    n = transition.shape[0]
    system = (transition.transpose() - sp.eye(n, format="csr")).tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    return system.tocsc(), rhs


def _embedded_matrices(integral_result: _IntegralResult) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if integral_result.transition is None or integral_result.correction is None:
        raise RuntimeError("This CME linear method needs embedded transition and correction matrices.")
    return integral_result.transition.tocsr(), integral_result.correction.tocsr()


def _regenerative_matrices(integral_result: _IntegralResult) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    if integral_result.transient is None or integral_result.completion is None:
        raise RuntimeError("This CME linear method needs regenerative transient/completion matrices.")
    return integral_result.transient.tocsr(), integral_result.completion.tocsr()


def _gmres_system(
    system: sp.csr_matrix,
    rhs: np.ndarray,
    tolerance: float,
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
        maxiter=_DEFAULT_KRYLOV_MAX_ITERATIONS,
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
        maxiter=_DEFAULT_KRYLOV_MAX_ITERATIONS,
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
) -> tuple[np.ndarray, int, bool]:
    n = transition.shape[0]
    probability = _initial_guess(n)
    converged = False
    iterations = 0

    for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
        previous = probability
        probability = _normalise_by_sum(np.asarray(previous @ transition).ravel())
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            converged = True
            break

    return probability, iterations, converged


def _gauss_seidel_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    transition_t = transition.transpose().tocsr()
    n = transition.shape[0]
    probability = _initial_guess(n)
    converged = False
    iterations = 0

    for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
        previous = probability.copy()
        for row in range(n):
            start, end = transition_t.indptr[row], transition_t.indptr[row + 1]
            probability[row] = sum(
                float(value) * probability[col]
                for col, value in zip(transition_t.indices[start:end], transition_t.data[start:end])
            )

        probability = _normalise_by_sum(probability)
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            converged = True
            break

    return probability, iterations, converged


def _cesaro_stationary(
    transition: sp.csr_matrix,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    n = transition.shape[0]
    current = _initial_guess(n)
    average_sum = np.zeros(n, dtype=float)
    average = current.copy()
    converged = False
    iterations = 0

    for iterations in range(1, _DEFAULT_MAX_ITERATIONS + 1):
        current = _normalise_by_sum(np.asarray(current @ transition).ravel())
        average_sum += current
        average = _normalise_by_sum(average_sum)
        if _embedded_residual_l1(average, transition) <= tolerance:
            converged = True
            break

    return average, iterations, converged


def _eigenvector_for_unit_root(transition: sp.csr_matrix) -> tuple[np.ndarray, float, int]:
    dense = transition.transpose().toarray()
    values, vectors = la.eig(dense)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real_if_close(vectors[:, index])
    vector = np.asarray(vector.real if np.iscomplexobj(vector) else vector, dtype=float)
    eigen_gap = float(abs(values[index] - 1.0))
    return _normalise_by_sum(vector), eigen_gap, int(dense.nbytes + vectors.nbytes + values.nbytes)


def _eigenvector_for_zero_root(
    generator: sp.csr_matrix,
    tolerance: float,
) -> tuple[np.ndarray, float, int]:
    n = generator.shape[0]
    if n <= _REAL_SCHUR_DENSE_LIMIT:
        dense = generator.transpose().toarray()
        values, vectors = la.eig(dense)
        index = int(np.argmin(np.abs(values)))
        vector = np.real_if_close(vectors[:, index])
        vector = np.asarray(vector.real if np.iscomplexobj(vector) else vector, dtype=float)
        eigen_gap = float(abs(values[index]))
        return _normalise_by_sum(vector), eigen_gap, int(dense.nbytes + vectors.nbytes + values.nbytes)

    values, vectors = spla.eigs(
        generator.transpose(),
        k=1,
        which="SM",
        tol=max(float(tolerance), 1e-10),
        maxiter=_DEFAULT_MAX_ITERATIONS,
    )
    vector = np.real_if_close(vectors[:, 0])
    vector = np.asarray(vector.real if np.iscomplexobj(vector) else vector, dtype=float)
    eigen_gap = float(abs(values[0]))
    return _normalise_by_sum(vector), eigen_gap, _vector_workspace_bytes(n, 8)


def _lu_preconditioner(system: sp.csr_matrix) -> tuple[spla.LinearOperator | None, int]:
    try:
        lu = spla.splu(system.tocsc())
    except RuntimeError:
        return None, 0
    return spla.LinearOperator(system.shape, matvec=lu.solve), _lu_memory_bytes(lu)


def _initial_guess(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _normalise(probability: np.ndarray) -> np.ndarray:
    probability = np.real(np.asarray(probability, dtype=float))
    total = float(probability.sum())
    if abs(total) <= _NZ_TOL:
        raise RuntimeError("CME solve produced an invalid probability vector.")
    return probability / total


def _normalise_by_sum(values: np.ndarray) -> np.ndarray:
    values = np.real(np.asarray(values, dtype=float))
    total = float(values.sum())
    if abs(total) <= _NZ_TOL:
        raise RuntimeError("CME embedded solve produced an invalid stationary vector.")
    return values / total


def _mpmath_level_distribution(
    probability: mps.MPVector,
    capacity: int,
    phase_count: int,
) -> np.ndarray:
    levels = [mp.mpf("0") for _ in range(capacity + 1)]
    levels[0] = probability[0]
    if capacity >= 1:
        levels[1] = mp.fsum(probability[1: 1 + phase_count])
    base = 1 + phase_count
    block = phase_count * phase_count
    for customers in range(2, capacity + 1):
        start = base + (customers - 2) * block
        levels[customers] = mp.fsum(probability[start: start + block])

    total = mp.fsum(levels)
    if abs(total) <= mp.mpf(str(_NZ_TOL)):
        raise RuntimeError("CME level probabilities are invalid.")
    return np.array([float(value / total) for value in levels], dtype=float)


def _embedded_residual_l1(embedded: np.ndarray, transition: sp.csr_matrix) -> float:
    residual = np.asarray(embedded @ transition - embedded).ravel()
    return float(np.linalg.norm(residual, ord=1))


def _stationary_residual_l1(probability: np.ndarray, generator: sp.csr_matrix) -> float:
    residual = np.asarray(probability @ generator).ravel()
    return float(np.linalg.norm(residual, ord=1))


def _generator_residual_tolerance(tolerance: float) -> float:
    return max(float(tolerance), 1e-10)


def _recovery_residual_tolerance(tolerance: float) -> float:
    return max(1_000.0 * float(tolerance), 1e-8)


def _uniformization_rate(generator: sp.csr_matrix) -> float:
    rate = float((-generator.diagonal()).max())
    if rate <= 0.0:
        raise RuntimeError("Cannot uniformize a generator with non-positive exit rates.")
    return rate


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


def _method_key(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return _METHOD_ALIASES.get(key, key)


_INTEGRAL_SOLVERS: dict[str, _IntegralSolver] = {
    "real_schur": _integrate_real_schur,
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
    "power_mpmath_generator": _solve_mpmath_power_generator,
    "cesaro": _solve_mpmath_cesaro,
}

_METHOD_ALIASES: dict[str, str] = {
    "real_schur": "real_schur",
    "realschur": "real_schur",
}
