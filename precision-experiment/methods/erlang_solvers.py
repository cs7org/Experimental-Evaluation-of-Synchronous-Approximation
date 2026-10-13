from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import warnings

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.special as sc
import scipy.stats as stats

from methods import mpmath_stationary as mps


_NZ_TOL = 1e-15
_DENSE_EIGEN_LIMIT = 1_000
_JACOBI_DAMPING = 0.75


@dataclass(frozen=True)
class StationarySolveResult:
    probability: np.ndarray
    solver_key: str
    solver_name: str
    converged: bool
    iterations: int | None
    residual_l1: float
    matrix_memory_bytes: int
    factor_memory_bytes: int

    @property
    def memory_bytes(self) -> int:
        return self.matrix_memory_bytes + self.factor_memory_bytes


@dataclass(frozen=True)
class StationarySolverSettings:
    gmres_restart: int = 50
    ilu_drop_tolerance: float = 1e-4
    ilu_fill_factor: float = 10.0
    transient_initial_time: float = 1.0
    transient_max_time: float = 500.0
    transient_time_growth: float = 2.0
    transient_poisson_tail: float = 1e-12
    tsit5_t1: float = 500.0
    tsit5_dt0: float = 0.05
    tsit5_rtol: float = 1e-9
    tsit5_atol: float = 1e-12


_StationarySolver = Callable[
    [sp.spmatrix, float, int, float | None, StationarySolverSettings | None],
    StationarySolveResult,
]
_MPMathStationarySolver = Callable[
    [sp.spmatrix, float, int, float | None, StationarySolverSettings | None, int],
    StationarySolveResult,
]


def solve_stationary(
    generator: sp.spmatrix,
    method: str = "sparse_lu_direct_q",
    tolerance: float = 1e-12,
    max_iterations: int = 50_000,
    relaxation: float | None = None,
    settings: StationarySolverSettings | None = None,
    precision_digits: int | None = None,
) -> StationarySolveResult:
    key = _method_key(method)
    try:
        solver = _SOLVERS[key]
    except KeyError as exc:
        valid = ", ".join(_SOLVERS)
        raise ValueError(f"Unsupported Erlang method: {method}. Valid methods: {valid}") from exc
    if precision_digits is not None:
        digits = mps.validate_precision_digits(precision_digits)
        try:
            mpmath_solver = _MPMATH_SOLVERS[key]
        except KeyError as exc:
            valid = ", ".join(_MPMATH_SOLVERS)
            raise ValueError(
                f"Erlang method '{key}' does not support precision_digits. "
                f"True mpmath implementations are available for: {valid}."
            ) from exc
        return mpmath_solver(
            generator,
            mps.precision_tolerance(digits),
            max_iterations,
            relaxation,
            settings,
            digits,
        )
    return solver(generator, tolerance, max_iterations, relaxation, settings)


def _solve_dense_lu_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _dense_lu_augmented(
        generator,
        tolerance,
        "dense_lu_direct_q",
        "Dense LU (Q_E)",
        "direct_q",
    )


def _solve_dense_lu_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _dense_lu_augmented(
        generator,
        tolerance,
        "dense_lu_uniformized_p",
        "Dense LU (uniformized P_u)",
        "uniformized_p",
    )


def _solve_sparse_lu_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _sparse_lu_augmented(
        generator,
        tolerance,
        "sparse_lu_direct_q",
        "Sparse LU (Q_E)",
        "direct_q",
    )


def _solve_sparse_lu_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _sparse_lu_augmented(
        generator,
        tolerance,
        "sparse_lu_uniformized_p",
        "Sparse LU (uniformized P_u)",
        "uniformized_p",
    )


def _solve_sparse_lu_refinement_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _sparse_lu_augmented(
        generator,
        tolerance,
        "sparse_lu_refinement_direct_q",
        "Sparse LU refinement (Q_E)",
        "direct_q",
    )


def _solve_sparse_lu_refinement_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _sparse_lu_augmented(
        generator,
        tolerance,
        "sparse_lu_refinement_uniformized_p",
        "Sparse LU refinement (uniformized P_u)",
        "uniformized_p",
    )


def _solve_jacobi_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _jacobi_stationary(
        generator,
        tolerance,
        max_iterations,
        "jacobi_direct_q",
        "Jacobi (Q_E)",
        "direct_q",
    )


def _solve_jacobi_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _jacobi_stationary(
        generator,
        tolerance,
        max_iterations,
        "jacobi_uniformized_p",
        "Jacobi (uniformized P_u)",
        "uniformized_p",
    )


def _solve_gauss_seidel_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _gauss_seidel_stationary(
        generator,
        tolerance,
        max_iterations,
        relaxation,
        "gauss_seidel_direct_q",
        "Gauss-Seidel (Q_E)",
        "direct_q",
    )


def _solve_gauss_seidel_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _gauss_seidel_stationary(
        generator,
        tolerance,
        max_iterations,
        relaxation,
        "gauss_seidel_uniformized_p",
        "Gauss-Seidel (uniformized P_u)",
        "uniformized_p",
    )


def _solve_gmres_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _gmres_augmented(
        generator,
        tolerance,
        max_iterations,
        settings,
        "gmres_direct_q",
        "GMRES (Q_E, ILU)",
        "direct_q",
    )


def _solve_gmres_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _gmres_augmented(
        generator,
        tolerance,
        max_iterations,
        settings,
        "gmres_uniformized_p",
        "GMRES (uniformized P_u, ILU)",
        "uniformized_p",
    )


def _solve_bicgstab_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _bicgstab_augmented(
        generator,
        tolerance,
        max_iterations,
        settings,
        "bicgstab_direct_q",
        "BiCGSTAB (Q_E, ILU)",
        "direct_q",
    )


def _solve_bicgstab_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _bicgstab_augmented(
        generator,
        tolerance,
        max_iterations,
        settings,
        "bicgstab_uniformized_p",
        "BiCGSTAB (uniformized P_u, ILU)",
        "uniformized_p",
    )


def _solve_power_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _power_stationary(
        generator,
        tolerance,
        max_iterations,
        "power_uniformized_p",
        "Power method (uniformized P_u)",
        "uniformized_p",
    )


def _solve_cesaro_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _cesaro_stationary(
        generator,
        tolerance,
        max_iterations,
        "cesaro_uniformized_p",
        "Cesaro-averaged power (uniformized P_u)",
        "uniformized_p",
    )


def _solve_eigenvector_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _eigenvector_stationary(
        generator,
        tolerance,
        max_iterations,
        "eigenvector_direct_q",
        "Eigenvector / nullspace (Q_E)",
        "direct_q",
    )


def _solve_eigenvector_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _eigenvector_stationary(
        generator,
        tolerance,
        max_iterations,
        "eigenvector_uniformized_p",
        "Eigenvector (uniformized P_u)",
        "uniformized_p",
    )


def _solve_matrix_exponential_transient_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _matrix_exponential_transient_direct_q(
        generator,
        tolerance,
        max_iterations,
        settings,
    )


def _solve_uniformization_transient_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _uniformization_transient_uniformized_p(
        generator,
        tolerance,
        max_iterations,
        settings,
    )


def _solve_tsit5_transient_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _tsit5_transient_uniformized_p(
        generator,
        tolerance,
        max_iterations,
        settings,
    )


def _solve_tsit5_transient_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    return _tsit5_transient_direct_q(
        generator,
        tolerance,
        max_iterations,
        settings,
    )


_SOLVERS: dict[str, _StationarySolver] = {
    "dense_lu_direct_q": _solve_dense_lu_direct_q,
    "dense_lu_uniformized_p": _solve_dense_lu_uniformized_p,
    "sparse_lu_direct_q": _solve_sparse_lu_direct_q,
    "sparse_lu_uniformized_p": _solve_sparse_lu_uniformized_p,
    "sparse_lu_refinement_direct_q": _solve_sparse_lu_refinement_direct_q,
    "sparse_lu_refinement_uniformized_p": _solve_sparse_lu_refinement_uniformized_p,
    "jacobi_direct_q": _solve_jacobi_direct_q,
    "jacobi_uniformized_p": _solve_jacobi_uniformized_p,
    "gauss_seidel_direct_q": _solve_gauss_seidel_direct_q,
    "gauss_seidel_uniformized_p": _solve_gauss_seidel_uniformized_p,
    "gmres_direct_q": _solve_gmres_direct_q,
    "gmres_uniformized_p": _solve_gmres_uniformized_p,
    "bicgstab_direct_q": _solve_bicgstab_direct_q,
    "bicgstab_uniformized_p": _solve_bicgstab_uniformized_p,
    "power_uniformized_p": _solve_power_uniformized_p,
    "cesaro_uniformized_p": _solve_cesaro_uniformized_p,
    "eigenvector_direct_q": _solve_eigenvector_direct_q,
    "eigenvector_uniformized_p": _solve_eigenvector_uniformized_p,
    "matrix_exponential_transient_direct_q": _solve_matrix_exponential_transient_direct_q,
    "uniformization_transient_uniformized_p": _solve_uniformization_transient_uniformized_p,
    "tsit5_transient_uniformized_p": _solve_tsit5_transient_uniformized_p,
    "tsit5_transient_direct_q": _solve_tsit5_transient_direct_q,
}


def _solve_mpmath_dense_lu_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_dense_lu_augmented(
        generator,
        tolerance,
        "dense_lu_direct_q",
        "Dense LU (Q_E, mpmath)",
        "direct_q",
        precision_digits,
    )


def _solve_mpmath_dense_lu_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    _max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_dense_lu_augmented(
        generator,
        tolerance,
        "dense_lu_uniformized_p",
        "Dense LU (uniformized P_u, mpmath)",
        "uniformized_p",
        precision_digits,
    )


def _solve_mpmath_sparse_lu_refinement_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_sparse_lu_refinement_augmented(
        generator,
        tolerance,
        max_iterations,
        "sparse_lu_refinement_direct_q",
        "Sparse LU refinement (Q_E, mpmath residual)",
        "direct_q",
        precision_digits,
    )


def _solve_mpmath_sparse_lu_refinement_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_sparse_lu_refinement_augmented(
        generator,
        tolerance,
        max_iterations,
        "sparse_lu_refinement_uniformized_p",
        "Sparse LU refinement (uniformized P_u, mpmath residual)",
        "uniformized_p",
        precision_digits,
    )


def _solve_mpmath_jacobi_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_jacobi_stationary(
        generator,
        tolerance,
        max_iterations,
        "jacobi_direct_q",
        "Jacobi (Q_E, mpmath)",
        "direct_q",
        precision_digits,
    )


def _solve_mpmath_jacobi_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_jacobi_stationary(
        generator,
        tolerance,
        max_iterations,
        "jacobi_uniformized_p",
        "Jacobi (uniformized P_u, mpmath)",
        "uniformized_p",
        precision_digits,
    )


def _solve_mpmath_gauss_seidel_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_gauss_seidel_stationary(
        generator,
        tolerance,
        max_iterations,
        relaxation,
        "gauss_seidel_direct_q",
        "Gauss-Seidel (Q_E, mpmath)",
        "direct_q",
        precision_digits,
    )


def _solve_mpmath_gauss_seidel_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_gauss_seidel_stationary(
        generator,
        tolerance,
        max_iterations,
        relaxation,
        "gauss_seidel_uniformized_p",
        "Gauss-Seidel (uniformized P_u, mpmath)",
        "uniformized_p",
        precision_digits,
    )


def _solve_mpmath_power_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_power_stationary(
        generator,
        tolerance,
        max_iterations,
        "power_uniformized_p",
        "Power method (uniformized P_u, mpmath)",
        "uniformized_p",
        precision_digits,
    )


def _solve_mpmath_cesaro_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    _relaxation: float | None,
    _settings: StationarySolverSettings | None,
    precision_digits: int,
) -> StationarySolveResult:
    return _mpmath_cesaro_stationary(
        generator,
        tolerance,
        max_iterations,
        "cesaro_uniformized_p",
        "Cesaro-averaged power (uniformized P_u, mpmath)",
        "uniformized_p",
        precision_digits,
    )


_MPMATH_SOLVERS: dict[str, _MPMathStationarySolver] = {
    "dense_lu_direct_q": _solve_mpmath_dense_lu_direct_q,
    "dense_lu_uniformized_p": _solve_mpmath_dense_lu_uniformized_p,
    "sparse_lu_refinement_direct_q": _solve_mpmath_sparse_lu_refinement_direct_q,
    "sparse_lu_refinement_uniformized_p": _solve_mpmath_sparse_lu_refinement_uniformized_p,
    "jacobi_direct_q": _solve_mpmath_jacobi_direct_q,
    "jacobi_uniformized_p": _solve_mpmath_jacobi_uniformized_p,
    "gauss_seidel_direct_q": _solve_mpmath_gauss_seidel_direct_q,
    "gauss_seidel_uniformized_p": _solve_mpmath_gauss_seidel_uniformized_p,
    "power_uniformized_p": _solve_mpmath_power_uniformized_p,
    "cesaro_uniformized_p": _solve_mpmath_cesaro_uniformized_p,
}


def _mpmath_dense_lu_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    with mp.workdps(precision_digits):
        generator, transition, system, rhs = _stationary_system(generator, equation_form)
        zero_tolerance = mp.mpf(str(_NZ_TOL))
        probability_mp, dense_bytes = mps.dense_lu_solve(
            system,
            rhs,
            lambda values: mps.normalise_nonnegative(
                values,
                zero_tolerance,
                "Stationary solve produced an invalid probability vector.",
            ),
        )
        generator_rows = mps.sparse_rows(generator)
        residual_mp = mps.stationary_residual_l1(probability_mp, generator_rows)
        residual_limit = mp.mpf(str(_residual_tolerance(generator, tolerance, equation_form)))
        probability = mps.vector_to_float(probability_mp)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_mp <= residual_limit),
        iterations=None,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition) + dense_bytes,
        factor_memory_bytes=dense_bytes,
    )


def _mpmath_sparse_lu_refinement_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    generator, transition, system, rhs = _stationary_system(generator, equation_form)
    lu = spla.splu(system)
    zero_tolerance = mp.mpf(str(_NZ_TOL))
    probability, iterations, converged, residual_mp, _probability_mp = mps.sparse_lu_refinement(
        system,
        rhs,
        lu.solve,
        precision_digits,
        _residual_tolerance(generator, tolerance, equation_form),
        max_iterations,
        lambda values: mps.normalise_by_sum(
            values,
            zero_tolerance,
            "Stationary solve produced an invalid probability vector.",
        ),
    )
    probability = _normalise(probability)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=converged,
        iterations=iterations,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_sparse_system_memory_bytes(generator, transition, system),
        factor_memory_bytes=_lu_memory_bytes(lu) + _vector_workspace_bytes(system.shape[0], 5),
    )


def _mpmath_jacobi_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_iteration_context(generator, tolerance, equation_form)
        probability = mps.initial_guess(context["n"])
        damping = mp.mpf(str(_JACOBI_DAMPING))
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            previous = probability
            candidate = [
                mps.stationary_iteration_value(
                    context["columns"],
                    previous,
                    previous,
                    col,
                    equation_form,
                    context["zero_tolerance"],
                )
                for col in range(context["n"])
            ]
            probability = [
                previous[col] + damping * (candidate[col] - previous[col])
                for col in range(context["n"])
            ]
            probability = mps.normalise_nonnegative(
                probability,
                context["zero_tolerance"],
                "Stationary solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return StationarySolveResult(
        probability=probability_np,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_mp <= context["residual_limit"]),
        iterations=iterations,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(context["generator"], context["transition"]),
        factor_memory_bytes=_vector_workspace_bytes(context["n"], 3),
    )


def _mpmath_gauss_seidel_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    omega = 1.0 if relaxation is None else float(relaxation)
    if omega <= 0.0 or omega >= 2.0:
        raise ValueError("Gauss-Seidel relaxation must be in the open interval (0, 2).")

    with mp.workdps(precision_digits):
        context = _mpmath_iteration_context(generator, tolerance, equation_form)
        omega_mp = mp.mpf(str(omega))
        probability = mps.initial_guess(context["n"])
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            previous = probability.copy()
            for col in range(context["n"]):
                candidate = mps.stationary_iteration_value(
                    context["columns"],
                    probability,
                    previous,
                    col,
                    equation_form,
                    context["zero_tolerance"],
                )
                probability[col] = previous[col] + omega_mp * (candidate - previous[col])
            probability = mps.normalise_nonnegative(
                probability,
                context["zero_tolerance"],
                "Stationary solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return StationarySolveResult(
        probability=probability_np,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_mp <= context["residual_limit"]),
        iterations=iterations,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(context["generator"], context["transition"]),
        factor_memory_bytes=_gauss_seidel_workspace_bytes(context["n"]),
    )


def _mpmath_power_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_iteration_context(generator, tolerance, equation_form)
        probability = mps.initial_guess(context["n"])
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            previous = probability
            if context["transition_rows"] is None:
                generator_step = mps.row_vector_times_sparse(previous, context["generator_rows"])
                probability = [
                    previous[index] + generator_step[index] / context["rate"]
                    for index in range(context["n"])
                ]
            else:
                probability = mps.row_vector_times_sparse(previous, context["transition_rows"])
            probability = mps.normalise_nonnegative(
                probability,
                context["zero_tolerance"],
                "Stationary solve produced an invalid probability vector.",
            )
            if mps.l1_delta(probability, previous) <= context["tolerance"]:
                if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                    break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return StationarySolveResult(
        probability=probability_np,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_mp <= context["residual_limit"]),
        iterations=iterations,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(context["generator"], context["transition"]),
        factor_memory_bytes=_power_workspace_bytes(context["n"]),
    )


def _mpmath_cesaro_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
    precision_digits: int,
) -> StationarySolveResult:
    with mp.workdps(precision_digits):
        context = _mpmath_iteration_context(generator, tolerance, equation_form)
        current = mps.initial_guess(context["n"])
        average_sum = mps.zeros(context["n"])
        average_count = 0
        probability = current.copy()
        average_start_limit = mp.mpf(str(max(100.0 * float(context["residual_limit"]), 1e-8)))
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            if context["transition_rows"] is None:
                generator_step = mps.row_vector_times_sparse(current, context["generator_rows"])
                current = [
                    current[index] + generator_step[index] / context["rate"]
                    for index in range(context["n"])
                ]
            else:
                current = mps.row_vector_times_sparse(current, context["transition_rows"])
            current = mps.normalise_nonnegative(
                current,
                context["zero_tolerance"],
                "Stationary solve produced an invalid probability vector.",
            )
            current_residual_l1 = mps.stationary_residual_l1(current, context["generator_rows"])
            if average_count > 0 or current_residual_l1 <= average_start_limit:
                average_sum = [total + value for total, value in zip(average_sum, current)]
                average_count += 1
                probability = mps.normalise_nonnegative(
                    average_sum,
                    context["zero_tolerance"],
                    "Stationary solve produced an invalid probability vector.",
                )
            else:
                probability = current
            if mps.stationary_residual_l1(probability, context["generator_rows"]) <= context["residual_limit"]:
                break

        residual_mp = mps.stationary_residual_l1(probability, context["generator_rows"])
        probability_np = mps.vector_to_float(probability)

    return StationarySolveResult(
        probability=probability_np,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_mp <= context["residual_limit"]),
        iterations=iterations,
        residual_l1=float(residual_mp),
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(context["generator"], context["transition"]),
        factor_memory_bytes=_vector_workspace_bytes(context["n"], 3),
    )


def _mpmath_iteration_context(
    generator: sp.spmatrix,
    tolerance: float,
    equation_form: str,
) -> dict[str, object]:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    iteration_matrix = transition if transition is not None else generator
    return {
        "generator": generator,
        "transition": transition,
        "columns": mps.sparse_columns(iteration_matrix),
        "generator_rows": mps.sparse_rows(generator),
        "transition_rows": mps.sparse_rows(transition) if transition is not None else None,
        "n": generator.shape[0],
        "rate": mp.mpf(str(_uniformization_rate(generator))),
        "tolerance": mp.mpf(str(tolerance)),
        "residual_limit": mp.mpf(str(_residual_tolerance(generator, tolerance, equation_form))),
        "zero_tolerance": mp.mpf(str(_NZ_TOL)),
    }


def _dense_lu_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator, transition, system, rhs = _stationary_system(generator, equation_form)
    dense_system = system.toarray()
    lu, pivots = la.lu_factor(dense_system)
    probability = _normalise(la.lu_solve((lu, pivots), rhs))
    residual_l1 = _stationary_residual_l1(probability, generator)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= _residual_tolerance(generator, tolerance, equation_form)),
        iterations=None,
        residual_l1=residual_l1,
        matrix_memory_bytes=_dense_system_memory_bytes(generator, transition, dense_system),
        factor_memory_bytes=int(lu.nbytes + pivots.nbytes),
    )


def _sparse_lu_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator, transition, system, rhs = _stationary_system(generator, equation_form)
    lu = spla.splu(system)
    probability = _normalise(lu.solve(rhs))
    residual_l1 = _stationary_residual_l1(probability, generator)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= _residual_tolerance(generator, tolerance, equation_form)),
        iterations=None,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_system_memory_bytes(generator, transition, system),
        factor_memory_bytes=_lu_memory_bytes(lu),
    )


def _gmres_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings | None,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    generator, transition, system, rhs = _stationary_system(generator, equation_form)
    n = generator.shape[0]
    restart = min(settings.gmres_restart, n)
    preconditioner, preconditioner_bytes = _ilu_preconditioner(
        system,
        settings.ilu_drop_tolerance,
        settings.ilu_fill_factor,
    )
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
        restart=restart,
        maxiter=max_iterations,
        M=preconditioner,
        callback=count_iteration,
        callback_type="legacy",
    )
    if info > 0:
        iteration_count = max(iteration_count, info)
    probability = _normalise(solution)
    residual_l1 = _stationary_residual_l1(probability, generator)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(info == 0 and residual_l1 <= _residual_tolerance(generator, tolerance, equation_form)),
        iterations=iteration_count,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_system_memory_bytes(generator, transition, system),
        factor_memory_bytes=_gmres_workspace_bytes(n, restart) + preconditioner_bytes,
    )


def _bicgstab_augmented(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings | None,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    generator, transition, system, rhs = _stationary_system(generator, equation_form)
    n = generator.shape[0]
    preconditioner, preconditioner_bytes = _ilu_preconditioner(
        system,
        settings.ilu_drop_tolerance,
        settings.ilu_fill_factor,
    )
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
        maxiter=max_iterations,
        M=preconditioner,
        callback=count_iteration,
    )
    if info > 0:
        iteration_count = max(iteration_count, info)
    probability = _normalise(solution)
    residual_l1 = _stationary_residual_l1(probability, generator)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(info == 0 and residual_l1 <= _residual_tolerance(generator, tolerance, equation_form)),
        iterations=iteration_count,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_system_memory_bytes(generator, transition, system),
        factor_memory_bytes=_bicgstab_workspace_bytes(n) + preconditioner_bytes,
    )


def _jacobi_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    columns = (transition if transition is not None else generator).tocsc()
    n = generator.shape[0]
    probability = _initial_guess(n)
    residual_limit = _residual_tolerance(generator, tolerance, equation_form)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability
        candidate = np.empty(n, dtype=float)
        for col in range(n):
            candidate[col] = _stationary_iteration_value(columns, previous, previous, col, equation_form)
        probability = previous + _JACOBI_DAMPING * (candidate - previous)
        probability = _normalise(probability)
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            if _stationary_residual_l1(probability, generator) <= residual_limit:
                break

    residual_l1 = _stationary_residual_l1(probability, generator)
    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= residual_limit),
        iterations=iterations,
        residual_l1=residual_l1,
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition),
        factor_memory_bytes=_vector_workspace_bytes(n, 3),
    )


def _gauss_seidel_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    relaxation: float | None,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    columns = (transition if transition is not None else generator).tocsc()
    n = generator.shape[0]
    omega = 1.0 if relaxation is None else float(relaxation)
    if omega <= 0.0 or omega >= 2.0:
        raise ValueError("Gauss-Seidel relaxation must be in the open interval (0, 2).")

    probability = _initial_guess(n)
    residual_limit = _residual_tolerance(generator, tolerance, equation_form)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability.copy()
        for col in range(n):
            candidate = _stationary_iteration_value(columns, probability, previous, col, equation_form)
            probability[col] = previous[col] + omega * (candidate - previous[col])
        probability = _normalise(probability)
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            if _stationary_residual_l1(probability, generator) <= residual_limit:
                break

    residual_l1 = _stationary_residual_l1(probability, generator)
    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= residual_limit),
        iterations=iterations,
        residual_l1=residual_l1,
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition),
        factor_memory_bytes=_gauss_seidel_workspace_bytes(n),
    )


def _power_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    rate = _uniformization_rate(generator)
    n = generator.shape[0]
    probability = _initial_guess(n)
    residual_limit = _residual_tolerance(generator, tolerance, equation_form)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        previous = probability
        if transition is None:
            probability = np.asarray(previous + (previous @ generator) / rate).ravel()
        else:
            probability = np.asarray(previous @ transition).ravel()
        probability = _normalise(probability)
        if np.linalg.norm(probability - previous, ord=1) <= tolerance:
            if _stationary_residual_l1(probability, generator) <= residual_limit:
                break

    residual_l1 = _stationary_residual_l1(probability, generator)
    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= residual_limit),
        iterations=iterations,
        residual_l1=residual_l1,
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition),
        factor_memory_bytes=_power_workspace_bytes(n),
    )


def _cesaro_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    rate = _uniformization_rate(generator)
    n = generator.shape[0]
    current = _initial_guess(n)
    average_sum = np.zeros(n, dtype=float)
    average_count = 0
    probability = current.copy()
    residual_limit = _residual_tolerance(generator, tolerance, equation_form)
    average_start_limit = max(100.0 * residual_limit, 1e-8)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        if transition is None:
            current = np.asarray(current + (current @ generator) / rate).ravel()
        else:
            current = np.asarray(current @ transition).ravel()
        current = _normalise(current)
        current_residual_l1 = _stationary_residual_l1(current, generator)
        if average_count > 0 or current_residual_l1 <= average_start_limit:
            average_sum += current
            average_count += 1
            probability = _normalise(average_sum)
        else:
            probability = current
        if _stationary_residual_l1(probability, generator) <= residual_limit:
            break

    residual_l1 = _stationary_residual_l1(probability, generator)
    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(residual_l1 <= residual_limit),
        iterations=iterations,
        residual_l1=residual_l1,
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition),
        factor_memory_bytes=_vector_workspace_bytes(n, 3),
    )


def _eigenvector_stationary(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    solver_key: str,
    solver_name: str,
    equation_form: str,
) -> StationarySolveResult:
    generator = generator.tocsr()
    transition = _uniformized_transition(generator) if equation_form == "uniformized_p" else None
    if transition is None:
        probability, eigen_gap, workspace_bytes = _left_nullspace_for_generator(
            generator,
            tolerance,
            max_iterations,
        )
    else:
        probability, eigen_gap, workspace_bytes = _left_eigenvector_for_transition(
            transition,
            tolerance,
            max_iterations,
        )
    residual_l1 = _stationary_residual_l1(probability, generator)
    residual_limit = max(_residual_tolerance(generator, tolerance, equation_form), 1e-10)

    return StationarySolveResult(
        probability=probability,
        solver_key=solver_key,
        solver_name=solver_name,
        converged=bool(eigen_gap <= residual_limit and residual_l1 <= residual_limit),
        iterations=None,
        residual_l1=residual_l1,
        matrix_memory_bytes=_stationary_iteration_matrix_bytes(generator, transition) + workspace_bytes,
        factor_memory_bytes=0,
    )


def _matrix_exponential_transient_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    generator = generator.tocsr()
    generator_t = generator.transpose().tocsr()
    n = generator.shape[0]
    initial = _transient_initial_distribution(n)
    probability = initial.copy()
    previous: np.ndarray | None = None
    residual_l1 = _stationary_residual_l1(probability, generator)
    residual_limit = _transient_residual_tolerance(tolerance)
    converged = False
    evaluations = 0

    for target_time in _transient_target_times(settings, max_iterations):
        probability = _normalise(spla.expm_multiply(generator_t * target_time, initial))
        residual_l1 = _stationary_residual_l1(probability, generator)
        evaluations += 1
        if previous is not None and _transient_has_converged(
            probability,
            previous,
            residual_l1,
            residual_limit,
        ):
            converged = True
            break
        previous = probability

    return StationarySolveResult(
        probability=probability,
        solver_key="matrix_exponential_transient_direct_q",
        solver_name="Matrix exponential transient (Q_E)",
        converged=converged,
        iterations=evaluations,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_memory_bytes(generator),
        factor_memory_bytes=_vector_workspace_bytes(n, 4),
    )


def _uniformization_transient_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings | None,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    generator = generator.tocsr()
    transition = _uniformized_transition(generator)
    rate = _uniformization_rate(generator)
    n = generator.shape[0]
    initial = _transient_initial_distribution(n)
    probability = initial.copy()
    previous: np.ndarray | None = None
    residual_l1 = _stationary_residual_l1(probability, generator)
    residual_limit = _transient_residual_tolerance(tolerance)
    converged = False
    used_powers = 0

    for target_time in _transient_target_times(settings, max_iterations):
        lower, upper, weights = _poisson_truncation_window(
            rate * target_time,
            settings.transient_poisson_tail,
        )
        remaining_powers = int(max_iterations) - used_powers
        if remaining_powers <= 0 or (previous is not None and upper > remaining_powers):
            break

        probability, powers, complete = _uniformization_probability_at_time(
            initial,
            transition,
            lower,
            upper,
            weights,
            remaining_powers,
        )
        used_powers += powers
        residual_l1 = _stationary_residual_l1(probability, generator)
        if complete and previous is not None and _transient_has_converged(
            probability,
            previous,
            residual_l1,
            residual_limit,
        ):
            converged = True
            break
        previous = probability
        if not complete:
            break

    return StationarySolveResult(
        probability=probability,
        solver_key="uniformization_transient_uniformized_p",
        solver_name="Uniformization transient (P_u)",
        converged=converged,
        iterations=used_powers,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_memory_bytes(generator) + _sparse_memory_bytes(transition),
        factor_memory_bytes=_vector_workspace_bytes(n, 4),
    )


def _tsit5_transient_uniformized_p(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    jax, jnp, diffrax, jsparse = _load_jax_tsit5_dependencies()

    generator = generator.tocsr()
    transition = _uniformized_transition(generator)
    transition_t = jsparse.BCOO.from_scipy_sparse(transition.transpose())
    rate = _uniformization_rate(generator)
    n = generator.shape[0]

    y0 = jnp.zeros(n, dtype=jnp.float64).at[0].set(1.0)

    def vector_field(_t, probability, _args):
        return rate * (transition_t @ probability - probability)

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(
        rtol=max(float(tolerance), settings.tsit5_rtol),
        atol=max(float(tolerance), settings.tsit5_atol),
    )
    steady_state_event = diffrax.SteadyStateEvent(
        rtol=max(float(tolerance) ** 0.5, settings.tsit5_rtol),
        atol=max(float(tolerance), settings.tsit5_atol),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*discrete_terminating_event=.*deprecated.*",
            category=UserWarning,
        )
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=settings.tsit5_t1,
            dt0=settings.tsit5_dt0,
            y0=y0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            discrete_terminating_event=steady_state_event,
            max_steps=max(4096, int(max_iterations)),
            throw=False,
        )

    probability = _normalise(np.array(solution.ys[-1]))
    residual_l1 = _stationary_residual_l1(probability, generator)
    residual_tolerance = max(10.0 * float(tolerance) ** 0.5, rate * float(tolerance))
    steps = int(solution.stats["num_steps"])

    return StationarySolveResult(
        probability=probability,
        solver_key="tsit5_transient_uniformized_p",
        solver_name="Tsit5 Transient Uniformized P (JAX)",
        converged=bool(residual_l1 <= residual_tolerance and steps < max(4096, int(max_iterations))),
        iterations=steps,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_memory_bytes(generator) + _sparse_memory_bytes(transition),
        factor_memory_bytes=_tsit5_workspace_bytes(n),
    )


def _tsit5_transient_direct_q(
    generator: sp.spmatrix,
    tolerance: float,
    max_iterations: int,
    settings: StationarySolverSettings,
) -> StationarySolveResult:
    settings = _solver_settings(settings)
    jax, jnp, diffrax, jsparse = _load_jax_tsit5_dependencies()

    generator = generator.tocsr()
    generator_t = jsparse.BCOO.from_scipy_sparse(generator.transpose())
    n = generator.shape[0]

    y0 = jnp.zeros(n, dtype=jnp.float64).at[0].set(1.0)

    def vector_field(_t, probability, _args):
        return generator_t @ probability

    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(
        rtol=max(float(tolerance), settings.tsit5_rtol),
        atol=max(float(tolerance), settings.tsit5_atol),
    )
    steady_state_event = diffrax.SteadyStateEvent(
        rtol=max(float(tolerance) ** 0.5, settings.tsit5_rtol),
        atol=max(float(tolerance), settings.tsit5_atol),
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*discrete_terminating_event=.*deprecated.*",
            category=UserWarning,
        )
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=settings.tsit5_t1,
            dt0=settings.tsit5_dt0,
            y0=y0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            discrete_terminating_event=steady_state_event,
            max_steps=max(4096, int(max_iterations)),
            throw=False,
        )

    probability = _normalise(np.array(solution.ys[-1]))
    residual_l1 = _stationary_residual_l1(probability, generator)
    residual_tolerance = max(10.0 * float(tolerance) ** 0.5, float(tolerance))
    steps = int(solution.stats["num_steps"])

    return StationarySolveResult(
        probability=probability,
        solver_key="tsit5_transient_direct_q",
        solver_name="Tsit5 Transient Direct Q (JAX)",
        converged=bool(residual_l1 <= residual_tolerance and steps < max(4096, int(max_iterations))),
        iterations=steps,
        residual_l1=residual_l1,
        matrix_memory_bytes=_sparse_memory_bytes(generator),
        factor_memory_bytes=_tsit5_workspace_bytes(n),
    )


def _augmented_stationary_system(generator: sp.csr_matrix) -> tuple[sp.csc_matrix, np.ndarray]:
    n = generator.shape[0]
    system = generator.transpose().tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    return system.tocsc(), rhs


def _augmented_dtmc_stationary_system(transition: sp.csr_matrix) -> tuple[sp.csc_matrix, np.ndarray]:
    n = transition.shape[0]
    system = (transition.transpose() - sp.eye(n, format="csc")).tolil()
    rhs = np.zeros(n)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    return system.tocsc(), rhs


def _stationary_system(
    generator: sp.spmatrix,
    equation_form: str,
) -> tuple[sp.csr_matrix, sp.csr_matrix | None, sp.csc_matrix, np.ndarray]:
    generator = generator.tocsr()
    if equation_form == "direct_q":
        system, rhs = _augmented_stationary_system(generator)
        return generator, None, system, rhs
    if equation_form == "uniformized_p":
        transition = _uniformized_transition(generator)
        system, rhs = _augmented_dtmc_stationary_system(transition)
        return generator, transition, system, rhs
    raise ValueError(f"Unsupported Erlang equation form: {equation_form}")


def _stationary_iteration_value(
    columns: sp.csc_matrix,
    current: np.ndarray,
    previous: np.ndarray,
    col: int,
    equation_form: str,
) -> float:
    start, end = columns.indptr[col], columns.indptr[col + 1]
    diagonal = 0.0
    subtotal = 0.0
    for row, value in zip(columns.indices[start:end], columns.data[start:end]):
        value = float(value)
        if row == col:
            diagonal = value
        elif row < col:
            subtotal += value * current[row]
        else:
            subtotal += value * previous[row]

    denominator = -diagonal if equation_form == "direct_q" else 1.0 - diagonal
    if abs(denominator) <= _NZ_TOL:
        raise RuntimeError("Stationary iteration encountered a zero diagonal entry.")
    return subtotal / denominator


def _residual_tolerance(
    generator: sp.csr_matrix,
    tolerance: float,
    equation_form: str,
) -> float:
    if equation_form == "uniformized_p":
        return max(float(tolerance), _uniformization_rate(generator) * float(tolerance))
    return float(tolerance)


def _sparse_system_memory_bytes(
    generator: sp.csr_matrix,
    transition: sp.csr_matrix | None,
    system: sp.csc_matrix,
) -> int:
    return _stationary_iteration_matrix_bytes(generator, transition) + _sparse_memory_bytes(system)


def _dense_system_memory_bytes(
    generator: sp.csr_matrix,
    transition: sp.csr_matrix | None,
    dense_system: np.ndarray,
) -> int:
    return _stationary_iteration_matrix_bytes(generator, transition) + int(dense_system.nbytes)


def _stationary_iteration_matrix_bytes(
    generator: sp.csr_matrix,
    transition: sp.csr_matrix | None,
) -> int:
    total = _sparse_memory_bytes(generator)
    if transition is not None:
        total += _sparse_memory_bytes(transition)
    return total


def _left_nullspace_for_generator(
    generator: sp.csr_matrix,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, int]:
    n = generator.shape[0]
    if n <= _DENSE_EIGEN_LIMIT:
        dense = generator.transpose().toarray()
        values, vectors = la.eig(dense)
        index = int(np.argmin(np.abs(values)))
        vector = np.real_if_close(vectors[:, index])
        eigen_gap = float(abs(values[index]))
        return _normalise_signed_probability(vector), eigen_gap, int(dense.nbytes + vectors.nbytes + values.nbytes)

    values, vectors = spla.eigs(
        generator.transpose(),
        k=1,
        which="SM",
        tol=max(float(tolerance), 1e-10),
        maxiter=max_iterations,
    )
    vector = np.real_if_close(vectors[:, 0])
    eigen_gap = float(abs(values[0]))
    return _normalise_signed_probability(vector), eigen_gap, _vector_workspace_bytes(n, 8)


def _left_eigenvector_for_transition(
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

    values, vectors = spla.eigs(
        transition.transpose(),
        k=1,
        which="LM",
        tol=max(float(tolerance), 1e-10),
        maxiter=max_iterations,
    )
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.real_if_close(vectors[:, index])
    eigen_gap = float(abs(values[index] - 1.0))
    return _normalise_signed_probability(vector), eigen_gap, _vector_workspace_bytes(n, 8)


def _solver_settings(settings: StationarySolverSettings | None) -> StationarySolverSettings:
    return settings if settings is not None else StationarySolverSettings()


def _method_key(method: str) -> str:
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in key:
        key = key.replace("__", "_")
    return key


def _initial_guess(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _transient_initial_distribution(n: int) -> np.ndarray:
    initial = np.zeros(n, dtype=float)
    initial[0] = 1.0
    return initial


def _transient_target_times(
    settings: StationarySolverSettings,
    max_iterations: int,
) -> list[float]:
    initial_time = float(settings.transient_initial_time)
    max_time = float(settings.transient_max_time)
    growth = float(settings.transient_time_growth)
    if initial_time <= 0.0:
        raise ValueError("Transient initial time must be positive.")
    if max_time < initial_time:
        raise ValueError("Transient max time must be at least the initial time.")
    if growth <= 1.0:
        raise ValueError("Transient time growth must be greater than 1.")

    times: list[float] = []
    target_time = initial_time
    evaluation_limit = max(1, int(max_iterations))
    while len(times) < evaluation_limit:
        times.append(min(target_time, max_time))
        if target_time >= max_time:
            break
        target_time *= growth
    return times


def _transient_residual_tolerance(tolerance: float) -> float:
    return max(10.0 * float(tolerance), 1e-10)


def _transient_has_converged(
    probability: np.ndarray,
    previous: np.ndarray,
    residual_l1: float,
    residual_limit: float,
) -> bool:
    return bool(
        residual_l1 <= residual_limit
        and np.linalg.norm(probability - previous, ord=1) <= residual_limit
    )


def _poisson_truncation_window(
    mean: float,
    tail_tolerance: float,
) -> tuple[int, int, np.ndarray]:
    if mean <= 0.0:
        return 0, 0, np.ones(1, dtype=float)

    tail = min(max(float(tail_tolerance), np.finfo(float).tiny), 0.5)
    lower = int(max(0, stats.poisson.ppf(0.5 * tail, mean)))
    upper = int(max(lower, stats.poisson.isf(0.5 * tail, mean)))
    terms = np.arange(lower, upper + 1, dtype=float)
    log_weights = -mean + terms * np.log(mean) - sc.gammaln(terms + 1.0)
    log_weights -= float(log_weights.max())
    weights = np.exp(log_weights)
    weights /= float(weights.sum())
    return lower, upper, weights


def _uniformization_probability_at_time(
    initial: np.ndarray,
    transition: sp.csr_matrix,
    lower: int,
    upper: int,
    weights: np.ndarray,
    max_power: int,
) -> tuple[np.ndarray, int, bool]:
    current = initial.copy()
    probability = np.zeros_like(initial)
    weight_sum = 0.0
    power_limit = min(int(upper), int(max_power))

    for power in range(power_limit + 1):
        if power >= lower:
            weight = float(weights[power - lower])
            probability += weight * current
            weight_sum += weight
        if power < power_limit:
            current = _normalise(np.asarray(current @ transition).ravel())

    if weight_sum <= 0.0:
        return current, power_limit, False
    return _normalise(probability), power_limit, bool(power_limit >= upper)


def _uniformized_transition(generator: sp.csr_matrix) -> sp.csr_matrix:
    rate = _uniformization_rate(generator)
    return sp.eye(generator.shape[0], format="csr") + generator / rate


def _uniformization_rate(generator: sp.csr_matrix) -> float:
    rate = float((-generator.diagonal()).max())
    if rate <= 0.0:
        raise RuntimeError("Cannot uniformize a generator with non-positive exit rates.")
    return rate


def _normalise(probability: np.ndarray) -> np.ndarray:
    probability = np.real(np.asarray(probability, dtype=float))
    probability[probability < 0.0] = 0.0
    total = float(probability.sum())
    if total <= 0.0:
        raise RuntimeError("Stationary solve produced an invalid probability vector.")
    return probability / total


def _normalise_signed_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.real(np.asarray(probability, dtype=float))
    if float(probability.sum()) < 0.0:
        probability = -probability
    probability[np.abs(probability) <= _NZ_TOL] = 0.0
    probability[probability < 0.0] = 0.0
    total = float(probability.sum())
    if total <= 0.0:
        raise RuntimeError("Eigenvector solve produced an invalid probability vector.")
    return probability / total


def _stationary_residual_l1(probability: np.ndarray, generator: sp.csr_matrix) -> float:
    residual = np.asarray(probability @ generator).ravel()
    return float(np.linalg.norm(residual, ord=1))


def _sparse_memory_bytes(matrix: sp.spmatrix) -> int:
    if sp.isspmatrix_csr(matrix) or sp.isspmatrix_csc(matrix):
        compressed = matrix
    else:
        compressed = matrix.tocsr()
    return int(compressed.data.nbytes + compressed.indices.nbytes + compressed.indptr.nbytes)


def _ilu_preconditioner(
    system: sp.csc_matrix,
    drop_tolerance: float,
    fill_factor: float,
) -> tuple[spla.LinearOperator | None, int]:
    try:
        ilu = spla.spilu(system, drop_tol=drop_tolerance, fill_factor=fill_factor)
    except RuntimeError:
        return None, 0
    preconditioner = spla.LinearOperator(system.shape, matvec=ilu.solve)
    return preconditioner, _lu_memory_bytes(ilu)


def _load_jax_tsit5_dependencies():
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import diffrax
        from jax.experimental import sparse as jsparse
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Tsit5 transient solver requires JAX and Diffrax. "
            "Install them with: python -m pip install \"jax[cpu]\" diffrax"
        ) from exc
    return jax, jnp, diffrax, jsparse


def _gmres_workspace_bytes(n: int, restart: int) -> int:
    return int((restart + 4) * n * np.dtype(float).itemsize)


def _bicgstab_workspace_bytes(n: int) -> int:
    return int(8 * n * np.dtype(float).itemsize)


def _gauss_seidel_workspace_bytes(n: int) -> int:
    return int(2 * n * np.dtype(float).itemsize)


def _power_workspace_bytes(n: int) -> int:
    return int(2 * n * np.dtype(float).itemsize)


def _vector_workspace_bytes(n: int, vector_count: int) -> int:
    return int(vector_count * n * np.dtype(float).itemsize)


def _tsit5_workspace_bytes(n: int) -> int:
    return int(12 * n * np.dtype(float).itemsize)


def _lu_memory_bytes(lu: spla.SuperLU) -> int:
    return (
        _sparse_memory_bytes(lu.L)
        + _sparse_memory_bytes(lu.U)
        + int(lu.perm_r.nbytes)
        + int(lu.perm_c.nbytes)
    )
