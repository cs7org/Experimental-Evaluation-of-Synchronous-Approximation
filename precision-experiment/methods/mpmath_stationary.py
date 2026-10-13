from __future__ import annotations

from collections.abc import Callable, Sequence

import mpmath as mp
import numpy as np
import scipy.sparse as sp


MPVector = list[mp.mpf]
MPSparseRows = list[list[tuple[int, mp.mpf]]]


def validate_precision_digits(precision_digits: int) -> int:
    digits = int(precision_digits)
    if digits <= 0:
        raise ValueError("precision_digits must be positive when provided.")
    return digits


def precision_tolerance(precision_digits: int) -> float:
    return 10.0 ** (-validate_precision_digits(precision_digits))


def sparse_rows(matrix: sp.spmatrix) -> MPSparseRows:
    csr = matrix.tocsr()
    rows: MPSparseRows = []
    for row in range(csr.shape[0]):
        start, end = csr.indptr[row], csr.indptr[row + 1]
        rows.append([
            (int(col), mp.mpf(str(value)))
            for col, value in zip(csr.indices[start:end], csr.data[start:end])
        ])
    return rows


def sparse_columns(matrix: sp.spmatrix) -> MPSparseRows:
    csc = matrix.tocsc()
    columns: MPSparseRows = []
    for col in range(csc.shape[1]):
        start, end = csc.indptr[col], csc.indptr[col + 1]
        columns.append([
            (int(row), mp.mpf(str(value)))
            for row, value in zip(csc.indices[start:end], csc.data[start:end])
        ])
    return columns


def dense_lu_solve(
    system: sp.spmatrix,
    rhs: np.ndarray,
    normalise: Callable[[Sequence[mp.mpf]], MPVector],
) -> tuple[MPVector, int]:
    dense = system.toarray()
    n = dense.shape[0]
    system_mp = mp.matrix([[mp.mpf(str(dense[i, j])) for j in range(n)] for i in range(n)])
    rhs_mp = mp.matrix([mp.mpf(str(value)) for value in rhs])
    solution = mp.lu_solve(system_mp, rhs_mp)
    return normalise([solution[i] for i in range(n)]), int(dense.nbytes)


def initial_guess(n: int) -> MPVector:
    value = mp.mpf(1) / int(n)
    return [value for _ in range(n)]


def zeros(n: int) -> MPVector:
    return [mp.mpf("0") for _ in range(n)]


def vector_to_float(values: Sequence[mp.mpf]) -> np.ndarray:
    return np.array([float(value) for value in values], dtype=float)


def normalise_by_sum(
    values: Sequence[mp.mpf],
    zero_tolerance: mp.mpf,
    error_message: str,
) -> MPVector:
    total = mp.fsum(values)
    if abs(total) <= zero_tolerance:
        raise RuntimeError(error_message)
    return [value / total for value in values]


def normalise_signed(
    values: Sequence[mp.mpf],
    zero_tolerance: mp.mpf,
    error_message: str,
) -> MPVector:
    total = mp.fsum(values)
    if abs(total) <= zero_tolerance:
        raise RuntimeError(error_message)
    if total < 0:
        total = -total
        return [(-value) / total for value in values]
    return [value / total for value in values]


def normalise_nonnegative(
    values: Sequence[mp.mpf],
    zero_tolerance: mp.mpf,
    error_message: str,
) -> MPVector:
    probability = [mp.mpf("0") if value < 0 else value for value in values]
    total = mp.fsum(probability)
    if total <= zero_tolerance:
        raise RuntimeError(error_message)
    return [value / total for value in probability]


def row_vector_times_sparse(vector: Sequence[mp.mpf], rows: MPSparseRows) -> MPVector:
    result = zeros(len(vector))
    for row, row_entries in enumerate(rows):
        left = vector[row]
        if left == 0:
            continue
        for col, value in row_entries:
            result[col] += left * value
    return result


def matrix_vector_times_sparse(rows: MPSparseRows, vector: Sequence[mp.mpf]) -> MPVector:
    result = zeros(len(rows))
    for row, row_entries in enumerate(rows):
        result[row] = mp.fsum(value * vector[col] for col, value in row_entries)
    return result


def l1_delta(left: Sequence[mp.mpf], right: Sequence[mp.mpf]) -> mp.mpf:
    return mp.fsum(abs(a - b) for a, b in zip(left, right))


def stationary_residual_l1(probability: Sequence[mp.mpf], generator_rows: MPSparseRows) -> mp.mpf:
    residual = row_vector_times_sparse(probability, generator_rows)
    return mp.fsum(abs(value) for value in residual)


def embedded_residual_l1(probability: Sequence[mp.mpf], transition_rows: MPSparseRows) -> mp.mpf:
    residual = row_vector_times_sparse(probability, transition_rows)
    return mp.fsum(abs(value - probability[index]) for index, value in enumerate(residual))


def sparse_lu_refinement(
    system: sp.spmatrix,
    rhs: np.ndarray,
    solve_correction: Callable[[np.ndarray], np.ndarray],
    precision_digits: int,
    tolerance: float,
    max_iterations: int,
    normalise: Callable[[Sequence[mp.mpf]], MPVector],
) -> tuple[np.ndarray, int, bool, mp.mpf, MPVector]:
    iteration_limit = max(1, min(int(max_iterations), 50))

    with mp.workdps(validate_precision_digits(precision_digits)):
        rows = sparse_rows(system)
        tolerance_mp = mp.mpf(str(tolerance))
        rhs_mp = [mp.mpf(str(value)) for value in rhs]
        solution = normalise([mp.mpf(str(value)) for value in solve_correction(rhs)])
        residual_l1 = _linear_residual_l1(rows, solution, rhs_mp)
        iterations = 0

        for iterations in range(1, iteration_limit + 1):
            if residual_l1 <= tolerance_mp:
                return vector_to_float(solution), iterations - 1, True, residual_l1, solution

            residual = _linear_residual(rows, solution, rhs_mp)
            correction = solve_correction(np.array([float(value) for value in residual], dtype=float))
            solution = normalise([
                value + mp.mpf(str(delta))
                for value, delta in zip(solution, correction)
            ])
            residual_l1 = _linear_residual_l1(rows, solution, rhs_mp)
            if residual_l1 <= tolerance_mp:
                return vector_to_float(solution), iterations, True, residual_l1, solution

        return vector_to_float(solution), iterations, residual_l1 <= tolerance_mp, residual_l1, solution


def _linear_residual(
    rows: MPSparseRows,
    solution: Sequence[mp.mpf],
    rhs: Sequence[mp.mpf],
) -> MPVector:
    product = matrix_vector_times_sparse(rows, solution)
    return [right - left for left, right in zip(product, rhs)]


def _linear_residual_l1(
    rows: MPSparseRows,
    solution: Sequence[mp.mpf],
    rhs: Sequence[mp.mpf],
) -> mp.mpf:
    return mp.fsum(abs(value) for value in _linear_residual(rows, solution, rhs))


def stationary_iteration_value(
    columns: MPSparseRows,
    current: Sequence[mp.mpf],
    previous: Sequence[mp.mpf],
    col: int,
    equation_form: str,
    zero_tolerance: mp.mpf,
) -> mp.mpf:
    diagonal = mp.mpf("0")
    subtotal = mp.mpf("0")
    for row, value in columns[col]:
        if row == col:
            diagonal = value
        elif row < col:
            subtotal += value * current[row]
        else:
            subtotal += value * previous[row]

    denominator = -diagonal if equation_form == "direct_q" else mp.mpf("1") - diagonal
    if abs(denominator) <= zero_tolerance:
        raise RuntimeError("Stationary iteration encountered a zero diagonal entry.")
    return subtotal / denominator
