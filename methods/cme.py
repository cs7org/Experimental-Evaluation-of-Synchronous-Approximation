from __future__ import annotations

import ast
import csv
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp


_CME_PARAMS: dict[int, dict[str, float | list[float]]] = {
    5: {
        "n": 4,
        "a": [-12.570924715391572, -8.9459493511561234, 5.7052396582304334, -0.42700075590220854],
        "b": [-24.149411578903063, 12.903827014483584, 0.59860030216227711, -0.92616168782546149],
        "c": 16.249821001551826,
        "omega": 0.83755,
        "mu1": 4.557592394391726,
        "cv2": 0.026156884691722396,
    },
    10: {
        "n": 9,
        "a": [27.325815328445792, -147.14404021599182, -48.399443546845028, 62.727735756141612, 26.035429053495786, -14.101244470709341, -5.1262979148338932, 1.311568616747915, 0.16521372215734895],
        "b": [-181.37253635334127, -44.982351178394893, 103.49037643332591, 39.879103640243876, -32.452855418929083, -13.347024109086677, 4.97817460700854, 1.3157057636324412, -0.19696012412047637],
        "c": 97.20698037778287,
        "omega": 0.706979,
        "mu1": 6.6461034098580143,
        "cv2": 0.0057367735345383161,
    },
    15: {
        "n": 14,
        "a": [22.754285785260819, -341.26631777643809, -52.359647495328687, 252.62665083198758, 58.099093683154678, -153.0725850490069, -47.925262331345287, 67.9991520039169, 25.913641035478527, -18.826674456282774, -7.1172892160093646, 2.6525085790451239, 0.61876304256993209, -0.094331268320701656],
        "b": [-369.7186783079747, -40.940382757750513, 300.27683731971155, 57.551373051293943, -202.56913105501914, -54.873840423398853, 107.10827500060043, 37.698682108140552, -38.362518398254714, -15.069629623103296, 7.834504641260164, 2.5597877294363172, -0.66619512229568678, -0.073089752368185615],
        "c": 189.99865355665719,
        "omega": 0.62695,
        "mu1": 7.488987595040129,
        "cv2": 0.0024052513649265544,
    },
}

_CME_MATRIX_FILES = {
    20: "CME20.csv",
}

_CME_MATRIX_SCV = {
    20: 0.0012568106539700918,
}

_CME_MATRIX_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def solve(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    targets: list[dict],
    solver: str = "recursive",
    tolerance: float = 1e-11,
    max_iterations: int | None = None,
    restart: int | None = None,
) -> list[dict[str, float | str]]:
    """
    Solve the M/CME/2/K queue with the concentrated matrix-exponential
    representations used in the paper.

    Unlike the earlier placeholder implementations, this routine is not tied to
    K=10. It uses a finite-capacity block recursion across queue levels, so it
    works for any finite K >= 2 as long as the requested CME order is one of the
    available optimized orders (5, 10, 15).
    """
    results: list[dict[str, float | str]] = []
    for target in targets:
        order = int(target["phases"])
        name = str(target["name"])
        alpha, matrix_t = _cme_representation(order, service_time)
        _validate_target(order, target.get("scv"), alpha, matrix_t)

        queue_dist = _compute_queue_distribution(
            arrival_rate,
            capacity,
            alpha,
            matrix_t,
            solver=solver,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
        )
        p_idle, p_full, response_time = _metrics(queue_dist, arrival_rate)

        results.append({
            "approach": name,
            "p_idle": p_idle,
            "p_full": p_full,
            "response_time": response_time,
        })
    return results


def solve_improved(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    targets: list[dict],
) -> list[dict[str, float | str]]:
    return solve(
        arrival_rate=arrival_rate,
        service_time=service_time,
        capacity=capacity,
        targets=targets,
        solver="recursive_improved",
    )


def solve_qbd(
    arrival_rate: float,
    service_time: float,
    capacity: int,
    targets: list[dict],
    tolerance: float = 1e-11,
    max_iterations: int = 500,
) -> list[dict[str, float | str]]:
    return solve(
        arrival_rate=arrival_rate,
        service_time=service_time,
        capacity=capacity,
        targets=targets,
        solver="qbd",
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _cme_representation(order: int, service_time: float) -> tuple[np.ndarray, np.ndarray]:
    if order in _CME_MATRIX_FILES:
        alpha, matrix_t = _load_cme_matrix_representation(order)
        return alpha.copy(), matrix_t.copy() / service_time

    if order not in _CME_PARAMS:
        available = sorted([*list(_CME_PARAMS), *list(_CME_MATRIX_FILES)])
        raise NotImplementedError(f"CME parameters are only available for orders {available}, got {order}.")

    params = _CME_PARAMS[order]
    base_n = int(params["n"])
    omega = float(params["omega"])
    mu1 = float(params["mu1"]) / service_time
    c = float(params["c"])
    a_vals = [float(v) for v in params["a"]]  # type: ignore[index]
    b_vals = [float(v) for v in params["b"]]  # type: ignore[index]

    phase_count = 2 * base_n + 1
    alpha = np.zeros(phase_count, dtype=float)
    alpha[0] = c

    matrix_t = np.zeros((phase_count, phase_count), dtype=float)
    matrix_t[0, 0] = -mu1

    for k in range(1, base_n + 1):
        kw = k * omega
        denom = 1.0 + kw * kw
        a_k = a_vals[k - 1]
        b_k = b_vals[k - 1]

        alpha[2 * k - 1] = 0.5 * (a_k * (1.0 + kw) - b_k * (1.0 - kw)) / denom
        alpha[2 * k] = 0.5 * (a_k * (1.0 - kw) + b_k * (1.0 + kw)) / denom

        block_start = 2 * k - 1
        matrix_t[block_start:block_start + 2, block_start:block_start + 2] = mu1 * np.array(
            [[-1.0, -kw], [kw, -1.0]],
            dtype=float,
        )

    return alpha, matrix_t


def _cme_representation_mp(
    order: int,
    service_time: float,
) -> tuple[list[mp.mpf], list[list[mp.mpf]]]:
    """Build a CME representation without intermediate double rounding."""
    if order in _CME_MATRIX_FILES:
        alpha, matrix_t = _cme_representation(order, service_time)
        return (
            [mp.mpf(str(value)) for value in alpha],
            [[mp.mpf(str(value)) for value in row] for row in matrix_t],
        )

    if order not in _CME_PARAMS:
        available = sorted([*list(_CME_PARAMS), *list(_CME_MATRIX_FILES)])
        raise NotImplementedError(f"CME parameters are only available for orders {available}, got {order}.")

    params = _CME_PARAMS[order]
    base_n = int(params["n"])
    omega = mp.mpf(str(params["omega"]))
    mu1 = mp.mpf(str(params["mu1"])) / mp.mpf(str(service_time))
    c = mp.mpf(str(params["c"]))
    a_vals = [mp.mpf(str(value)) for value in params["a"]]  # type: ignore[index]
    b_vals = [mp.mpf(str(value)) for value in params["b"]]  # type: ignore[index]

    phase_count = 2 * base_n + 1
    alpha = [mp.mpf("0") for _ in range(phase_count)]
    alpha[0] = c

    matrix_t = [
        [mp.mpf("0") for _ in range(phase_count)]
        for _ in range(phase_count)
    ]
    matrix_t[0][0] = -mu1

    for k in range(1, base_n + 1):
        kw = mp.mpf(k) * omega
        denom = mp.mpf("1") + kw * kw
        a_k = a_vals[k - 1]
        b_k = b_vals[k - 1]

        alpha[2 * k - 1] = (
            mp.mpf("0.5") * (a_k * (mp.mpf("1") + kw) - b_k * (mp.mpf("1") - kw)) / denom
        )
        alpha[2 * k] = (
            mp.mpf("0.5") * (a_k * (mp.mpf("1") - kw) + b_k * (mp.mpf("1") + kw)) / denom
        )

        block_start = 2 * k - 1
        matrix_t[block_start][block_start] = -mu1
        matrix_t[block_start][block_start + 1] = -mu1 * kw
        matrix_t[block_start + 1][block_start] = mu1 * kw
        matrix_t[block_start + 1][block_start + 1] = -mu1

    return alpha, matrix_t


def _load_cme_matrix_representation(order: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _CME_MATRIX_CACHE.get(order)
    if cached is not None:
        return cached

    csv_path = _cme_matrix_path(order)
    rows = list(csv.reader(csv_path.open(newline="", encoding="utf-8-sig")))
    if len(rows) != 2:
        raise RuntimeError(f"Expected {csv_path} to contain alpha row and matrix row; got {len(rows)} rows.")

    alpha = np.array([float(value) for value in rows[0]], dtype=float)
    matrix_t = np.array(
        [
            ast.literal_eval(row_text.replace("{", "[").replace("}", "]"))
            for row_text in rows[1]
        ],
        dtype=float,
    )

    if matrix_t.shape != (len(alpha), len(alpha)):
        raise RuntimeError(
            f"CME order {order} matrix shape {matrix_t.shape} does not match alpha length {len(alpha)}.",
        )

    _CME_MATRIX_CACHE[order] = (alpha, matrix_t)
    return alpha, matrix_t


def _cme_matrix_path(order: int) -> Path:
    file_name = _CME_MATRIX_FILES[order]
    code_dir = Path(__file__).resolve().parents[2]
    candidates = [
        code_dir / "CME parameters" / file_name,
        Path(__file__).resolve().parents[1] / "CME parameters" / file_name,
    ]

    for path in candidates:
        if path.exists():
            return path

    locations = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find CME order {order} parameter file. Looked in: {locations}")


def _validate_target(
    order: int,
    target_scv: float | None,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> None:
    mean, scv = _moments(alpha, matrix_t)
    if not np.isclose(mean, 1.0, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"CME order {order} does not preserve mean 1.0; got {mean}.")

    param_scv = _stored_scv(order)
    if not np.isclose(scv, param_scv, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"CME order {order} moments do not match the stored CME SCV {param_scv}.")

    if target_scv is not None and not math.isclose(float(target_scv), scv, rel_tol=5e-2, abs_tol=5e-5):
        raise RuntimeError(
            f"CME order {order} has SCV {scv}, which is not consistent with requested target {target_scv}.",
        )


def _stored_scv(order: int) -> float:
    if order in _CME_PARAMS:
        return float(_CME_PARAMS[order]["cv2"])
    if order in _CME_MATRIX_SCV:
        return float(_CME_MATRIX_SCV[order])
    raise NotImplementedError(f"No stored SCV is available for CME order {order}.")


def _moments(alpha: np.ndarray, matrix_t: np.ndarray) -> tuple[float, float]:
    ones = np.ones((matrix_t.shape[0], 1), dtype=float)
    alpha_row = alpha.reshape(1, -1)
    first_solve = _solve_right(-matrix_t, ones)
    second_solve = _solve_right(-matrix_t, first_solve)

    mean = float((alpha_row @ first_solve)[0, 0])
    second = float((2.0 * alpha_row @ second_solve)[0, 0])
    scv = (second - mean * mean) / (mean * mean)
    return mean, scv


def _compute_queue_distribution(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
    *,
    solver: str = "recursive",
    tolerance: float = 1e-11,
    max_iterations: int | None = None,
    restart: int | None = None,
) -> np.ndarray:
    mode = solver.strip().lower()
    if mode in {"recursive", "direct"}:
        return _compute_queue_distribution_recursive(arrival_rate, capacity, alpha, matrix_t)
    if mode in {"recursive_improved", "improved"}:
        return _compute_queue_distribution_recursive_improved(arrival_rate, capacity, alpha, matrix_t)
    if mode in {"qbd", "matrix_geometric"}:
        return _compute_queue_distribution_qbd(
            arrival_rate,
            capacity,
            alpha,
            matrix_t,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
    if mode in {"iterative", "gmres"}:
        return _compute_queue_distribution_iterative(
            arrival_rate,
            capacity,
            alpha,
            matrix_t,
            tolerance=tolerance,
            max_iterations=max_iterations,
            restart=restart,
        )
    raise ValueError(f"Unsupported CME solver mode: {solver}")


def _compute_queue_distribution_recursive(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> np.ndarray:
    if capacity < 2:
        raise NotImplementedError("The CME solver expects capacity at least 2.")

    # Levels 2..K share the same block structure, with only the final level
    # differing by the absence of arrivals. This lets us compute the finite-K
    # stationary distribution by a stable backward recursion instead of solving
    # one huge signed sparse system directly.
    (
        q00,
        level0_to_1,
        level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    ) = _build_level_blocks(arrival_rate, alpha, matrix_t)

    level_factor: dict[int, np.ndarray] = {}
    level_factor[capacity - 1] = -up_block @ np.linalg.inv(full_block)

    for level in range(capacity - 2, 1, -1):
        level_factor[level] = -up_block @ np.linalg.inv(middle_block + level_factor[level + 1] @ down_block)

    level1_factor = -level1_to_2 @ np.linalg.inv(middle_block + level_factor[2] @ down_block)
    level1_row = -level0_to_1 @ np.linalg.inv(level1_within + level1_factor @ level2_to_1)

    idle_residual = float(q00[0, 0] + (level1_row @ level1_to_0)[0, 0])
    if abs(idle_residual) > 1e-8:
        raise RuntimeError(f"CME block recursion lost level-0 balance: residual={idle_residual}.")

    levels: list[np.ndarray] = [np.array([1.0], dtype=float)]
    levels.append(np.asarray(level1_row, dtype=float).ravel())

    current = np.asarray(level1_row @ level1_factor, dtype=float)
    levels.append(current.ravel())
    for level in range(2, capacity):
        current = np.asarray(current @ level_factor[level], dtype=float)
        levels.append(current.ravel())

    norm = sum(float(np.sum(level)) for level in levels)
    if norm <= 0.0:
        raise RuntimeError("CME block recursion produced an invalid normalization constant.")

    dist = np.zeros(capacity + 1, dtype=float)
    dist[0] = levels[0][0] / norm
    dist[1] = float(np.sum(levels[1])) / norm
    for customers in range(2, capacity + 1):
        dist[customers] = float(np.sum(levels[customers])) / norm
    return dist


def _compute_queue_distribution_recursive_improved(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> np.ndarray:
    if capacity < 2:
        raise NotImplementedError("The CME solver expects capacity at least 2.")

    (
        q00,
        level0_to_1,
        level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    ) = _build_level_blocks(arrival_rate, alpha, matrix_t)

    level_factor: dict[int, np.ndarray] = {}
    level_factor[capacity - 1] = -_solve_left(up_block, full_block)

    for level in range(capacity - 2, 1, -1):
        system = middle_block + level_factor[level + 1] @ down_block
        level_factor[level] = -_solve_left(up_block, system)

    level1_system = middle_block + level_factor[2] @ down_block
    level1_factor = -_solve_left(level1_to_2, level1_system)
    level0_system = level1_within + level1_factor @ level2_to_1
    level1_row = -_solve_left(level0_to_1, level0_system)

    idle_residual = float(q00[0, 0] + (level1_row @ level1_to_0)[0, 0])
    if abs(idle_residual) > 1e-8:
        raise RuntimeError(f"CME improved recursion lost level-0 balance: residual={idle_residual}.")

    levels: list[np.ndarray] = [np.array([1.0], dtype=float)]
    levels.append(np.asarray(level1_row, dtype=float).ravel())

    current = np.asarray(level1_row @ level1_factor, dtype=float)
    levels.append(current.ravel())
    for level in range(2, capacity):
        current = np.asarray(current @ level_factor[level], dtype=float)
        levels.append(current.ravel())

    dist = _aggregate_levels(levels, capacity)
    _validate_recursive_levels(
        levels,
        level0_to_1,
        level1_within,
        level1_to_2,
        level2_to_1,
        middle_block,
        full_block,
        up_block,
        down_block,
    )
    return dist


def _compute_queue_distribution_iterative(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int | None,
    restart: int | None,
) -> np.ndarray:
    if capacity < 2:
        raise NotImplementedError("The CME solver expects capacity at least 2.")

    (
        _q00,
        level0_to_1,
        _level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    ) = _build_level_blocks(arrival_rate, alpha, matrix_t)

    phase_count = len(alpha)
    both_dim = phase_count * phase_count
    inv_level1 = np.linalg.inv(level1_within)
    inv_middle = np.linalg.inv(middle_block)
    inv_full = np.linalg.inv(full_block)

    levels: list[np.ndarray] = [np.array([1.0], dtype=float)]
    level_weights = np.exp(-np.arange(1, capacity + 1, dtype=float))
    levels.append(np.full(2 * phase_count, level_weights[0] / (2 * phase_count), dtype=float))
    for index in range(2, capacity + 1):
        levels.append(np.full(both_dim, level_weights[index - 1] / both_dim, dtype=float))

    previous_dist = _aggregate_levels(levels, capacity)
    effective_maxiter = max_iterations if max_iterations is not None else 2000

    for _ in range(effective_maxiter):
        updated: list[np.ndarray] = [np.array([1.0], dtype=float)]
        level1 = -(levels[0] @ level0_to_1 + levels[2] @ level2_to_1) @ inv_level1
        updated.append(np.asarray(level1, dtype=float).ravel())

        if capacity == 2:
            last = -(updated[1] @ level1_to_2) @ inv_full
            updated.append(np.asarray(last, dtype=float).ravel())
        else:
            level2 = -(updated[1] @ level1_to_2 + levels[3] @ down_block) @ inv_middle
            updated.append(np.asarray(level2, dtype=float).ravel())

            for level in range(3, capacity):
                block = -(updated[level - 1] @ up_block + levels[level + 1] @ down_block) @ inv_middle
                updated.append(np.asarray(block, dtype=float).ravel())

            last = -(updated[capacity - 1] @ up_block) @ inv_full
            updated.append(np.asarray(last, dtype=float).ravel())

        current_dist = _aggregate_levels(updated, capacity)
        if np.linalg.norm(current_dist - previous_dist, ord=1) < tolerance:
            return current_dist

        levels = updated
        previous_dist = current_dist

    raise RuntimeError("CME iterative solver did not converge within the iteration limit.")


def _compute_queue_distribution_qbd(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int | None,
) -> np.ndarray:
    if capacity < 2:
        raise NotImplementedError("The CME solver expects capacity at least 2.")

    (
        q00,
        level0_to_1,
        level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    ) = _build_level_blocks(arrival_rate, alpha, matrix_t)

    phase_count = len(alpha)
    both_dim = phase_count * phase_count
    effective_maxiter = max_iterations if max_iterations is not None else 500

    if capacity == 2:
        boundary_matrix = np.block([
            [level1_within, level1_to_2],
            [level2_to_1, full_block],
        ])
        rhs = np.hstack([-level0_to_1, np.zeros((1, both_dim), dtype=float)])
        boundary_row = _solve_left(rhs, boundary_matrix)
        pi1 = boundary_row[:, : 2 * phase_count]
        pi2 = boundary_row[:, 2 * phase_count :]
        levels = [np.array([1.0], dtype=float), np.asarray(pi1, dtype=float).ravel(), np.asarray(pi2, dtype=float).ravel()]
        return _aggregate_levels(levels, capacity)

    tail_factor = -_solve_left(up_block, full_block)
    rate_matrix = _solve_qbd_rate_matrix(
        up_block,
        middle_block,
        down_block,
        initial=tail_factor,
        tolerance=tolerance,
        max_iterations=effective_maxiter,
    )

    boundary_matrix = np.block([
        [level1_within, level1_to_2],
        [level2_to_1, middle_block + rate_matrix @ down_block],
    ])
    rhs = np.hstack([-level0_to_1, np.zeros((1, both_dim), dtype=float)])
    boundary_row = _solve_left(rhs, boundary_matrix)
    pi1 = np.asarray(boundary_row[:, : 2 * phase_count], dtype=float)
    pi2 = np.asarray(boundary_row[:, 2 * phase_count :], dtype=float)

    levels: list[np.ndarray] = [np.array([1.0], dtype=float), pi1.ravel(), pi2.ravel()]
    current = pi2
    for _customers in range(3, capacity):
        current = current @ rate_matrix
        levels.append(np.asarray(current, dtype=float).ravel())
    final_level = current @ tail_factor
    levels.append(np.asarray(final_level, dtype=float).ravel())

    idle_residual = float(q00[0, 0] + (pi1 @ level1_to_0)[0, 0])
    if abs(idle_residual) > 1e-6:
        raise RuntimeError(f"CME QBD lost level-0 balance: residual={idle_residual}.")
    return _aggregate_levels(levels, capacity)


def _solve_qbd_rate_matrix(
    up_block: np.ndarray,
    middle_block: np.ndarray,
    down_block: np.ndarray,
    *,
    initial: np.ndarray | None,
    tolerance: float,
    max_iterations: int,
) -> np.ndarray:
    seeds: list[np.ndarray] = [np.zeros_like(up_block)]
    if initial is not None:
        seeds.insert(0, np.array(initial, dtype=float, copy=True))

    for seed in seeds:
        rate = seed.copy()
        for omega in (1.0, 0.5, 0.25):
            current = rate.copy()
            for _ in range(max_iterations):
                raw_next = -_solve_left(up_block, middle_block + current @ down_block)
                next_rate = omega * raw_next + (1.0 - omega) * current
                if np.linalg.norm(next_rate - current, ord=1) < tolerance:
                    return next_rate
                current = next_rate
    raise RuntimeError("CME QBD rate-matrix iteration did not converge.")


def _build_balance_matrix(
    arrival_rate: float,
    capacity: int,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> sp.csr_matrix:
    (
        q00,
        level0_to_1,
        level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    ) = _build_level_blocks(arrival_rate, alpha, matrix_t)

    phase_count = len(alpha)
    both_dim = phase_count * phase_count
    block_count = capacity + 1
    blocks: list[list[sp.spmatrix | None]] = [[None for _ in range(block_count)] for _ in range(block_count)]

    blocks[0][0] = sp.csr_matrix(q00)
    blocks[0][1] = sp.csr_matrix(level0_to_1)
    blocks[1][0] = sp.csr_matrix(level1_to_0)
    blocks[1][1] = sp.csr_matrix(level1_within)
    blocks[1][2] = sp.csr_matrix(level1_to_2)

    down_sparse = sp.csr_matrix(down_block)
    middle_sparse = sp.csr_matrix(middle_block)
    full_sparse = sp.csr_matrix(full_block)
    up_sparse = sp.csr_matrix(up_block)
    level2_to_1_sparse = sp.csr_matrix(level2_to_1)

    for level in range(2, capacity + 1):
        blocks[level][level - 1] = level2_to_1_sparse if level == 2 else down_sparse
        blocks[level][level] = full_sparse if level == capacity else middle_sparse
        if level < capacity:
            blocks[level][level + 1] = up_sparse

    return sp.bmat(blocks, format="csr")


def _level_slices(phase_count: int, capacity: int) -> list[slice]:
    both_dim = phase_count * phase_count
    slices = [slice(0, 1), slice(1, 1 + 2 * phase_count)]
    start = 1 + 2 * phase_count
    for _ in range(2, capacity + 1):
        slices.append(slice(start, start + both_dim))
        start += both_dim
    return slices


def _aggregate_levels(levels: list[np.ndarray], capacity: int) -> np.ndarray:
    norm = sum(float(np.sum(level)) for level in levels)
    if norm <= 0.0:
        raise RuntimeError("CME iterative sweep produced an invalid normalization constant.")

    dist = np.zeros(capacity + 1, dtype=float)
    dist[0] = float(np.sum(levels[0])) / norm
    dist[1] = float(np.sum(levels[1])) / norm
    for customers in range(2, capacity + 1):
        dist[customers] = float(np.sum(levels[customers])) / norm
    return dist


def _solve_right(system: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lu, piv = la.lu_factor(system, check_finite=False)
    return la.lu_solve((lu, piv), rhs, check_finite=False)


def _solve_left(lhs: np.ndarray, system: np.ndarray) -> np.ndarray:
    return _solve_right(system.T, lhs.T).T


def _validate_recursive_levels(
    levels: list[np.ndarray],
    level0_to_1: np.ndarray,
    level1_within: np.ndarray,
    level1_to_2: np.ndarray,
    level2_to_1: np.ndarray,
    middle_block: np.ndarray,
    full_block: np.ndarray,
    up_block: np.ndarray,
    down_block: np.ndarray,
) -> None:
    level1_residual = levels[0] @ level0_to_1 + levels[1] @ level1_within + levels[2] @ level2_to_1
    if float(np.linalg.norm(level1_residual, ord=1)) > 1e-6:
        raise RuntimeError("CME improved recursion failed level-1 residual validation.")

    level2_residual = levels[1] @ level1_to_2 + levels[2] @ middle_block + levels[3] @ down_block
    if float(np.linalg.norm(level2_residual, ord=1)) > 1e-6:
        raise RuntimeError("CME improved recursion failed level-2 residual validation.")

    for level in range(3, len(levels) - 1):
        current_residual = levels[level - 1] @ up_block + levels[level] @ middle_block + levels[level + 1] @ down_block
        if float(np.linalg.norm(current_residual, ord=1)) > 1e-6:
            raise RuntimeError(f"CME improved recursion failed level-{level} residual validation.")

    final_residual = levels[-2] @ up_block + levels[-1] @ full_block
    if float(np.linalg.norm(final_residual, ord=1)) > 1e-6:
        raise RuntimeError("CME improved recursion failed final-level residual validation.")


def _build_level_blocks(
    arrival_rate: float,
    alpha: np.ndarray,
    matrix_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_count = len(alpha)
    completion = -matrix_t @ np.ones(phase_count, dtype=float)
    both_dim = phase_count * phase_count

    q00 = np.array([[-arrival_rate]], dtype=float)

    level0_to_1 = np.zeros((1, 2 * phase_count), dtype=float)
    level0_to_1[0, :phase_count] = arrival_rate / 2.0 * alpha
    level0_to_1[0, phase_count:] = arrival_rate / 2.0 * alpha

    level1_to_0 = np.zeros((2 * phase_count, 1), dtype=float)
    level1_to_0[:phase_count, 0] = completion
    level1_to_0[phase_count:, 0] = completion

    level1_within = np.zeros((2 * phase_count, 2 * phase_count), dtype=float)
    level1_within[:phase_count, :phase_count] = matrix_t - arrival_rate * np.eye(phase_count)
    level1_within[phase_count:, phase_count:] = matrix_t - arrival_rate * np.eye(phase_count)

    level1_to_2 = np.zeros((2 * phase_count, both_dim), dtype=float)
    for i in range(phase_count):
        for j in range(phase_count):
            level1_to_2[i, i * phase_count + j] = arrival_rate * alpha[j]
            level1_to_2[phase_count + i, j * phase_count + i] = arrival_rate * alpha[j]

    level2_to_1 = np.zeros((both_dim, 2 * phase_count), dtype=float)
    for a in range(phase_count):
        for b in range(phase_count):
            row = a * phase_count + b
            level2_to_1[row, phase_count + b] += completion[a]
            level2_to_1[row, a] += completion[b]

    down_block = np.zeros((both_dim, both_dim), dtype=float)
    for a in range(phase_count):
        for b in range(phase_count):
            row = a * phase_count + b
            for j in range(phase_count):
                down_block[row, j * phase_count + b] += completion[a] * alpha[j]
                down_block[row, a * phase_count + j] += completion[b] * alpha[j]

    middle_block = (
        np.kron(matrix_t, np.eye(phase_count))
        + np.kron(np.eye(phase_count), matrix_t)
        - arrival_rate * np.eye(both_dim)
    )
    full_block = middle_block + arrival_rate * np.eye(both_dim)
    up_block = arrival_rate * np.eye(both_dim)

    return (
        q00,
        level0_to_1,
        level1_to_0,
        level1_within,
        level1_to_2,
        level2_to_1,
        down_block,
        middle_block,
        full_block,
        up_block,
    )


def _metrics(dist: np.ndarray, arrival_rate: float) -> tuple[float, float, float]:
    cleaned = np.real_if_close(dist).astype(float)
    cleaned[cleaned < 0.0] = 0.0

    total = float(cleaned.sum())
    if total <= 0.0:
        raise RuntimeError("Aggregated CME distribution is invalid.")

    probabilities = cleaned / total
    capacity = len(probabilities) - 1
    p_idle = float(probabilities[0])
    p_full = max(float(probabilities[capacity]), 0.0)
    mean_n = float(np.dot(np.arange(capacity + 1), probabilities))
    throughput = arrival_rate * (1.0 - p_full)
    response_time = mean_n / throughput if throughput > 0 else float("inf")
    return p_idle, p_full, response_time
