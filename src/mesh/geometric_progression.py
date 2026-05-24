from dataclasses import dataclass

import numpy as np
from scipy.optimize import root_scalar


@dataclass
class GeometricAxisMesh:
    coordinates: np.ndarray
    spacings: np.ndarray
    refined_bounds: np.ndarray
    num_stretched: np.ndarray
    stretch_ratios: np.ndarray
    stretched_lengths: np.ndarray


def geometric_series_terms(initial_term: float, ratio: float, count: int):
    indices = np.arange(count)
    return initial_term * ratio**indices


def geometric_series_sum(initial_term: float, ratio: float, count: int):
    if np.isclose(ratio, 1.0):
        return initial_term * count
    return initial_term * (ratio**count - 1.0) / (ratio - 1.0)


def equivalent_initial_term_from_last_term(last_term: float, ratio: float, count: int):
    if np.isclose(ratio, 1.0):
        return last_term / count
    return last_term * (ratio - 1.0) / (ratio**count - 1.0)


def solve_geometric_ratio(
    first_spacing: float,
    num_cells: int,
    stretched_length: float,
    tol: float = 1.0e-12,
):
    def residual(ratio):
        log_term = num_cells * np.log(ratio)
        if log_term > 700.0:
            geometric_tail = np.inf
        else:
            geometric_tail = np.expm1(log_term)
        return stretched_length * (ratio - 1.0) - first_spacing * geometric_tail

    if num_cells == 0 or stretched_length == 0 or np.abs(stretched_length - num_cells * first_spacing) < tol:
        return 1.0
    if stretched_length < num_cells * first_spacing:
        solution = root_scalar(residual, bracket=[tol, 1.0 - tol])
    else:
        solution = root_scalar(residual, bracket=[1.0 + tol, 1.0e5 - tol])

    return solution.root


def build_geometric_axis_mesh(
    lower: float,
    upper: float,
    refined_lower: float,
    refined_upper: float,
    num_elements: int,
    num_refined_elements: int,
):
    stretched_lengths = np.array([refined_lower - lower, upper - refined_upper])
    num_stretched = np.zeros(2, dtype=int)

    if stretched_lengths[0] <= 0:
        num_stretched[0] = 0
        num_stretched[1] = num_elements - num_refined_elements
        refined_lower = lower
    elif stretched_lengths[1] <= 0:
        num_stretched[0] = num_elements - num_refined_elements
        num_stretched[1] = 0
        refined_upper = upper
    else:
        matrix = np.array(
            [
                [1.0, -stretched_lengths[0] / stretched_lengths[1]],
                [1.0, 1.0],
            ]
        )
        rhs = np.array([0.0, num_elements - num_refined_elements])
        num_stretched[:] = np.rint(np.linalg.solve(matrix, rhs)).astype(int)

    refined_spacings = np.full(
        num_refined_elements,
        (refined_upper - refined_lower) / num_refined_elements,
    )
    first_spacing = refined_spacings[0]

    stretch_ratios = np.array(
        [
            solve_geometric_ratio(first_spacing, num_stretched[0], stretched_lengths[0]),
            solve_geometric_ratio(first_spacing, num_stretched[1], stretched_lengths[1]),
        ]
    )

    left_spacings = (
        first_spacing * stretch_ratios[0] ** (num_stretched[0] - 1 - np.arange(num_stretched[0]))
        if num_stretched[0] > 0
        else np.array([])
    )
    right_spacings = (
        first_spacing * stretch_ratios[1] ** np.arange(num_stretched[1])
        if num_stretched[1] > 0
        else np.array([])
    )

    spacings = np.concatenate([left_spacings, refined_spacings, right_spacings])
    coordinates = np.cumsum(np.concatenate([[lower], spacings]))

    return GeometricAxisMesh(
        coordinates=coordinates,
        spacings=spacings,
        refined_bounds=np.array([refined_lower, refined_upper]),
        num_stretched=num_stretched,
        stretch_ratios=stretch_ratios,
        stretched_lengths=stretched_lengths,
    )
