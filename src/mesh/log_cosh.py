from dataclasses import dataclass

import numpy as np
from scipy.optimize import fsolve


@dataclass
class LogCoshAxisMesh:
    auxiliary_coordinates: np.ndarray
    stretched_coordinates: np.ndarray
    auxiliary_bounds: np.ndarray


def stretch_transform(x, stretch_start, stretch_end, alpha, domain_length):
    return (
        x
        + x / alpha
        * (
            np.log(np.cosh(alpha * (x - stretch_start) / domain_length))
            + np.log(np.cosh(alpha * (x - stretch_end) / domain_length))
            - 2.0 * np.log(np.cosh(alpha * (stretch_end - stretch_start) / (2.0 * domain_length)))
        )
    )


def boundary_system(bounds, lower, upper, stretch_start, stretch_end, alpha):
    lower_aux, upper_aux = bounds
    domain_length = upper_aux - lower_aux

    equation_1 = stretch_transform(lower_aux, stretch_start, stretch_end, alpha, domain_length) - lower
    equation_2 = stretch_transform(upper_aux, stretch_start, stretch_end, alpha, domain_length) - upper
    return [equation_1, equation_2]


def build_log_cosh_axis_mesh(
    lower: float,
    upper: float,
    stretch_start: float,
    stretch_end: float,
    alpha: float,
    num_boundary_points: int,
):
    lower_aux, upper_aux = fsolve(
        boundary_system,
        [lower, upper],
        args=(lower, upper, stretch_start, stretch_end, alpha),
    )
    auxiliary_bounds = np.array([lower_aux, upper_aux])
    auxiliary_coordinates = np.linspace(lower_aux, upper_aux, num_boundary_points)
    stretched_coordinates = stretch_transform(
        auxiliary_coordinates,
        stretch_start,
        stretch_end,
        alpha,
        upper_aux - lower_aux,
    )

    return LogCoshAxisMesh(
        auxiliary_coordinates=auxiliary_coordinates,
        stretched_coordinates=stretched_coordinates,
        auxiliary_bounds=auxiliary_bounds,
    )


def print_mesh_stats(deltas, reference_length, labels=("x", "y")):
    for axis, label in enumerate(labels):
        abs_delta = np.abs(deltas[axis])
        print(
            f"{label}: max Δ/RL={np.max(abs_delta)/reference_length:.3e}, "
            f"min Δ/RL={np.min(abs_delta)/reference_length:.3e}, "
            f"ratio={np.max(abs_delta)/np.min(abs_delta):.3e}"
        )
