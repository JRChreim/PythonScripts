from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class KellerMiksisCase:
    liquid_pressure: float
    bubble_pressure: float
    liquid_temperature: float
    bubble_temperature: float
    initial_radius: float
    liquid_cv: float = 1816.0
    liquid_cp: float = 4267.0
    liquid_pi_inf: float = 1.0e9
    bubble_cv: float = 717.5
    bubble_cp: float = 1006.0
    liquid_viscosity: float = 5.0e-2
    surface_tension: float = 72.8e-3


def normalize_radius_history(time_values, radius_values):
    time_values = np.asarray(time_values, dtype=float)
    radius_values = np.asarray(radius_values, dtype=float)

    if time_values.ndim != 1 or radius_values.ndim != 1:
        raise ValueError("time_values and radius_values must be one-dimensional.")
    if time_values.size == 0 or radius_values.size == 0:
        raise ValueError("time_values and radius_values cannot be empty.")
    if time_values.size != radius_values.size:
        raise ValueError("time_values and radius_values must have the same length.")

    initial_radius = radius_values[0]
    collapse_index = int(np.argmin(radius_values))
    collapse_time = time_values[collapse_index]

    if initial_radius == 0.0:
        raise ValueError("Initial radius cannot be zero.")
    if collapse_time == 0.0:
        raise ValueError("Collapse time cannot be zero.")

    return time_values / collapse_time, radius_values / initial_radius, collapse_time


def solve_keller_miksis(
    case: KellerMiksisCase,
    heat_transfer_coefficient: float,
    min_normalized_time_end: float | None = None,
):
    liquid_gamma = case.liquid_cp / case.liquid_cv
    bubble_gas_constant = case.bubble_cp - case.bubble_cv

    liquid_density = (case.liquid_pressure + case.liquid_pi_inf) / (
        (liquid_gamma - 1.0) * case.liquid_cv * case.liquid_temperature
    )
    liquid_sound_speed = np.sqrt(
        liquid_gamma * (case.liquid_pressure + case.liquid_pi_inf) / liquid_density
    )

    initial_volume = 4.0 / 3.0 * np.pi * case.initial_radius**3
    bubble_mass = (
        case.bubble_pressure
        * initial_volume
        / (bubble_gas_constant * case.bubble_temperature)
    )
    initial_velocity = -(
        case.liquid_pressure - case.bubble_pressure
    ) / (liquid_sound_speed * liquid_density)
    rayleigh_collapse_time = 0.915 * case.initial_radius * np.sqrt(
        liquid_density / (case.liquid_pressure - case.bubble_pressure)
    )

    def rhs(_time, state):
        radius = max(state[0], case.initial_radius * 1.0e-6)
        radius_velocity = state[1]
        bubble_temperature = state[2]

        area = 4.0 * np.pi * radius**2
        volume = 4.0 / 3.0 * np.pi * radius**3
        volume_rate = area * radius_velocity

        bubble_pressure = bubble_mass * bubble_gas_constant * bubble_temperature / volume
        temperature_rate = -(
            bubble_pressure * volume_rate
            + heat_transfer_coefficient
            * area
            * (bubble_temperature - case.liquid_temperature)
        ) / (bubble_mass * case.bubble_cv)
        pressure_rate = bubble_mass * bubble_gas_constant / volume * (
            temperature_rate - bubble_temperature / volume * volume_rate
        )

        acceleration_factor = (
            1.0
            - radius_velocity / liquid_sound_speed
            + 4.0
            * case.liquid_viscosity
            / (liquid_density * liquid_sound_speed * radius)
        ) * liquid_density * radius
        inertia_term = -(
            1.0 - radius_velocity / (3.0 * liquid_sound_speed)
        ) * 1.5 * liquid_density * radius_velocity**2
        pressure_term = (1.0 + radius_velocity / liquid_sound_speed) * (
            bubble_pressure - case.liquid_pressure
        )
        surface_tension_term = -2.0 * case.surface_tension / radius
        viscosity_term = -4.0 * case.liquid_viscosity * radius_velocity / radius
        compressibility_term = radius / liquid_sound_speed * pressure_rate

        radius_acceleration = (
            inertia_term
            + pressure_term
            + surface_tension_term
            + viscosity_term
            + compressibility_term
        ) / acceleration_factor

        return np.array([radius_velocity, radius_acceleration, temperature_rate])

    def build_clustered_time_grid(time_end, collapse_time):
        early_end = max(0.85 * collapse_time, 1.0e-12)
        collapse_start = max(0.0, 0.85 * collapse_time)
        collapse_end = min(time_end, 1.10 * collapse_time)
        rebound_start = collapse_end

        early_segment = np.linspace(0.0, early_end, 700, endpoint=False)
        collapse_segment = np.linspace(collapse_start, collapse_end, 1800, endpoint=False)
        rebound_segment = np.linspace(rebound_start, time_end, 700)

        time_grid = np.unique(
            np.concatenate((early_segment, collapse_segment, rebound_segment))
        )
        if time_grid[0] != 0.0:
            time_grid = np.insert(time_grid, 0, 0.0)
        if time_grid[-1] != time_end:
            time_grid = np.append(time_grid, time_end)
        return time_grid

    time_span_factor = 1.5
    solution = None
    collapse_time_estimate = rayleigh_collapse_time
    for _ in range(4):
        time_end = time_span_factor * rayleigh_collapse_time
        time_eval = build_clustered_time_grid(time_end, collapse_time_estimate)
        trial_solution = solve_ivp(
            rhs,
            (0.0, time_end),
            np.array([case.initial_radius, initial_velocity, case.bubble_temperature]),
            method="BDF",
            t_eval=time_eval,
            rtol=1.0e-9,
            atol=1.0e-12,
        )
        if not trial_solution.success:
            raise RuntimeError(trial_solution.message)

        min_index = int(np.argmin(trial_solution.y[0]))
        collapse_time_estimate = trial_solution.t[min_index]
        solution = trial_solution
        if min_index < trial_solution.y.shape[1] - 1:
            break
        time_span_factor *= 1.5

    if solution is None:
        raise RuntimeError("Keller-Miksis solve did not produce a solution.")

    time_values = solution.t
    radius_values = solution.y[0]
    temperature_values = solution.y[2]

    normalized_time, normalized_radius, collapse_time = normalize_radius_history(
        time_values, radius_values
    )

    if (
        min_normalized_time_end is not None
        and normalized_time[-1] < min_normalized_time_end
    ):
        target_time_end = min_normalized_time_end * collapse_time
        if target_time_end > solution.t[-1]:
            continuation = solve_ivp(
                rhs,
                (solution.t[-1], target_time_end),
                solution.y[:, -1],
                method="BDF",
                t_eval=np.linspace(solution.t[-1], target_time_end, 700),
                rtol=1.0e-9,
                atol=1.0e-12,
            )
            if not continuation.success:
                raise RuntimeError(continuation.message)

            time_values = np.concatenate((time_values, continuation.t[1:]))
            radius_values = np.concatenate((radius_values, continuation.y[0, 1:]))
            temperature_values = np.concatenate((temperature_values, continuation.y[2, 1:]))

            normalized_time = time_values / collapse_time
            normalized_radius = radius_values / radius_values[0]

    return {
        "time": time_values,
        "radius": radius_values,
        "temperature": temperature_values,
        "normalized_time": normalized_time,
        "normalized_radius": normalized_radius,
        "collapse_time": collapse_time,
        "rayleigh_collapse_time": rayleigh_collapse_time,
    }


def build_ecogen_strong_collapse_case():
    return KellerMiksisCase(
        liquid_pressure=5.0e6,
        bubble_pressure=3550.0,
        liquid_temperature=298.15,
        bubble_temperature=298.15,
        initial_radius=30.0e-6,
    )


def build_mfc_strong_collapse_case():
    return KellerMiksisCase(
        liquid_pressure=5.0e6,
        bubble_pressure=3550.0,
        liquid_temperature=298.15,
        bubble_temperature=298.15,
        initial_radius=30.0e-6,
    )
