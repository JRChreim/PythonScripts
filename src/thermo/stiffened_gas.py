from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


ROOT_SEARCH_MIN = 1.0e-8
ROOT_SEARCH_MAX = 1.0e14
ROOT_SEARCH_POINTS = 800
TEMPERATURE_SEARCH_MIN = 1.0
TEMPERATURE_SEARCH_MAX = 5.0e3
TEMPERATURE_SEARCH_POINTS = 800
SOLVER_XTOL = 1.0e-14
SOLVER_MAXFEV = 1_000_000
SATURATION_PRESSURE_GUESS = 1.0e5
SATURATION_TEMPERATURE_GUESS = 400.0
THREE_PHASE_PRESSURE_GUESS = 1.0e4
THREE_PHASE_M1_SAMPLES = 100


@dataclass(frozen=True)
class PhaseParameters:
    name: str
    p_inf: float
    q: float
    qp: float
    cv: float
    cp: float
    color: tuple[float, float, float]

    @property
    def gamma(self) -> float:
        return self.cp / self.cv

    @property
    def pressure_offset(self) -> float:
        return self.p_inf * self.gamma / (self.gamma - 1.0)


@dataclass(frozen=True)
class Preset:
    name: str
    phases: tuple[PhaseParameters, ...]
    initial_volume_fractions: np.ndarray
    reference_temperature: float
    reference_pressure: float
    reference_entropy: np.ndarray | None = None


@dataclass(frozen=True)
class PhaseFields:
    rho: np.ndarray
    rhoe: np.ndarray
    s: np.ndarray
    s_reference: np.ndarray
    h: np.ndarray
    e: np.ndarray
    g: np.ndarray


@dataclass(frozen=True)
class SaturationCurve:
    temperature: np.ndarray
    pressure: np.ndarray
    rho: np.ndarray
    mass_density: np.ndarray
    internal_energy: np.ndarray
    total_energy_density: np.ndarray


@dataclass(frozen=True)
class ThreePhaseBranch:
    m1: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    mass_density: np.ndarray
    rho: np.ndarray
    internal_energy: np.ndarray
    total_energy_density: np.ndarray
    alpha: np.ndarray
    alpha_rho: np.ndarray


@dataclass(frozen=True)
class SaturationConstants:
    A: float
    B: float
    C: float
    D: float


def compute_reference_entropy(
    phases: tuple[PhaseParameters, ...],
    reference_temperature: float,
    reference_pressure: float,
) -> np.ndarray:
    reference_entropy = []
    for phase in phases:
        gamma = phase.gamma
        reference_entropy.append(
            phase.qp
            - phase.cp
            * np.log(
                ((reference_pressure + phase.p_inf) ** ((gamma - 1.0) / gamma))
                / reference_temperature
            )
        )
    return np.asarray(reference_entropy, dtype=float)


def compute_phase_fields(
    pressure_grid: np.ndarray,
    temperature_grid: np.ndarray,
    phase: PhaseParameters,
    reference_entropy: float,
    reference_temperature: float,
    reference_pressure: float,
) -> PhaseFields:
    gamma = phase.gamma
    pressure_plus = pressure_grid + phase.p_inf

    rho = pressure_plus / (temperature_grid * phase.cv * (gamma - 1.0))
    rhoe = pressure_grid / (gamma - 1.0) + phase.pressure_offset + phase.q * rho
    s = phase.cv * np.log(
        temperature_grid**gamma / pressure_plus ** (gamma - 1.0)
    ) + phase.qp
    s_reference = phase.cp * np.log(
        (temperature_grid / reference_temperature)
        * ((reference_pressure + phase.p_inf) / pressure_plus)
        ** ((gamma - 1.0) / gamma)
    ) + reference_entropy
    h = gamma * phase.cv * temperature_grid + phase.q
    e = (
        (pressure_grid + gamma * phase.p_inf) / pressure_plus * phase.cv * temperature_grid
        + phase.q
    )
    g = h - temperature_grid * s

    return PhaseFields(
        rho=rho,
        rhoe=rhoe,
        s=s,
        s_reference=s_reference,
        h=h,
        e=e,
        g=g,
    )


def build_saturation_constants(
    liquid: PhaseParameters, vapor: PhaseParameters
) -> SaturationConstants:
    gamma_liquid = liquid.gamma
    gamma_vapor = vapor.gamma
    denominator = (gamma_vapor - 1.0) * vapor.cv

    return SaturationConstants(
        A=(
            gamma_liquid * liquid.cv
            - gamma_vapor * vapor.cv
            + vapor.qp
            - liquid.qp
        )
        / denominator,
        B=(liquid.q - vapor.q) / denominator,
        C=(gamma_vapor * vapor.cv - gamma_liquid * liquid.cv) / denominator,
        D=((gamma_liquid - 1.0) * liquid.cv) / denominator,
    )


def saturation_residual(
    pressure: float,
    temperature: float,
    liquid: PhaseParameters,
    vapor: PhaseParameters,
    constants: SaturationConstants,
) -> float:
    return (
        constants.A
        + constants.B / temperature
        + constants.C * np.log(temperature)
        + constants.D * np.log(pressure + liquid.p_inf)
        - np.log(pressure + vapor.p_inf)
    )


def _solve_root_from_grid(
    residual: Callable[[float], float],
    initial_guess: float,
    *,
    search_min: float,
    search_max: float,
    search_points: int,
) -> float:
    guess = max(float(initial_guess), search_min)
    search_grid = np.logspace(
        np.log10(search_min),
        np.log10(search_max),
        search_points,
    )
    residual_values = np.array(
        [residual(float(value)) for value in search_grid],
        dtype=float,
    )

    valid_mask = np.isfinite(residual_values)
    search_grid = search_grid[valid_mask]
    residual_values = residual_values[valid_mask]
    if search_grid.size < 2:
        raise RuntimeError("No valid samples were available for root bracketing.")

    brackets: list[tuple[float, float]] = []
    for lower_value, upper_value, lower_residual, upper_residual in zip(
        search_grid[:-1],
        search_grid[1:],
        residual_values[:-1],
        residual_values[1:],
        strict=True,
    ):
        if lower_residual == 0.0:
            brackets.append((float(lower_value), float(lower_value)))
        elif upper_residual == 0.0:
            brackets.append((float(upper_value), float(upper_value)))
        elif np.sign(lower_residual) != np.sign(upper_residual):
            brackets.append((float(lower_value), float(upper_value)))

    if not brackets:
        raise RuntimeError(
            "No positive root was bracketed in the search interval "
            f"[{search_min:g}, {search_max:g}]."
        )

    roots: list[float] = []
    for lower_value, upper_value in brackets:
        if lower_value == upper_value:
            roots.append(lower_value)
        else:
            roots.append(
                brentq(
                    residual,
                    lower_value,
                    upper_value,
                    xtol=SOLVER_XTOL,
                    maxiter=SOLVER_MAXFEV,
                )
            )

    return min(roots, key=lambda candidate: abs(np.log(candidate / guess)))


def solve_positive_root(
    residual: Callable[[float], float],
    initial_guess: float,
) -> float:
    return _solve_root_from_grid(
        residual,
        initial_guess,
        search_min=ROOT_SEARCH_MIN,
        search_max=ROOT_SEARCH_MAX,
        search_points=ROOT_SEARCH_POINTS,
    )


def solve_saturation_temperature(
    pressure: float,
    liquid: PhaseParameters,
    vapor: PhaseParameters,
    constants: SaturationConstants,
    initial_guess: float = SATURATION_TEMPERATURE_GUESS,
) -> float:
    return _solve_root_from_grid(
        lambda temperature: saturation_residual(
            pressure,
            temperature,
            liquid,
            vapor,
            constants,
        ),
        initial_guess,
        search_min=TEMPERATURE_SEARCH_MIN,
        search_max=TEMPERATURE_SEARCH_MAX,
        search_points=TEMPERATURE_SEARCH_POINTS,
    )


def solve_saturation_curve(
    temperatures: np.ndarray,
    liquid: PhaseParameters,
    vapor: PhaseParameters,
    constants: SaturationConstants,
    initial_guess: float = SATURATION_PRESSURE_GUESS,
    *,
    report: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    temperature_values: list[float] = []
    pressure_values: list[float] = []
    guess = float(initial_guess)

    for temperature in temperatures:
        try:
            pressure = solve_positive_root(
                lambda pressure, t=float(temperature): saturation_residual(
                    pressure, t, liquid, vapor, constants
                ),
                guess,
            )
        except RuntimeError as exc:
            if report is not None:
                report(
                    f"Stopping the saturation sweep at T={float(temperature):.2f} K "
                    f"because no positive root was found: {exc}"
                )
            break

        temperature_values.append(float(temperature))
        pressure_values.append(float(pressure))
        guess = float(pressure)

    if not temperature_values:
        raise RuntimeError("The saturation sweep did not produce any valid points.")

    return (
        np.asarray(pressure_values, dtype=float),
        np.asarray(temperature_values, dtype=float),
    )


def build_saturation_curve(
    temperatures: np.ndarray,
    preset: Preset,
    constants: SaturationConstants,
    *,
    initial_guess: float = SATURATION_PRESSURE_GUESS,
    report: Callable[[str], None] | None = None,
) -> SaturationCurve:
    phases = preset.phases
    alpha = preset.initial_volume_fractions
    if len(phases) != len(alpha):
        raise ValueError(
            "The number of initial volume fractions must match the number of phases."
        )

    pressures, temperatures = solve_saturation_curve(
        temperatures,
        phases[0],
        phases[1],
        constants,
        initial_guess,
        report=report,
    )

    gamma = np.array([phase.gamma for phase in phases], dtype=float)[:, None]
    cv = np.array([phase.cv for phase in phases], dtype=float)[:, None]
    p_inf = np.array([phase.p_inf for phase in phases], dtype=float)[:, None]
    q = np.array([phase.q for phase in phases], dtype=float)[:, None]

    temperature_row = temperatures[None, :]
    pressure_row = pressures[None, :]

    rho = (pressure_row + p_inf) / ((gamma - 1.0) * cv * temperature_row)
    mass_density = alpha[:, None] * rho
    internal_energy = (
        (pressure_row + gamma * p_inf) / (pressure_row + p_inf) * cv * temperature_row
        + q
    )
    total_energy_density = np.sum(mass_density * internal_energy, axis=0)

    return SaturationCurve(
        temperature=temperatures,
        pressure=pressures,
        rho=rho,
        mass_density=mass_density,
        internal_energy=internal_energy,
        total_energy_density=total_energy_density,
    )


def build_three_phase_branch(
    preset: Preset,
    constants: SaturationConstants,
    saturation_curve: SaturationCurve,
    *,
    report: Callable[[str], None] | None = None,
) -> ThreePhaseBranch | None:
    if len(preset.phases) < 3:
        return None

    phases = preset.phases
    m12 = saturation_curve.mass_density[0, 0] + saturation_curve.mass_density[1, 1]
    m3 = saturation_curve.mass_density[2, 0]
    candidate_m1 = np.linspace(0.0, m12, THREE_PHASE_M1_SAMPLES)
    pressure_values: list[float] = []
    temperature_values: list[float] = []
    m1_values: list[float] = []
    guess = THREE_PHASE_PRESSURE_GUESS

    gamma = np.array([phase.gamma for phase in phases], dtype=float)
    cv = np.array([phase.cv for phase in phases], dtype=float)
    p_inf = np.array([phase.p_inf for phase in phases], dtype=float)
    q = np.array([phase.q for phase in phases], dtype=float)

    def mixture_temperature_argument(pressure: float, m1_value: float) -> float:
        phase_argument = (
            m1_value
            * (
                cv[0] * (gamma[0] - 1.0) / (pressure + p_inf[0])
                - cv[1] * (gamma[1] - 1.0) / (pressure + p_inf[1])
            )
            + m12 * cv[1] * (gamma[1] - 1.0) / (pressure + p_inf[1])
            + m3 * cv[2] * (gamma[2] - 1.0) / (pressure + p_inf[2])
        )
        if phase_argument <= 0.0:
            raise ValueError(
                "The mixture temperature argument became non-positive during "
                "the three-phase saturation solve."
            )
        return phase_argument

    for m1_value in candidate_m1:
        def residual(pressure: float, m_value: float = float(m1_value)) -> float:
            argument = mixture_temperature_argument(pressure, m_value)
            return (
                constants.A
                + constants.B * argument
                - constants.C * np.log(argument)
                + constants.D * np.log(pressure + p_inf[0])
                - np.log(pressure + p_inf[1])
            )

        try:
            pressure = solve_positive_root(residual, guess)
        except RuntimeError as exc:
            if report is not None:
                report(
                    "Stopping the three-phase saturation sweep at "
                    f"m_1={float(m1_value):.6e} kg/m^3 because no positive root was found: "
                    f"{exc}"
                )
            break

        pressure_values.append(float(pressure))
        m1_values.append(float(m1_value))
        guess = float(pressure)
        temperature_values.append(mixture_temperature_argument(float(pressure), float(m1_value)))

    if not pressure_values:
        return None

    m1 = np.asarray(m1_values, dtype=float)
    pressures = np.asarray(pressure_values, dtype=float)
    temperature = 1.0 / np.asarray(temperature_values, dtype=float)

    mass_density = np.vstack([m1, m12 - m1, np.full_like(m1, m3)])
    rho = (pressures[None, :] + p_inf[:, None]) / (
        (gamma[:, None] - 1.0) * cv[:, None] * temperature[None, :]
    )
    internal_energy = (
        (pressures[None, :] + gamma[:, None] * p_inf[:, None])
        / (pressures[None, :] + p_inf[:, None])
        * cv[:, None]
        * temperature[None, :]
        + q[:, None]
    )
    total_energy_density = np.sum(mass_density * internal_energy, axis=0)

    weighted_cp = np.sum(
        mass_density * np.array([phase.cp for phase in phases])[:, None],
        axis=0,
    )
    weighted_q = np.sum(mass_density * q[:, None], axis=0)
    alpha = (
        ((gamma - 1.0) / gamma)[:, None]
        * np.array([phase.cp for phase in phases], dtype=float)[:, None]
        * mass_density
        / weighted_cp
        * (total_energy_density + pressures - weighted_q)[None, :]
        / (pressures[None, :] + p_inf[:, None])
    )
    alpha_rho = alpha * rho

    return ThreePhaseBranch(
        m1=m1,
        pressure=pressures,
        temperature=temperature,
        mass_density=mass_density,
        rho=rho,
        internal_energy=internal_energy,
        total_energy_density=total_energy_density,
        alpha=alpha,
        alpha_rho=alpha_rho,
    )
