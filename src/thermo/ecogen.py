from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class EquationOfStateParameters:
    name: str
    eos_type: str
    gamma: float
    cv: float
    cp: float
    p_inf: float
    energy_ref: float
    entropy_ref: float


def read_ecogen_eos(eos_path: Path) -> EquationOfStateParameters:
    xml_root = ET.parse(eos_path).getroot()
    eos_node = xml_root.find("EOS")
    parameter_node = xml_root.find("parameters")

    if eos_node is None or parameter_node is None:
        raise ValueError(f"Invalid EOS file: {eos_path}")

    gamma = float(parameter_node.attrib["gamma"])
    cv = float(parameter_node.attrib["cv"])

    return EquationOfStateParameters(
        name=eos_path.stem,
        eos_type=eos_node.attrib.get("type", ""),
        gamma=gamma,
        cv=cv,
        cp=gamma * cv,
        p_inf=float(parameter_node.attrib.get("pInf", "0.0")),
        energy_ref=float(parameter_node.attrib.get("energyRef", "0.0")),
        entropy_ref=float(parameter_node.attrib.get("entropyRef", "0.0")),
    )


def compute_fluid_thermo_fields(
    pressure: np.ndarray,
    temperature: np.ndarray,
    eos: EquationOfStateParameters,
):
    denominator = pressure + eos.p_inf
    energy = np.full_like(pressure, np.nan, dtype=float)
    energy_mask = denominator != 0.0
    energy[energy_mask] = (
        (pressure[energy_mask] + eos.gamma * eos.p_inf)
        / denominator[energy_mask]
        * eos.cv
        * temperature[energy_mask]
        + eos.energy_ref
    )

    enthalpy = eos.gamma * eos.cv * temperature + eos.energy_ref

    entropy = np.full_like(pressure, np.nan, dtype=float)
    entropy_mask = (temperature > 0.0) & (denominator > 0.0)
    entropy[entropy_mask] = eos.cv * np.log(
        (temperature[entropy_mask] ** eos.gamma)
        / denominator[entropy_mask] ** (eos.gamma - 1.0)
    ) + eos.entropy_ref

    # Preserved from the current Matlab implementation for consistency.
    gibbs = enthalpy - temperature * enthalpy

    return {
        "e": energy,
        "h": enthalpy,
        "s": entropy,
        "g": gibbs,
    }


def build_mixture_properties(
    fluid_fields: list[dict[str, np.ndarray]],
    eos_parameters: list[EquationOfStateParameters],
):
    if not fluid_fields:
        return {}

    array_shape = fluid_fields[0]["alpha"].shape
    mixture = {
        name: np.zeros(array_shape)
        for name in ("Gama", "mCp", "PI", "Q", "Rho", "ET", "S", "H", "G", "E")
    }

    gamma_coefficients = np.array(
        [1.0 / (eos.gamma - 1.0) for eos in eos_parameters],
        dtype=float,
    )
    pressure_offsets = np.array(
        [eos.p_inf * eos.gamma / (eos.gamma - 1.0) for eos in eos_parameters],
        dtype=float,
    )

    for fluid_index, (fields, eos) in enumerate(zip(fluid_fields, eos_parameters)):
        mixture["Rho"] += fields["alpha_rho"]
        mixture["Gama"] += fields["alpha"] * gamma_coefficients[fluid_index]
        mixture["PI"] += fields["alpha"] * pressure_offsets[fluid_index]
        mixture["Q"] += fields["alpha_rho"] * eos.energy_ref
        mixture["mCp"] += fields["alpha_rho"] * eos.cp
        mixture["ET"] += _safe_weighted_product(fields["alpha_rho"], fields["e"])
        mixture["S"] += _safe_weighted_product(fields["alpha_rho"], fields["s"])
        mixture["H"] += _safe_weighted_product(fields["alpha_rho"], fields["h"])
        mixture["G"] += _safe_weighted_product(fields["alpha_rho"], fields["g"])

    mixture["E"] = mixture["Gama"] * fluid_fields[0]["p"] + mixture["PI"] + mixture["Q"]
    return mixture


def _safe_weighted_product(weights: np.ndarray, values: np.ndarray):
    weighted_values = np.zeros_like(weights)
    active_mask = weights != 0.0
    weighted_values[active_mask] = weights[active_mask] * values[active_mask]
    return weighted_values
