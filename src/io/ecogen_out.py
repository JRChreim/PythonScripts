from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from src.thermo.ecogen import (
    EquationOfStateParameters,
    build_mixture_properties,
    compute_fluid_thermo_fields,
    read_ecogen_eos,
)

RESULT_FILENAME_RE = re.compile(r"result_CPU(?P<cpu>\d+)_TIME(?P<time>\d+)\.(?P<suffix>\w+)$")
PHASE_FIELD_NAMES = ("alpha", "rho", "p", "T", "Y")
FLUID_DERIVED_FIELD_NAMES = ("alpha_rho", "e", "h", "s", "g")
MULTIPHASE_MIXTURE_FIELD_NAMES = ("Rho", "p", "E", "u", "velocityMagnitude", "amrLevel", "xi")
SINGLE_PHASE_MIXTURE_FIELD_NAMES = ("Rho", "p", "T", "u", "velocityMagnitude", "amrLevel", "xi")
SINGLE_PHASE_FLUID_RAW_FIELD_NAMES = ("rho", "p", "T", "u", "velocityMagnitude")
SINGLE_PHASE_FLUID_DERIVED_FIELD_NAMES = ("alpha", "Y", "alpha_rho", "e", "h", "s", "g")
MIXTURE_DERIVED_FIELD_NAMES = ("Gama", "mCp", "PI", "Q", "ET", "S", "H", "G", "E")


@dataclass(frozen=True)
class ECOGENCaseMetadata:
    name: str
    case_root: Path
    datasets_root: Path
    flow_model: str
    num_fluids: int
    eos_names: tuple[str, ...]
    dataset_extensions: tuple[str, ...]
    available_out_times: tuple[int, ...]
    available_out_cpus: tuple[int, ...]

    @property
    def has_out_datasets(self) -> bool:
        return bool(self.available_out_times)


@dataclass
class FieldBlock:
    fields: dict[str, np.ndarray]
    raw_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]


@dataclass
class FluidBlock:
    index: int
    eos_name: str
    eos_parameters: EquationOfStateParameters | None
    data: FieldBlock


@dataclass
class ECOGENOutCaseData:
    metadata: ECOGENCaseMetadata
    x: np.ndarray
    saved_times: np.ndarray
    fluids: list[FluidBlock]
    mixture: FieldBlock


def default_ecogen_repo_root() -> Path:
    return Path(__file__).resolve().parents[3] / "ECOGEN"


def default_ecogen_results_root() -> Path:
    return default_ecogen_repo_root() / "results"


def default_ecogen_eos_root() -> Path:
    return default_ecogen_repo_root() / "libEOS"


def discover_ecogen_cases(results_root: Path | None = None):
    if results_root is None:
        results_root = default_ecogen_results_root()

    metadata_by_name = {}
    for case_root in sorted(path for path in results_root.iterdir() if path.is_dir()):
        datasets_root = case_root / "datasets"
        model_root = case_root / "savesInput"

        flow_model, num_fluids, eos_names = read_ecogen_case_model_metadata(model_root)
        dataset_files = sorted(
            file_path for file_path in datasets_root.glob("result_CPU*_TIME*.*") if file_path.is_file()
        )
        dataset_extensions = tuple(sorted({file_path.suffix for file_path in dataset_files}))

        out_times = set()
        out_cpus = set()
        for file_path in dataset_files:
            match = RESULT_FILENAME_RE.match(file_path.name)
            if match is None or file_path.suffix != ".out":
                continue
            out_times.add(int(match.group("time")))
            out_cpus.add(int(match.group("cpu")))

        metadata_by_name[case_root.name] = ECOGENCaseMetadata(
            name=case_root.name,
            case_root=case_root,
            datasets_root=datasets_root,
            flow_model=flow_model,
            num_fluids=num_fluids,
            eos_names=tuple(eos_names),
            dataset_extensions=dataset_extensions,
            available_out_times=tuple(sorted(out_times)),
            available_out_cpus=tuple(sorted(out_cpus)),
        )

    return metadata_by_name


def read_ecogen_case_model_metadata(model_root: Path):
    model_path = model_root / "model.xml"
    if not model_path.is_file():
        return "", 1, []

    xml_root = ET.parse(model_path).getroot()
    flow_node = xml_root.find("flowModel")
    eos_nodes = xml_root.findall("EOS")

    flow_model = ""
    num_fluids = 1
    if flow_node is not None:
        flow_model = flow_node.attrib.get("name", "")
        num_fluids = int(flow_node.attrib.get("numberPhases", str(max(1, len(eos_nodes)))))

    eos_names = [Path(node.attrib.get("name", "")).stem for node in eos_nodes]
    if eos_names and num_fluids <= 0:
        num_fluids = len(eos_names)

    return flow_model, num_fluids, eos_names


def load_ecogen_out_case(
    metadata: ECOGENCaseMetadata,
    cpu: int = 0,
    times: list[int] | tuple[int, ...] | None = None,
    eos_root: Path | None = None,
):
    if not metadata.has_out_datasets:
        raise FileNotFoundError(f"No dataset .out files were found for case {metadata.name}.")
    if cpu not in metadata.available_out_cpus:
        raise ValueError(
            f"CPU {cpu} is not available for {metadata.name}. "
            f"Available CPUs: {metadata.available_out_cpus}"
        )

    if eos_root is None:
        eos_root = default_ecogen_eos_root()

    requested_times = metadata.available_out_times if times is None else tuple(times)
    eos_parameters = [
        read_ecogen_eos(eos_root / f"{eos_name}.xml")
        for eos_name in metadata.eos_names
    ]

    x_coordinates = None
    fluid_fields = None
    fluid_raw_names = None
    fluid_derived_names = None
    mixture_raw_fields = None
    mixture_raw_names = None

    for time_index, saved_time in enumerate(requested_times):
        file_path = metadata.datasets_root / f"result_CPU{cpu}_TIME{saved_time}.out"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing expected dataset file: {file_path}")

        data = np.loadtxt(file_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]

        if x_coordinates is None:
            x_coordinates = data[:, 0].copy()
            (
                fluid_fields,
                fluid_raw_names,
                fluid_derived_names,
                mixture_raw_fields,
                mixture_raw_names,
            ) = _allocate_ecogen_blocks(metadata, data.shape[0], len(requested_times), data.shape[1])
        elif not np.allclose(x_coordinates, data[:, 0]):
            raise ValueError(f"Inconsistent x-grid found while reading {file_path}")

        _fill_time_slice(
            metadata,
            data,
            time_index,
            fluid_fields,
            mixture_raw_fields,
        )

    for fluid_index, fluid_dict in enumerate(fluid_fields):
        eos = eos_parameters[fluid_index] if fluid_index < len(eos_parameters) else None
        if eos is None:
            continue

        fluid_dict["alpha_rho"] = fluid_dict["alpha"] * fluid_dict["rho"]
        thermo_fields = compute_fluid_thermo_fields(
            fluid_dict["p"],
            fluid_dict["T"],
            eos,
        )
        for field_name, values in thermo_fields.items():
            fluid_dict[field_name] = values

    mixture_fields = build_mixture_properties(fluid_fields, eos_parameters)
    if mixture_raw_fields is not None:
        for field_name, values in mixture_raw_fields.items():
            mixture_fields[field_name] = values

    mixture_derived_names = tuple(
        field_name for field_name in MIXTURE_DERIVED_FIELD_NAMES if field_name not in mixture_raw_names
    )

    fluids = []
    for fluid_index, fluid_dict in enumerate(fluid_fields):
        eos_name = metadata.eos_names[fluid_index] if fluid_index < len(metadata.eos_names) else f"fluid_{fluid_index + 1}"
        eos = eos_parameters[fluid_index] if fluid_index < len(eos_parameters) else None
        fluids.append(
            FluidBlock(
                index=fluid_index + 1,
                eos_name=eos_name,
                eos_parameters=eos,
                data=FieldBlock(
                    fields=fluid_dict,
                    raw_fields=fluid_raw_names,
                    derived_fields=fluid_derived_names,
                ),
            )
        )

    return ECOGENOutCaseData(
        metadata=metadata,
        x=x_coordinates,
        saved_times=np.array(requested_times, dtype=int),
        fluids=fluids,
        mixture=FieldBlock(
            fields=mixture_fields,
            raw_fields=mixture_raw_names,
            derived_fields=mixture_derived_names,
        ),
    )


def summarize_ecogen_out_case(case_data: ECOGENOutCaseData):
    fluid_summaries = []
    for fluid in case_data.fluids:
        sample_shape = next(iter(fluid.data.fields.values())).shape
        fluid_summaries.append(
            {
                "index": fluid.index,
                "eos_name": fluid.eos_name,
                "raw_fields": list(fluid.data.raw_fields),
                "derived_fields": list(fluid.data.derived_fields),
                "shape": list(sample_shape),
            }
        )

    mixture_shape = next(iter(case_data.mixture.fields.values())).shape
    return {
        "name": case_data.metadata.name,
        "case_root": str(case_data.metadata.case_root),
        "flow_model": case_data.metadata.flow_model,
        "num_fluids": case_data.metadata.num_fluids,
        "eos_names": list(case_data.metadata.eos_names),
        "dataset_extensions": list(case_data.metadata.dataset_extensions),
        "available_out_times": list(case_data.metadata.available_out_times),
        "loaded_times": [int(saved_time) for saved_time in case_data.saved_times],
        "available_out_cpus": list(case_data.metadata.available_out_cpus),
        "num_cells": int(case_data.x.size),
        "fluids": fluid_summaries,
        "mixture": {
            "raw_fields": list(case_data.mixture.raw_fields),
            "derived_fields": list(case_data.mixture.derived_fields),
            "shape": list(mixture_shape),
        },
    }


def _allocate_ecogen_blocks(
    metadata: ECOGENCaseMetadata,
    num_cells: int,
    num_times: int,
    num_columns: int,
):
    fluid_fields = []
    if _is_multiphase_layout(metadata, num_columns):
        fluid_raw_names = PHASE_FIELD_NAMES
        fluid_derived_names = FLUID_DERIVED_FIELD_NAMES
        for _ in range(metadata.num_fluids):
            field_names = fluid_raw_names + fluid_derived_names
            fluid_fields.append(
                {field_name: np.zeros((num_cells, num_times)) for field_name in field_names}
            )

        mixture_raw_names = MULTIPHASE_MIXTURE_FIELD_NAMES
        mixture_raw_fields = {
            field_name: np.zeros((num_cells, num_times))
            for field_name in mixture_raw_names
        }
        return fluid_fields, fluid_raw_names, fluid_derived_names, mixture_raw_fields, mixture_raw_names

    if _is_single_phase_layout(metadata, num_columns):
        fluid_raw_names = SINGLE_PHASE_FLUID_RAW_FIELD_NAMES
        fluid_derived_names = SINGLE_PHASE_FLUID_DERIVED_FIELD_NAMES
        field_names = fluid_raw_names + fluid_derived_names
        fluid_fields.append(
            {field_name: np.zeros((num_cells, num_times)) for field_name in field_names}
        )

        mixture_raw_names = SINGLE_PHASE_MIXTURE_FIELD_NAMES
        mixture_raw_fields = {
            field_name: np.zeros((num_cells, num_times))
            for field_name in mixture_raw_names
        }
        return fluid_fields, fluid_raw_names, fluid_derived_names, mixture_raw_fields, mixture_raw_names

    raise ValueError(
        f"Unsupported .out layout for case {metadata.name}: "
        f"{num_columns} columns with {metadata.num_fluids} fluids."
    )


def _fill_time_slice(
    metadata: ECOGENCaseMetadata,
    data: np.ndarray,
    time_index: int,
    fluid_fields: list[dict[str, np.ndarray]],
    mixture_raw_fields: dict[str, np.ndarray] | None,
):
    if _is_multiphase_layout(metadata, data.shape[1]):
        num_phase_fields = len(PHASE_FIELD_NAMES)
        for fluid_index in range(metadata.num_fluids):
            for field_index, field_name in enumerate(PHASE_FIELD_NAMES):
                column_index = 1 + field_index + num_phase_fields * fluid_index
                fluid_fields[fluid_index][field_name][:, time_index] = data[:, column_index]

        mixture_offset = 1 + num_phase_fields * metadata.num_fluids
        for field_index, field_name in enumerate(MULTIPHASE_MIXTURE_FIELD_NAMES):
            mixture_raw_fields[field_name][:, time_index] = data[:, mixture_offset + field_index]
        return

    fluid_fields[0]["rho"][:, time_index] = data[:, 1]
    fluid_fields[0]["p"][:, time_index] = data[:, 2]
    fluid_fields[0]["T"][:, time_index] = data[:, 3]
    fluid_fields[0]["u"][:, time_index] = data[:, 4]
    fluid_fields[0]["velocityMagnitude"][:, time_index] = data[:, 5]
    fluid_fields[0]["alpha"][:, time_index] = 1.0
    fluid_fields[0]["Y"][:, time_index] = 1.0

    mixture_raw_fields["Rho"][:, time_index] = data[:, 1]
    mixture_raw_fields["p"][:, time_index] = data[:, 2]
    mixture_raw_fields["T"][:, time_index] = data[:, 3]
    mixture_raw_fields["u"][:, time_index] = data[:, 4]
    mixture_raw_fields["velocityMagnitude"][:, time_index] = data[:, 5]
    mixture_raw_fields["amrLevel"][:, time_index] = data[:, 6]
    mixture_raw_fields["xi"][:, time_index] = data[:, 7]


def _is_multiphase_layout(metadata: ECOGENCaseMetadata, num_columns: int):
    expected_columns = 1 + len(PHASE_FIELD_NAMES) * metadata.num_fluids + len(MULTIPHASE_MIXTURE_FIELD_NAMES)
    return metadata.num_fluids >= 1 and num_columns == expected_columns


def _is_single_phase_layout(metadata: ECOGENCaseMetadata, num_columns: int):
    return metadata.num_fluids == 1 and num_columns == 1 + len(SINGLE_PHASE_MIXTURE_FIELD_NAMES)
