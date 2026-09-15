from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np

FORTRAN_RECORD_MARKER_SIZE = 4
FIELD_NAME_BYTES = 50
PARTITION_DIR_RE = re.compile(r"p\d+$")
MASS_FRACTION_FIELDS = ("Y1", "Y2", "Y3")
_MASS_FRACTION_SOURCE_FIELDS = ("alpha_rho1", "alpha_rho2", "alpha_rho3")


@dataclass(frozen=True)
class MFCBinarySnapshot:
    step: int
    path: Path
    header: tuple[int, int, int, int]
    x_faces: np.ndarray
    fields: dict[str, np.ndarray]

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    @property
    def num_cells(self) -> int:
        return int(self.x_faces.size - 1)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.fields.keys())


def available_mfc_binary_variables(snapshot: MFCBinarySnapshot) -> tuple[str, ...]:
    variables = list(snapshot.field_names)
    if _can_compute_mass_fractions(snapshot):
        for name in MASS_FRACTION_FIELDS:
            if name not in variables:
                variables.append(name)
    return tuple(variables)


def resolve_mfc_binary_variable(snapshot: MFCBinarySnapshot, variable: str) -> np.ndarray:
    if variable in snapshot.fields:
        return snapshot.fields[variable]

    if variable in MASS_FRACTION_FIELDS:
        mass_fractions = _compute_mass_fractions(snapshot)
        return mass_fractions[MASS_FRACTION_FIELDS.index(variable)]

    available = ", ".join(sorted(available_mfc_binary_variables(snapshot)))
    raise ValueError(
        f"Variable '{variable}' is not available in {snapshot.path.name}. "
        f"Available fields: {available}"
    )


def discover_mfc_binary_snapshot_directory(base_folder: Path) -> Path:
    base_folder = Path(base_folder)
    if not base_folder.exists():
        raise FileNotFoundError(f"MFC case directory not found: {base_folder}")
    if base_folder.is_dir() and any(base_folder.glob("*.dat")):
        return base_folder

    root_folder = base_folder / "root"
    if root_folder.is_dir() and any(root_folder.glob("*.dat")):
        return root_folder

    partition_dirs = sorted(
        (
            path
            for path in base_folder.iterdir()
            if path.is_dir() and PARTITION_DIR_RE.fullmatch(path.name)
        ),
        key=lambda path: int(path.name[1:]),
    )
    for partition_dir in partition_dirs:
        if any(partition_dir.glob("*.dat")):
            return partition_dir

    raise FileNotFoundError(
        f"No MFC binary snapshot files were found under {base_folder}."
    )


def discover_mfc_binary_steps(snapshot_directory: Path) -> tuple[int, ...]:
    snapshot_directory = Path(snapshot_directory)
    snapshot_files = [
        file_path
        for file_path in snapshot_directory.glob("*.dat")
        if file_path.stem.isdigit()
    ]
    return tuple(sorted(int(file_path.stem) for file_path in snapshot_files))


def load_mfc_binary_snapshot(filepath: Path) -> MFCBinarySnapshot:
    filepath = Path(filepath)
    records = _read_fortran_unformatted_records(filepath)
    if len(records) < 3:
        raise ValueError(
            f"{filepath} does not contain the expected MFC record layout."
        )

    header = struct.unpack("<4I", records[0])
    real_size = _infer_real_size(header, records[1], filepath)
    dtype = np.dtype("<f8" if real_size == 8 else "<f4")

    x_faces = np.frombuffer(records[1], dtype=dtype).copy()
    expected_cells = x_faces.size - 1

    if header[0] + 1 != expected_cells:
        raise ValueError(
            f"{filepath} reports {header[0] + 1} cells in the header but the x-grid "
            f"contains {expected_cells} cells."
        )

    fields: dict[str, np.ndarray] = {}
    for record in records[2:]:
        field_name = record[:FIELD_NAME_BYTES].decode("ascii", errors="ignore").strip()
        if not field_name:
            continue

        values = np.frombuffer(record[FIELD_NAME_BYTES:], dtype=dtype).copy()
        if values.size != expected_cells:
            raise ValueError(
                f"Field '{field_name}' in {filepath} contains {values.size} values "
                f"but {expected_cells} cell values were expected."
            )
        fields[field_name] = values

    return MFCBinarySnapshot(
        step=int(filepath.stem),
        path=filepath,
        header=header,
        x_faces=x_faces,
        fields=fields,
    )


def _can_compute_mass_fractions(snapshot: MFCBinarySnapshot) -> bool:
    return all(field_name in snapshot.fields for field_name in _MASS_FRACTION_SOURCE_FIELDS)


def _compute_mass_fractions(snapshot: MFCBinarySnapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing = [
        field_name
        for field_name in _MASS_FRACTION_SOURCE_FIELDS
        if field_name not in snapshot.fields
    ]
    if missing:
        raise ValueError(
            f"Cannot compute mass fractions for {snapshot.path.name}; missing fields: "
            f"{missing}"
        )

    m1 = np.asarray(snapshot.fields[_MASS_FRACTION_SOURCE_FIELDS[0]], dtype=float)
    m2 = np.asarray(snapshot.fields[_MASS_FRACTION_SOURCE_FIELDS[1]], dtype=float)
    m3 = np.asarray(snapshot.fields[_MASS_FRACTION_SOURCE_FIELDS[2]], dtype=float)
    total = m1 + m2 + m3

    if np.any(total == 0.0):
        raise ValueError(
            f"Cannot compute mass fractions for {snapshot.path.name}; the total "
            "mass density is zero in at least one cell."
        )

    return (m1 / total, m2 / total, m3 / total)


def _read_fortran_unformatted_records(filepath: Path) -> list[bytes]:
    data = filepath.read_bytes()
    records: list[bytes] = []
    offset = 0

    while offset + FORTRAN_RECORD_MARKER_SIZE <= len(data):
        record_length = struct.unpack(
            "<I", data[offset : offset + FORTRAN_RECORD_MARKER_SIZE]
        )[0]
        offset += FORTRAN_RECORD_MARKER_SIZE

        if record_length == 0:
            continue

        end = offset + record_length
        trailer_end = end + FORTRAN_RECORD_MARKER_SIZE
        if trailer_end > len(data):
            raise ValueError(f"{filepath} ended mid-record.")

        trailer_length = struct.unpack("<I", data[end:trailer_end])[0]
        if trailer_length != record_length:
            raise ValueError(
                f"{filepath} has mismatched Fortran record markers at byte offset "
                f"{offset - FORTRAN_RECORD_MARKER_SIZE}: {record_length} vs "
                f"{trailer_length}."
            )

        records.append(data[offset:end])
        offset = trailer_end

    return records


def _infer_real_size(
    header: tuple[int, int, int, int],
    x_record: bytes,
    filepath: Path,
) -> int:
    if header[3] in (4, 8):
        return header[3]

    for real_size in (8, 4):
        if len(x_record) % real_size != 0:
            continue
        if len(x_record) // real_size == header[0] + 2:
            return real_size

    raise ValueError(
        f"Unable to infer floating-point precision for {filepath} from header "
        f"{header} and x-record length {len(x_record)}."
    )
