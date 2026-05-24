from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np

FORTRAN_RECORD_MARKER_SIZE = 4
FIELD_NAME_BYTES = 50
PARTITION_DIR_RE = re.compile(r"p\d+$")


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
