"""Track pressure histories at fixed x locations for MFC binary snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.io.mfc_binary import (
    discover_mfc_binary_snapshot_directory,
    discover_mfc_binary_steps,
    load_mfc_binary_snapshot,
)
from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    THESIS_TITLE_FONT_SIZE,
    add_thesis_export_argument,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FOLDER = Path(
    "/disk/simulations/Relaxation/CylinderAerobreakup/IC/pTg/3F/NoStretch/"
    "PureFluid/NoDroplet/binary"
)
DEFAULT_PROBE_X_MM = (208.0, 248.0)
DEFAULT_PRESSURE_SCALE = 1.0e5
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts" / "figures" / "mfc_binary_pressure_probe_timestep.png"
)
THESIS_EXPORT_STEM = "MFC_CylinderAerobreakup_pressure_probes"
LINEWIDTH = 1.8
STEP_LABEL = r"$n_{\mathrm{step}}$"


@dataclass(frozen=True)
class ProbeSelection:
    target_x_mm: float
    selected_index: int
    selected_x_m: float

    @property
    def selected_x_mm(self) -> float:
        return self.selected_x_m * 1.0e3

    @property
    def delta_mm(self) -> float:
        return self.selected_x_mm - self.target_x_mm


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track pressure at fixed x locations across MFC binary snapshots."
        )
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=DEFAULT_DATA_FOLDER,
        help=(
            "Path to the MFC case directory or directly to a snapshot "
            "directory."
        ),
    )
    parser.add_argument(
        "--probe-x-mm",
        nargs="+",
        type=float,
        default=list(DEFAULT_PROBE_X_MM),
        metavar="X_MM",
        help=(
            "Probe locations in millimetres. The script chooses the nearest "
            "existing cell center for each target and does not interpolate. "
            "Defaults to 208 248."
        ),
    )
    parser.add_argument(
        "--pressure-scale",
        type=float,
        default=DEFAULT_PRESSURE_SCALE,
        help=(
            "Divide pressure by this factor before plotting. Use 1 for raw "
            "pressure in Pa, 1e5 for the current normalization, 1e6 for MPa, "
            "or any other positive scale."
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        help=(
            "Optional saved step numbers to include. Omit this flag to use "
            "all available snapshots."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path to save the figure. Defaults to "
            f"{DEFAULT_OUTPUT} unless --to-thesis is used."
        ),
    )
    parser.add_argument(
        "--title",
        help="Optional overall figure title.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive window.",
    )
    return parser


def select_steps(
    available_steps: tuple[int, ...],
    *,
    requested_steps: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if not available_steps:
        raise ValueError("No MFC snapshots were found in the selected directory.")

    if requested_steps is None:
        return list(available_steps)

    missing_steps = [step for step in requested_steps if step not in available_steps]
    if missing_steps:
        raise ValueError(
            "The following requested steps are not available: "
            f"{missing_steps}. Available steps include {list(available_steps[:10])}"
        )

    return sorted(set(int(step) for step in requested_steps))


def select_probe_locations(
    reference_x_centers_m: np.ndarray,
    probe_x_mm: list[float] | tuple[float, ...],
) -> list[ProbeSelection]:
    if reference_x_centers_m.size == 0:
        raise ValueError("The reference snapshot does not contain any x centers.")

    reference_x_centers_mm = reference_x_centers_m * 1.0e3
    probe_selections: list[ProbeSelection] = []

    for target_x_mm in probe_x_mm:
        selected_index = int(np.abs(reference_x_centers_mm - target_x_mm).argmin())
        probe_selections.append(
            ProbeSelection(
                target_x_mm=float(target_x_mm),
                selected_index=selected_index,
                selected_x_m=float(reference_x_centers_m[selected_index]),
            )
        )

    return probe_selections


def load_pressure_histories(
    snapshot_directory: Path,
    selected_steps: list[int],
    *,
    reference_snapshot,
    probe_selections: list[ProbeSelection],
    pressure_scale: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if pressure_scale <= 0.0:
        raise ValueError(
            f"The pressure scale must be positive. Got {pressure_scale:g}."
        )

    reference_x_centers = reference_snapshot.x_centers

    if "pres" not in reference_snapshot.fields:
        available = ", ".join(sorted(reference_snapshot.fields))
        raise ValueError(
            f"Pressure field 'pres' is not available in {reference_snapshot.path.name}. "
            f"Available fields: {available}"
        )

    step_values: list[int] = []
    histories: list[list[float]] = [[] for _ in probe_selections]

    for step in selected_steps:
        snapshot = (
            reference_snapshot
            if step == reference_snapshot.step
            else load_mfc_binary_snapshot(snapshot_directory / f"{step}.dat")
        )

        if not np.allclose(snapshot.x_centers, reference_x_centers):
            raise ValueError(
                f"Snapshot {snapshot.path} does not share the same x-grid as the "
                f"reference snapshot {reference_snapshot.path}."
            )

        pressure = snapshot.fields.get("pres")
        if pressure is None:
            available = ", ".join(sorted(snapshot.fields))
            raise ValueError(
                f"Pressure field 'pres' is not available in {snapshot.path.name}. "
                f"Available fields: {available}"
            )

        step_values.append(int(step))
        for history, probe_selection in zip(histories, probe_selections):
            history.append(float(pressure[probe_selection.selected_index]) / pressure_scale)

    return np.asarray(step_values, dtype=int), [
        np.asarray(history, dtype=float) for history in histories
    ]


def _format_probe_title(index: int, selection: ProbeSelection) -> str:
    return (
        f"Probe {index}: target x = {selection.target_x_mm:g} mm, "
        f"nearest cell center x = {selection.selected_x_mm:.3f} mm"
    )


def build_pressure_axis_label(pressure_scale: float) -> str:
    if np.isclose(pressure_scale, 1.0):
        return r"$p\ [\mathrm{Pa}]$"
    if np.isclose(pressure_scale, 1.0e6):
        return r"$p\ [\mathrm{MPa}]$"

    exponent = round(np.log10(pressure_scale))
    if np.isclose(pressure_scale, 10.0**exponent):
        scale_text = rf"10^{{{int(exponent)}}}"
    else:
        scale_text = f"{pressure_scale:g}"

    return rf"$p / ({scale_text}\ \mathrm{{Pa}})$"


def build_pressure_probe_figure(
    steps: np.ndarray,
    probe_histories: list[np.ndarray],
    probe_selections: list[ProbeSelection],
    *,
    title: str | None,
    pressure_scale: float,
):
    if not probe_histories:
        raise ValueError("At least one probe history is required.")

    figure_height_ratio = max(0.42 * len(probe_histories), 0.85)
    figure, axes = plt.subplots(
        len(probe_histories),
        1,
        figsize=thesis_figure_size(figure_height_ratio),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    axes_array = np.atleast_1d(axes).ravel()
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(probe_histories)))

    for index, (axis, history, selection, color) in enumerate(
        zip(axes_array, probe_histories, probe_selections, colors),
        start=1,
    ):
        axis.plot(steps, history, color=color, linewidth=LINEWIDTH)
        axis.grid(True, alpha=0.25)
        axis.tick_params(labelsize=THESIS_TICK_FONT_SIZE)
        axis.set_ylabel(
            build_pressure_axis_label(pressure_scale),
            fontsize=THESIS_LABEL_FONT_SIZE,
        )
        axis.set_title(
            _format_probe_title(index, selection),
            fontsize=THESIS_TITLE_FONT_SIZE,
        )

    axes_array[-1].set_xlabel(STEP_LABEL, fontsize=THESIS_LABEL_FONT_SIZE)

    if title:
        figure.suptitle(title, fontsize=THESIS_TITLE_FONT_SIZE)

    return figure


def _describe_steps(steps: list[int]) -> str:
    if not steps:
        return "no steps"
    if len(steps) == 1:
        return f"step {steps[0]}"
    if all((b - a) == 1 for a, b in zip(steps, steps[1:])):
        return f"steps {steps[0]} through {steps[-1]}"
    if len(steps) <= 10:
        return "steps " + ", ".join(str(step) for step in steps)
    return f"{len(steps)} steps from {steps[0]} through {steps[-1]}"


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    apply_thesis_style()

    snapshot_directory = discover_mfc_binary_snapshot_directory(args.data_folder)
    available_steps = discover_mfc_binary_steps(snapshot_directory)
    selected_steps = select_steps(
        available_steps,
        requested_steps=args.steps,
    )

    reference_step = selected_steps[0]
    reference_snapshot = load_mfc_binary_snapshot(
        snapshot_directory / f"{reference_step}.dat"
    )
    probe_selections = select_probe_locations(
        reference_snapshot.x_centers,
        args.probe_x_mm,
    )
    step_values, probe_histories = load_pressure_histories(
        snapshot_directory,
        selected_steps,
        reference_snapshot=reference_snapshot,
        probe_selections=probe_selections,
        pressure_scale=float(args.pressure_scale),
    )

    figure = build_pressure_probe_figure(
        step_values,
        probe_histories,
        probe_selections,
        title=args.title,
        pressure_scale=float(args.pressure_scale),
    )

    print(
        f"Loaded {len(selected_steps)} snapshot(s) from {snapshot_directory} "
        f"({_describe_steps(selected_steps)})."
    )
    print(f"Pressure values divided by {float(args.pressure_scale):g}.")
    print("Nearest cell-center probes:")
    for index, selection in enumerate(probe_selections, start=1):
        print(
            f"  Probe {index}: target {selection.target_x_mm:g} mm -> "
            f"{selection.selected_x_mm:.3f} mm "
            f"(delta {selection.delta_mm:+.3f} mm)"
        )

    output_path = args.output
    if output_path is None and not args.to_thesis:
        output_path = DEFAULT_OUTPUT

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200)
        print(f"Figure written to {output_path}")

    thesis_path = save_thesis_figure_from_args(
        figure,
        args,
        stem=THESIS_EXPORT_STEM,
    )
    if thesis_path is not None:
        print(f"Thesis PDF written to {thesis_path}")

    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
