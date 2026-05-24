import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.io.ecogen_out import (
    default_ecogen_results_root,
    discover_ecogen_cases,
    load_ecogen_out_case,
)
from src.plots.publication import apply_thesis_style, latex_text

# Edit this block for quick exploratory plotting.
QUICKLOOK_CONFIG = {
    "case_names": ["6Eq_pEq"],
    "cpu": 0,
    "times": [0, 10],
    "figure_size": (13, 9),
    "title": latex_text("ECOGEN Quicklook"),
    "panels": [
        {
            "domain": "mixture",
            "variable": "p",
            "title": latex_text("Mixture pressure"),
        },
        {
            "domain": "mixture",
            "variable": "Rho",
            "title": latex_text("Mixture density"),
        },
        {
            "domain": "fluid",
            "fluid_index": 1,
            "variable": "alpha",
            "title": latex_text("Fluid 1 volume fraction"),
        },
        {
            "domain": "fluid",
            "fluid_index": 2,
            "variable": "Y",
            "title": latex_text("Fluid 2 mass fraction"),
            "case_names": ["6Eq_pEq"],
        },
    ],
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Quick exploratory multi-panel plotting for ECOGEN dataset .out data. "
            "The main configuration lives at the top of scripts/ecogen_quicklook.py."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_ecogen_results_root(),
        help="Path to the ECOGEN results directory.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        help="Optional override for QUICKLOOK_CONFIG['case_names']. Repeat for multiple cases.",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=int,
        help="Optional override for QUICKLOOK_CONFIG['times'].",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Ignore configured times and use only the latest available .out time per loaded case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the quicklook figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive matplotlib window.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    config = _apply_cli_overrides(QUICKLOOK_CONFIG, args)
    metadata_by_name = discover_ecogen_cases(args.results_root)
    requested_times = None if config["latest_only"] else _collect_requested_times(config)

    loaded_cases = []
    skipped_cases = []
    missing_cases = []

    for case_name in config["case_names"]:
        metadata = metadata_by_name.get(case_name)
        if metadata is None:
            missing_cases.append(case_name)
            continue
        if not metadata.has_out_datasets:
            skipped_cases.append(case_name)
            continue

        selected_times = _select_case_times(metadata, requested_times)
        if not selected_times:
            print(
                f"Skipping case {case_name}: none of the requested times are available in "
                f"{list(metadata.available_out_times)}"
            )
            continue

        loaded_cases.append(
            load_ecogen_out_case(
                metadata,
                cpu=config["cpu"],
                times=selected_times,
            )
        )

    if not loaded_cases:
        raise ValueError(
            "No quicklook cases could be loaded from dataset .out files. "
            f"Skipped without .out data: {skipped_cases}; missing: {missing_cases}"
        )

    figure = build_quicklook_figure(config, loaded_cases)

    if skipped_cases:
        print("Selected cases without dataset .out files:")
        for case_name in skipped_cases:
            print(f"  - {case_name}")

    if missing_cases:
        print("Selected cases not found under the results root:")
        for case_name in missing_cases:
            print(f"  - {case_name}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=200)
        print(f"Quicklook figure written to {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


def build_quicklook_figure(config: dict, loaded_cases: list):
    apply_thesis_style()

    num_panels = len(config["panels"])
    num_columns = 2 if num_panels > 1 else 1
    num_rows = math.ceil(num_panels / num_columns)

    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=config["figure_size"],
        squeeze=False,
    )
    flat_axes = list(axes.ravel())

    for axis, panel in zip(flat_axes, config["panels"]):
        _plot_quicklook_panel(axis, panel, loaded_cases)

    for axis in flat_axes[num_panels:]:
        axis.set_visible(False)

    if config.get("title"):
        figure.suptitle(_ensure_latex_title(config["title"]))
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    else:
        figure.tight_layout()

    return figure


def _plot_quicklook_panel(axis, panel: dict, loaded_cases: list):
    compatible_cases = [
        case_data
        for case_data in loaded_cases
        if not panel.get("case_names") or case_data.metadata.name in panel["case_names"]
    ]
    if not compatible_cases:
        axis.text(
            0.5,
            0.5,
            latex_text("No compatible cases"),
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return

    domain = panel["domain"]
    variable = panel["variable"]
    fluid_index = panel.get("fluid_index", 1)
    plotted_anything = False

    for case_data in compatible_cases:
        field_block, domain_label = _resolve_field_block(case_data, domain, fluid_index)
        if field_block is None:
            print(
                f"Skipping panel '{panel.get('title', variable)}' for case {case_data.metadata.name}: "
                f"fluid {fluid_index} is not available."
            )
            continue
        if variable not in field_block.fields:
            print(
                f"Skipping panel '{panel.get('title', variable)}' for case {case_data.metadata.name}: "
                f"variable '{variable}' is unavailable."
            )
            continue

        time_indices, time_labels = _resolve_time_selection(case_data, panel.get("times"))
        if not time_indices:
            print(
                f"Skipping panel '{panel.get('title', variable)}' for case {case_data.metadata.name}: "
                f"none of the requested times are loaded."
            )
            continue

        values = field_block.fields[variable]
        for time_index, saved_time in zip(time_indices, time_labels):
            axis.plot(
                case_data.x,
                values[:, time_index],
                linewidth=2,
                label=_build_series_label(
                    case_name=case_data.metadata.name,
                    saved_time=saved_time,
                    domain_label=domain_label,
                    num_cases=len(compatible_cases),
                    num_times=len(time_indices),
                ),
            )
            plotted_anything = True

    axis.set_xlabel(r"$x\ [\mathrm{m}]$")
    axis.set_ylabel(_build_variable_label(variable))
    axis.set_title(
        _ensure_latex_title(
            panel.get("title") or _build_panel_title(domain, variable, fluid_index)
        )
    )
    axis.grid(True)

    if plotted_anything:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    else:
        axis.text(
            0.5,
            0.5,
            latex_text("No data plotted"),
            ha="center",
            va="center",
            transform=axis.transAxes,
        )


def _apply_cli_overrides(base_config: dict, args):
    config = {
        "case_names": list(base_config["case_names"]),
        "cpu": base_config["cpu"],
        "times": None if base_config.get("times") is None else list(base_config["times"]),
        "figure_size": tuple(base_config["figure_size"]),
        "title": base_config.get("title"),
        "latest_only": False,
        "panels": [dict(panel) for panel in base_config["panels"]],
    }

    if args.cases:
        config["case_names"] = list(args.cases)
    if args.times is not None:
        config["times"] = list(args.times)
    if args.latest_only:
        config["latest_only"] = True
        config["times"] = None

    return config


def _collect_requested_times(config: dict):
    requested_times = set()
    if config.get("times"):
        requested_times.update(int(time_value) for time_value in config["times"])

    for panel in config["panels"]:
        if panel.get("times"):
            requested_times.update(int(time_value) for time_value in panel["times"])

    return sorted(requested_times) if requested_times else None


def _select_case_times(metadata, requested_times):
    if requested_times is None:
        return [metadata.available_out_times[-1]]

    available = set(metadata.available_out_times)
    selected = [time_value for time_value in requested_times if time_value in available]
    missing = [time_value for time_value in requested_times if time_value not in available]
    if missing:
        print(
            f"Case {metadata.name} is missing requested times {missing}. "
            f"Using {selected} from available times {list(metadata.available_out_times)}."
        )
    return selected


def _resolve_field_block(case_data, domain: str, fluid_index: int):
    if domain == "mixture":
        return case_data.mixture, "mixture"

    fluid_position = fluid_index - 1
    if fluid_position < 0 or fluid_position >= len(case_data.fluids):
        return None, None

    fluid = case_data.fluids[fluid_position]
    return fluid.data, f"fluid {fluid.index}"


def _resolve_time_selection(case_data, requested_times):
    if not requested_times:
        return list(range(len(case_data.saved_times))), [int(value) for value in case_data.saved_times]

    mapping = {int(saved_time): index for index, saved_time in enumerate(case_data.saved_times)}
    selected_indices = []
    selected_labels = []

    for requested_time in requested_times:
        time_value = int(requested_time)
        if time_value in mapping:
            selected_indices.append(mapping[time_value])
            selected_labels.append(time_value)

    return selected_indices, selected_labels


def _build_series_label(
    case_name: str,
    saved_time: int,
    domain_label: str,
    num_cases: int,
    num_times: int,
):
    parts = []
    if num_cases > 1:
        parts.append(latex_text(case_name))
    if domain_label != "mixture":
        parts.append(latex_text(domain_label))
    if num_times > 1:
        parts.append(latex_text(f"time {saved_time}"))

    return r" $\mid$ ".join(parts) if parts else latex_text(case_name)


def _build_panel_title(domain: str, variable: str, fluid_index: int):
    variable_label = _build_variable_label(variable)
    if domain == "mixture":
        return f"{latex_text('Mixture field')} {variable_label}"
    return f"{latex_text(f'Fluid {fluid_index} field')} {variable_label}"


def _build_variable_label(variable: str) -> str:
    field_labels = {
        "E": r"$E$",
        "Rho": r"$\rho\ [\mathrm{kg\,m^{-3}}]$",
        "T": r"$T\ [\mathrm{K}]$",
        "Y": r"$Y$",
        "alpha": r"$\alpha$",
        "p": r"$p\ [\mathrm{Pa}]$",
        "rho": r"$\rho\ [\mathrm{kg\,m^{-3}}]$",
        "u": r"$u\ [\mathrm{m\,s^{-1}}]$",
        "velocityMagnitude": r"$|\mathbf{u}|\ [\mathrm{m\,s^{-1}}]$",
    }
    return field_labels.get(variable, latex_text(variable))


def _ensure_latex_title(title: str) -> str:
    if "$" in title:
        return title
    return latex_text(title)


if __name__ == "__main__":
    main()
