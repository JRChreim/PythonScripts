import argparse
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

DEFAULT_CASES = ("6Eq_pEq",)
DEFAULT_DOMAIN = "mixture"
DEFAULT_VARIABLE = "p"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot ECOGEN dataset .out fields against x for selected cases and saved times."
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
        help="Case name to plot. Repeat the flag to compare multiple cases.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=0,
        help="CPU index to read from result_CPU*_TIME*.out files.",
    )
    parser.add_argument(
        "--domain",
        choices=("mixture", "fluid"),
        default=DEFAULT_DOMAIN,
        help="Whether to plot a mixture field or a fluid field.",
    )
    parser.add_argument(
        "--variable",
        default=DEFAULT_VARIABLE,
        help="Field name to plot, for example p, Rho, alpha, Y, or E.",
    )
    parser.add_argument(
        "--fluid-index",
        type=int,
        default=1,
        help="1-based fluid index when --domain fluid is selected.",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=int,
        help="Saved-time indices to plot. Defaults to the last available .out time in each case.",
    )
    parser.add_argument(
        "--title",
        help="Optional custom figure title.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Build or save the figure without opening an interactive matplotlib window.",
    )
    return parser


def plot_ecogen_cases(
    case_data_list,
    domain: str,
    variable: str,
    fluid_index: int = 1,
    title: str | None = None,
):
    apply_thesis_style()
    figure, axis = plt.subplots(figsize=(10, 6))

    for case_data in case_data_list:
        field_block, label_prefix = _resolve_field_block(case_data, domain, fluid_index)
        if variable not in field_block.fields:
            available_fields = sorted(field_block.fields)
            raise ValueError(
                f"Variable '{variable}' is not available for {label_prefix} in case "
                f"{case_data.metadata.name}. Available fields: {available_fields}"
            )

        values = field_block.fields[variable]
        for time_column, saved_time in enumerate(case_data.saved_times):
            axis.plot(
                case_data.x,
                values[:, time_column],
                linewidth=2,
                label=_build_series_label(
                    case_name=case_data.metadata.name,
                    saved_time=int(saved_time),
                    label_prefix=label_prefix,
                    num_cases=len(case_data_list),
                    num_times=len(case_data.saved_times),
                ),
            )

    axis.set_xlabel(r"$x\ [\mathrm{m}]$")
    axis.set_ylabel(_build_variable_label(variable))
    axis.set_title(
        _ensure_latex_title(title) if title is not None else _build_default_title(domain, variable, fluid_index)
    )
    axis.grid(True)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels)
    figure.tight_layout()

    return figure, axis


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    selected_case_names = args.cases or list(DEFAULT_CASES)
    metadata_by_name = discover_ecogen_cases(args.results_root)

    loaded_cases = []
    skipped_cases = []
    missing_cases = []

    for case_name in selected_case_names:
        metadata = metadata_by_name.get(case_name)
        if metadata is None:
            missing_cases.append(case_name)
            continue
        if not metadata.has_out_datasets:
            skipped_cases.append(case_name)
            continue

        selected_times = args.times if args.times is not None else [metadata.available_out_times[-1]]
        loaded_cases.append(
            load_ecogen_out_case(
                metadata,
                cpu=args.cpu,
                times=selected_times,
            )
        )

    if not loaded_cases:
        raise ValueError(
            "No selected cases could be loaded from dataset .out files. "
            f"Skipped without .out data: {skipped_cases}; missing: {missing_cases}"
        )

    plot_ecogen_cases(
        loaded_cases,
        domain=args.domain,
        variable=args.variable,
        fluid_index=args.fluid_index,
        title=args.title,
    )

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
        plt.savefig(args.output, dpi=200)
        print(f"Figure written to {args.output}")

    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


def _resolve_field_block(case_data, domain: str, fluid_index: int):
    if domain == "mixture":
        return case_data.mixture, "mixture"

    fluid_position = fluid_index - 1
    if fluid_position < 0 or fluid_position >= len(case_data.fluids):
        raise ValueError(
            f"Fluid index {fluid_index} is invalid for case {case_data.metadata.name}. "
            f"Available fluid indices: 1..{len(case_data.fluids)}"
        )

    fluid = case_data.fluids[fluid_position]
    return fluid.data, f"fluid {fluid.index} ({fluid.eos_name})"


def _build_series_label(
    case_name: str,
    saved_time: int,
    label_prefix: str,
    num_cases: int,
    num_times: int,
):
    parts = []
    if num_cases > 1:
        parts.append(latex_text(case_name))
    if label_prefix != "mixture":
        parts.append(latex_text(label_prefix))
    if num_times > 1:
        parts.append(latex_text(f"time {saved_time}"))
    if not parts:
        parts.append(latex_text(case_name))
    return r" $\mid$ ".join(parts)


def _build_default_title(domain: str, variable: str, fluid_index: int):
    variable_label = _build_variable_label(variable)
    if domain == "mixture":
        return f"{latex_text('ECOGEN mixture field')} {variable_label}"
    return f"{latex_text(f'ECOGEN fluid {fluid_index} field')} {variable_label}"


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
