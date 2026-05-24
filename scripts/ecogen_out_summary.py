import argparse
import json
from pathlib import Path

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.io.ecogen_out import (
    default_ecogen_results_root,
    discover_ecogen_cases,
    load_ecogen_out_case,
    summarize_ecogen_out_case,
)

DEFAULT_CASES = ("Euler_IG_air", "6Eq_pTEq", "6Eq_pEq")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory ECOGEN dataset .out files and load the data into fluid "
            "and mixture blocks."
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
        help=(
            "Case name to inspect. Repeat the flag to select multiple cases. "
            "Defaults to Euler_IG_air, 6Eq_pTEq, and 6Eq_pEq."
        ),
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=0,
        help="CPU index to load from dataset .out files.",
    )
    parser.add_argument(
        "--times",
        nargs="*",
        type=int,
        help="Optional saved-time indices to load. Defaults to all available .out times.",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        help="Optional path to write a JSON summary of the selected cases.",
    )
    return parser


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    selected_case_names = args.cases or list(DEFAULT_CASES)
    metadata_by_name = discover_ecogen_cases(args.results_root)

    summaries = []
    missing_cases = []
    cases_without_out = []

    for case_name in selected_case_names:
        metadata = metadata_by_name.get(case_name)
        if metadata is None:
            missing_cases.append(case_name)
            continue
        if not metadata.has_out_datasets:
            cases_without_out.append(case_name)
            continue

        case_data = load_ecogen_out_case(
            metadata,
            cpu=args.cpu,
            times=args.times,
        )
        summary = summarize_ecogen_out_case(case_data)
        summaries.append(summary)
        _print_case_summary(summary)

    if cases_without_out:
        print("Selected cases without dataset .out files:")
        for case_name in cases_without_out:
            print(f"  - {case_name}")

    if missing_cases:
        print("Selected cases not found under the results root:")
        for case_name in missing_cases:
            print(f"  - {case_name}")

    if args.dump_json is not None:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "results_root": str(args.results_root),
            "selected_cases": selected_case_names,
            "loaded_cases": summaries,
            "cases_without_out": cases_without_out,
            "missing_cases": missing_cases,
        }
        args.dump_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON summary written to {args.dump_json}")


def _print_case_summary(summary: dict):
    print(f"Case: {summary['name']}")
    print(f"  flow model: {summary['flow_model'] or 'unknown'}")
    print(f"  EOS: {', '.join(summary['eos_names']) if summary['eos_names'] else 'unknown'}")
    print(f"  cells: {summary['num_cells']}")
    print(f"  available .out times: {summary['available_out_times']}")
    print(f"  loaded times: {summary['loaded_times']}")
    print(f"  available CPUs: {summary['available_out_cpus']}")
    for fluid_summary in summary["fluids"]:
        print(
            f"  fluid {fluid_summary['index']} ({fluid_summary['eos_name']}): "
            f"raw={fluid_summary['raw_fields']} derived={fluid_summary['derived_fields']}"
        )
    print(
        f"  mixture: raw={summary['mixture']['raw_fields']} "
        f"derived={summary['mixture']['derived_fields']}"
    )


if __name__ == "__main__":
    main()
