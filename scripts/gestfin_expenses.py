from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.gestfin.pipeline import (
    DEFAULT_INPUT_DIRNAME,
    DEFAULT_OUTPUT_FILENAME,
    DEFAULT_RULES_FILENAME,
    build_report,
    load_rules_config,
    parse_pdf_directory,
)
from src.gestfin.xlsx_writer import write_xlsx


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse monthly expense statements from PDF files, classify transactions "
            "with rules, and rebuild a rolling Excel workbook."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "GestFin" / DEFAULT_INPUT_DIRNAME,
        help="Directory containing the statement PDF files.",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "GestFin" / DEFAULT_RULES_FILENAME,
        help="Path to the category rules JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "GestFin" / DEFAULT_OUTPUT_FILENAME,
        help="Path to the rolling Excel workbook to write.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_output",
        help="Open the generated workbook with the default spreadsheet app after writing it.",
    )
    return parser


def open_generated_workbook(path: Path) -> None:
    path = Path(path)
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(path)], check=False)
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
    except FileNotFoundError:
        print(f"Could not open {path}: no system file opener was found.")
    except OSError as exc:
        print(f"Could not open {path}: {exc}")


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    rules_config = load_rules_config(args.rules)
    transactions = parse_pdf_directory(args.input_dir, rules_config)
    if not transactions:
        raise ValueError(f"No PDF transactions were parsed from {args.input_dir}")

    sheets = build_report(
        transactions=transactions,
        rules_config=rules_config,
        input_dir=args.input_dir,
        rules_path=args.rules,
        output_path=args.output,
    )
    write_xlsx(args.output, sheets)

    if args.open_output:
        open_generated_workbook(args.output)

    print(f"Parsed {len(transactions)} transactions from {args.input_dir}")
    print(f"Workbook written to {args.output}")
    print(f"Rules file: {args.rules}")


if __name__ == "__main__":
    main()
