from pathlib import Path

import matplotlib.pyplot as plt

THESIS_EXPORT_DIR = Path(
    "/home/user/Documents/GitHub/Caltech-Thesis---JRChreim/Figures"
)

_SERIF_LATEX_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
}

PUBLICATION_STYLE = {
    **_SERIF_LATEX_STYLE,
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 16,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
}

# Caltech thesis class uses letter paper with 1.5 in left/right margins, which
# leaves 5.5 in of usable text width for figures placed near \linewidth.
THESIS_TEXT_WIDTH_IN = 5.5

THESIS_STYLE = {
    **_SERIF_LATEX_STYLE,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

THESIS_LAYOUT_PADS = {
    "w_pad": 0.01,
    "h_pad": 0.01,
    "wspace": 0.02,
    "hspace": 0.02,
}

THESIS_TITLE_FONT_SIZE = 11
THESIS_LABEL_FONT_SIZE = 10
THESIS_TICK_FONT_SIZE = 9

THESIS_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "pad_inches": 0.01,
}


def _apply_style(style):
    plt.rcParams.update(style)


def escape_latex_text(text: str) -> str:
    """Escape plain text so it can be embedded safely in LaTeX math mode."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        " ": r"\ ",
    }
    return "".join(replacements.get(character, character) for character in str(text))


def latex_text(text: str) -> str:
    """Return a LaTeX math string for plain descriptive text."""

    return rf"$\mathrm{{{escape_latex_text(text)}}}$"


def apply_publication_style():
    _apply_style(PUBLICATION_STYLE)


def apply_thesis_style():
    _apply_style(THESIS_STYLE)


def thesis_figure_size(height_ratio: float, width_fraction: float = 1.0):
    """Return a thesis-friendly figure size in inches."""
    width = THESIS_TEXT_WIDTH_IN * width_fraction
    return (width, width * height_ratio)


def add_thesis_export_argument(parser, default_stem: str | None = None):
    parser.add_argument(
        "--to-thesis",
        action="store_true",
        help=(
            "Use thesis plotting defaults and export a PDF into the thesis "
            "Figures directory."
        ),
    )
    parser.add_argument(
        "--thesis-stem",
        default=default_stem,
        help=(
            "Override the exported PDF filename stem used with --to-thesis. "
            "The file is always written as PDF."
        ),
    )
    return parser


def add_show_titles_argument(parser, default: bool = False):
    parser.add_argument(
        "--show-titles",
        action="store_true",
        default=default,
        help="Show plot titles on generated figures.",
    )
    return parser


def thesis_output_path(
    stem: str,
    output_dir: Path | None = None,
) -> Path:
    target_dir = THESIS_EXPORT_DIR if output_dir is None else Path(output_dir)
    return target_dir / f"{stem}.pdf"


def save_thesis_figure(
    figure,
    path: Path | str | None = None,
    *,
    stem: str | None = None,
    output_dir: Path | None = None,
    dpi: int = 600,
):
    if path is None:
        if stem is None:
            raise ValueError("Either path or stem must be provided.")
        path = thesis_output_path(stem, output_dir=output_dir)

    path = Path(path)
    if path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, **THESIS_SAVEFIG_KWARGS)
    return path


def save_thesis_figure_from_args(
    figure,
    args,
    *,
    stem: str | None = None,
    stem_suffix: str = "",
    output_dir: Path | None = None,
    dpi: int = 600,
):
    if not getattr(args, "to_thesis", False):
        return None
    resolved_stem = getattr(args, "thesis_stem", None) or stem
    if resolved_stem is None:
        raise ValueError("A thesis stem is required when --to-thesis is set.")
    if stem_suffix:
        resolved_stem = f"{resolved_stem}{stem_suffix}"
    return save_thesis_figure(
        figure,
        stem=resolved_stem,
        output_dir=output_dir,
        dpi=dpi,
    )
