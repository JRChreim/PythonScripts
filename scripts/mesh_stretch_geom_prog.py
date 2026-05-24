import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.mesh.geometric_progression import build_geometric_axis_mesh
from src.plots.publication import (
    THESIS_LABEL_FONT_SIZE,
    THESIS_LAYOUT_PADS,
    THESIS_TICK_FONT_SIZE,
    THESIS_TITLE_FONT_SIZE,
    add_thesis_export_argument,
    add_show_titles_argument,
    apply_thesis_style,
    save_thesis_figure_from_args,
    thesis_figure_size,
)

REFERENCE_LENGTH = 22
NUM_DIMS = 2
DOMAIN_X = np.array([0.0, 250.0/22.0]) * REFERENCE_LENGTH
DOMAIN_Y = np.array([0.0, 37.0/22.0]) * REFERENCE_LENGTH
# NUM_ELEMENTS = np.array([880, 400])
NUM_ELEMENTS = np.array([600, 240])
NUM_REFINED_ELEMENTS = np.array([320, 160])
DROPLET_DIAMETER = 1.2 * REFERENCE_LENGTH
CENTER_OFFSET = np.array([1.44, -0.840909091]) * REFERENCE_LENGTH
THESIS_HEIGHT_RATIO = 0.56
THESIS_EXPORT_STEM = "MTGP"
FIGURE_SIZE = thesis_figure_size(THESIS_HEIGHT_RATIO)
TITLE_FONT_SIZE = THESIS_TITLE_FONT_SIZE
LABEL_FONT_SIZE = THESIS_LABEL_FONT_SIZE
TICK_FONT_SIZE = THESIS_TICK_FONT_SIZE
MESH_LINE_COLOR = "0.55"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stretched 2D mesh using geometric progressions."
    )
    add_thesis_export_argument(parser, default_stem=THESIS_EXPORT_STEM)
    add_show_titles_argument(parser)
    parser.add_argument(
        "--no-show",
        "--no-plot",
        dest="no_show",
        action="store_true",
        help="Skip the mesh plots and print only the mesh parameters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the figure, for example thesis/mesh.pdf or thesis/mesh.png.",
    )
    return parser


def build_mesh():
    lower = np.array([DOMAIN_X[0], DOMAIN_Y[0]])
    upper = np.array([DOMAIN_X[1], DOMAIN_Y[1]])
    center = 0.5 * (upper + lower) + CENTER_OFFSET
    refined_lower = center - DROPLET_DIAMETER / 2.0
    refined_upper = center + DROPLET_DIAMETER / 2.0
    boundary_points = NUM_ELEMENTS + 1

    coords_uniform = [
        np.linspace(lower[axis], upper[axis], boundary_points[axis])
        for axis in range(NUM_DIMS)
    ]

    coords_stretched = []
    num_stretched = np.zeros((2, NUM_DIMS), dtype=int)
    stretch_ratios = np.zeros((2, NUM_DIMS))

    for axis in range(NUM_DIMS):
        axis_mesh = build_geometric_axis_mesh(
            lower[axis],
            upper[axis],
            refined_lower[axis],
            refined_upper[axis],
            NUM_ELEMENTS[axis],
            NUM_REFINED_ELEMENTS[axis],
        )
        coords_stretched.append(axis_mesh.coordinates)
        refined_lower[axis], refined_upper[axis] = axis_mesh.refined_bounds
        num_stretched[:, axis] = axis_mesh.num_stretched
        stretch_ratios[:, axis] = axis_mesh.stretch_ratios

    return {
        "center": center,
        "coords_stretched": coords_stretched,
        "coords_uniform": coords_uniform,
        "droplet_diameter": DROPLET_DIAMETER,
        "num_refined_elements": NUM_REFINED_ELEMENTS,
        "num_stretched": num_stretched,
        "refined_lower": refined_lower,
        "refined_upper": refined_upper,
        "stretch_ratios": stretch_ratios,
    }


def print_mesh_summary(mesh_data):
    print("1st, 2nd, and 3rd, refer to x,y,z")
    print("minimum coordinates for the refined region")
    print(mesh_data["refined_lower"])
    print("maximum coordinates for the refined region")
    print(mesh_data["refined_upper"])
    print("stretching factors")
    print(mesh_data["stretch_ratios"])
    print("Number of elements into the refined region")
    print(mesh_data["num_refined_elements"])
    print("Number of elements into the stretched region")
    print(mesh_data["num_stretched"])


def plot_mesh(mesh_data, show_titles: bool = False):
    apply_thesis_style()
    mesh_uniform_x, mesh_uniform_y = np.meshgrid(*mesh_data["coords_uniform"])
    mesh_stretched_x, mesh_stretched_y = np.meshgrid(*mesh_data["coords_stretched"])

    figure, axes = plt.subplots(2, 1, figsize=FIGURE_SIZE, constrained_layout=True)
    figure.set_constrained_layout_pads(**THESIS_LAYOUT_PADS)

    axes[0].plot(
        mesh_uniform_x,
        mesh_uniform_y,
        color=MESH_LINE_COLOR,
        linewidth=0.2,
    )
    axes[0].plot(
        mesh_uniform_x.T,
        mesh_uniform_y.T,
        color=MESH_LINE_COLOR,
        linewidth=0.2,
    )
    axes[0].add_patch(
        Circle(
            (mesh_data["center"][0], mesh_data["center"][1]),
            mesh_data["droplet_diameter"] / 2.0,
            fill=True,
            linewidth=2,
        )
    )
    if show_titles:
        axes[0].set_title(r"$\mathrm{Uniform}$", fontsize=TITLE_FONT_SIZE)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$x\ [\mathrm{m}]$", fontsize=LABEL_FONT_SIZE)
    axes[0].set_ylabel(r"$y\ [\mathrm{m}]$", fontsize=LABEL_FONT_SIZE)
    axes[0].tick_params(labelsize=TICK_FONT_SIZE)

    axes[1].plot(
        mesh_stretched_x,
        mesh_stretched_y,
        color=MESH_LINE_COLOR,
        linewidth=0.2,
    )
    axes[1].plot(
        mesh_stretched_x.T,
        mesh_stretched_y.T,
        color=MESH_LINE_COLOR,
        linewidth=0.2,
    )
    axes[1].add_patch(
        Circle(
            (mesh_data["center"][0], mesh_data["center"][1]),
            mesh_data["droplet_diameter"] / 2.0,
            fill=True,
            linewidth=2,
        )
    )
    if show_titles:
        axes[1].set_title(r"$\mathrm{Stretched}$", fontsize=TITLE_FONT_SIZE)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel(r"$x\ [\mathrm{m}]$", fontsize=LABEL_FONT_SIZE)
    axes[1].set_ylabel(r"$y\ [\mathrm{m}]$", fontsize=LABEL_FONT_SIZE)
    axes[1].tick_params(labelsize=TICK_FONT_SIZE)

    return figure, axes


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    mesh_data = build_mesh()
    print_mesh_summary(mesh_data)

    figure = None
    if not args.no_show:
        figure, _ = plot_mesh(mesh_data, show_titles=args.show_titles)

    if args.to_thesis:
        if figure is None:
            figure, _ = plot_mesh(mesh_data, show_titles=args.show_titles)
        output_path = save_thesis_figure_from_args(
            figure,
            args,
            stem=THESIS_EXPORT_STEM,
        )
        print(f"Figure written to {output_path}")
    elif args.output is not None:
        if figure is None:
            figure, _ = plot_mesh(mesh_data, show_titles=args.show_titles)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=600, bbox_inches="tight", pad_inches=0.02)
        print(f"Figure written to {args.output}")

    if figure is not None and args.no_show:
        plt.close(figure)

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
