import argparse

import matplotlib.pyplot as plt
import numpy as np

try:
    from _bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from src.mesh.log_cosh import build_log_cosh_axis_mesh, print_mesh_stats
from src.plots.publication import apply_thesis_style

REFERENCE_LENGTH = 0.048
DROPLET_DIAMETER = REFERENCE_LENGTH
LOWER_BOUNDS = np.array([-6.25 * REFERENCE_LENGTH, 0.0 * REFERENCE_LENGTH])
UPPER_BOUNDS = np.array([17.0 * REFERENCE_LENGTH, 6.0 * REFERENCE_LENGTH])
CENTER = np.array([0.0, 0.0])
NUM_X_ELEMENTS = 1800
STRETCH_START = np.array([-0.6 * REFERENCE_LENGTH, 0.0 * REFERENCE_LENGTH])
STRETCH_END = np.array([0.6 * REFERENCE_LENGTH, 0.6 * REFERENCE_LENGTH])
STRETCH_ALPHA = np.array([20.0, 1.3])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stretched 2D mesh using a log-cosh mapping."
    )
    parser.add_argument(
        "--no-show",
        "--no-plot",
        dest="no_show",
        action="store_true",
        help="Skip the mesh plots and print only the mesh parameters.",
    )
    return parser


def build_mesh():
    domain_lengths = UPPER_BOUNDS - LOWER_BOUNDS
    num_elements = np.array(
        [NUM_X_ELEMENTS, round(NUM_X_ELEMENTS * domain_lengths[1] / domain_lengths[0])],
        dtype=int,
    )
    num_boundary_points = num_elements + 1

    coords_aux = []
    coords_stretched = []
    auxiliary_bounds = []

    for axis in range(2):
        axis_mesh = build_log_cosh_axis_mesh(
            LOWER_BOUNDS[axis],
            UPPER_BOUNDS[axis],
            STRETCH_START[axis],
            STRETCH_END[axis],
            STRETCH_ALPHA[axis],
            num_boundary_points[axis],
        )
        coords_aux.append(axis_mesh.auxiliary_coordinates)
        coords_stretched.append(axis_mesh.stretched_coordinates)
        auxiliary_bounds.append(axis_mesh.auxiliary_bounds)

    return {
        "auxiliary_bounds": np.array(auxiliary_bounds),
        "center": CENTER,
        "coords_aux": coords_aux,
        "coords_stretched": coords_stretched,
        "droplet_diameter": DROPLET_DIAMETER,
        "lower_bounds": LOWER_BOUNDS,
        "num_elements": num_elements,
        "stretch_alpha": STRETCH_ALPHA,
        "stretch_bounds": np.column_stack((STRETCH_START, STRETCH_END)),
        "upper_bounds": UPPER_BOUNDS,
    }


def print_mesh_summary(mesh_data):
    deltas = [np.diff(coord) for coord in mesh_data["coords_stretched"]]

    print("difference between desired and obtained boundaries")
    print(
        "beginning:",
        np.abs(
            mesh_data["lower_bounds"]
            - np.array([coord[0] for coord in mesh_data["coords_stretched"]])
        ),
    )
    print(
        "end:",
        np.abs(
            mesh_data["upper_bounds"]
            - np.array([coord[-1] for coord in mesh_data["coords_stretched"]])
        ),
    )

    print_mesh_stats(deltas, REFERENCE_LENGTH)

    elements_inside_reference_length = (
        np.sum(
            (mesh_data["center"][0] - mesh_data["droplet_diameter"] / 2.0 <= mesh_data["coords_stretched"][0])
            & (
                mesh_data["coords_stretched"][0]
                <= mesh_data["center"][0] + mesh_data["droplet_diameter"] / 2.0
            )
        )
        - 1
    )
    print("elements inside RL:", elements_inside_reference_length)

    print("\nInput the following on your file:")
    print("Nx,Ny", mesh_data["num_elements"])
    print("ax, ay:", mesh_data["stretch_alpha"])
    print("[x,y]_[i,f]/RL:\n", mesh_data["auxiliary_bounds"] / REFERENCE_LENGTH)
    print("[x,y]_[a,b]/RL:\n", mesh_data["stretch_bounds"] / REFERENCE_LENGTH)


def plot_mesh(mesh_data):
    apply_thesis_style()

    mesh_aux_x, mesh_aux_y = np.meshgrid(
        mesh_data["coords_aux"][0],
        mesh_data["coords_aux"][1],
    )
    mesh_stretched_x, mesh_stretched_y = np.meshgrid(
        mesh_data["coords_stretched"][0],
        mesh_data["coords_stretched"][1],
    )

    figure, axes = plt.subplots(2, 2)

    axes[0][0].plot(mesh_aux_x / REFERENCE_LENGTH, mesh_aux_y / REFERENCE_LENGTH, "k-", linewidth=0.2)
    axes[0][0].plot(mesh_aux_x.T / REFERENCE_LENGTH, mesh_aux_y.T / REFERENCE_LENGTH, "k-", linewidth=0.2)
    axes[0][0].add_patch(
        plt.Circle(
            (
                mesh_data["center"][0] / REFERENCE_LENGTH,
                mesh_data["center"][1] / REFERENCE_LENGTH,
            ),
            mesh_data["droplet_diameter"] / (2.0 * REFERENCE_LENGTH),
            fill=False,
            linewidth=2,
        )
    )
    axes[0][0].set_xlabel(r"$x/R_L$")
    axes[0][0].set_ylabel(r"$y/R_L$")
    axes[0][0].set_title(r"$\mathrm{Auxiliary\ Uniform}$")
    axes[0][0].set_aspect("equal")

    axes[0][1].plot(mesh_aux_x, mesh_aux_y, "k-", linewidth=0.2)
    axes[0][1].plot(mesh_aux_x.T, mesh_aux_y.T, "k-", linewidth=0.2)
    axes[0][1].add_patch(
        plt.Circle(
            (mesh_data["center"][0], mesh_data["center"][1]),
            mesh_data["droplet_diameter"] / 2.0,
            fill=False,
            linewidth=2,
        )
    )
    axes[0][1].set_xlabel(r"$x\ [\mathrm{m}]$")
    axes[0][1].set_ylabel(r"$y\ [\mathrm{m}]$")
    axes[0][1].set_title(r"$\mathrm{Auxiliary\ Uniform}$")
    axes[0][1].set_aspect("equal")

    axes[1][0].plot(
        mesh_stretched_x / REFERENCE_LENGTH,
        mesh_stretched_y / REFERENCE_LENGTH,
        "b-",
        linewidth=0.2,
    )
    axes[1][0].plot(
        mesh_stretched_x.T / REFERENCE_LENGTH,
        mesh_stretched_y.T / REFERENCE_LENGTH,
        "b-",
        linewidth=0.2,
    )
    axes[1][0].add_patch(
        plt.Circle(
            (
                mesh_data["center"][0] / REFERENCE_LENGTH,
                mesh_data["center"][1] / REFERENCE_LENGTH,
            ),
            mesh_data["droplet_diameter"] / (2.0 * REFERENCE_LENGTH),
            fill=False,
            linewidth=2,
        )
    )
    axes[1][0].set_xlabel(r"$x/R_L$")
    axes[1][0].set_ylabel(r"$y/R_L$")
    axes[1][0].set_title(r"$\mathrm{Stretched}$")
    axes[1][0].set_aspect("equal")

    axes[1][1].plot(mesh_stretched_x, mesh_stretched_y, "b-", linewidth=0.2)
    axes[1][1].plot(mesh_stretched_x.T, mesh_stretched_y.T, "b-", linewidth=0.2)
    axes[1][1].add_patch(
        plt.Circle(
            (mesh_data["center"][0], mesh_data["center"][1]),
            mesh_data["droplet_diameter"] / 2.0,
            fill=False,
            linewidth=2,
        )
    )
    axes[1][1].set_xlabel(r"$x\ [\mathrm{m}]$")
    axes[1][1].set_ylabel(r"$y\ [\mathrm{m}]$")
    axes[1][1].set_title(r"$\mathrm{Stretched}$")
    axes[1][1].set_aspect("equal")

    figure.tight_layout()
    return figure, axes


def main(argv=None):
    args = build_argument_parser().parse_args(argv)
    mesh_data = build_mesh()
    print_mesh_summary(mesh_data)

    if not args.no_show:
        plot_mesh(mesh_data)
        plt.show()


if __name__ == "__main__":
    main()
