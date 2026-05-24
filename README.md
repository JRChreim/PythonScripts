# PythonScripts

This repository contains research-oriented Python code for mesh generation, small numerical experiments, and post-processing or plotting of simulation data.

The repository is organized so runnable scripts live under `scripts/`, reusable helpers live under `src/`, and generated outputs go under `artifacts/`. The root folder is kept intentionally light.

## Current Layout

- `scripts/`: canonical runnable analyses and experiments.
- `src/io/`: lightweight file readers and data-loading helpers.
- `src/mesh/`: reusable geometric-progression and log-cosh mesh helpers.
- `src/plots/`: shared plotting defaults.
- `src/relaxation/`: pressure-relaxation helpers.
- `src/thermo/`: EOS and mixture-property helpers.
- `tests/`: home for regression checks and future test helpers.
- `archive/`: home for dated or superseded script variants.
- `artifacts/`: home for generated figures and other outputs.
- `external/`: home for imported or third-party code snapshots.

## Notes

- New runnable analyses should live under `scripts/`.
- Several scripts still use hard-coded physical parameters and filesystem paths; prefer turning new configuration into top-level parameters, helper functions, or command-line arguments instead of scattering additional literals.
- Keep plotting and post-processing workflows reproducible: new scripts should state required inputs near the top of the file and avoid hidden runtime assumptions.
- Plotting convention: use SI units in square brackets for any unit-bearing axis, colorbar, or legend label, and keep figure text LaTeX-rendered through the shared helpers in `src/plots/publication.py`.
- Plotting scripts should use `--no-show` to suppress the GUI window; `--no-plot` may remain only as a compatibility alias.
- Reusable numerical code should live under `src/`, with the root kept clean for documentation and shared project notes.

## Usage

Install the project in editable mode once to make the command-line entry points available:

```bash
python3 -m pip install -e .
```

Then run:

- `ecogen-bd-analysis --data-folder /path/to/output/folder`
- `mfc-bd-analysis --data-folder /disk/simulations/Relaxation/BubbleCollapse/2D/Sphere/StrongCollapse/pT/6Eqn/Axisymmetric`
- `geom-progression --ratio 1.03 --count 36`
- `mesh-stretch-geom-prog`
- `mesh-stretch-log-cosh`
- `p-relaxation`
- `ecogen-out-summary --case Euler_IG_air --case 6Eq_pTEq --case 6Eq_pEq`
- `plot-ecogen-out --case 6Eq_pEq --domain mixture --variable p --times 0 10`
- `ecogen-quicklook`
- `mfc-binary-quicklook --data-folder /disk/simulations/Relaxation/Thesis/ExpansionTube/pT/5Eqn/binary`
- `mfc-binary-comparison --to-thesis`

The MFC comparison workflow now produces three figures by default: an overview by simulation percentage, a zoom/inset figure for the front region, and a summary table of max differences.
When `--to-thesis` is not used, the comparison script saves figures into case-organized folders under `artifacts/figures/mfc/<tube>/<mode>/`, so runs like `ExpansionTube_pT` and `ShockTube_pTg` stay distinct.
Both MFC plotting commands also support `--to-thesis`, which keeps the preview titles visible but omits them from the thesis-exported PDFs.

If you prefer not to install the project, the equivalent module form still works with `python3 -m scripts.<module_name>`.

## ECOGEN Results

- The ECOGEN `.out` loader is organized around the sibling results tree `../ECOGEN/results/<case>/datasets`.
- The current Python structure supports dataset `.out` files and separates loaded data into per-fluid fields and mixture fields.
- For the currently selected ECOGEN cases, `Euler_IG_air` and `6Eq_pEq` contain dataset `.out` files, while `6Eq_pTEq` currently does not.
- Programmatic access starts from `src.io.ecogen_out`:

```python
from src.io import discover_ecogen_cases, load_ecogen_out_case

cases = discover_ecogen_cases()
case = load_ecogen_out_case(cases["6Eq_pEq"])

water_pressure = case.fluids[0].data.fields["p"]
air_mass_fraction = case.fluids[1].data.fields["Y"]
mixture_pressure = case.mixture.fields["p"]
mixture_energy = case.mixture.fields["E"]
```

- First plotting entry point:

```bash
plot-ecogen-out --case 6Eq_pEq --domain mixture --variable p --times 0 10
plot-ecogen-out --case 6Eq_pEq --domain fluid --fluid-index 1 --variable alpha --times 0 10
```

- Notebook-style exploratory plotting starts from [scripts/ecogen_quicklook.py](/home/user/Documents/GitHub/PythonScripts/scripts/ecogen_quicklook.py), where you can edit `QUICKLOOK_CONFIG` directly and then run:

```bash
ecogen-quicklook
ecogen-quicklook --latest-only
ecogen-quicklook --case Euler_IG_air --case 6Eq_pEq --times 0 10
```

## Environment

- Python 3
- `numpy`
- `matplotlib`
- `scipy` for mesh-stretching scripts
- `scipy` is also used by the Keller-Miksis comparison overlays in the bubble-dynamics scripts
- A LaTeX installation is currently required by `scripts.ecogen_bd_analysis` and `scripts.mfc_bd_analysis` because `text.usetex` is enabled

## Naming Convention

- Use `snake_case` for functions, variables, and new module names.
- Use `UPPER_SNAKE_CASE` for true constants and fixed configuration values shared across a script.
- Prefer descriptive names such as `stretch_transform` and `print_mesh_stats` over abbreviations when touching owned code.
- Keep canonical runnable implementations in `snake_case.py` files under `scripts/` or `src/`.
- Avoid hard-coded absolute paths in new code. Use `pathlib.Path`, function arguments, or environment variables to describe external data locations.
