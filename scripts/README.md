# scripts

Runnable entry points for analyses, plotting, and numerical experiments.

The canonical script implementations live here. Install the project in editable mode with `python3 -m pip install -e .` to get the command-line entry points from the repository root; otherwise you can still run them with `python3 -m scripts.<module_name>`.

Current highlights:

- `ecogen_out_summary.py`: inventory selected ECOGEN result cases and load dataset `.out` files into fluid and mixture blocks
- `plot_ecogen_out.py`: plot selected ECOGEN `.out` variables versus `x` for mixture or fluid fields
- `ecogen_quicklook.py`: notebook-style multi-panel exploratory plotting with an editable config block
- `mfc_binary_quicklook.py`: read MFC binary `root/*.dat` snapshots and plot a handful of 1D profiles across saved steps, with `--to-thesis` PDF export support
- `mfc_binary_comparison.py`: compare `5Eqn` and `6Eqn` MFC binary snapshots with percentage-based overview plots, zoom insets, a summary table, case-organized output folders such as `artifacts/figures/mfc/expansion_tube/pT/` or `artifacts/figures/mfc/shock_tube/pTg/`, and `--to-thesis` PDF export support
- `droplet_min_pressure_map.py`: plot `(x, y)` locations of minimum pressure events colored by `min_pres`
- `droplet_min_pressure_timestep.py`: plot `min_pres` versus timestep for the pure-fluid and mixture histories
- `droplet_max_alpha_rho2_map.py`: plot `(x, y)` locations of maximum `alpha_rho2` events colored by `max_alpha_rho2`
- `droplet_max_alpha_rho2_timestep.py`: plot `max_alpha_rho2` versus timestep for the droplet breakup history
- `analysis_simoes_moreira.py`: plot the Simoes-Moreira density profile and `U_F` versus `T` in a 1x2 layout, with `--to-thesis` PDF export support
- `stiffened_gas_eos.py`: reproduce the stiffened-gas EoS pressure-temperature, entropy, Gibbs, and saturation sweeps from the MATLAB script, with the `p-v` diagram as the default output and the auxiliary MATLAB-style figures available via `--all-figures`, plus `--to-thesis` PDF export support
