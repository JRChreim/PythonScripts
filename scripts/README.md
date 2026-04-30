# scripts

Runnable entry points for analyses, plotting, and numerical experiments.

The canonical script implementations live here. Install the project in editable mode with `python3 -m pip install -e .` to get the command-line entry points from the repository root; otherwise you can still run them with `python3 -m scripts.<module_name>`.

Current highlights:

- `ecogen_out_summary.py`: inventory selected ECOGEN result cases and load dataset `.out` files into fluid and mixture blocks
- `plot_ecogen_out.py`: plot selected ECOGEN `.out` variables versus `x` for mixture or fluid fields
- `ecogen_quicklook.py`: notebook-style multi-panel exploratory plotting with an editable config block
