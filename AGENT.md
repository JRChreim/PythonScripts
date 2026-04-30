# AGENT.md

Local working rules for agents and contributors in `PythonScripts`, adapted from `MatlabScripts/AGENTS.md` for this Python workspace.

## Working Style

- Preserve existing scientific logic and numerical assumptions unless a change is explicitly requested.
- Prefer small, targeted edits over broad refactors.
- Keep file formats, variable naming patterns, and plotting conventions consistent with nearby code.
- Avoid changing hard-coded simulation paths unless the task specifically calls for path cleanup or portability work.

## Python Notes

- Favor compatibility with the style already used in the repo.
- Add comments only where the numerical intent is not obvious from the code.
- When modifying data-loading, optimization, or mesh-generation scripts, keep units, indexing, array shapes, and plotting assumptions aligned with the current implementation.
- Place runnable analyses and experiments under `scripts/`; keep root-level filenames only as thin compatibility wrappers when needed.
- Place reusable logic under `src/` instead of duplicating helper code across scripts.
- Use `snake_case` for repo-owned functions, variables, and new module names.
- Use descriptive names instead of near-duplicates that differ only by abbreviation or capitalization.
- Use `UPPER_SNAKE_CASE` for fixed configuration values that are intended to behave like constants.
- Keep legacy script names only as thin compatibility entry points during future cleanups; new reusable modules should prefer canonical `snake_case.py` names.
- Prefer `pathlib.Path` over manual path concatenation when adding new filesystem logic.
- Centralize machine-specific roots near the top of the script or in a small helper instead of scattering absolute paths across the file.
- Route generated outputs to `artifacts/figures`, `artifacts/movies`, or `artifacts/data` rather than saving them in the repo root.

## Validation

- If a task changes behavior, prefer lightweight verification steps and report what was or was not validated.
