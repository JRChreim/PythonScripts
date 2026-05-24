# src

Reusable helpers for the Python analysis scripts.

- `io/`: lightweight file readers and data-loading helpers
  - `mfc_binary.py`: parser for MFC binary post-process snapshots stored as Fortran unformatted records
- `mesh/`: mesh-stretching and geometric-progression helpers
- `plots/`: shared plotting defaults
- `relaxation/`: pressure-relaxation utilities
- `thermo/`: EOS and mixture-property helpers
  - `stiffened_gas.py`: reusable stiffened-gas thermodynamics, saturation, and continuation helpers used by the analysis script
