from pathlib import Path

import numpy as np


def load_time_radius_history(filepath: Path):
    return np.loadtxt(
        filepath,
        skiprows=2,
        usecols=(1, 4),
        unpack=True,
    )
