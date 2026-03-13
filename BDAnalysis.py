import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 16,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"]
})

data_folder = Path("/disk/simulations/PhaseChange/BubbleCollapse/ECOGEN/1D/SphericalCollapse/Pratio1427")

P_types = ["P", "PT"]
N_types = ["N160E3", "N320E3", "N640E3", "N128E4", "N256E4"]

# Color-blind safe palette (Okabe-Ito)
colors = {
    "P": "#0072B2",   # blue
    "PT": "#E69F00"   # orange
}

# Resolution styles
styles = {
    "N160E3": {"linestyle": "-",  "marker": "o"},
    "N320E3": {"linestyle": "--", "marker": "s"},
    "N640E3": {"linestyle": "-.", "marker": "^"},
    "N128E4": {"linestyle": ":",  "marker": "d"},
    "N256E4": {"linestyle": "-",  "marker": "x"}
}

plt.figure(figsize=(8,5))

for P in P_types:
    for N in N_types:

        filename = f"{P}{N}.xyz"
        filepath = data_folder / filename

        try:
            t, R = np.loadtxt(
                filepath,
                skiprows=2,
                usecols=(1,4),
                unpack=True
            )

            plt.plot(
                t,
                R/R[0],
                color=colors[P],
                linestyle=styles[N]["linestyle"],
                marker=styles[N]["marker"],
                markersize=14,
                markevery=50,
                linewidth=2,
                label=f"{P} {N}"
            )

        except OSError:
            print(f"{filepath} not found")

plt.xlabel("time-step []")
plt.ylabel("$R/R_0 []$")
plt.title("Radial evolution, bubble collapse problem")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()