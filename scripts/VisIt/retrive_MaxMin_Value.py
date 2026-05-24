import re
import csv

variable = "alpha_rho2"
output_file = "max_alpha_rho2_location.csv"

# Make sure the plot already exists and is selected/highlighted.
# For example, you should already have a Pseudocolor plot of "alpha_rho2".

n_states = TimeSliderGetNStates()

pattern = re.compile(
    r"Max\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    r".*coord\s*<([^>]*)>"
)

rows = []

for state in range(n_states):
    SetTimeSliderState(state)
    DrawPlots()

    Query("Max")
    result = GetQueryOutputString()

    match = pattern.search(result)

    if match:
        min_pres = float(match.group(1))
        coords = [float(c.strip()) for c in match.group(2).split(",")]

        x = coords[0] if len(coords) > 0 else ""
        y = coords[1] if len(coords) > 1 else ""
        z = coords[2] if len(coords) > 2 else ""

        rows.append([state, min_pres, x, y, z])
    else:
        rows.append([state, "", "", "", ""])

with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["timestep", "max_alpha_rho2", "x", "y", "z"])
    writer.writerows(rows)

print("Saved results to:", output_file)
