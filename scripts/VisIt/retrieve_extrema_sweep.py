import csv
import re


# Edit these four blocks for your case.
variables = ["pres", "alpha_b", "alpha_2", "Y2"]
query_modes = ["Min", "Max"]
start_state = 208
end_state = 340
stride = 1


SetQueryOutputToString()
SuppressQueryOutputOn()


def parse_query_output(text, query_name):
    pattern = re.compile(
        rf"{re.escape(query_name)}\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
        r".*coord\s*<([^>]*)>",
        re.S,
    )

    match = pattern.search(text)
    if not match:
        return None

    try:
        value = float(match.group(1))
        coords = [float(coord.strip()) for coord in match.group(2).split(",")]
    except ValueError:
        return None

    while len(coords) < 3:
        coords.append("")

    return value, coords[0], coords[1], coords[2]


try:
    for variable in variables:
        DeleteAllPlots()
        AddPlot("Pseudocolor", variable)
        DrawPlots()

        rows_by_query = {query_name: [] for query_name in query_modes}

        for state in range(start_state, end_state + 1, stride):
            SetTimeSliderState(state)
            DrawPlots()

            Query("Time")
            sim_time = GetQueryOutputValue()

            for query_name in query_modes:
                Query(query_name)
                parsed = parse_query_output(GetQueryOutputString(), query_name)

                if parsed is None:
                    rows_by_query[query_name].append(
                        [state, sim_time, "", "", "", ""]
                    )
                    continue

                value, x, y, z = parsed
                rows_by_query[query_name].append([state, sim_time, value, x, y, z])

        for query_name in query_modes:
            output_file = f"{variable}_{query_name.lower()}_location.csv"
            with open(output_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestep", "time", "value", "x", "y", "z"])
                writer.writerows(rows_by_query[query_name])

            print("Saved results to:", output_file)
finally:
    SuppressQueryOutputOff()
