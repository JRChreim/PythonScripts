import re
import csv

outfile = "alpha_rho2_max_by_timestep.csv"

# Parse lines like:
# alpha_rho2 -- Max = 0 (zone 0 in domain 1 at coord <-3.90628, 0.0009375>)
def parse_max_text(text):
    value_m = re.search(r"Max\s*=\s*([^\s(]+)", text)
    coord_m = re.search(r"coord\s*<([^>]+)>", text)
    zone_m = re.search(r"zone\s+(\d+)", text)
    domain_m = re.search(r"domain\s+(\d+)", text)

    if not value_m or not coord_m:
        return None

    try:
        value = float(value_m.group(1))
        coords = [float(x.strip()) for x in coord_m.group(1).split(",")]
    except ValueError:
        return None

    while len(coords) < 3:
        coords.append("")

    zone = zone_m.group(1) if zone_m else ""
    domain = domain_m.group(1) if domain_m else ""
    return value, coords, zone, domain

# Make sure the alpha_rho2 plot you want is active before running this.
# If you want the script to build the plot itself, add OpenDatabase/AddPlot here.

SetQueryOutputToString()
SuppressQueryOutputOn()

fp = open(outfile, "w")
try:
    writer = csv.writer(fp)
    writer.writerow(["timestep", "time", "max_value", "zone", "domain", "x", "y", "z"])

    nstates = TimeSliderGetNStates()
    for ts in range(nstates):
        SetTimeSliderState(ts)

        Query("Time")
        sim_time = GetQueryOutputValue()

        Query("Max", use_actual_data=1)
        parsed = parse_max_text(GetQueryOutputString())
        if parsed is None:
            print("Could not parse timestep %d" % ts)
            continue

        max_value, coords, zone, domain = parsed
        writer.writerow([ts, sim_time, max_value, zone, domain, coords[0], coords[1], coords[2]])
        print("timestep=%d time=%s max=%g coord=%s" % (ts, str(sim_time), max_value, coords[:3]))
finally:
    fp.close()
    SuppressQueryOutputOff()

print("Wrote %s" % outfile)
