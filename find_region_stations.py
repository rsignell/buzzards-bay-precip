"""
Find every CoCoRaHS station inside the map extent used by
compare_buzzards_bay_mrms.py (Buzzards Bay + upper Cape Cod + MA south coast),
not just the curated 16-station watershed set.

The CoCoRaHS web API has no working server-side geographic filter, so this
pages through several days of *national* DailyPrecipObs and keeps whatever
falls in the bounding box. Scanning a multi-day window (not one day) catches
stations that skipped a report on any given morning.

Outputs:
  region_cocorahs_stations.csv   one row per station: id, name, lat/lon,
                                 n_reports in the scan window, and the two
                                 KEY_DATES report totals as precip_MMDD (in)
"""

import os
import time
from datetime import date

import geopandas as gpd
import pandas as pd
import requests

# Map extent from compare_buzzards_bay_mrms.py (Buzzards Bay basin bounds + PAD,
# pushed east to the Cape Cod basin divide).
PAD = 0.12
EAST_EDGE = -70.34
minx, miny, maxx, maxy = gpd.read_file("buzzards_bay_watershed.geojson").to_crs(4326).total_bounds
LON_MIN, LON_MAX = minx - PAD, max(maxx + PAD, EAST_EDGE)
LAT_MIN, LAT_MAX = miny - PAD, maxy + PAD
print(f"bbox lon [{LON_MIN:.3f}, {LON_MAX:.3f}]  lat [{LAT_MIN:.3f}, {LAT_MAX:.3f}]")

BASE = "https://api2.cocorahs.org/api/DailyPrecipObs"
# Last report day to scan. Defaults to today (this morning's 7am readout);
# override for a back-fill with  REPORT_DATE=2026-09-04 python find_region_stations.py
END_DATE = os.environ.get("REPORT_DATE") or date.today().isoformat()
SCAN_DAYS = pd.date_range(pd.Timestamp(END_DATE) - pd.Timedelta(days=15),
                          END_DATE, freq="D")
# report-day columns to carry through to the csv (yesterday + END_DATE)
KEY_DATES = [(pd.Timestamp(END_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
             END_DATE]
LIMIT = 2000

stations = {}           # stationNumber -> metadata + report counter
day_precip = {}         # (stationNumber, date) -> precip_in

for day in SCAN_DAYS:
    d = day.strftime("%Y-%m-%d")
    offset, total = 0, None
    while total is None or offset < total:
        r = requests.get(BASE, params={
            "startDate": d, "endDate": d, "offset": offset, "limit": LIMIT,
            "units": "in",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        total = data["metadata"]["resultset"]["totalCount"]
        for o in data["results"]:
            lat, lon = o["latitude"], o["longitude"]
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                continue
            num = o["stationNumber"]
            st = stations.setdefault(num, {
                "stationNumber": num, "stationName": o["stationName"],
                "latitude": lat, "longitude": lon, "n_reports": 0,
            })
            st["n_reports"] += 1
            day_precip[(num, o["obsDateTime"][:10])] = o["precip"]
        offset += LIMIT
        time.sleep(0.15)
    print(f"  {d}: {total:5d} national obs scanned, {len(stations)} region stations so far")

df = pd.DataFrame(stations.values())
for kd in KEY_DATES:
    df[f"precip_{kd[5:].replace('-', '')}"] = df["stationNumber"].map(
        lambda n, kd=kd: day_precip.get((n, kd))
    )

# tag the curated 16-station Buzzards Bay watershed set
CURATED = {
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",
}
df["in_curated_16"] = df["stationNumber"].isin(CURATED)
df = df.sort_values(["stationNumber"]).reset_index(drop=True)
df.to_csv("region_cocorahs_stations.csv", index=False)

last_col = f"precip_{END_DATE[5:].replace('-', '')}"
n_rep = df[last_col].notna().sum()
print(f"\n{len(df)} CoCoRaHS stations in the map extent "
      f"({df['in_curated_16'].sum()} of the curated 16 among them)")
prev = (pd.Timestamp(END_DATE) - pd.Timedelta(days=1)).strftime("%b %-d")
print(f"{n_rep} of them filed a {END_DATE} report "
      f"(the {prev} 7am -> {pd.Timestamp(END_DATE):%b %-d} 7am window)")
print("\n" + df.to_string(index=False))
