import requests
import json
import time
from datetime import date
import pandas as pd

with open("buzzards_bay_stations.json") as f:
    all_stations = json.load(f)

# Curated 16-station Buzzards Bay list (6 official MassDEP Buzzards Bay basin
# + 10 Buzzards-facing Cape Cod stations), matching STATIONS in
# compare_buzzards_bay_v2.py. Drops Martha's Vineyard (separate island
# watershed), Taunton River basin towns (drain to Narragansett Bay, not
# Buzzards Bay), and a handful of other bbox stations outside the true
# watershed polygon (see refilter_stations.py).
STATIONS = [
    "MA-BR-14", "MA-BR-18", "MA-BR-52", "MA-BR-79", "MA-PL-63", "MA-PL-66",
    "MA-BA-115", "MA-BA-101", "MA-BA-109", "MA-BA-112", "MA-BA-105",
    "MA-BA-113", "MA-BA-57", "MA-BA-2", "MA-BA-87", "MA-BA-13",
]
stations = {k: v for k, v in all_stations.items() if k in STATIONS}
assert sorted(stations) == sorted(STATIONS), f"missing stations: {set(STATIONS) - set(stations)}"
print(f"{len(stations)} stations after filtering to curated list")

BASE = "https://api2.cocorahs.org/api/DailyPrecipObs"
START, END = "2025-01-01", date.today().isoformat()

all_rows = []
for num, info in stations.items():
    r = requests.get(BASE, params={
        "startDate": START, "endDate": END,
        "stationField": "StationNumber", "stationFieldValue": num,
        "limit": 1000, "units": "in",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    n = data["metadata"]["resultset"]["totalCount"]
    for obs in data["results"]:
        all_rows.append({
            "stationNumber": num,
            "stationName": info["stationName"],
            "latitude": info["latitude"],
            "longitude": info["longitude"],
            "date": obs["obsDateTime"][:10],
            "precip_in": obs["precip"],
        })
    print(f"  {num:12s} {info['stationName']:30s} n={n}")
    time.sleep(0.2)

df = pd.DataFrame(all_rows)
df.to_csv("buzzards_bay_cocorahs_daily.csv", index=False)
print(f"\nTotal rows: {len(df)}, stations: {df['stationNumber'].nunique()}")
print(df.head())
