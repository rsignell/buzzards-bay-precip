import requests
import json
import time
import pandas as pd

with open("buzzards_bay_stations.json") as f:
    stations = json.load(f)

# Drop Martha's Vineyard (Dukes County - separate island watershed) and
# Taunton River basin towns (Somerset, Dighton - drain to Narragansett Bay,
# not Buzzards Bay), which fell inside the raw bbox but aren't in-watershed.
EXCLUDE = {"MA-BR-72", "MA-BR-8", "MA-PL-2", "MA-PL-54"}
stations = {k: v for k, v in stations.items() if not k.startswith("MA-DK") and k not in EXCLUDE}
print(f"{len(stations)} stations after exclusions")

BASE = "https://api2.cocorahs.org/api/DailyPrecipObs"
START, END = "2025-01-01", "2026-07-31"

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
