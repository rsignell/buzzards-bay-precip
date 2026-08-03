import requests
import json

# Buzzards Bay watershed approx bbox (MA south coast)
LAT_MIN, LAT_MAX = 41.35, 41.85
LON_MIN, LON_MAX = -71.15, -70.55

BASE = "https://api2.cocorahs.org/api/DailyPrecipObs"

stations = {}
offset = 0
limit = 1000
day = "2026-06-01"
while True:
    r = requests.get(BASE, params={
        "startDate": day, "endDate": day,
        "offset": offset, "limit": limit,
    }, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data["results"]
    total = data["metadata"]["resultset"]["totalCount"]
    for obs in results:
        lat, lon = obs["latitude"], obs["longitude"]
        if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
            stations[obs["stationNumber"]] = {
                "stationId": obs["stationId"],
                "stationName": obs["stationName"],
                "latitude": lat,
                "longitude": lon,
            }
    offset += limit
    if offset >= total:
        break

print(f"Scanned {total} national obs for {day}; found {len(stations)} stations in Buzzards Bay bbox:")
for num, info in sorted(stations.items()):
    print(f"  {num:12s} {info['stationName']:35s} {info['latitude']:.4f} {info['longitude']:.4f}")

with open("buzzards_bay_stations.json", "w") as f:
    json.dump(stations, f, indent=2)
