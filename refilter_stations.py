import json
import geopandas as gpd
from shapely.geometry import Point

watershed = gpd.read_file("/home/rsignell/documents/cso/buzzards_bay_watershed.geojson")
union = watershed.union_all()

with open("buzzards_bay_stations.json") as f:
    stations = json.load(f)

inside, outside = {}, {}
for num, info in stations.items():
    pt = Point(info["longitude"], info["latitude"])
    (inside if union.contains(pt) else outside)[num] = info

print(f"{len(inside)} inside the real Buzzards Bay basin polygon, {len(outside)} outside\n")
print("INSIDE:")
for num, info in sorted(inside.items()):
    print(f"  {num:12s} {info['stationName']:30s} {info['latitude']:.4f} {info['longitude']:.4f}")
print("\nOUTSIDE (previously included/excluded by hand):")
for num, info in sorted(outside.items()):
    print(f"  {num:12s} {info['stationName']:30s} {info['latitude']:.4f} {info['longitude']:.4f}")

with open("buzzards_bay_stations_verified.json", "w") as f:
    json.dump(inside, f, indent=2)
