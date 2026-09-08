"""
SSURGO soil layers for the Buzzards Bay + Cape Cod watersheds.

Soil is what turns a rainfall total into a moisture state. On this landscape
that matters more than usual: Cape Cod is glacial outwash sand, so a soak drains
away in days on the uplands while the kettle-hole swales beside it stay wet for
weeks. Two stands with identical 1 km rainfall can be in completely different
condition, and the difference is the soil.

Pulled from USDA Soil Data Access. The IIPP soil ImageServers look convenient
but are cached display-only tile services -- they render pictures, not values --
so this goes to the tabular/spatial API instead.

Attributes come from `muaggatt`, which is already aggregated to one row per map
unit, so the 62k polygons over this domain resolve to just 694 distinct mukeys
and one small attribute query. Geometry is fetched tile by tile because a single
whole-domain geometry request is far too large.

Outputs (under soil/):
  soil_awc.tif        available water storage, 0-100 cm (cm) -- the moisture
                      capacity that sets how long a soak lasts
  soil_drainage.tif   drainage class, 1 = excessively drained .. 7 = very poorly
  soil_wtdepth.tif    annual minimum water table depth (cm)
  soil_summary.csv    area by drainage class and by map unit

Run on the oak-mapping cluster:
  /home/ubuntu/miniconda3/bin/python pull_soil.py
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import rasterize
from shapely import wkt as shapely_wkt

SDA = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
BBOX_LL = (-71.20, 41.43, -69.87, 42.16)
CRS = "EPSG:32619"
RES = 30
BBOX_UTM = (319920, 4591950, 425820, 4662630)  # same grid as the host layers
OUTDIR = Path("soil")
NTILE_X, NTILE_Y = 8, 5

# NRCS drainage classes, ordered dry -> wet. The ordinal is what makes this
# usable as a moisture-decay modifier; the names alone are not sortable.
DRAINAGE = {
    "Excessively drained": 1,
    "Somewhat excessively drained": 2,
    "Well drained": 3,
    "Moderately well drained": 4,
    "Somewhat poorly drained": 5,
    "Poorly drained": 6,
    "Very poorly drained": 7,
}


def sda(query, timeout=600):
    r = requests.post(SDA, json={"format": "JSON+COLUMNNAME", "query": query},
                      timeout=timeout)
    r.raise_for_status()
    body = r.json()
    if "Table" not in body:
        raise RuntimeError(f"SDA: {json.dumps(body)[:300]}")
    rows = body["Table"]
    return pd.DataFrame(rows[1:], columns=rows[0])


def wkt_box(x0, y0, x1, y1):
    return (f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, "
            f"{x0} {y1}, {x0} {y0}))")


def fetch_tile(box):
    q = f"""SELECT mukey, mupolygongeo.STAsText() AS wkt
FROM mupolygon
WHERE mupolygongeo.STIntersects(geometry::STGeomFromText('{box}',4326))=1"""
    for attempt in range(3):
        try:
            return sda(q)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"    retry after {type(e).__name__}", flush=True)
    return pd.DataFrame()


def main():
    OUTDIR.mkdir(exist_ok=True)
    x0, y0, x1, y1 = BBOX_LL

    # --- geometry, tile by tile ------------------------------------------- #
    xs = np.linspace(x0, x1, NTILE_X + 1)
    ys = np.linspace(y0, y1, NTILE_Y + 1)
    boxes = [wkt_box(xs[i], ys[j], xs[i + 1], ys[j + 1])
             for i in range(NTILE_X) for j in range(NTILE_Y)]
    print(f"fetching geometry in {len(boxes)} tiles ...")

    with ThreadPoolExecutor(max_workers=5) as pool:
        frames = list(pool.map(fetch_tile, boxes))
    for i, f in enumerate(frames):
        print(f"  tile {i + 1:2d}/{len(boxes)}: {len(f):6d} polygons")

    poly = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["wkt"])
    print(f"\n{len(poly)} unique polygons, {poly.mukey.nunique()} map units")

    gdf = gpd.GeoDataFrame(
        poly[["mukey"]],
        geometry=[shapely_wkt.loads(w) for w in poly.wkt],
        crs="EPSG:4326",
    ).to_crs(CRS)
    gdf["mukey"] = gdf.mukey.astype(str)

    # --- attributes, one query -------------------------------------------- #
    keys = ",".join(f"'{k}'" for k in sorted(gdf.mukey.unique()))
    att = sda(f"""SELECT mukey, muname, drclassdcd, aws0100wta, wtdepannmin,
 hydgrpdcd FROM muaggatt WHERE mukey IN ({keys})""")
    att["mukey"] = att.mukey.astype(str)
    att["awc"] = pd.to_numeric(att.aws0100wta, errors="coerce")
    att["wtdep"] = pd.to_numeric(att.wtdepannmin, errors="coerce")
    att["drain"] = att.drclassdcd.map(DRAINAGE)
    print(f"attributes for {len(att)} map units "
          f"({att.drain.notna().sum()} with a drainage class)")

    gdf = gdf.merge(att[["mukey", "muname", "awc", "wtdep", "drain"]],
                    on="mukey", how="left")

    # --- rasterise --------------------------------------------------------- #
    width = int((BBOX_UTM[2] - BBOX_UTM[0]) / RES)
    height = int((BBOX_UTM[3] - BBOX_UTM[1]) / RES)
    transform = rasterio.transform.from_origin(
        BBOX_UTM[0], BBOX_UTM[3], RES, RES)
    print(f"rasterising to {width} x {height} @ {RES} m")

    for name, col, dtype, nodata in [
        ("soil_awc", "awc", "float32", np.nan),
        ("soil_drainage", "drain", "float32", np.nan),
        ("soil_wtdepth", "wtdep", "float32", np.nan),
    ]:
        sub = gdf[gdf[col].notna()]
        arr = rasterize(
            ((g, v) for g, v in zip(sub.geometry, sub[col])),
            out_shape=(height, width), transform=transform,
            fill=np.nan, dtype="float32", all_touched=False,
        )
        path = OUTDIR / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=height,
                           width=width, count=1, dtype=dtype, crs=CRS,
                           transform=transform, nodata=nodata,
                           compress="lzw", tiled=True) as dst:
            dst.write(arr, 1)
        v = arr[np.isfinite(arr)]
        print(f"  {name:16s} coverage {100 * len(v) / arr.size:5.1f}%  "
              f"min {v.min():7.2f}  median {np.median(v):7.2f}  max {v.max():7.2f}")

    # --- summary ----------------------------------------------------------- #
    gdf["area_km2"] = gdf.geometry.area / 1e6
    by_drain = (gdf.groupby("drclassdcd" if "drclassdcd" in gdf else "drain")
                if False else gdf.groupby("drain"))
    inv = {v: k for k, v in DRAINAGE.items()}
    rows = []
    for d, g in by_drain:
        rows.append({"drainage_code": int(d), "drainage_class": inv.get(int(d), "?"),
                     "area_km2": g.area_km2.sum(),
                     "mean_awc_cm": g.awc.mean()})
    summary = pd.DataFrame(rows).sort_values("drainage_code")
    summary["pct"] = 100 * summary.area_km2 / summary.area_km2.sum()
    print("\nby drainage class (whole bbox, incl. area outside the basins):")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    summary.to_csv(OUTDIR / "soil_summary.csv", index=False)

    top = (gdf.groupby("muname").area_km2.sum().sort_values(ascending=False)
           .head(12))
    print("\nlargest map units:")
    for name, a in top.items():
        print(f"  {a:7.1f} km2  {name[:80]}")


if __name__ == "__main__":
    main()
