"""
Does the Sentinel-2 deciduous index find the oak that FHP and BIGMAP missed?

Three checks, in order of how much they matter:

  1. The acceptance test. A known 50-acre white/red oak forest at
     41.7267, -70.5967 that both USFS products call non-forest. If the S2 layer
     does not light this up, it is not done.
  2. Separability. Does the index actually split BIGMAP's oak classes from its
     pine class? BIGMAP is imperfect but independent, so agreement where it is
     confident is evidence the index measures what it claims to.
  3. Coverage. How much oak-bearing ground does S2 find that FHP zeroed out?

Run on the oak-mapping cluster after s2_deciduous.py.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
from pyproj import Transformer

CRS = "EPSG:32619"
SITE = (41.7266604, -70.5966575)  # the oak forest both products miss
FTG = {0: "Non-forest", 100: "White/red/jack pine", 400: "Oak/pine",
       500: "Oak/hickory", 700: "Elm/ash/cottonwood", 800: "Maple/beech/birch"}


def basins():
    parts = [gpd.read_file(f).to_crs(CRS) for f in
             ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]]
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            crs=CRS).geometry.union_all()


def main():
    decid = rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
    summer = rioxarray.open_rasterio("s2/s2_ndvi_summer.tif").squeeze(drop=True)
    leafoff = rioxarray.open_rasterio("s2/s2_ndvi_leafoff.tif").squeeze(drop=True)

    # ---- 1. the acceptance test ------------------------------------------ #
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    X, Y = tr.transform(SITE[1], SITE[0])
    print(f"=== Site: {SITE[0]}, {SITE[1]} (the 50-acre oak forest) ===")
    for name, da in [("summer NDVI", summer), ("leaf-off NDVI", leafoff),
                     ("DECIDUOUS INDEX", decid)]:
        pt = float(da.sel(x=X, y=Y, method="nearest"))
        box = da.sel(x=slice(X - 225, X + 225), y=slice(Y + 225, Y - 225)).values
        print(f"  {name:16s} point {pt:6.3f}   450 m box: mean {np.nanmean(box):6.3f} "
              f"p10 {np.nanpercentile(box, 10):6.3f} p90 {np.nanpercentile(box, 90):6.3f}")

    # ---- 2. separability against BIGMAP ---------------------------------- #
    ftg = (rioxarray.open_rasterio("forest/bigmap_forest_type_group.tif")
           .squeeze(drop=True)
           .rio.reproject_match(decid, resampling=0))  # 0 = nearest
    geom = basins()
    d = decid.rio.clip([geom], CRS, drop=False).values
    f = ftg.rio.clip([geom], CRS, drop=False).values

    print("\n=== Deciduous index by BIGMAP 2018 class (in basins) ===")
    print(f"  {'class':24s} {'n (Mpx)':>8s} {'p25':>7s} {'median':>7s} {'p75':>7s}")
    rows = {}
    for code, label in FTG.items():
        m = np.isfinite(d) & (f == code)
        if m.sum() < 1000:
            continue
        v = d[m]
        rows[label] = np.median(v)
        print(f"  {label:24s} {v.size / 1e6:8.2f} {np.percentile(v, 25):7.3f} "
              f"{np.median(v):7.3f} {np.percentile(v, 75):7.3f}")

    if "Oak/hickory" in rows and "White/red/jack pine" in rows:
        print(f"\n  oak/hickory - pine separation: "
              f"{rows['Oak/hickory'] - rows['White/red/jack pine']:+.3f}")

    # ---- 3. coverage vs FHP ---------------------------------------------- #
    oak = (rioxarray.open_rasterio("forest/fhp_ba_oak_deciduous_spp.tif")
           .squeeze(drop=True).rio.reproject_match(decid, resampling=0))
    o = oak.rio.clip([geom], CRS, drop=False).values

    px_km2 = 10 * 10 / 1e6
    valid = np.isfinite(d) & np.isfinite(o)
    print(f"\n=== Coverage (basin, {valid.sum() * px_km2:.0f} km2 with both layers) ===")
    for thr in (0.25, 0.30, 0.35):
        s2_yes = valid & (d >= thr)
        fhp_no = valid & (o == 0)
        both = s2_yes & fhp_no
        print(f"  decid >= {thr:.2f}: S2 {s2_yes.sum() * px_km2:7.1f} km2   "
              f"of which FHP says zero oak: {both.sum() * px_km2:7.1f} km2 "
              f"({100 * both.sum() / max(s2_yes.sum(), 1):.0f}%)")


if __name__ == "__main__":
    main()
