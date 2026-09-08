"""
How badly does salt marsh masquerade as oak in the deciduous index?

Spartina greens up in late spring and browns off in winter, which is the same
signature the index uses to find oak. Buzzards Bay has a lot of tidal marsh
along its shoreline, so if marsh scores like oak the layer will paint the whole
coast as prime foraging ground.

Tidal marsh is identified from SSURGO rather than guessed: the map units that
are very poorly drained AND very frequently flooded are, on this coast, the
Ipswich / Pawcatuck / Matunuck tidal peats.

Run on the oak-mapping cluster after pull_soil.py.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
import rioxarray  # noqa: F401
from rasterio.features import geometry_mask, rasterize
import rasterio

SDA = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
CRS = "EPSG:32619"
BBOX_LL = (-71.20, 41.43, -69.87, 42.16)


def sda(query, timeout=600):
    r = requests.post(SDA, json={"format": "JSON+COLUMNNAME", "query": query},
                      timeout=timeout)
    r.raise_for_status()
    rows = r.json()["Table"]
    return pd.DataFrame(rows[1:], columns=rows[0])


def main():
    decid = rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
    drain = rioxarray.open_rasterio("soil/soil_drainage.tif").squeeze(drop=True)

    x0, y0, x1, y1 = BBOX_LL
    box = (f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, "
           f"{x0} {y1}, {x0} {y0}))")

    # Which map units in this domain are tidal?
    att = sda(f"""SELECT mukey, muname, drclassdcd, flodfreqdcd FROM muaggatt
WHERE mukey IN (SELECT DISTINCT mukey FROM mupolygon
 WHERE mupolygongeo.STIntersects(geometry::STGeomFromText('{box}',4326))=1)
 AND flodfreqdcd IN ('Very frequent','Frequent')""")
    print(f"{len(att)} frequently-flooded map units in the domain:")
    for _, r in att.head(12).iterrows():
        print(f"  {r.muname[:78]}")

    keys = ",".join(f"'{k}'" for k in att.mukey)
    geo = sda(f"""SELECT mukey, mupolygongeo.STAsText() AS wkt FROM mupolygon
WHERE mukey IN ({keys})
 AND mupolygongeo.STIntersects(geometry::STGeomFromText('{box}',4326))=1""")
    print(f"\n{len(geo)} tidal-marsh polygons")

    from shapely import wkt as W
    gdf = gpd.GeoDataFrame(geometry=[W.loads(w) for w in geo.wkt],
                           crs="EPSG:4326").to_crs(CRS)

    # Rasterise the marsh mask onto the 10 m deciduous grid.
    marsh = rasterize(
        [(g, 1) for g in gdf.geometry], out_shape=decid.shape,
        transform=decid.rio.transform(), fill=0, dtype="uint8")
    print(f"tidal marsh covers {marsh.sum() * 100 / 1e6:.1f} km2 of the grid")

    d = decid.values
    ok = np.isfinite(d)
    m = (marsh == 1) & ok
    print(f"\n=== deciduous index over tidal marsh (n={m.sum() / 1e6:.2f} Mpx) ===")
    v = d[m]
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:<3d} {np.percentile(v, p):6.3f}")

    # Compare against BIGMAP's confident classes on upland.
    ftg = (rioxarray.open_rasterio("forest/bigmap_forest_type_group.tif")
           .squeeze(drop=True).rio.reproject_match(decid, resampling=0))
    f = ftg.values
    print("\n=== for comparison, upland classes ===")
    for code, label in [(500, "Oak/hickory"), (400, "Oak/pine"),
                        (100, "White/red/jack pine")]:
        sel = ok & (f == code) & (marsh == 0)
        if sel.sum() > 1000:
            print(f"  {label:22s} median {np.median(d[sel]):6.3f} "
                  f"(n={sel.sum() / 1e6:.2f} Mpx)")

    frac = (v > 0.30).mean()
    print(f"\n{100 * frac:.0f}% of tidal marsh scores above 0.30 -- "
          f"the level of a confirmed oak stand")

    # Write the mask so downstream layers can exclude it.
    with rasterio.open(
            "soil/tidal_marsh_mask.tif", "w", driver="GTiff",
            height=marsh.shape[0], width=marsh.shape[1], count=1,
            dtype="uint8", crs=CRS, transform=decid.rio.transform(),
            nodata=0, compress="lzw", tiled=True) as dst:
        dst.write(marsh, 1)
    print("\nwrote soil/tidal_marsh_mask.tif")


if __name__ == "__main__":
    main()
