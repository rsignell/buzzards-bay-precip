"""
The oak layer: deciduous index, gated to real closed canopy.

Neither Sentinel-2 index works alone, and they fail in opposite directions.

The deciduous index splits hardwood from conifer well (Oak/hickory 0.375 vs
pine 0.098, separation-to-noise 1.56) but cannot tell oak from anything else
with a seasonal swing -- salt marsh medians 0.254 and mown grass is similar, so
on its own it paints 209 km2 of Buzzards Bay shoreline as prime ground.

Summer NDVI barely separates oak from pine at all (0.896 vs 0.865) but is
excellent at saying what is closed canopy: marsh sits at 0.719, far below any
forest. That is a gate, not a discriminator -- which is what walking the ground
suggested, and what the numbers bear out.

So: gate on summer NDVI, discriminate with the deciduous index, and mask tidal
marsh from SSURGO for the part the gate misses. Measured over the gate sweep,
0.80 is the knee -- it keeps 94% of Oak/hickory while removing 63% of marsh, and
nudges the oak/pine separation up to 1.63. Above 0.84 the gate starts eating
real pine-barrens canopy without buying much more marsh rejection.

Outputs (under s2/):
  oak_index.tif   deciduous index where canopy is closed and ground is not
                  tidal marsh; NaN elsewhere

Run on the oak-mapping cluster after s2_deciduous.py and marsh_confounder.py.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import rioxarray  # noqa: F401
from pyproj import Transformer
from rasterio.features import geometry_mask

CRS = "EPSG:32619"
CANOPY_GATE = 0.80   # summer NDVI; see module docstring
OAK_CUT = 0.30       # index at/above which a pixel reads as oak-bearing

SITES = {
    "confirmed-ish oak (marsh edge)": (41.72656, -70.60387),
    "Oak/hickory control": (41.72572, -70.53661),
    "pitch pine control": (41.90640, -70.69646),
    "cranberry bog": (41.7266604, -70.5966575),
}


def main():
    decid = rioxarray.open_rasterio("s2/s2_decid.tif").squeeze(drop=True)
    summer = (rioxarray.open_rasterio("s2/s2_ndvi_summer.tif")
              .squeeze(drop=True).rio.reproject_match(decid, resampling=1))
    marsh = (rioxarray.open_rasterio("soil/tidal_marsh_mask.tif")
             .squeeze(drop=True).rio.reproject_match(decid, resampling=0))

    d, s, m = decid.values, summer.values, marsh.values
    keep = np.isfinite(d) & np.isfinite(s) & (s > CANOPY_GATE) & (m != 1)
    oak = np.where(keep, d, np.nan).astype("float32")

    out = decid.copy(data=oak).rio.write_nodata(np.nan)
    out.rio.to_raster("s2/oak_index.tif", compress="lzw")

    # --- basin statistics -------------------------------------------------- #
    parts = [gpd.read_file(f).to_crs(CRS) for f in
             ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]]
    geom = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                            crs=CRS).geometry.union_all()
    basin = ~geometry_mask([geom], out_shape=d.shape,
                           transform=decid.rio.transform(), invert=False)
    px = 100 / 1e6  # km2 per 10 m pixel

    print(f"basin                     {basin.sum() * px:8.1f} km2")
    print(f"closed canopy, not marsh  {(basin & keep).sum() * px:8.1f} km2  "
          f"({100 * (basin & keep).sum() / basin.sum():.0f}% of basin)")
    v = oak[basin & keep]
    print(f"\noak index over that canopy: p10 {np.percentile(v, 10):.3f}  "
          f"median {np.median(v):.3f}  p90 {np.percentile(v, 90):.3f}")
    for cut in (0.20, 0.25, 0.30, 0.35, 0.40):
        n = (v >= cut).sum()
        print(f"  index >= {cut:.2f}: {n * px:7.1f} km2  "
              f"({100 * n / len(v):4.0f}% of canopy, "
              f"{100 * n / basin.sum():4.0f}% of basin)")

    # --- what it says at the reference sites ------------------------------- #
    TR = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    print(f"\n{'site':34s}{'canopy%':>9}{'oak index':>11}")
    for name, (lat, lon) in SITES.items():
        x, y = TR.transform(lon, lat)
        sub = out.sel(x=slice(x - 225, x + 225), y=slice(y + 225, y - 225))
        raw = decid.sel(x=slice(x - 225, x + 225), y=slice(y + 225, y - 225))
        vv = sub.values[np.isfinite(sub.values)]
        frac = 100 * len(vv) / raw.values.size
        med = np.median(vv) if len(vv) else float("nan")
        print(f"  {name:32s}{frac:8.0f}%{med:11.3f}")

    print(f"\nwrote s2/oak_index.tif")


if __name__ == "__main__":
    main()
