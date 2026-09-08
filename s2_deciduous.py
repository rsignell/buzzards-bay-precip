"""
Sentinel-2 deciduous-fraction layer for the Buzzards Bay + Cape Cod watersheds.

Why this exists: both USFS products classify the 50-acre oak forest at
41.7267, -70.5967 as *non-forest* -- zero modeled basal area for every species,
across a 500 m box, and BIGMAP calls the site pixel "Non-forest". The most
likely cause is low-density residential canopy tripping a developed mask, and
that failure mode covers much of the upper Cape: mature oak with houses under
it. Which is precisely the ground a forager walks.

So this layer inherits no one else's land-cover classification. It measures
leaf-on / leaf-off phenology directly from surface reflectance:

    decid = NDVI(summer median) - NDVI(leaf-off median)

Oak canopy swings hard across that pair (roughly 0.85 -> 0.25); pitch and white
pine barely move (roughly 0.75 -> 0.65). On this landscape the difference is
very nearly a pure oak map, because the deciduous canopy here is overwhelmingly
Quercus -- red maple in the wet swales being the main confuser.

Two details that matter for correctness:
  * The STAC metadata advertises a BOA_ADD_OFFSET that the delivered pixels do
    not actually carry -- see the SCALE comment below. Applying it silently
    drives NDVI above 1.
  * SCL is 20 m and must be resampled nearest, never interpolated between
    class codes.

Outputs (under s2/):
  s2_ndvi_summer.tif    median NDVI, peak growing season
  s2_ndvi_leafoff.tif   median NDVI, leaf-off
  s2_decid.tif          summer - leaf-off, the deciduous index
  s2_nobs_*.tif         valid-observation count per window (QA)

Run on the oak-mapping cluster:
  /home/ubuntu/miniconda3/bin/python s2_deciduous.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import odc.stac
import rioxarray  # noqa: F401  (registers .rio accessor)
from odc.geo.geobox import GeoBox
from pystac_client import Client

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Same extent and CRS as the 30 m host grid from pull_forest_hosts.py, at 10 m.
# Both bounds are multiples of 10, so this grid lands exactly on Sentinel-2's
# native 10 m grid and the 10 m bands need no resampling at all.
BBOX_UTM = (319920, 4591950, 425820, 4662630)
CRS = "EPSG:32619"
RES = 10
BBOX_LL = [-71.20, 41.43, -69.87, 42.16]

WINDOWS = {
    "leafoff": ("2026-02-15", "2026-03-31"),
    "summer": ("2026-07-01", "2026-08-15"),
}
MAX_CLOUD = 40  # permissive; per-pixel SCL masking does the real work

# SCL classes to keep: vegetation, not-vegetated, unclassified.
# Dropped: nodata(0), saturated(1), cast shadow(2), cloud shadow(3),
# water(6), cloud medium/high(8,9), thin cirrus(10), snow(11).
# Water is dropped deliberately -- see MIN_DENOM.
SCL_KEEP = [4, 5, 7]

# Reflectance = DN * SCALE, with NO additive offset -- despite what the STAC
# metadata says.
#
# Every item here is processing baseline 05.12 and advertises
# raster:bands = {scale: 0.0001, offset: -0.1}, i.e. the baseline-04.00
# BOA_ADD_OFFSET convention. The pixels delivered by sentinel-cogs are NOT in
# that convention: Element84 harmonises them back to the classic 0-10000 scale,
# and the offset field is stale boilerplate. Measured over oak canopy in July:
#
#     raw DN        red ~435-472   nir ~3400-3500
#     DN * 1e-4     red 0.044      nir 0.34    -> NDVI 0.81   (textbook)
#     with -0.1     red -0.056     nir 0.24    -> NDVI 1.57   (impossible)
#
# Trusting the metadata drove NDVI past 1 for 42% of pixels and pinned the
# median at the clip bound. Verify this empirically again if the collection is
# ever reprocessed.
SCALE = 0.0001

# Floor on (nir + red) reflectance. Open water and deep shadow sit near zero,
# where the NDVI ratio is numerically unstable; vegetation and bare soil are
# both far above this.
MIN_DENOM = 0.05

OUTDIR = Path("s2")


def search(window):
    d0, d1 = WINDOWS[window]
    cat = Client.open(STAC)
    items = list(
        cat.search(
            collections=[COLLECTION],
            bbox=BBOX_LL,
            datetime=f"{d0}T00:00:00Z/{d1}T23:59:59Z",
            query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        ).items()
    )
    dates = sorted({i.datetime.date().isoformat() for i in items})
    print(f"  {window:8s} {len(items):3d} items over {len(dates)} solar days")
    print(f"           {', '.join(dates)}")
    return items


def composite(items, gbox):
    """Cloud-masked median NDVI over the window, plus valid-obs count."""
    ds = odc.stac.load(
        items,
        bands=("red", "nir", "scl"),
        geobox=gbox,
        groupby="solar_day",  # mosaics the 4 MGRS tiles within each date
        resampling="nearest",
        chunks={"time": -1, "x": 2048, "y": 2048},
        dtype="uint16",
        nodata=0,
    )
    print(f"           stack {dict(ds.sizes)}")

    good = ds.scl.isin(SCL_KEEP)

    def reflectance(band):
        # DN 0 is the nodata flag. Use .where() rather than xr.where(), which
        # silently drops the spatial_ref coord and leaves the written GeoTIFF
        # with no CRS.
        return band.where(band > 0).astype("float32") * SCALE

    red = reflectance(ds.red).where(good)
    nir = reflectance(ds.nir).where(good)

    denom = nir + red
    ndvi = ((nir - red) / denom).where(denom > MIN_DENOM).clip(-1, 1)

    return ndvi.median("time", skipna=True), ndvi.notnull().sum("time")


def main():
    OUTDIR.mkdir(exist_ok=True)
    gbox = GeoBox.from_bbox(BBOX_UTM, crs=CRS, resolution=RES)
    print(f"grid {gbox.shape[1]} x {gbox.shape[0]} @ {RES} m "
          f"({gbox.shape[0] * gbox.shape[1] / 1e6:.1f} Mpx)\n")

    out = {}
    for window in WINDOWS:
        t0 = time.time()
        items = search(window)
        if not items:
            sys.exit(f"no items for {window}")
        ndvi, nobs = composite(items, gbox)

        print(f"           computing median ...", flush=True)
        ndvi = ndvi.compute().odc.assign_crs(CRS)
        nobs = nobs.compute().odc.assign_crs(CRS)
        out[window] = ndvi

        ndvi.odc.write_cog(OUTDIR / f"s2_ndvi_{window}.tif", overwrite=True)
        nobs.astype("int16").odc.write_cog(
            OUTDIR / f"s2_nobs_{window}.tif", overwrite=True
        )

        v = basin_only(ndvi)
        n = nobs.values
        print(f"           NDVI in basins  n={v.size / 1e6:.1f}M  "
              f"p5 {np.percentile(v, 5):.3f}  median {np.median(v):.3f}  "
              f"p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}")
        print(f"           obs/px  min {n.min()}  median {int(np.median(n))}  "
              f"max {n.max()}")
        print(f"           {time.time() - t0:.0f}s\n", flush=True)

    decid = (out["summer"] - out["leafoff"]).rename("decid").odc.assign_crs(CRS)
    decid.odc.write_cog(OUTDIR / "s2_decid.tif", overwrite=True)
    d = basin_only(decid)
    print(f"deciduous index in basins: p5 {np.percentile(d, 5):.3f}  "
          f"median {np.median(d):.3f}  p95 {np.percentile(d, 95):.3f}")
    print(f"\nwrote {OUTDIR}/")


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
_BASIN = None


def basin_geom():
    """Union of the two MassDEP basins, cached."""
    global _BASIN
    if _BASIN is None:
        import geopandas as gpd
        import pandas as pd

        parts = [
            gpd.read_file(f).to_crs(CRS)
            for f in ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]
        ]
        _BASIN = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS) \
            .geometry.union_all()
    return _BASIN


def basin_only(da):
    """Finite values inside the basins. The bbox is two-thirds ocean, so
    whole-grid statistics are dominated by water and say nothing useful."""
    v = da.rio.clip([basin_geom()], CRS, drop=False).values
    return v[np.isfinite(v)]


if __name__ == "__main__":
    main()
