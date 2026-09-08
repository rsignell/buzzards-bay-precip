"""
Sentinel-2 deciduous-fraction layer for the Buzzards Bay + Cape Cod watersheds.

Why this exists: USFS FHP reports zero basal area -- for oak, pitch pine and
white pine alike -- across the whole neighbourhood of a confirmed ~50-acre
white/red oak forest at 41.72656, -70.60387, including pixels where BIGMAP maps
Oak/pine. FHP is a national model keyed on climate, terrain and soils, and on
flat, climatically uniform, uniformly sandy Cape Cod those predictors carry
almost no local signal.

So this layer inherits no one else's land-cover classification. It measures
leaf-on / leaf-off phenology directly from surface reflectance:

    decid = NDVI(summer median) - NDVI(leaf-off median)

Oak canopy swings hard across that pair; pitch and white pine barely move. On
this landscape the difference is very nearly a pure oak map, because the
deciduous canopy here is overwhelmingly Quercus -- red maple in the wet swales
being the main confuser. Measured over ~50-acre circles: pine control 0.11,
cranberry bog 0.13, the confirmed oak site 0.33, a pure Oak/hickory control
0.43.

(Cranberry bogs are worth calling out: cranberry is an EVERGREEN dwarf shrub,
so a bog holds NDVI ~0.75 all winter and reads as strongly non-deciduous. Given
how much of this landscape is bog, that is a useful property, not a nuisance.)

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
import xarray as xr
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

# The leaf-off window is wide on purpose. Feb 15 - Mar 31 looks like the
# obvious choice and is the worst of the five tested (tune_leafoff_window.py):
# it left a median of ONE valid observation at the confirmed oak site, and at
# the pine control it caught snow and low-sun-angle scenes whose depressed NDVI
# made pine look deciduous. Widening to Nov 15 - Apr 15 gives ~7 observations
# and lets the median reject those outliers, which nearly doubles the
# oak-vs-pine separation (0.178 -> 0.326). Oaks here are bare by mid-November
# and do not leaf out until mid-May, so the window stays genuinely leaf-off.
WINDOWS = {
    "leafoff": ("2025-11-15", "2026-04-15"),
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

# Rows per block in composite(). 1024 rows x 10590 px x 33 dates x 3 bands of
# uint16 is ~2 GB, which leaves plenty of headroom on a 32 GiB box.
BLOCK_ROWS = 1024

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
    """Cloud-masked median NDVI over the window, plus valid-obs count.

    Processed in horizontal blocks rather than as one dask graph. With a wide
    leaf-off window the stack is ~33 dates x 75 Mpx x 3 bands, which is ~15 GB
    of uint16 before any float intermediates -- enough to OOM a 32 GiB box. The
    first attempt did exactly that, and the kill was not clean: GDAL had already
    flushed partial output, so the GeoTIFFs looked valid, carried a sane CRS and
    a plausible value distribution, but were silently WRONG (at the reference
    site the raster said 1 valid observation and NDVI 0.439 where a direct
    small-window computation gives 7 and 0.495). Blocking keeps peak memory at a
    couple of GB and makes the result trustworthy.
    """
    ny, nx = gbox.shape
    out = np.full((ny, nx), np.nan, dtype="float32")
    nobs = np.zeros((ny, nx), dtype="int16")

    for y0 in range(0, ny, BLOCK_ROWS):
        y1 = min(y0 + BLOCK_ROWS, ny)
        sub = gbox[y0:y1, 0:nx]
        ds = odc.stac.load(
            items,
            bands=("red", "nir", "scl"),
            geobox=sub,
            groupby="solar_day",  # mosaics the 4 MGRS tiles within each date
            resampling="nearest",
            dtype="uint16",
            nodata=0,
        )
        good = ds.scl.isin(SCL_KEEP)

        def reflectance(band):
            # DN 0 is the nodata flag. Use .where() rather than xr.where(),
            # which silently drops the spatial_ref coord and leaves the written
            # GeoTIFF with no CRS.
            return band.where(band > 0).astype("float32") * SCALE

        red = reflectance(ds.red).where(good)
        nir = reflectance(ds.nir).where(good)
        denom = nir + red
        ndvi = ((nir - red) / denom).where(denom > MIN_DENOM).clip(-1, 1)

        out[y0:y1] = ndvi.median("time", skipna=True).values
        nobs[y0:y1] = ndvi.notnull().sum("time").values.astype("int16")
        ndates = ds.sizes["time"]
        del ds, good, red, nir, denom, ndvi
        print(f"             rows {y0:5d}-{y1:5d}  {ndates:2d} dates  "
              f"median obs {int(np.median(nobs[y0:y1]))}", flush=True)

    coords = {"y": gbox.coordinates["y"].values, "x": gbox.coordinates["x"].values}

    def wrap(a):
        return xr.DataArray(a, dims=("y", "x"), coords=coords).odc.assign_crs(CRS)

    return wrap(out), wrap(nobs)


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
        print("           compositing by block ...", flush=True)
        ndvi, nobs = composite(items, gbox)

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


_MASK = None


def basin_mask(da):
    """Boolean basin mask on the grid, rasterised once and reused."""
    global _MASK
    if _MASK is None:
        from rasterio.features import geometry_mask

        _MASK = ~geometry_mask(
            [basin_geom()], out_shape=da.shape,
            transform=da.rio.transform(), invert=False)
    return _MASK


def basin_only(da):
    """Finite values inside the basins. The bbox is two-thirds ocean, so
    whole-grid statistics are dominated by water and say nothing useful."""
    v = da.values
    return v[basin_mask(da) & np.isfinite(v)]


if __name__ == "__main__":
    main()
