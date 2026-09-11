"""
Per-species foraging conditions: favorable / marginal / unfavorable.

Combines the static habitat layers with the current moisture state into one
score per species, on the shared 30 m grid, for the four species on the curated
list: chanterelle, bolete, chicken-of-the-woods, hen-of-the-woods.

Three sub-scores, each 0-1, combined by Liebig's law of the minimum rather than
averaged. A stand with perfect moisture and no oak is not "half good" for
hen-of-the-woods, it is useless, and averaging would hide that. Taking the
minimum also means every pixel has a named limiting factor, which is written out
alongside the score -- "marginal because it is too dry" and "marginal because
the oak is thin" are different messages to a forager, and only one of them is
worth waiting out.

  host      from the Sentinel-2 oak index, Cape-calibrated. Boletes also take
            conifers, so they get the better of an oak ramp and a conifer ramp
            derived from the same index read the other way.
  season    a day-of-year trapezoid per species.
  moisture  soil moisture averaged over the species' fruiting lag window, and
            rainfall over the same window. Fruiting follows a soak by one to
            three weeks depending on species, so the window is the point.

Everything is masked to closed non-marsh canopy: outside that, there is no
habitat to rate and the map stays transparent rather than painting the ocean
and the parking lots "unfavorable".

THE SPECIES PARAMETERS ARE EXPERT-GUESS, NOT FITTED. There is no fruiting
record for this region to calibrate against. They encode ordinary mycological
knowledge for southeastern Massachusetts and are isolated in SPECIES below so a
season of observations can replace them.

Outputs (under scores/):
  <species>_score.tif    0-1 continuous
  <species>_class.tif    1 unfavorable, 2 marginal, 3 favorable
  <species>_limiter.tif  1 host, 2 season, 3 moisture
  summary.json           areas by class, as-of date

Run on the oak-mapping cluster after mrms_moisture.py.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401

CRS = "EPSG:32619"
OUTDIR = Path("scores")

FAVORABLE, MARGINAL = 0.60, 0.35  # score thresholds for the three classes

# --------------------------------------------------------------------------- #
# Species definitions -- the domain knowledge, all in one place
# --------------------------------------------------------------------------- #
# host       : "oak" (mycorrhizal/decay on Quercus) or "oak_or_conifer"
# oak_lo/hi  : oak index ramp. Cape-calibrated -- confirmed Cape oak stands
#              median 0.27 and an inland Oak/hickory stand 0.43, so these are
#              deliberately lower than an inland scale would suggest.
# season     : (start, full, end_full, end) day-of-year trapezoid
# window     : trailing days over which moisture is judged = the fruiting lag
# sm_lo/hi   : soil moisture fraction ramp over that window
# rain_lo/hi : rainfall total ramp over that window, mm
SPECIES = {
    "chanterelle": dict(
        label="Chanterelle (Cantharellus)",
        note="Mycorrhizal with oak. Fruits 1-3 weeks after a soak and persists; "
             "wants sustained moisture rather than one downpour.",
        host="oak", oak_lo=0.12, oak_hi=0.30,
        season=(166, 186, 243, 268),        # mid-Jun .. late Sep, peak Jul-Aug
        window=21, sm_lo=0.20, sm_hi=0.45, rain_lo=25, rain_hi=60,
    ),
    "bolete": dict(
        label="Boletes (edible Boletus spp.)",
        note="Mycorrhizal with both oak and pine, so pine barrens count. "
             "Flushes hard 5-14 days after rain and passes quickly.",
        host="oak_or_conifer", oak_lo=0.12, oak_hi=0.30,
        season=(161, 182, 273, 293),        # mid-Jun .. mid-Oct
        window=14, sm_lo=0.18, sm_hi=0.40, rain_lo=20, rain_hi=50,
    ),
    "chicken": dict(
        label="Chicken-of-the-woods (Laetiporus)",
        note="Decays oak wood, living or dead, so scattered big trees count "
             "and it is the least soil-moisture dependent of the four.",
        host="oak", oak_lo=0.08, oak_hi=0.25,
        season=(152, 222, 283, 309),        # Jun .. early Nov, peak late summer
        window=21, sm_lo=0.12, sm_hi=0.35, rain_lo=15, rain_hi=45,
    ),
    "hen": dict(
        label="Hen-of-the-woods (Grifola frondosa)",
        note="At the base of large, mature, often stressed oaks. Tight autumn "
             "window. NOTE: the real trigger is the first run of cool nights, "
             "which this calendar window only approximates -- no temperature "
             "layer is wired in yet.",
        host="oak", oak_lo=0.20, oak_hi=0.40,
        season=(244, 269, 298, 314),        # Sep .. mid-Nov, peak Oct
        window=21, sm_lo=0.15, sm_hi=0.38, rain_lo=20, rain_hi=50,
    ),
}


def ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear between. NaN-safe."""
    with np.errstate(invalid="ignore"):
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def season_score(doy, s):
    """Trapezoid: 0 outside (start,end), 1 across the plateau."""
    a, b, c, d = s
    if doy <= a or doy >= d:
        return 0.0
    if doy < b:
        return (doy - a) / (b - a)
    if doy <= c:
        return 1.0
    return (d - doy) / (d - c)


def coarsen3(a):
    """10 m -> 30 m block mean; the grids are exact multiples.

    A plain nanmean would call a 30 m cell canopy on the strength of a single
    valid 10 m pixel, dilating the mask along every forest edge and road. A
    majority of the block has to be canopy for the cell to count.
    """
    h, w = a.shape[0] // 3 * 3, a.shape[1] // 3 * 3
    blk = a[:h, :w].reshape(h // 3, 3, w // 3, 3)
    n = np.isfinite(blk).sum(axis=(1, 3))
    with np.errstate(invalid="ignore"):
        m = np.nanmean(blk, axis=(1, 3))
    return np.where(n >= 5, m, np.nan)


def read(path):
    return rioxarray.open_rasterio(path).squeeze(drop=True)


def write(arr, name, profile, dtype="float32", nodata=np.nan):
    OUTDIR.mkdir(exist_ok=True)
    p = dict(profile, dtype=dtype, nodata=nodata, count=1,
             compress="lzw", tiled=True, driver="GTiff")
    with rasterio.open(OUTDIR / f"{name}.tif", "w", **p) as dst:
        dst.write(arr.astype(dtype), 1)


def main():
    meta = json.loads(Path("moisture/meta.json").read_text())
    as_of = pd.Timestamp(meta["as_of"])
    doy = as_of.dayofyear
    print(f"scoring for {as_of.date()} (day {doy})\n")

    sm_ref = read("moisture/sm_now.tif")
    profile = dict(driver="GTiff", height=sm_ref.shape[0], width=sm_ref.shape[1],
                   crs=CRS, transform=sm_ref.rio.transform())

    oak10 = read("s2/oak_index.tif").values.astype("float32")
    oak = coarsen3(oak10).astype("float32")
    assert oak.shape == sm_ref.shape, f"{oak.shape} != {sm_ref.shape}"

    # Clip to the watersheds: the raster grid is a bbox two-thirds of which is
    # ocean and neighbouring towns, and rating ground outside the basins would
    # quietly inflate every area figure reported below.
    import geopandas as gpd
    from rasterio.features import geometry_mask
    parts = [gpd.read_file(f).to_crs(CRS) for f in
             ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]]
    basin_geom = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True),
                                  crs=CRS).geometry.union_all()
    in_basin = ~geometry_mask([basin_geom], out_shape=oak.shape,
                              transform=sm_ref.rio.transform(), invert=False)

    canopy = np.isfinite(oak) & in_basin
    px_km2 = 30 * 30 / 1e6
    print(f"rateable canopy: {canopy.sum() * px_km2:.0f} km2 "
          f"(basin {in_basin.sum() * px_km2:.0f} km2)")

    sm = {w: read(f"moisture/sm_mean_{w}.tif").values.astype("float32")
          for w in (7, 14, 21)}
    rain = {w: read(f"moisture/rain_{w}.tif").values.astype("float32")
            for w in (7, 14, 21, 30)}

    summary = {"as_of": str(as_of.date()), "day_of_year": int(doy),
               "rateable_canopy_km2": round(float(canopy.sum() * px_km2), 1),
               "calibrated": False, "species": {}}

    for key, sp in SPECIES.items():
        oak_term = ramp(oak, sp["oak_lo"], sp["oak_hi"])
        if sp["host"] == "oak_or_conifer":
            # The same index read the other way: inside closed canopy, a low
            # deciduous signal is conifer, and conifer hosts boletes too.
            conifer_term = 1.0 - ramp(oak, 0.04, 0.22)
            host = np.maximum(oak_term, conifer_term)
        else:
            host = oak_term

        season = season_score(doy, sp["season"])
        w = sp["window"]
        moisture = np.minimum(ramp(sm[w], sp["sm_lo"], sp["sm_hi"]),
                              ramp(rain[w], sp["rain_lo"], sp["rain_hi"]))

        terms = np.stack([host, np.full_like(host, season), moisture])
        score = np.where(canopy, terms.min(axis=0), np.nan)

        cls = np.where(~canopy, 0,
                       np.where(score >= FAVORABLE, 3,
                                np.where(score >= MARGINAL, 2, 1)))

        # Only ground that is NOT already favorable has a limiting factor worth
        # naming. Writing one everywhere would shade favorable stands in the
        # "why not" view and contradict both its legend and the percentages
        # below, which are computed over non-favorable canopy.
        limiter = np.where(canopy & (cls < 3), terms.argmin(axis=0) + 1, 0)

        write(score, f"{key}_score", profile)
        write(cls, f"{key}_class", profile, dtype="uint8", nodata=0)
        write(limiter, f"{key}_limiter", profile, dtype="uint8", nodata=0)

        n = canopy.sum()
        areas = {nm: round(float((cls == v).sum() * px_km2), 1)
                 for v, nm in [(3, "favorable"), (2, "marginal"), (1, "unfavorable")]}
        lim_counts = {nm: round(100 * float(((limiter == v) & (cls < 3)).sum() / n), 0)
                      for v, nm in [(1, "host"), (2, "season"), (3, "moisture")]}
        summary["species"][key] = {
            "label": sp["label"], "note": sp["note"],
            "season_score": round(season, 2), "window_days": w,
            "areas_km2": areas, "limiting_pct_of_canopy": lim_counts,
        }
        print(f"{sp['label']}")
        print(f"  season {season:.2f} | favorable {areas['favorable']:6.1f} km2 | "
              f"marginal {areas['marginal']:6.1f} | unfavorable {areas['unfavorable']:6.1f}")
        print(f"  limiting factor where not favorable: "
              + ", ".join(f"{k} {v:.0f}%" for k, v in lim_counts.items()))

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTDIR}/")


if __name__ == "__main__":
    main()
