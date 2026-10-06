"""
Per-species foraging conditions: favorable / marginal / unfavorable.

Combines the static habitat layers with the current moisture state into one
score per species, on the shared 30 m grid, for the five species on the
curated list: chanterelle, black trumpet, bolete, chicken-of-the-woods,
hen-of-the-woods.

Three sub-scores, each 0-1, combined by Liebig's law of the minimum rather than
averaged. A stand with perfect moisture and no oak is not "half good" for
hen-of-the-woods, it is useless, and averaging would hide that. Taking the
minimum also means every pixel has a named limiting factor, which is written out
alongside the score -- "marginal because it is too dry" and "marginal because
the oak is thin" are different messages to a forager, and only one of them is
worth waiting out.

  host      how much oak canopy there is comes from the Sentinel-2 oak index,
            Cape-calibrated. Whether it is host ground at all comes from the
            TESSERA class probabilities (tessera_hosts.py): the oak term is
            scaled by 1 - P(forested wetland or bog), because red maple swamp
            is deciduous and reads as oak on the index alone (it calls half of
            all mapped forested wetland deciduous). Boletes also take
            conifers, so they get the better of the oak term and P(evergreen).
            Where TESSERA has no value (~1% of canopy) the index is used alone
            and conifer is inferred from it read the other way, as before.
  season    a day-of-year trapezoid per species, then two temperature
            gates from the HRRR analysis (moisture/): a hard frost
            (moisture/days_since_frost.tif, <= -2 C) since 1 Sep ends the
            season for every species, and species with a `cool_nights` ramp
            (hen) also need the 7-day mean night low (moisture/tmin_mean_7.tif)
            to have come down -- the real fall trigger, which the calendar
            alone only approximates. A warm October no longer scores like a
            cold one.
  moisture  the minimum of two things: (1) is it sustained -- soil moisture
            and rainfall averaged over the species' window, so one downpour in
            an otherwise dry spell doesn't read as wet; and (2) has enough
            time passed -- days since the current wet spell BEGAN
            (moisture/days_since_wetup.tif) must clear the species' lag_days
            before any credit is given at all. Without (2), a single big storm
            can max out a wide trailing window the very next day, which
            contradicts every species' own "fruits N days after a soak" note.
            The clock is deliberately NOT days since the most recent soak:
            that resets on every rainy day, so a week of steady rain -- the
            best fruiting weather there is -- would read as "too soon" until
            it stopped. Only a soak onto soil that had dried out starts it.

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
  <species>_limiter.tif  1 host, 2 season, 3 too dry, 4 wet spell too new
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
# cool_nights: optional (none_at, full_at) deg C on the 7-day mean daily
#              minimum: 0 credit at or above none_at, full at or below full_at
# window     : trailing days over which moisture is judged = "is it sustained"
# sm_lo/hi   : soil moisture fraction ramp over that window
# rain_lo/hi : rainfall total ramp over that window, mm
# lag_days   : minimum days since the current wet spell began (moisture/
#              days_since_wetup.tif) before fruiting is credited at all. A wide
#              trailing window alone doesn't stop a single huge storm from
#              maxing out sm/rain the very next day -- lag_days is what
#              actually encodes "fruits N days after a soak", separately from
#              "is moisture sustained".
SPECIES = {
    "chanterelle": dict(
        label="Chanterelle (Cantharellus)",
        note="Mycorrhizal with oak. Fruits 1-3 weeks after a soak and persists; "
             "wants sustained moisture rather than one downpour.",
        host="oak", oak_lo=0.12, oak_hi=0.30,
        season=(166, 186, 243, 268),        # mid-Jun .. late Sep, peak Jul-Aug
        window=21, sm_lo=0.20, sm_hi=0.45, rain_lo=25, rain_hi=60, lag_days=7,
    ),
    "black_trumpet": dict(
        label="Black trumpet (Craterellus)",
        note="Mycorrhizal with oak, usually in mossy, low, damp ground nearby. "
             "Wants sustained moisture even more than chanterelles and hides "
             "well in leaf litter -- look, don't just glance.",
        host="oak", oak_lo=0.12, oak_hi=0.30,
        season=(182, 205, 262, 283),        # early Jul .. early Oct, peak Aug-Sep
        window=21, sm_lo=0.22, sm_hi=0.48, rain_lo=28, rain_hi=65, lag_days=8,
    ),
    "bolete": dict(
        label="Boletes (edible Boletus spp.)",
        note="Mycorrhizal with both oak and pine, so pine barrens count. "
             "Flushes hard 5-14 days after rain and passes quickly.",
        host="oak_or_conifer", oak_lo=0.12, oak_hi=0.30,
        season=(161, 182, 273, 293),        # mid-Jun .. mid-Oct
        window=14, sm_lo=0.18, sm_hi=0.40, rain_lo=20, rain_hi=50, lag_days=4,
    ),
    "chicken": dict(
        label="Chicken-of-the-woods (Laetiporus)",
        note="Decays oak wood, living or dead, so scattered big trees count "
             "and it is the least soil-moisture dependent of the five.",
        host="oak", oak_lo=0.08, oak_hi=0.25,
        season=(152, 222, 283, 309),        # Jun .. early Nov, peak late summer
        window=21, sm_lo=0.12, sm_hi=0.35, rain_lo=15, rain_hi=45, lag_days=3,
    ),
    "hen": dict(
        label="Hen-of-the-woods (Grifola frondosa)",
        note="At the base of large, mature, often stressed oaks. Tight autumn "
             "window, triggered by the first run of cool nights (nights in the "
             "50s F) -- the season term below waits for them.",
        host="oak", oak_lo=0.20, oak_hi=0.40,
        season=(244, 269, 298, 314),        # Sep .. mid-Nov, peak Oct
        cool_nights=(18.0, 13.0),           # 64 F -> 55 F mean night low
        window=21, sm_lo=0.15, sm_hi=0.38, rain_lo=20, rain_hi=50, lag_days=7,
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


def season_phase(doy, s):
    """Plain-language read of where today sits in the same trapezoid."""
    a, b, c, d = s
    if doy <= a or doy >= d:
        return "out of season"
    if doy < b:
        return "season just starting"
    if doy <= c:
        return "in season"
    return "season winding down"


FALL_START = 244     # 1 Sep: a frost after this ends the season
FROST_ENDS_SEASON = "season ended by frost"


def phase_now(doy, sp, frost_free_frac, tmin7_median):
    """season_phase, overridden by the temperature gates where they bind."""
    phase = season_phase(doy, sp["season"])
    if phase == "out of season":
        return phase
    if frost_free_frac < 0.5:
        return FROST_ENDS_SEASON
    if "cool_nights" in sp and tmin7_median > sp["cool_nights"][1]:
        return (f"waiting for cooler nights (7-day mean low {tmin7_median:.0f} C, "
                f"full credit at {sp['cool_nights'][1]:.0f} C)")
    return phase


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
    del oak10

    # TESSERA class probabilities, percent, same 10 m grid as the oak index.
    # One band at a time: four float32 copies of the full 10 m grid would be
    # 1.2 GB, which the Lambda can't spare alongside everything else.
    with rasterio.open("tessera/host_proba.tif") as src:
        def band(i):
            b = src.read(i).astype("float32")
            b[b == 255] = np.nan
            return (coarsen3(b) / 100.0).astype("float32")
        p_evergreen = band(2)
        p_nonhost = np.minimum(band(3) + band(4), 1.0)   # forested wetland + bog
    has_tessera = np.isfinite(p_nonhost)

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
    # NaN means no wet spell began anywhere in the record -- treat that as
    # "long past any lag" so the sm/rain ramps (which will themselves be low)
    # are what rules it out, not a stale lag gate.
    tmin7 = read("moisture/tmin_mean_7.tif").values.astype("float32")
    days_since_frost = read("moisture/days_since_frost.tif").values.astype("float32")
    # A hard frost since 1 Sep ends the season. NaN = no frost in the record.
    frost_ok = ~(np.nan_to_num(days_since_frost, nan=9999.0) <= doy - FALL_START)
    days_since_wetup = np.nan_to_num(
        read("moisture/days_since_wetup.tif").values.astype("float32"), nan=9999.0)

    summary = {"as_of": str(as_of.date()), "day_of_year": int(doy),
               "rateable_canopy_km2": round(float(canopy.sum() * px_km2), 1),
               "calibrated": False, "species": {}}

    for key, sp in SPECIES.items():
        # The index says how much oak; TESSERA says whether it is swamp or bog
        # instead. Scaling (not a min against P(deciduous)) leaves dry mixed
        # oak/pine stands nearly untouched -- they read only ~40 % deciduous
        # but are real host ground.
        oak_term = ramp(oak, sp["oak_lo"], sp["oak_hi"]) * np.where(
            has_tessera, 1.0 - p_nonhost, 1.0)
        if sp["host"] == "oak_or_conifer":
            # Conifer directly from TESSERA. The fallback reads the index the
            # other way (low deciduous = conifer), which also counts bogs and
            # evergreen-shrub swamps as pine.
            conifer_term = np.where(has_tessera, ramp(p_evergreen, 0.3, 0.7),
                                    1.0 - ramp(oak, 0.04, 0.22))
            host = np.maximum(oak_term, conifer_term)
        else:
            host = oak_term

        season = season_score(doy, sp["season"])
        season_t = np.where(frost_ok, season, 0.0).astype("float32")
        if "cool_nights" in sp:
            warm, cool = sp["cool_nights"]
            # Lower nights are better, so the ramp runs backwards.
            cool_term = ramp(warm - tmin7, 0.0, warm - cool)
            season_t = np.minimum(season_t, np.nan_to_num(cool_term, nan=1.0))
        w = sp["window"]
        # Sustained (is the window wet enough) and lag (has enough time passed
        # since the soak) are kept as separate terms, not folded into one
        # "moisture" number, so the limiter below can tell a forager "too dry"
        # from "too soon" -- different, actionable messages.
        sustained = np.minimum(ramp(sm[w], sp["sm_lo"], sp["sm_hi"]),
                               ramp(rain[w], sp["rain_lo"], sp["rain_hi"]))
        lag = ramp(days_since_wetup, 0.0, sp["lag_days"])

        terms = np.stack([host, season_t, sustained, lag])
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
                      for v, nm in [(1, "host"), (2, "season"), (3, "moisture"), (4, "lag")]}
        summary["species"][key] = {
            "label": sp["label"], "note": sp["note"],
            "season_score": round(season, 2),
            "season_phase": phase_now(doy, sp, frost_ok[canopy].mean(),
                                      np.nanmedian(tmin7[canopy])),
            "window_days": w, "lag_days": sp["lag_days"],
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
