"""
Soil-moisture state for the Buzzards Bay + Cape Cod watersheds, from MRMS.

This is the dynamic half of the foraging model. The habitat layers say where the
oaks are; this says whether the ground under them is currently in a state that
can fruit a mushroom.

Rainfall alone will not do it. Half this basin is excessively drained outwash
sand where a soak is gone in days, and the kettle swales beside it hold water
for weeks -- two stands under the same 1 km MRMS cell can be in completely
different condition. So rain is run through a single-layer soil bucket whose
capacity is the SSURGO available water storage of the top 25 cm, which is where
mycelium and leaf litter actually live and which wets and dries far faster than
the full 1 m root zone.

    S(t) = clip( S(t-1) + throughfall - AET , 0 , capacity )
    AET  = PET(t) * S/capacity                    (moisture-limited)
    PET  = Hargreaves, from that day's HRRR 2 m air temperature max/min, so a
           heat wave dries litter faster than an average July day and a cool,
           cloudy spell holds it (falls back to a day-of-year climatology on
           any day the temperature analysis is missing)
    sm   = S / capacity                           (0 = wilting, 1 = field cap)

Grid is the 30 m SSURGO grid, because capacity varies at that scale while the
MRMS forcing is smooth at 1 km. Windowed rain totals are computed on the native
1 km grid and upsampled at the end, since they carry no sub-kilometre structure.

EVERY PARAMETER BELOW IS UNCALIBRATED. There is no fruiting record to fit them
to yet, so they are physically reasonable starting values, isolated in one block
so they can be tuned once a season of observations exists.

Outputs (under moisture/):
  sm_now.tif         soil moisture fraction, most recent complete day
  sm_mean_7/14/21    mean soil moisture over trailing windows
  rain_7/14/21/30    rainfall totals over trailing windows (mm)
  days_since_soak    days since a day delivering >= SOAK_MM
  tmin_mean_7        mean daily minimum air temperature over the last 7 days
                     (deg C, HRRR 3 km) -- the fall "cool nights" signal
  days_since_frost   days since the last day with a minimum <= FROST_C
                     (deg C); NaN if none in the record
  days_since_wetup   days since the current wet spell began: the first soak
                     after the soil had dried below DRY_SM. Further soaks while
                     the ground stays wet do NOT restart it -- this is the
                     fruiting clock, and days of rain keep it running.
  meta.json          as-of date, window, provenance

Run on the oak-mapping cluster in the `mrms` env (needs Python >= 3.11):
  /home/ubuntu/miniconda3/envs/mrms/bin/python mrms_moisture.py
"""

import json
from collections import deque
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from pyproj import CRS as CRS_, Transformer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MRMS_BUCKET = "dynamical-noaa-mrms"
MRMS_PREFIX = "noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk"
VAR = "precipitation_surface"  # radar/gauge MultiSensor QPE pass 2, kg m-2 s-1

# HRRR analysis (dynamical.org): hourly 2 m air temperature, 3 km, 2014-on.
HRRR_BUCKET = "dynamical-noaa-hrrr"
HRRR_PREFIX = "noaa-hrrr-analysis/v0.2.0.icechunk"
HRRR_VAR = "temperature_2m"    # deg C, instantaneous at each hourly stamp
HRRR_PAD_M = 6000

CRS = "EPSG:32619"
BBOX_UTM = (319920, 4591950, 425820, 4662630)  # the shared 30 m habitat grid
RES = 30
PAD_DEG = 0.05

SOIL_AWC25 = "soil/soil_awc25.tif"   # cm of available water, 0-25 cm
SOIL_AWC100 = "soil/soil_awc.tif"    # fallback if the shallow layer is missing
OUTDIR = Path("moisture")

# Days pulled. The first SPINUP_DAYS are burned so the bucket forgets its
# initial guess; the remainder is what the windows are computed over.
HISTORY_DAYS = 150
SPINUP_DAYS = 90

# --- uncalibrated parameters ---------------------------------------------- #
# Fraction of gross rainfall reaching the forest floor. Closed hardwood canopy
# intercepts roughly 10-20% of growing-season rainfall.
THROUGHFALL = 0.85

# Potential ET as a seasonal sinusoid for coastal SE Massachusetts, mm/day:
# ~4.4 mid-July, ~0.3 mid-January. Avoids a temperature dependency entirely,
# at the cost of missing individual hot or cool spells.
PET_MEAN, PET_AMP, PET_PEAK_DOY = 2.35, 2.05, 196.0

# A "soak": the daily total that actually recharges litter and triggers
# fruiting. 12.7 mm is half an inch, the number foragers use.
SOAK_MM = 12.7

# Soil moisture fraction below which the ground counts as having dried out.
# A soak only starts a new wet spell (restarts the fruiting clock) if the soil
# had dropped below this first; soaks onto already-wet ground extend the spell.
# On excessively drained sand this takes ~a week of dry weather in late summer.
DRY_SM = 0.30

# A hard ("killing") frost, deg C daily minimum. Ends the fruiting season in
# score_species.py.
FROST_C = -2.0

# Latitude for extraterrestrial radiation in Hargreaves PET. Ra varies by
# ~1 % across this domain, so one value serves.
PET_LAT = 41.7

# Floor on bucket capacity (mm). Guards against map units with implausibly
# small available water reporting instant saturation and instant drought.
MIN_CAPACITY_MM = 8.0


def pet_mm(doy):
    """Climatological potential ET for a day of year, mm/day."""
    return PET_MEAN + PET_AMP * np.cos(2 * np.pi * (doy - PET_PEAK_DOY) / 365.25)


def hargreaves_mm(tmax, tmin, doy):
    """Hargreaves (1985) reference ET, mm/day, from daily max/min air temp (C).

    Checked against the climatology above on 2026 HRRR: within ~10 % May-Aug
    with no rescaling, and lower in a cool, cloudy Sep-Oct, as it should be.
    """
    lat = np.radians(PET_LAT)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(lat) * np.tan(dec))
    ra = (24 * 60 / np.pi * 0.0820 * dr                     # MJ m-2 day-1
          * (ws * np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.sin(ws)))
    tmean = (tmax + tmin) / 2
    return 0.0023 * 0.408 * ra * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))


def target_grid():
    """Coordinates of the 30 m grid, plus its lon/lat for MRMS lookup."""
    x0, y0, x1, y1 = BBOX_UTM
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    xs = x0 + (np.arange(nx) + 0.5) * RES
    ys = y1 - (np.arange(ny) + 0.5) * RES
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(xx, yy)
    transform = rasterio.transform.from_origin(x0, y1, RES, RES)
    return (ny, nx), transform, lon, lat


def load_precip(lon, lat):
    """Daily MRMS rainfall (mm) over the domain, days ending 12 UTC."""
    storage = icechunk.s3_storage(bucket=MRMS_BUCKET, prefix=MRMS_PREFIX,
                                  region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(storage)
    ds = xr.open_zarr(repo.readonly_session("main").store, chunks=None,
                      decode_timedelta=True)

    avail = pd.DatetimeIndex(ds.time.values)
    end = avail.max().floor("h")
    start = end - pd.Timedelta(days=HISTORY_DAYS)
    print(f"  MRMS available through {end} UTC")

    sub = ds[VAR].sel(
        time=slice(start, end),
        latitude=slice(lat.max() + PAD_DEG, lat.min() - PAD_DEG),  # descending
        longitude=slice(lon.min() - PAD_DEG, lon.max() + PAD_DEG),
    )
    print(f"  subset {dict(sub.sizes)}; loading ...", flush=True)
    sub = sub.load()

    # The hourly stamp t labels the (t-1h, t] accumulation, so a day ending at
    # 12 UTC (08 EDT) is every stamp in (D-1 12:00, D 12:00] -- which is exactly
    # what ceil gives after shifting back 12 h.
    tt = pd.DatetimeIndex(sub.time.values)
    day = (tt - pd.Timedelta(hours=12)).ceil("D")
    sub = sub.assign_coords(day=("time", day))

    mm_per_step = sub * 3600.0  # mm/s over a 1 h step
    daily = mm_per_step.groupby("day").sum(skipna=True)
    counts = mm_per_step.groupby("day").count()

    # Drop leading/trailing partial days -- a day missing hours understates rain.
    full = (counts.max(("latitude", "longitude")) >= 24).values
    daily = daily.isel(day=full)
    print(f"  {daily.sizes['day']} complete days: "
          f"{pd.Timestamp(daily.day.values[0]).date()} -> "
          f"{pd.Timestamp(daily.day.values[-1]).date()}")
    return daily


def load_temperature(days, lon, lat):
    """Daily max/min 2 m air temperature (C) on the HRRR grid, for `days`.

    Days follow the MRMS convention (ending 12 UTC), so each day's max is the
    previous afternoon and its min the night just ended. A day short of 20
    analysed hours is left NaN, and the bucket falls back to climatology.
    Returns (tmax, tmin) as (day, y, x) arrays plus the 30 m -> HRRR index.
    """
    storage = icechunk.s3_storage(bucket=HRRR_BUCKET, prefix=HRRR_PREFIX,
                                  region="us-west-2", anonymous=True)
    ds = xr.open_zarr(icechunk.Repository.open(storage).readonly_session("main").store,
                      chunks=None, decode_timedelta=True)
    to_hrrr = Transformer.from_crs("EPSG:4326", CRS_.from_wkt(ds.spatial_ref.attrs["crs_wkt"]),
                                   always_xy=True)
    hx, hy = to_hrrr.transform(lon, lat)
    start = days[0] - pd.Timedelta(hours=12)
    sub = ds[HRRR_VAR].sel(
        time=slice(start, days[-1] + pd.Timedelta(hours=12)),
        x=slice(hx.min() - HRRR_PAD_M, hx.max() + HRRR_PAD_M),
        y=slice(hy.max() + HRRR_PAD_M, hy.min() - HRRR_PAD_M),      # y descends
    )
    print(f"  HRRR subset {dict(sub.sizes)}; loading ...", flush=True)
    sub = sub.load()
    tt = pd.DatetimeIndex(sub.time.values)
    sub = sub.assign_coords(day=("time", (tt - pd.Timedelta(hours=12)).ceil("D")))
    g = sub.groupby("day")
    full = g.count().min(("y", "x")) >= 20
    tmax = g.max().where(full).reindex(day=days)
    tmin = g.min().where(full).reindex(day=days)
    print(f"  temperature for {int(np.isfinite(tmin.values[:, 0, 0]).sum())} of {len(days)} days")
    xs, ys = sub.x.values, sub.y.values
    j = np.clip(np.round((hx - xs[0]) / (xs[1] - xs[0])).astype("int32"), 0, len(xs) - 1)
    i = np.clip(np.round((hy - ys[0]) / (ys[1] - ys[0])).astype("int32"), 0, len(ys) - 1)
    return tmax.values.astype("float32"), tmin.values.astype("float32"), i, j


def nearest_index(daily, lon, lat):
    """Map every 30 m cell to its MRMS cell (nearest neighbour)."""
    mlat = daily.latitude.values
    mlon = daily.longitude.values
    # latitude descends, so search the reversed axis and flip back
    i = len(mlat) - 1 - np.searchsorted(mlat[::-1], lat)
    i = np.clip(i, 0, len(mlat) - 2)
    i += (np.abs(mlat[i] - lat) > np.abs(mlat[i + 1] - lat))
    j = np.clip(np.searchsorted(mlon, lon) - 1, 0, len(mlon) - 2)
    j += (np.abs(mlon[j] - lon) > np.abs(mlon[j + 1] - lon))
    return i.astype("int32"), j.astype("int32")


def capacity_mm(shape):
    """Bucket capacity from SSURGO available water in the top 25 cm."""
    path = SOIL_AWC25 if Path(SOIL_AWC25).exists() else SOIL_AWC100
    scale = 10.0 if path == SOIL_AWC25 else 10.0 * 0.30
    if path == SOIL_AWC100:
        print("  NOTE: 0-25 cm capacity missing; approximating from 0-100 cm")
    awc = rioxarray.open_rasterio(path).squeeze(drop=True)
    assert awc.shape == shape, f"soil grid {awc.shape} != target {shape}"
    cap = awc.values.astype("float32") * scale
    valid = np.isfinite(cap)
    cap = np.where(valid, np.maximum(cap, MIN_CAPACITY_MM), np.nan)
    print(f"  capacity: {100 * valid.mean():.0f}% coverage, "
          f"median {np.nanmedian(cap):.0f} mm, "
          f"p10 {np.nanpercentile(cap, 10):.0f}, p90 {np.nanpercentile(cap, 90):.0f}")
    return cap.astype("float32")


def write(arr, name, transform, nodata=np.nan, dtype="float32"):
    OUTDIR.mkdir(exist_ok=True)
    with rasterio.open(OUTDIR / f"{name}.tif", "w", driver="GTiff",
                       height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=dtype, crs=CRS, transform=transform,
                       nodata=nodata, compress="lzw", tiled=True) as dst:
        dst.write(arr.astype(dtype), 1)


def main():
    shape, transform, lon, lat = target_grid()
    print(f"target grid {shape[1]} x {shape[0]} @ {RES} m")

    print("opening MRMS ...")
    daily = load_precip(lon, lat)
    cap = capacity_mm(shape)

    i, j = nearest_index(daily, lon, lat)
    days = pd.DatetimeIndex(daily.day.values)
    P = daily.values.astype("float32")          # (day, lat, lon), mm
    P = np.nan_to_num(P)

    print("opening HRRR temperature ...")
    tmax, tmin, ti, tj = load_temperature(days, lon, lat)

    # --- run the bucket ---------------------------------------------------- #
    print(f"running the bucket over {len(days)} days "
          f"({SPINUP_DAYS} of spin-up) ...", flush=True)
    S = 0.5 * cap                               # initial guess, burned off
    keep = deque(maxlen=21)                     # trailing soil-moisture states
    last_soak = np.full(shape, np.nan, dtype="float32")
    # The wet-spell clock runs over the whole record, spin-up included, so a
    # spell that began during spin-up is still dated correctly. `armed` means
    # the soil has dried out since the last spell began, so the next soak
    # starts a new one.
    wetup = np.full(shape, np.nan, dtype="float32")
    armed = np.ones(shape, dtype=bool)

    for n, d in enumerate(days):
        armed |= (S / cap) < DRY_SM             # state at the start of the day
        soaked = P[n][i, j] >= SOAK_MM
        start = soaked & armed
        wetup = np.where(start, 0.0, wetup + 1.0)
        armed &= ~start
        rain = P[n][i, j] * THROUGHFALL
        S = np.minimum(S + rain, cap)
        if np.isfinite(tmin[n]).all():
            pet = hargreaves_mm(tmax[n], tmin[n], d.dayofyear)[ti, tj]
        else:
            pet = pet_mm(d.dayofyear)               # temperature analysis missing
        S = np.maximum(S - pet * (S / cap), 0.0)
        sm = (S / cap).astype("float32")
        if n >= SPINUP_DAYS:
            keep.append(sm)
            last_soak = np.where(soaked, 0.0, last_soak + 1.0)

    # Accumulate the trailing means in place rather than np.stack-ing the
    # deque: stacking 21 arrays of 8.3 M float32 copies ~700 MB on top of the
    # deque already holding it, which is the difference between this fitting
    # comfortably on a 6 GB laptop and not.
    arrs = list(keep)

    def trailing_mean(w):
        acc = np.zeros(shape, dtype="float32")
        cnt = np.zeros(shape, dtype="int16")
        for a in arrs[-w:]:
            m = np.isfinite(a)
            acc[m] += a[m]
            cnt[m] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)

    now = arrs[-1]
    print(f"  soil moisture now: median {np.nanmedian(now):.2f}, "
          f"p10 {np.nanpercentile(now, 10):.2f}, "
          f"p90 {np.nanpercentile(now, 90):.2f}")

    write(now, "sm_now", transform)
    for w in (7, 14, 21):
        write(trailing_mean(w), f"sm_mean_{w}", transform)
    write(last_soak, "days_since_soak", transform)
    write(wetup, "days_since_wetup", transform)

    # --- temperature signals, on the HRRR grid then out to 30 m ----------- #
    with np.errstate(invalid="ignore"):
        tmin7 = np.nanmean(tmin[-7:], axis=0)
    frosted = tmin <= FROST_C
    last = np.where(frosted.any(axis=0),
                    len(days) - 1 - np.argmax(frosted[::-1], axis=0), -1)
    since_frost = np.where(last >= 0, len(days) - 1 - last, np.nan).astype("float32")
    write(tmin7[ti, tj], "tmin_mean_7", transform)
    write(since_frost[ti, tj], "days_since_frost", transform)
    print(f"  7-day mean night low: median {np.nanmedian(tmin7):.1f} C "
          f"(range {np.nanmin(tmin7):.1f} to {np.nanmax(tmin7):.1f}); "
          f"hard frost (<= {FROST_C} C) in record: {int(frosted.any(axis=0).sum())} "
          f"of {frosted[0].size} HRRR cells")
    # Only where there is soil: without a capacity the bucket never dries, so
    # those cells carry a meaningless record-long clock and swamp the median.
    soil = np.isfinite(cap)
    print(f"  days since wet spell began: median {np.median(wetup[soil]):.0f}, "
          f"days since last soak: median {np.nanmedian(last_soak[soil]):.0f}")

    # --- rain windows: 1 km, then upsampled -------------------------------- #
    for w in (7, 14, 21, 30):
        acc = P[-w:].sum(axis=0)
        write(acc[i, j], f"rain_{w}", transform)
        print(f"  rain_{w:<2d} basin median {np.median(acc):6.1f} mm  "
              f"max {acc.max():6.1f} mm")

    meta = {
        "as_of": str(days[-1].date()),
        "day_ends_utc": "12:00",
        "first_day": str(days[SPINUP_DAYS].date()),
        "n_days": int(len(days)),
        "spinup_days": SPINUP_DAYS,
        "source": f"MRMS {VAR} (radar/gauge QPE), dynamical.org icechunk",
        "temperature": f"HRRR analysis {HRRR_VAR}, dynamical.org icechunk",
        "pet": "Hargreaves from daily HRRR max/min; climatology on missing days",
        "capacity": "SSURGO available water storage 0-25 cm",
        "parameters": {"throughfall": THROUGHFALL, "pet_mean": PET_MEAN,
                       "pet_amp": PET_AMP, "soak_mm": SOAK_MM,
                       "dry_sm": DRY_SM, "frost_c": FROST_C,
                       "min_capacity_mm": MIN_CAPACITY_MM},
        "calibrated": False,
    }
    (OUTDIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nas of {meta['as_of']}; wrote {OUTDIR}/")


if __name__ == "__main__":
    main()
