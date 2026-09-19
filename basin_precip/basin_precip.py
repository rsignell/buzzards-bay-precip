"""
Daily precipitation for each Buzzards Bay v2 basin polygon, from MRMS.

For every report date D the accumulation window is (D-1) 7am -> D 7am US
Eastern local time (the CoCoRaHS convention, converted to UTC per date so the
offset follows EDT/EST; DST-change days are 23 or 25 h). MRMS is the
dynamical.org hourly 1 km radar/gauge QPE Icechunk store on us-west-2; hourly
stamp t labels the (t-1h, t] accumulation. Only windows lying wholly inside the
store's time range are written. Hours inside the range that have no data over a
polygon lower that basin's `n_hours` below `n_hours_expected`; its total is then a
lower bound (NaN if n_hours is 0). Filter on n_hours == n_hours_expected for clean days.

Polygon means are area-weighted: each polygon is intersected with the MRMS grid
cells once (MA State Plane, metres) and the weights are reused for every day.
Cell-centre sampling would return NaN for the small polygons (a 6-acre marsh is
smaller than one 1 km cell). volume_m3 = mean depth x polygon area.

The store is chunked (648 h, 100, 100): one chunk is 27 days of a 100x100-cell
tile, so days are loaded in long blocks and sliced, never one day at a time.

Output is a plain time-series parquet, one row per (basin, report date), keyed
by basin_id = BBP_SYS_ID + "_" + TYPE. Geometry stays in the v2 geojson. Re-running
a date replaces its rows; other dates are kept. With --r2-key the existing file is
pulled from Cloudflare R2, merged, and pushed back.

  python basin_precip.py --dates 2014-11-02:2014-11-08 --out out/p.parquet
  python basin_precip.py --last 3 --out out/p.parquet --r2-key some/key.parquet
  python basin_precip.py --all --out out/p.parquet --r2-key some/key.parquet

R2 credentials: R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY, else the SSM
SecureStrings the foraging Lambda uses (needs AWS creds for the OSC account).
"""

import argparse
import os
import sys
import time
from zoneinfo import ZoneInfo

import geopandas as gpd
import icechunk
import numpy as np
import pandas as pd
import shapely
import xarray as xr

POLYGONS = ("https://raw.githubusercontent.com/rsignell/buzzards-bay-precip/"
            "main/basins/bbnep_subbasins_2026_v2.geojson")
MRMS_BUCKET = "dynamical-noaa-mrms"
MRMS_PREFIX = "noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk"
AREA_CRS = "EPSG:26986"          # MA State Plane, metres -- same CRS as Joe's source
PAD = 0.05                       # deg of grid margin around the polygons
BLOCK_DAYS = 120                 # report dates loaded per block
ET = ZoneInfo("America/New_York")

R2_ENDPOINT = "https://9cbdcb4884f86a6779032ae561e474a5.r2.cloudflarestorage.com"
R2_BUCKET = "osc-pub"
SSM_PREFIX = "/buzzards-bay-foraging/r2"


def local_7am_utc(day):
    return (pd.Timestamp(f"{day:%Y-%m-%d} 07:00", tz=ET)
            .tz_convert("UTC").tz_localize(None))


def window(report_date):
    """(start, end] of the report date's accumulation window, naive UTC."""
    return local_7am_utc(report_date - pd.Timedelta(days=1)), local_7am_utc(report_date)


def stamps_for(report_date):
    start, end = window(report_date)
    return pd.date_range(start + pd.Timedelta(hours=1), end, freq="1h")


def report_date_range(t_min, t_max):
    """First and last report dates whose windows lie wholly inside [t_min, t_max]."""
    d = pd.Timestamp(t_min.tz_localize("UTC").tz_convert(ET).date())
    while stamps_for(d)[0] < t_min:
        d += pd.Timedelta(days=1)
    last = pd.Timestamp(t_max.tz_localize("UTC").tz_convert(ET).date())
    while window(last)[1] > t_max:
        last -= pd.Timedelta(days=1)
    return d, last


def parse_dates(spec):
    out = set()
    for tok in spec.split(","):
        if ":" in tok:
            a, b = tok.split(":")
            out.update(pd.date_range(a.strip(), b.strip()))
        else:
            out.add(pd.Timestamp(tok.strip()))
    return sorted(out)


def load_basins(path):
    g = gpd.read_file(path).to_crs(4326)
    g["basin_id"] = g["BBP_SYS_ID"] + "_" + g["TYPE"]
    assert g["basin_id"].is_unique, "basin_id collision"
    assert set(g["TYPE"]) <= {"LAND", "WATER"}, set(g["TYPE"])
    g = g.rename(columns={"BBP_SYS_ID": "sys_id", "TYPE": "type"})
    return g[["basin_id", "sys_id", "type", "geometry"]]


def cell_weights(basins, lat, lon):
    """Polygon x cell intersection areas (m2) on the MRMS grid, as index arrays."""
    dy, dx = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    boxes = shapely.box(lon2.ravel() - dx / 2, lat2.ravel() - dy / 2,
                        lon2.ravel() + dx / 2, lat2.ravel() + dy / 2)
    cells = np.asarray(gpd.GeoSeries(boxes, crs=4326).to_crs(AREA_CRS))
    polys = shapely.make_valid(np.asarray(basins.to_crs(AREA_CRS).geometry))
    p_idx, c_idx = shapely.STRtree(cells).query(polys, predicate="intersects")
    area = shapely.area(shapely.intersection(polys[p_idx], cells[c_idx]))
    keep = area > 0
    return p_idx[keep], c_idx[keep], area[keep], shapely.area(polys)


def polygon_means(x, w):
    """Area-weighted mean of flat array x per polygon, ignoring NaN cells."""
    p_idx, c_idx, area, poly_area = w
    n = len(poly_area)
    ok = np.isfinite(x)[c_idx]
    den = np.bincount(p_idx, weights=area * ok, minlength=n)
    num = np.bincount(p_idx, weights=area * ok * np.nan_to_num(x)[c_idx], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan), den / poly_area


def valid_hours(x, w):
    """Per polygon: number of time steps with any non-NaN cell under it. x is (time, cells)."""
    p_idx, c_idx, area, poly_area = w
    n = len(poly_area)
    ok = np.isfinite(x)
    count = np.zeros(n, dtype=int)
    for t in range(x.shape[0]):
        count += np.bincount(p_idx, weights=area * ok[t, c_idx], minlength=n) > 0
    return count


def open_mrms():
    storage = icechunk.s3_storage(bucket=MRMS_BUCKET, prefix=MRMS_PREFIX,
                                   region="us-west-2", anonymous=True)
    repo = icechunk.Repository.open(storage)
    return xr.open_zarr(repo.readonly_session("main").store, chunks=None,
                        decode_timedelta=True)


def r2_client():
    import boto3
    key_id, secret = os.environ.get("R2_ACCESS_KEY_ID"), os.environ.get("R2_SECRET_ACCESS_KEY")
    if not (key_id and secret):
        ssm = boto3.client("ssm", region_name="us-west-2")
        get = lambda n: ssm.get_parameter(Name=f"{SSM_PREFIX}/{n}", WithDecryption=True)["Parameter"]["Value"]
        key_id, secret = get("access-key-id"), get("secret-access-key")
    return boto3.client("s3", endpoint_url=R2_ENDPOINT, region_name="auto",
                        aws_access_key_id=key_id, aws_secret_access_key=secret)


def pull_existing(s3, key, path):
    from botocore.exceptions import ClientError
    try:
        s3.download_file(R2_BUCKET, key, path)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def blocks(dates, max_days=BLOCK_DAYS, max_gap_days=3):
    """Split sorted dates into runs that are close together and at most max_days long."""
    run = [dates[0]]
    for d in dates[1:]:
        if (d - run[-1]).days > max_gap_days or (d - run[0]).days >= max_days:
            yield run
            run = []
        run.append(d)
    yield run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--dates", help="report dates: comma list of YYYY-MM-DD or START:END ranges")
    sel.add_argument("--last", type=int, help="the N most recent complete report dates")
    sel.add_argument("--all", action="store_true", help="every complete report date in the store")
    ap.add_argument("--polygons", default=POLYGONS)
    ap.add_argument("--out", required=True, help="local parquet path")
    ap.add_argument("--r2-key", help="merge with / upload to this key in the osc-pub bucket")
    args = ap.parse_args()

    t0 = time.time()
    basins = load_basins(args.polygons)
    print(f"{len(basins)} polygons from {args.polygons}")

    ds = open_mrms()
    avail = pd.DatetimeIndex(ds.time.values)
    first, last = report_date_range(avail.min(), avail.max())
    print(f"MRMS {avail.min()} -> {avail.max()} UTC; complete report dates "
          f"{first:%Y-%m-%d} .. {last:%Y-%m-%d}")
    if args.all:
        dates = list(pd.date_range(first, last))
    elif args.last:
        dates = list(pd.date_range(end=last, periods=args.last))
    else:
        dates = parse_dates(args.dates)
    out_of_range = [d for d in dates if d < first or d > last]
    if out_of_range:
        print(f"  skipping {len(out_of_range)} date(s) whose window is not fully in the "
              f"store: {out_of_range[0]:%Y-%m-%d}..{out_of_range[-1]:%Y-%m-%d}", file=sys.stderr)
        dates = [d for d in dates if first <= d <= last]
    if not dates:
        sys.exit("no complete report dates to write")

    minx, miny, maxx, maxy = basins.total_bounds
    box = dict(latitude=slice(maxy + PAD, miny - PAD),     # MRMS latitude descends
               longitude=slice(minx - PAD, maxx + PAD))
    pr = ds["precipitation_surface"]
    grid = pr.isel(time=0).sel(**box)
    weights = cell_weights(basins, grid.latitude.values, grid.longitude.values)
    poly_area_m2 = weights[3]
    print(f"MRMS box {dict(grid.sizes)}; {len(weights[0])} polygon/cell intersections; "
          f"setup {time.time() - t0:.1f}s")

    frames, t_load, t_calc, n_hours_loaded = [], 0.0, 0.0, 0
    for run in blocks(dates):
        span_start = window(run[0])[0] + pd.Timedelta(hours=1)
        span_end = window(run[-1])[1]
        t1 = time.time()
        block = pr.sel(time=slice(span_start, span_end), **box).load()
        dt = time.time() - t1
        t_load += dt
        n_hours_loaded += block.sizes["time"]
        print(f"  block {run[0]:%Y-%m-%d}..{run[-1]:%Y-%m-%d}: loaded {block.sizes['time']} h "
              f"({block.nbytes / 1e6:.0f} MB) in {dt:.1f}s", flush=True)

        t2 = time.time()
        for d in run:
            stamps = stamps_for(d)
            seg = block.sel(time=stamps)
            hours = valid_hours(seg.values.reshape(len(stamps), -1), weights)
            depth_mm, cover = polygon_means(
                seg.sum("time", min_count=1).values.ravel() * 3600.0, weights)  # mm/s -> mm
            secs = len(stamps) * 3600.0
            vol = depth_mm / 1000.0 * poly_area_m2
            frames.append(pd.DataFrame({
                "report_date": d, "basin_id": basins["basin_id"].values,
                "sys_id": basins["sys_id"].values, "type": basins["type"].values,
                "precip_mm": np.round(depth_mm, 3),
                "precip_in": np.round(depth_mm / 25.4, 4),
                "volume_m3": np.round(vol, 1),
                "volume_flux_m3s": np.round(vol / secs, 6),
                "cover_frac": np.round(cover, 4),
                "n_hours": hours, "n_hours_expected": len(stamps)}))
        t_calc += time.time() - t2

    new = pd.concat(frames, ignore_index=True)
    short = (new.n_hours < new.n_hours_expected).mean()
    print(f"{new.report_date.nunique()} dates computed ({short:.0%} of basin-days have missing hours); "
          f"load {t_load:.1f}s ({n_hours_loaded} h), compute {t_calc:.1f}s")

    s3 = r2_client() if args.r2_key else None
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    old = None
    if args.r2_key and pull_existing(s3, args.r2_key, args.out + ".prev"):
        old = pd.read_parquet(args.out + ".prev")
        os.remove(args.out + ".prev")
        print(f"merging with {len(old)} existing rows from R2")
    elif not args.r2_key and os.path.exists(args.out):
        old = pd.read_parquet(args.out)
    if old is not None:
        new = pd.concat([old[~old.report_date.isin(new.report_date.unique())], new],
                        ignore_index=True)
    new = new.sort_values(["report_date", "basin_id"]).reset_index(drop=True)
    new.to_parquet(args.out, compression="zstd", index=False)
    print(f"wrote {args.out}: {len(new)} rows, {new.report_date.nunique()} dates "
          f"({new.report_date.min():%Y-%m-%d}..{new.report_date.max():%Y-%m-%d}), "
          f"{os.path.getsize(args.out) / 1e6:.2f} MB")

    if args.r2_key:
        s3.upload_file(args.out, R2_BUCKET, args.r2_key,
                       ExtraArgs={"ContentType": "application/vnd.apache.parquet",
                                  "CacheControl": "no-cache"})
        print(f"uploaded https://r2-pub.openscicomp.io/{args.r2_key}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
