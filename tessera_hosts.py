"""
Fungal-host probability layers from TESSERA embeddings.

The S2 oak index (s2_deciduous.py) is one number -- summer minus leaf-off
NDVI -- and on this landscape its known failure is red maple swamp: deciduous,
so it reads as oak, and wet, so the moisture model then flags it too. Pine is
only ever inferred as "canopy with a low deciduous signal".

TESSERA (Cambridge; on AWS Open Data / Source Cooperative) compresses a full
year of Sentinel-1 radar + Sentinel-2 optical behaviour into 128 numbers per
10 m pixel. A plain logistic regression on those, trained against MassGIS 2016
land cover, separates the classes that matter here far better than the index:
in a two-tile pilot around Pocasset (spatially blocked CV), balanced accuracy
0.78 vs 0.44, and forested wetland recall 0.81 vs 0.19. Basin-wide (this
script) it is 0.89 vs 0.49 on the same samples and blocks.

Ground check (2026-10-01): at one owner-confirmed oak stand the classifier
split ~50/50 deciduous/wetland and placed a narrow wet strip along its edge
(swale_map.py); the owner confirmed the wet strip is there. The oak index
gave no sign of it.

Classes (MassGIS 2016 COVERCODE):
   9 Deciduous Forest            -> on Cape/Buzzards Bay uplands, ~oak
  10 Evergreen Forest            -> pitch / white pine
  13 Palustrine Forested Wetland -> mostly red maple swamp
   6 Cultivated                  -> here mostly cranberry bog

Labels are large (>~1 ha) polygons eroded 20 m, so training pixels sit well
inside stands, not on edges. Caveats: the labels are 2016 and the embeddings
2025 (development and the 2016-18 spongy moth oak mortality fall between),
and "deciduous" is not literally "oak". The classifier is only meaningful
inside closed canopy; score_species.py already masks to that.

Outputs (under tessera/):
  host_proba.tif       4 bands, percent probability on the shared 10 m grid
                       (same grid as s2/oak_index.tif): 1 deciduous,
                       2 evergreen, 3 forested wetland, 4 bog. nodata 255.
  host_cv.json         spatially blocked CV scores + confusion matrix
  lclu2016_labels.gpkg label polygons (re-fetchable cache)
  cache/               downloaded TESSERA tiles (re-fetchable cache)

Runs locally: tiles are processed one at a time (~0.5 GB each in memory).
"""

import json
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from geotessera import GeoTessera
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from shapely.ops import unary_union
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("tessera")
GRID = "s2/oak_index.tif"   # the shared 10 m grid everything is written on
YEAR = 2025
WATERSHEDS = ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]

CLASSES = {9: "deciduous", 10: "evergreen", 13: "forested_wetland", 6: "bog"}
LCLU = ("https://arcgisserver.digital.mass.gov/arcgisserver/rest/services/"
        "AGOL/LandCoverLandUse2016/FeatureServer/0/query")
MIN_AREA = 18000        # Shape__Area is web-mercator m2: ~1 ha on the ground here
ERODE_M = 20
PER_TILE_CAP = 3000     # training pixels per class per tile
PER_CLASS_CAP = 30000   # overall, after pooling
BLOCK_M = 5000          # spatial CV block size

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def esri_polygon(rings):
    """Esri rings -> shapely: outer rings are clockwise, holes counter-clockwise."""
    polys = [Polygon(r) for r in rings if len(r) >= 4]
    # Server-side simplification (maxAllowableOffset) can leave self-touching
    # rings, which GEOS then refuses to difference; repair each ring first.
    outer = [make_valid(p) for p in polys if not p.exterior.is_ccw]
    holes = [make_valid(p) for p in polys if p.exterior.is_ccw]
    g = unary_union(outer)
    return g.difference(unary_union(holes)) if holes else g


def fetch_labels(bounds):
    path = OUT / "lclu2016_labels.gpkg"
    if path.exists():
        return gpd.read_file(path)
    raw = OUT / "cache" / "lclu2016_raw.json"   # the slow part, kept separately
    if raw.exists():
        feats = json.loads(raw.read_text())
        return _to_gdf(feats, path)
    codes = ",".join(str(c) for c in CLASSES)
    query = dict(where=f"COVERCODE IN ({codes}) AND Shape__Area > {MIN_AREA}",
                 geometry=",".join(map(str, bounds)), geometryType="esriGeometryEnvelope",
                 inSR=4326, spatialRel="esriSpatialRelIntersects", f="json")
    total = requests.get(LCLU, timeout=300, params=dict(query, returnCountOnly="true")).json()["count"]

    def page(off):
        # The server 500s on large pages and takes ~30 s per query whatever
        # the size, so: small pages, simplified outlines, retries, and several
        # pages in flight at once.
        for attempt in range(5):
            try:
                r = requests.get(LCLU, timeout=300, params=dict(
                    query, outFields="COVERCODE", outSR=32619, orderByFields="OBJECTID",
                    maxAllowableOffset=5, geometryPrecision=0,
                    resultOffset=off, resultRecordCount=200)).json()
                return r["features"]
            except (requests.RequestException, KeyError, ValueError):
                time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"label fetch failed at offset {off}")

    feats = []
    with ThreadPoolExecutor(6) as ex:
        for n, p in enumerate(ex.map(page, range(0, total, 200)), 1):
            feats += p
            print(f"  labels: {len(feats)}/{total}", end="\r", flush=True)
    print()
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(feats))
    return _to_gdf(feats, path)


def _to_gdf(feats, path):
    g = gpd.GeoDataFrame(
        [{"COVERCODE": f["attributes"]["COVERCODE"],
          "geometry": esri_polygon(f["geometry"]["rings"])} for f in feats], crs=32619)
    g["geometry"] = g.buffer(0)          # polygonal part only (make_valid can add lines)
    g = g[~g.is_empty & ~g.geometry.isna()]
    g.to_file(path)
    return g


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
def basin_tiles(gt, basin_ll):
    tiles = gt.registry.load_blocks_for_region(bounds=tuple(basin_ll.bounds), year=YEAR)
    return [t for t in tiles
            if box(t[1] - 0.05, t[2] - 0.05, t[1] + 0.05, t[2] + 0.05).intersects(basin_ll)]


def valid_mask(emb):
    """TESSERA marks no-data as NaN or as an all-zero vector."""
    return np.isfinite(emb[..., 0]) & (np.abs(emb).sum(axis=-1) > 0)


def sample_tile(emb, transform, labels):
    """Random labelled pixels from one tile, in the tile's own native grid."""
    h, w, _ = emb.shape
    tb = rasterio.transform.array_bounds(h, w, transform)
    lab = labels.cx[tb[0]:tb[2], tb[1]:tb[3]]
    if lab.empty:
        return None
    y_img = rasterize(zip(lab.geometry, lab.COVERCODE), out_shape=(h, w),
                      transform=transform, fill=0, dtype="int16")
    y_img[~valid_mask(emb)] = 0
    out = []
    for c in CLASSES:
        rr, cc = np.nonzero(y_img == c)
        if rr.size == 0:
            continue
        k = rng.permutation(rr.size)[:PER_TILE_CAP]
        xs, ys = rasterio.transform.xy(transform, rr[k], cc[k])
        out.append(pd.DataFrame({"y": c, "x_m": xs, "y_m": ys})
                   .join(pd.DataFrame(emb[rr[k], cc[k]], columns=[f"e{i}" for i in range(128)])))
    return pd.concat(out, ignore_index=True) if out else None


def main():
    OUT.mkdir(exist_ok=True)
    parts = [gpd.read_file(f).to_crs(4326) for f in WATERSHEDS]
    basin_ll = pd.concat(parts).union_all()

    print("labels ...")
    labels = fetch_labels(tuple(np.round(basin_ll.bounds, 3)))
    labels = labels.assign(geometry=labels.buffer(-ERODE_M))
    labels = labels[~labels.is_empty]
    print("  polygons per class:",
          {CLASSES[c]: int(n) for c, n in labels.COVERCODE.value_counts().items()})

    gt = GeoTessera(embeddings_dir=OUT / "cache")
    tiles = basin_tiles(gt, basin_ll)
    print(f"{len(tiles)} TESSERA tiles ({YEAR}) intersect the watersheds")

    # --- pass 1: training samples ------------------------------------------ #
    samples_path = OUT / "cache" / "samples.parquet"
    if samples_path.exists():
        df = pd.read_parquet(samples_path)
    else:
        dfs = []
        for n, (_, lon, lat, emb, crs, tf) in enumerate(gt.fetch_embeddings(tiles), 1):
            assert crs.to_epsg() == 32619, crs
            s = sample_tile(emb, tf, labels)
            if s is not None:
                dfs.append(s)
            print(f"  sampled {n}/{len(tiles)} ({lon:.2f},{lat:.2f})", flush=True)
        df = pd.concat(dfs, ignore_index=True)
        # Tiles overlap slightly, so the same ground can be drawn twice; the
        # caps keep any one tile or class from dominating.
        keep = [g.sample(min(len(g), PER_CLASS_CAP), random_state=0).index
                for _, g in df.groupby("y")]
        df = df.loc[np.concatenate(keep)].reset_index(drop=True)
        df.to_parquet(samples_path)
    X = df[[f"e{i}" for i in range(128)]].to_numpy("float32")
    y = df["y"].to_numpy()
    groups = (df.x_m // BLOCK_M).astype(int) * 100000 + (df.y_m // BLOCK_M).astype(int)
    print("training pixels:", {CLASSES[c]: int((y == c).sum()) for c in CLASSES},
          f"| {groups.nunique()} {BLOCK_M // 1000} km blocks")

    # --- spatially blocked CV ---------------------------------------------- #
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1))
    pred = cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5), n_jobs=5)
    order = list(CLASSES)
    cm = confusion_matrix(y, pred, labels=order, normalize="true")
    bal = balanced_accuracy_score(y, pred)
    print(f"\nblocked CV balanced accuracy {bal:.2f}")
    print(pd.DataFrame(cm.round(2), index=[CLASSES[c] for c in order],
                       columns=[CLASSES[c] for c in order]).to_string())
    (OUT / "host_cv.json").write_text(json.dumps({
        "year": YEAR, "block_m": BLOCK_M, "balanced_accuracy": round(float(bal), 3),
        "n_train": {CLASSES[c]: int((y == c).sum()) for c in CLASSES},
        "confusion_row_normalized": {CLASSES[r]: {CLASSES[c]: round(float(v), 3)
                                     for c, v in zip(order, row)} for r, row in zip(order, cm)},
    }, indent=2))

    model.fit(X, y)
    col = [list(model.classes_).index(c) for c in CLASSES]   # band order = CLASSES

    # --- pass 2: predict every tile onto the shared grid -------------------- #
    with rasterio.open(GRID) as g:
        prof, shape, gtf = g.profile, g.shape, g.transform
    proba = np.full((len(CLASSES),) + shape, 255, dtype="uint8")
    for n, (_, lon, lat, emb, crs, tf) in enumerate(gt.fetch_embeddings(tiles), 1):
        h, w, _ = emb.shape
        ok = valid_mask(emb)
        p = np.full((len(CLASSES), h, w), np.nan, dtype="float32")
        p[:, ok] = model.predict_proba(emb[ok])[:, col].T
        dst = np.full((len(CLASSES),) + shape, np.nan, dtype="float32")
        # Only the part of the shared grid this tile covers.
        tb = rasterio.transform.array_bounds(h, w, tf)
        win = rasterio.windows.from_bounds(*tb, transform=gtf).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, shape[1], shape[0]))
        r0, c0, hh, ww = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
        sub = np.full((len(CLASSES), hh, ww), np.nan, dtype="float32")
        reproject(p, sub, src_transform=tf, src_crs=crs,
                  dst_transform=rasterio.windows.transform(win, gtf), dst_crs=prof["crs"],
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        del dst
        tgt = proba[:, r0:r0 + hh, c0:c0 + ww]
        fill = (tgt[0] == 255) & np.isfinite(sub[0])   # first tile wins in overlaps
        tgt[:, fill] = np.round(sub[:, fill] * 100).astype("uint8")
        print(f"  predicted {n}/{len(tiles)} ({lon:.2f},{lat:.2f})", flush=True)

    out = dict(prof, count=len(CLASSES), dtype="uint8", nodata=255,
               compress="lzw", tiled=True, blockxsize=512, blockysize=512)
    with rasterio.open(OUT / "host_proba.tif", "w", **out) as dst:
        dst.write(proba)
        for i, c in enumerate(CLASSES, 1):
            dst.set_band_description(i, f"P({CLASSES[c]}) %")
    print(f"\nwrote {OUT}/host_proba.tif")


if __name__ == "__main__":
    main()
