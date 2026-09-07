"""
Fungal-host forest layers for the Buzzards Bay / Cape Cod watersheds.

The four target species (chanterelle, bolete, chicken-of-the-woods,
hen-of-the-woods) are all oak-associated here -- the first two mycorrhizally,
the last two as wood/root decayers -- so "where are the oaks" is the static
layer that a foraging-conditions map is built on. Two independent sources are
pulled so they can be cross-checked against ground truth:

  1. USFS FHP Tree Species Metrics, basal area (30 m, ~2002, MODELED).
     Per-species basal area (sq ft/acre) fit from >300k FIA plots against
     climate/terrain/soil/Landsat predictors. Species-level detail is its
     selling point; it is a smooth statistical surface, not an observed stand
     map, and it predates the 2016-2018 spongy moth oak mortality on the Cape.

  2. USFS FIA BIGMAP forest type group (30 m, 2018).
     Coarser -- "Oak / hickory", "Oak / pine" rather than species -- but 16
     years fresher, so it is the better recency check on source 1.

Both are ArcGIS ImageServers on DOI's Interdepartmental Imagery Publication
Platform (IIPP); each layer is one exportImage call clipped to the domain.

Outputs (under forest/):
  fhp_ba_<species>.tif          per-species basal area, sq ft/acre
  oak_basal_area.tif            sum over all oak species
  bigmap_forest_type_group.tif  FIA forest type group code
  forest_hosts_summary.csv      area by class / basal-area distribution
  forest_hosts_map.png          quick-look 4-panel for eyeballing

Run in the `protocoast-notebook` conda env.
"""

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IIPP = "https://imagery.geoplatform.gov/iipp/rest/services/Vegetation"
FHP_BA = f"{IIPP}/USFS_EDW_FHP_TreeSpeciesMetrics_BasalArea/ImageServer"
BIGMAP_FTG = f"{IIPP}/USFS_FIA_BIGMAP_CONUS_ForestTypeGroup_2018/ImageServer"

WATERSHEDS = ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]

# UTM 19N: true 30 m pixels, so no Web Mercator latitude stretch to undo.
CRS = "EPSG:32619"
RES = 30.0
PAD = 2000.0  # m of slop around the watershed union

OUTDIR = Path("forest")

# Every oak FIA maps in this zone. All four target fungi key on these.
OAKS = [
    "white_oak",
    "black_oak",
    "scarlet_oak",
    "northern_red_oak",
    "chestnut_oak",
    "post_oak",
    "swamp_white_oak",
    "pin_oak",
    "scrub_oak",
]

# FHP also ships an all-deciduous-oak rollup. It is NOT a species to be added to
# the list above: over this domain it matches the sum of the individual oaks on
# 94.5% of pixels (r = 0.99), the residual being FHP's own histogram-matching
# post-process. Use it as the oak total so the per-species layers and the total
# stay internally consistent with FIA rather than double-counting.
OAK_TOTAL = "oak_deciduous_spp"

# Secondary hosts, kept separate because they discriminate between the targets:
# beech/hemlock/pine carry chanterelles and boletes but not hen-of-the-woods;
# cherry and maple carry chicken-of-the-woods.
COMPANIONS = [
    "American_beech",
    "eastern_hemlock",
    "pitch_pine",
    "eastern_white_pine",
    "black_cherry",
    "red_maple",
    "birch_spp",
]

# FIA forest type group codes (BIGMAP pixel values).
FTG_CODES = {
    0: "Non-forest",
    100: "White/red/jack pine",
    120: "Spruce/fir",
    140: "Longleaf/slash pine",
    160: "Loblolly/shortleaf pine",
    180: "Other eastern softwoods",
    200: "Douglas-fir",
    220: "Ponderosa pine",
    240: "Western white pine",
    260: "Fir/spruce/mtn hemlock",
    280: "Lodgepole pine",
    300: "Hemlock/Sitka spruce",
    320: "Western larch",
    340: "Redwood",
    360: "Other western softwoods",
    370: "California mixed conifer",
    380: "Exotic softwoods",
    390: "Other softwoods",
    400: "Oak/pine",
    500: "Oak/hickory",
    600: "Oak/gum/cypress",
    700: "Elm/ash/cottonwood",
    800: "Maple/beech/birch",
    900: "Aspen/birch",
    910: "Alder/maple",
    920: "Western oak",
    940: "Tanoak/laurel",
    950: "Other hardwoods",
    980: "Woodland hardwoods",
    990: "Exotic hardwoods",
    999: "Nonstocked",
}


# --------------------------------------------------------------------------- #
# Domain
# --------------------------------------------------------------------------- #
def domain_grid():
    """Union of the two MassDEP basins -> a snapped 30 m UTM 19N grid."""
    parts = [gpd.read_file(f).to_crs(CRS) for f in WATERSHEDS]
    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    x0, y0, x1, y1 = basins.total_bounds
    x0, y0 = np.floor((x0 - PAD) / RES) * RES, np.floor((y0 - PAD) / RES) * RES
    x1, y1 = np.ceil((x1 + PAD) / RES) * RES, np.ceil((y1 + PAD) / RES) * RES
    return basins, (x0, y0, x1, y1), int((x1 - x0) / RES), int((y1 - y0) / RES)


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def export_image(service, bbox, w, h, out, mosaic_where=None, pixel_type="U16"):
    """One exportImage call, clipped and reprojected server-side. Cached."""
    if out.exists():
        return out
    params = {
        "bbox": ",".join(f"{v}" for v in bbox),
        "bboxSR": CRS.split(":")[1],
        "imageSR": CRS.split(":")[1],
        "size": f"{w},{h}",
        "format": "tiff",
        "pixelType": pixel_type,
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation": "RSP_NearestNeighbor",
        "f": "image",
    }
    if mosaic_where:
        params["mosaicRule"] = json.dumps(
            {"mosaicMethod": "esriMosaicNone", "where": mosaic_where}
        )
    r = requests.get(f"{service}/exportImage", params=params, timeout=600)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image"):
        raise RuntimeError(f"{out.name}: {r.text[:300]}")
    out.write_bytes(r.content)
    return out


def recompress(path):
    """Rewrite as LZW-compressed -- these rasters are mostly zeros."""
    with rasterio.open(path) as src:
        prof = src.profile | {"compress": "lzw", "predictor": 2, "tiled": True}
        data = src.read()
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    OUTDIR.mkdir(exist_ok=True)
    basins, bbox, w, h = domain_grid()
    print(f"domain {w} x {h} px @ {RES:.0f} m  ({w * h / 1e6:.1f} Mpx)")
    print(f"bbox {CRS}: {bbox}")

    # --- per-species basal area ------------------------------------------- #
    species = OAKS + [OAK_TOTAL] + COMPANIONS
    for i, sp in enumerate(species, 1):
        out = OUTDIR / f"fhp_ba_{sp}.tif"
        if out.exists():
            print(f"  [{i:2d}/{len(species)}] {sp:22s} cached")
            continue
        print(f"  [{i:2d}/{len(species)}] {sp:22s} fetching...", flush=True)
        export_image(FHP_BA, bbox, w, h, out, mosaic_where=f"variable='{sp}'")
        recompress(out)

    # --- BIGMAP forest type group ----------------------------------------- #
    ftg_path = OUTDIR / "bigmap_forest_type_group.tif"
    if not ftg_path.exists():
        print("  BIGMAP forest type group fetching...", flush=True)
        export_image(BIGMAP_FTG, bbox, w, h, ftg_path)
        recompress(ftg_path)

    # --- load, clipped to the basins --------------------------------------- #
    # rio.clip on an integer raster back-fills with the raster's nodata rather
    # than NaN, so cast to float first -- otherwise everything outside the basin
    # counts as a real zero and the areas come out several times too large.
    geom = basins.geometry.union_all()

    def load(sp, clip=True):
        da = rioxarray.open_rasterio(OUTDIR / f"fhp_ba_{sp}.tif").squeeze(drop=True)
        da = da.astype("float32").where(da < 65535, 0.0)
        da.rio.write_crs(CRS, inplace=True)
        da.rio.write_nodata(np.nan, inplace=True)
        return da.rio.clip([geom], CRS, drop=False) if clip else da

    oak = load(OAK_TOTAL, clip=False)
    oak.rio.to_raster(OUTDIR / "oak_basal_area.tif", compress="lzw")
    oak_m = oak.rio.clip([geom], CRS, drop=False)

    ftg = rioxarray.open_rasterio(ftg_path).squeeze(drop=True).astype("float32")
    ftg.rio.write_crs(CRS, inplace=True)
    ftg.rio.write_nodata(np.nan, inplace=True)
    ftg_m = ftg.rio.clip([geom], CRS, drop=False)

    # --- summary ----------------------------------------------------------- #
    px_km2 = RES * RES / 1e6
    rows = []
    for sp in species:
        v = load(sp).values
        v = v[np.isfinite(v)]
        present = v > 0
        rows.append(
            {
                "species": sp,
                "group": ("oak_total" if sp == OAK_TOTAL
                          else "oak" if sp in OAKS else "companion"),
                "area_km2_present": present.sum() * px_km2,
                "pct_of_basin": 100 * present.mean(),
                "mean_ba_where_present": v[present].mean() if present.any() else 0.0,
                "p90_ba_where_present": (
                    np.percentile(v[present], 90) if present.any() else 0.0
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values(
        ["group", "area_km2_present"], ascending=[True, False]
    )

    ov = oak_m.values
    ov = ov[np.isfinite(ov)]
    print(f"\nbasin area           {ov.size * px_km2:8.1f} km2")
    print(f"any oak present      {(ov > 0).sum() * px_km2:8.1f} km2"
          f"  ({100 * (ov > 0).mean():.1f}%)")
    for t in (10, 25, 50, 75):
        print(f"oak BA >= {t:3d} sqft/ac {(ov >= t).sum() * px_km2:8.1f} km2"
              f"  ({100 * (ov >= t).mean():.1f}%)")

    fv = ftg_m.values
    fv = fv[np.isfinite(fv)]
    ftg_counts = pd.Series(fv.astype(int)).value_counts()
    ftg_rows = [
        {
            "code": int(c),
            "forest_type_group": FTG_CODES.get(int(c), f"unmapped ({int(c)})"),
            "area_km2": n * px_km2,
            "pct_of_basin": 100 * n / len(fv),
        }
        for c, n in ftg_counts.items()
    ]
    ftg_df = pd.DataFrame(ftg_rows).sort_values("area_km2", ascending=False)
    oak_ftg = ftg_df[ftg_df.code.isin([400, 500])].area_km2.sum()
    print(f"\nBIGMAP 2018 oak groups {oak_ftg:8.1f} km2"
          f"  ({100 * oak_ftg / (len(fv) * px_km2):.1f}%)")
    print(ftg_df.to_string(index=False))

    print("\nFHP ~2002 basal area by species:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    summary.to_csv(OUTDIR / "forest_hosts_summary.csv", index=False)
    ftg_df.to_csv(OUTDIR / "forest_type_group_summary.csv", index=False)

    quicklook(basins, oak_m, ftg_m, load)
    print(f"\nwrote {OUTDIR}/forest_hosts_map.png")


def quicklook(basins, oak_m, ftg_m, load):
    """4-panel: oak total, BIGMAP oak groups, pitch pine, beech+hemlock."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 13), constrained_layout=True)
    ext = [
        float(oak_m.x.min()), float(oak_m.x.max()),
        float(oak_m.y.min()), float(oak_m.y.max()),
    ]

    def frame(ax, title):
        basins.boundary.plot(ax=ax, color="k", lw=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

    ax = axes[0, 0]
    im = ax.imshow(oak_m.values, extent=ext, origin="upper",
                   cmap="YlGn", vmin=0, vmax=80, interpolation="nearest")
    frame(ax, "Total oak basal area (all Quercus spp.)\nUSFS FHP, 30 m, circa 2002 (modeled)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="sq ft / acre")

    # BIGMAP: highlight just the oak-bearing groups against everything else.
    ax = axes[0, 1]
    fv = ftg_m.values
    cls = np.full(fv.shape, np.nan)
    cls[np.isfinite(fv) & (fv > 0)] = 0          # other forest / nonstocked
    cls[fv == 400] = 1                            # oak/pine
    cls[fv == 500] = 2                            # oak/hickory
    cmap = ListedColormap(["#e8e4dc", "#7fb069", "#1b5e20"])
    ax.imshow(cls, extent=ext, origin="upper", cmap=cmap,
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest")
    frame(ax, "BIGMAP 2018 forest type group, 30 m\ndark = Oak/hickory, mid = Oak/pine")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#1b5e20", label="Oak/hickory (500)"),
            plt.Rectangle((0, 0), 1, 1, fc="#7fb069", label="Oak/pine (400)"),
            plt.Rectangle((0, 0), 1, 1, fc="#e8e4dc", label="other / nonstocked"),
        ],
        loc="lower left", fontsize=8, framealpha=0.9,
    )

    ax = axes[1, 0]
    pp = load("pitch_pine")
    im = ax.imshow(pp.values, extent=ext, origin="upper",
                   cmap="Oranges", vmin=0, vmax=50, interpolation="nearest")
    frame(ax, "Pitch pine basal area\n(Suillus / bolete host, pine-barrens signal)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="sq ft / acre")

    ax = axes[1, 1]
    bh = load("American_beech") + load("eastern_hemlock")
    im = ax.imshow(bh.values, extent=ext, origin="upper",
                   cmap="BuPu", vmin=0, vmax=40, interpolation="nearest")
    frame(ax, "Beech + hemlock basal area\n(secondary chanterelle / bolete hosts)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="sq ft / acre")

    fig.suptitle(
        "Fungal host layers - Buzzards Bay + Cape Cod watersheds",
        fontsize=14, fontweight="bold",
    )
    fig.savefig(OUTDIR / "forest_hosts_map.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
