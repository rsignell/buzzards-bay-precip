"""
Which index actually separates hardwood from pine on this landscape?

Ground-truth observation from someone who walks this forest: summer canopy
greenness looks like a better hardwood map than the leaf-on/leaf-off deciduous
index. That is worth testing rather than assuming, because it runs against the
usual reasoning -- phenology is the textbook way to split deciduous from
evergreen, and summer NDVI is nominally just a greenness measure.

There is a plausible mechanism for it being right here. Cape Cod pine is pitch
pine barrens: open, sparse, low-LAI canopy over bright sand, which reads low in
summer NDVI. Closed broadleaf canopy reads high. Meanwhile the deciduous index
inherits every problem of the leaf-off composite -- few clear winter scenes,
snow and low-sun artefacts, evergreen understory under bare oaks -- and is
badly confounded by anything else with a seasonal swing, notably salt marsh and
mown grass.

Labels are BIGMAP forest type groups plus the SSURGO tidal-marsh mask. BIGMAP
is imperfect but independent of anything Sentinel-2 derived.

Scored on:
  gap        median(Oak/hickory) - median(pine), in index units
  sep        that gap divided by the mean IQR of the two classes -- the part
             that matters for a map, since a large gap between noisy classes
             still produces a speckled picture
  oak>pine   fraction of Oak/hickory pixels above the pine median (0.5 = no
             skill, 1.0 = perfect)
  marsh      median over tidal marsh, which must stay well below oak or the
             shoreline lights up as prime ground

Run on the oak-mapping cluster.
"""

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401

LAYERS = [
    ("deciduous index", "s2/s2_decid.tif", 1),
    ("summer NDVI", "s2/s2_ndvi_summer.tif", 1),
    ("leaf-off NDVI (inverted)", "s2/s2_ndvi_leafoff.tif", -1),
]

CLASSES = [(500, "Oak/hickory"), (400, "Oak/pine"), (100, "W/R/J pine")]


def main():
    ref = rioxarray.open_rasterio(LAYERS[0][1]).squeeze(drop=True)
    ftg = (rioxarray.open_rasterio("forest/bigmap_forest_type_group.tif")
           .squeeze(drop=True).rio.reproject_match(ref, resampling=0)).values
    marsh = (rioxarray.open_rasterio("soil/tidal_marsh_mask.tif")
             .squeeze(drop=True).rio.reproject_match(ref, resampling=0)).values

    rows = []
    for name, path, sign in LAYERS:
        a = rioxarray.open_rasterio(path).squeeze(drop=True)
        if a.shape != ref.shape:
            a = a.rio.reproject_match(ref, resampling=1)
        v = sign * a.values
        ok = np.isfinite(v)

        stats = {}
        for code, label in CLASSES:
            sel = ok & (ftg == code) & (marsh == 0)
            d = v[sel]
            stats[label] = (np.median(d), np.percentile(d, 75) - np.percentile(d, 25),
                            d)

        oak_med, oak_iqr, oak_d = stats["Oak/hickory"]
        pine_med, pine_iqr, pine_d = stats["W/R/J pine"]
        gap = oak_med - pine_med
        sep = gap / (0.5 * (oak_iqr + pine_iqr))
        skill = (oak_d > pine_med).mean()
        msel = ok & (marsh == 1)
        marsh_med = np.median(v[msel])
        # How far marsh sits from oak, in the same normalised units.
        marsh_gap = (oak_med - marsh_med) / (0.5 * (oak_iqr + pine_iqr))

        rows.append({
            "index": name,
            "Oak/hickory": oak_med, "Oak/pine": stats["Oak/pine"][0],
            "pine": pine_med, "marsh": marsh_med,
            "gap": gap, "sep": sep, "oak>pine": skill,
            "marsh_sep": marsh_gap,
        })

    df = pd.DataFrame(rows)
    print(f"{'index':26s}{'Oak/hick':>9}{'Oak/pine':>9}{'pine':>8}{'marsh':>8}"
          f"{'gap':>8}{'sep':>7}{'oak>pine':>10}{'marsh sep':>11}")
    for _, r in df.iterrows():
        print(f"{r['index']:26s}{r['Oak/hickory']:9.3f}{r['Oak/pine']:9.3f}"
              f"{r['pine']:8.3f}{r['marsh']:8.3f}{r['gap']:8.3f}{r['sep']:7.2f}"
              f"{r['oak>pine']:10.2f}{r['marsh_sep']:11.2f}")

    best_sep = df.loc[df.sep.idxmax()]
    best_skill = df.loc[df["oak>pine"].idxmax()]
    print(f"\nbest separation-to-noise : {best_sep['index']} (sep {best_sep.sep:.2f})")
    print(f"best oak/pine skill      : {best_skill['index']} "
          f"({best_skill['oak>pine']:.2f})")
    print(f"\nmarsh rejection (higher is better):")
    for _, r in df.sort_values("marsh_sep", ascending=False).iterrows():
        print(f"  {r['index']:26s} {r['marsh_sep']:5.2f}")


if __name__ == "__main__":
    main()
