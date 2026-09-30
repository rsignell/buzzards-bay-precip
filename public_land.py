"""
Public-land mask for the Buzzards Bay + Cape Cod watersheds.

Two MassGIS sources, unioned, rasterized onto the same 30 m grid as the
moisture and species layers -- so it overlays pixel-for-pixel with everything
else make_foraging_app.py draws:

1. Protected and Recreational OpenSpace polygons, filtered to federal/state/
   county/municipal ownership (OWNER_TYPE in F/S/C/M). Clean, but only covers
   land MassGIS has formally entered as protected open space -- an ordinary,
   never-registered town lot (a DPW yard, an unregistered forest holding) is
   invisible to it. This alone was the first cut and under-counted public
   land -- e.g. most town-owned parcels in Bourne never showed up.
2. Statewide property tax parcels (assessors' data), kept where OWNER1 matches
   a government-ownership name pattern (TOWN OF, COMMONWEALTH OF MASS,
   COUNTY OF, UNITED STATES, HOUSING AUTHORITY, WATER/FIRE DISTRICT, etc).
   This is the actual per-parcel ownership record, so it catches ordinary
   municipal/state/federal parcels the open-space layer misses. It is fetched
   as a server-side name filter (not every parcel in the watershed) because
   the underlying service times out on an unfiltered scan and 400s on a
   too-large query geometry -- hence the tiling below.

Static, like the oak index and soil layers: rerun by hand when the source data
changes, not part of run_daily.sh.

COVERAGE CAVEAT: source (1) misses unregistered public parcels; source (2)
is a free-text OWNER1 match against known naming conventions ("TOWN OF X",
"X TOWN OF", "COMMONWEALTH OF MASS...") which will still miss an unusual or
misspelled owner string, and could in principle mis-tag a private entity whose
name happens to match (none observed in spot checks -- land trusts, "TRUSTEES
OF RESERVATIONS", NSTAR, and private LLCs sampled around Bourne all correctly
stayed unmatched). Together these two sources are a materially better first
cut than open-space alone, not a legal record of ownership.

Sources (both public, no key required):
  MassGIS "Protected and Recreational OpenSpace" polygons -- ArcGIS
    FeatureServer at gis.eea.mass.gov
  MassGIS "Massachusetts Property Tax Parcels" (assessors' L3 parcels) --
    ArcGIS FeatureServer at services1.arcgis.com (hosted by MassGIS)

Output: land/public_land.tif  (2 = public, 1 = not public/unknown)
Code 1/2 rather than 0/1 so that, after reprojection, true "outside the basin"
nodata (which collapses to 0) stays distinguishable from "inside the basin but
not public" -- see the PUBLIC_LAND_STYLE comment in make_foraging_app.py.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize

CRS = "EPSG:32619"
BBOX_UTM = (319920, 4591950, 425820, 4662630)  # the shared 30 m habitat grid
RES = 30
OUT = "land/public_land.tif"

OPENSPACE_SERVICE = ("https://gis.eea.mass.gov/server/rest/services/"
                      "Protected_and_Recreational_OpenSpace_Polygons/FeatureServer/0/query")
PUBLIC_OWNER_TYPES = {"F", "S", "C", "M"}

PARCELS_SERVICE = ("https://services1.arcgis.com/hGdibHYSPO59RG1h/arcgis/rest/services/"
                    "Massachusetts_Property_Tax_Parcels/FeatureServer/0/query")
# OWNER1 patterns for government ownership, built from a spot check of actual
# owner strings around Bourne (both "TOWN OF X" and "X TOWN OF" appear).
# LIKE, not regex -- this ArcGIS service's SQL layer doesn't support REGEXP,
# and times out on an unfiltered/unbounded scan (hence tiling the geometry
# too, not just the owner filter).
OWNER_PATTERNS = [
    "OWNER1 LIKE 'TOWN OF%'", "OWNER1 LIKE '%TOWN OF'",
    "OWNER1 LIKE 'CITY OF%'", "OWNER1 LIKE '%CITY OF'",
    "OWNER1 LIKE 'COMMONWEALTH OF MASS%'",
    "OWNER1 LIKE 'MASSACHUSETTS %'",
    "OWNER1 LIKE 'MASS DEPT%'", "OWNER1 LIKE 'MASS DIV%'", "OWNER1 LIKE 'MASS DCR%'",
    "OWNER1 LIKE '%COUNTY OF%'",
    "OWNER1 LIKE 'BARNSTABLE COUNTY%'", "OWNER1 LIKE 'PLYMOUTH COUNTY%'",
    "OWNER1 LIKE 'UNITED STATES%'", "OWNER1 LIKE 'USA %'", "OWNER1 = 'USA'",
    "OWNER1 LIKE '%US ARMY%'", "OWNER1 LIKE '%US FISH%'",
    "OWNER1 LIKE '%US DEPT%'", "OWNER1 LIKE '%US GOVERNMENT%'",
    "OWNER1 LIKE '%NATIONAL PARK SERVICE%'",
    "OWNER1 LIKE '%HOUSING AUTHORITY%'",
    "OWNER1 LIKE '%WATER DISTRICT%'", "OWNER1 LIKE '%FIRE DISTRICT%'",
    "OWNER1 LIKE '%SCHOOL DISTRICT%'",
    "OWNER1 LIKE '%CONSERVATION COMMISSION%'",
    "OWNER1 LIKE '%REDEVELOPMENT AUTHORITY%'",
]
PARCELS_WHERE = " OR ".join(OWNER_PATTERNS)
# The parcels service 400s on a query geometry as large as the whole
# watershed bbox, so it's fetched in small tiles instead.
TILE_DEG = 0.15

PAGE = 2000


def _fetch(url, retries=4):
    """These ArcGIS endpoints occasionally 504 under load; retry with backoff."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _query_all(service, params_base, page=PAGE):
    """Page an ArcGIS FeatureServer query via resultOffset until exhausted."""
    features, offset = [], 0
    while True:
        params = dict(params_base, resultOffset=offset, resultRecordCount=page)
        url = service + "?" + urllib.parse.urlencode(params)
        fc = _fetch(url)
        feats = fc.get("features", [])
        features.extend(feats)
        if len(feats) < page:
            break
        offset += page
    return features


def fetch_open_space(bounds):
    """Protected/Recreational OpenSpace polygons in bounds (EPSG:32619)."""
    minx, miny, maxx, maxy = bounds
    params = {
        "where": "1=1",
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "32619", "outSR": "32619",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "OWNER_TYPE",
        "f": "geojson",
    }
    feats = _query_all(OPENSPACE_SERVICE, params)
    return gpd.GeoDataFrame.from_features(feats, crs=CRS)


def fetch_public_parcels(lonlat_bounds):
    """Government-owned tax parcels, tiled across lonlat_bounds (EPSG:4326)."""
    minlon, minlat, maxlon, maxlat = lonlat_bounds
    import numpy as np
    lons = np.arange(minlon, maxlon, TILE_DEG)
    lats = np.arange(minlat, maxlat, TILE_DEG)
    all_feats, n_tiles = [], len(lons) * len(lats)
    tile_n = 0
    for lon0 in lons:
        for lat0 in lats:
            tile_n += 1
            lon1, lat1 = min(lon0 + TILE_DEG, maxlon), min(lat0 + TILE_DEG, maxlat)
            params = {
                "where": PARCELS_WHERE,
                "geometry": f"{lon0},{lat0},{lon1},{lat1}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326", "outSR": "32619",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "OWNER1",
                "f": "geojson",
            }
            feats = _query_all(PARCELS_SERVICE, params)
            all_feats.extend(feats)
            print(f"  tile {tile_n}/{n_tiles} "
                  f"({lon0:.2f},{lat0:.2f}): {len(feats)} public parcels, "
                  f"{len(all_feats)} total", flush=True)
    return gpd.GeoDataFrame.from_features(all_feats, crs=CRS)


def main():
    parts = [gpd.read_file(f).to_crs(CRS) for f in
             ["buzzards_bay_watershed.geojson", "cape_cod_watershed.geojson"]]
    basins = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    geom = basins.geometry.union_all()
    lonlat_bounds = basins.to_crs(4326).total_bounds

    print("querying MassGIS protected/recreational open space ...")
    os_gdf = fetch_open_space(geom.bounds)
    os_gdf = os_gdf[os_gdf.geometry.notna() & os_gdf.is_valid & ~os_gdf.geometry.is_empty]
    print(f"  {len(os_gdf)} polygons in the watershed bbox")
    print(os_gdf["OWNER_TYPE"].value_counts().to_string())
    open_space_public = os_gdf[os_gdf["OWNER_TYPE"].isin(PUBLIC_OWNER_TYPES)]
    print(f"  {len(open_space_public)} are federal/state/county/municipal")

    print("\nquerying MassGIS property tax parcels for government owners ...")
    parcels = fetch_public_parcels(lonlat_bounds)
    parcels = parcels[parcels.geometry.notna() & parcels.is_valid & ~parcels.geometry.is_empty]
    print(f"  {len(parcels)} government-owned parcels")
    print("  sample owners:")
    print(parcels["OWNER1"].value_counts().head(15).to_string())

    public = pd.concat(
        [open_space_public.geometry, parcels.geometry], ignore_index=True)

    x0, y0, x1, y1 = BBOX_UTM
    nx, ny = int((x1 - x0) / RES), int((y1 - y0) / RES)
    transform = rasterio.transform.from_origin(x0, y1, RES, RES)

    mask = rasterize(
        [(g, 2) for g in public if g is not None],
        out_shape=(ny, nx), transform=transform, fill=1, dtype="uint8")

    os.makedirs("land", exist_ok=True)
    with rasterio.open(OUT, "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="uint8", crs=CRS, transform=transform,
                       compress="lzw", tiled=True) as dst:
        dst.write(mask, 1)

    px_km2 = RES * RES / 1e6
    n_public = int((mask == 2).sum())
    print(f"\npublic land: {n_public * px_km2:.0f} km2 of a "
          f"{nx * ny * px_km2:.0f} km2 grid -> wrote {OUT}")


if __name__ == "__main__":
    main()
