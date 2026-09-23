# Animated basin-rainfall choropleth in GeoLibre Web

No server, no Python: GeoLibre's in-browser DuckDB (SQL Workspace) can join
the basin polygons straight to the R2 time series and drive a real animated
choropleth off the Time Slider. Verified end to end in a live browser
(join -> style -> Time Slider recoloring the map as it scrubs across dates).

Both source files send `Access-Control-Allow-Origin: *` and the parquet
supports HTTP range requests, so this works with no credentials and no CORS
proxy.

## Steps

1. Open [web.geolibre.app](https://web.geolibre.app/).
2. **Processing -> SQL Workspace**. Paste the query below (pick your own
   date range -- a query pulls the *whole* basins geojson before DuckDB can
   filter it, so keep the range modest, a few weeks at most, or the browser
   tab can run out of memory) and click **Run**.
3. Type a layer name, click **Add as layer** (not *Add as query layer* --
   only a plain layer gets the full Style panel and the "Bind to Time
   Slider" action).
4. Select the new layer, open its **⋯** menu -> **Open Style panel**. A
   "Style suggestions" box offers **Graduate by precip_mm** -- click it.
   (Or set it by hand: Style type "Graduated", Attribute `precip_mm`.)
5. **⋯** menu again -> **Bind to Time Slider…**. It auto-detects
   `report_date`; leave "Show features: In the current step" and click
   **Bind**.
6. **Plugins -> Time Slider -> Activate** if the slider bar isn't already
   showing at the bottom. Scrub it, or hit play -- the basins recolor by
   that date's `precip_mm`.

`Project -> Save` keeps this in the browser (GeoLibre auto-persists to
local storage), so it's a once-per-browser setup, not a once-per-visit one.

## Query

```sql
SELECT g.TMDL_BASIN, g.BBP_SYS_ID, g.TYPE, t.report_date, t.precip_mm,
       t.precip_in, t.n_hours, t.n_hours_expected, g.geom
FROM ST_Read('https://raw.githubusercontent.com/rsignell/buzzards-bay-precip/main/basins/bbnep_subbasins_2026_v2.geojson') g
JOIN 'https://r2-pub.openscicomp.io/buzzards-bay-precip/basins_v2_precip_ts.parquet' t
  ON g.BBP_SYS_ID || '_' || g.TYPE = t.basin_id
WHERE t.report_date BETWEEN DATE '2026-09-08' AND DATE '2026-09-21'
  AND g.TYPE = 'WATER'   -- or 'LAND', or drop this line for both (66 features/day instead of 33)
```

## What's still manual

GeoLibre's project-format docs don't document the exact JSON schema for a
styled + time-slider-bound layer (classification breakpoints, the
Time-Slider-binding record), so this isn't baked into a one-click
`.geolibre.json` link the way `basins_precip.geolibre.json` and
`basins_precip_v2.geolibre.json` are -- steps 2-5 above are a ~30 second
manual setup, once per browser (it survives reloads via local storage; it
does not survive switching browsers/devices without redoing it or using
GeoLibre's own Project -> Share).
