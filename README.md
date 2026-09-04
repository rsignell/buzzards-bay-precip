# buzzards-bay-precip — published map

This branch (`gh-pages`) exists only to serve a static page via GitHub Pages.

**Live map:** https://rsignell.github.io/buzzards-bay-precip/

`index.html` — interactive hvplot/Bokeh map of 24-hour rainfall over the
Buzzards Bay / upper Cape Cod / Massachusetts south-coast region for one
CoCoRaHS reporting day (7am ET → 7am ET). **Currently showing:
Sep 3 7am → Sep 4 7am ET 2026.**

- **Shaded:** NOAA MRMS radar/gauge-corrected ~1 km QPE
  (`precipitation_surface`), from the dynamical.org public Icechunk/Zarr store,
  accumulated over the 24 hourly steps in the CoCoRaHS 7am–7am reporting window.
- **Basemap:** OpenStreetMap.
- **Points:** every CoCoRaHS station in the map extent that filed a report
  (squares = the curated 16-station Buzzards Bay watershed set, circles = the
  rest), colored on the same scale. Hover for station id/name, gauge total,
  MRMS total, and their difference.
- **Outlines:** MassDEP major basins — Buzzards Bay (gray solid) and
  Cape Cod (orange dashed).

Regenerate with `compare_buzzards_bay_mrms.py` on `main` (set `REPORT_DATE`);
rebuild the roster with `find_region_stations.py` (set `END_DATE`).
