# Buzzards Bay precip comparison: IMERG vs. AORC vs. CoCoRaHS

This started from a simple question while working on a [New Bedford, MA
combined sewer overflow (CSO) discharge analysis](new_bedford_cso_analysis.ipynb):
how good is the rain-gauge data (`rainfall_in`) in MassDEP's CSO discharge
reports, and how does it compare to modern gridded precipitation products?
That led to comparing three very different precipitation data sources over
the Buzzards Bay watershed:

- **[NASA IMERG Analysis-Late](https://dynamical.org/catalog/nasa-imerg-analysis-late/)**
  — global satellite precip, 0.1° (~10 km), half-hourly, 1998-present.
- **[NOAA AORC](https://registry.opendata.aws/noaa-nws-aorc/)** (Analysis of
  Record for Calibration) — CONUS radar/gauge blend, bias-corrected, ~800 m
  (1 km grid), hourly, 1979-present (currently through 2025 in Zarr form).
- **[CoCoRaHS](https://www.cocorahs.org/)** — volunteer rain-gauge network,
  daily manual reports.

Both IMERG and AORC are published as cloud-native Zarr stores on AWS Open
Data (`us-east-1`), so no download is needed — just point `xarray` at them.
No AWS credentials are required to read either one (IMERG via its public
HTTPS/Icechunk endpoint, AORC via anonymous S3). A small EC2 VM in
`us-east-1` (via [SkyPilot](https://skypilot.co/)) was used purely to be
network-local to the data, not for authentication.

## What's here

| File | What it is |
|---|---|
| `new_bedford_cso_analysis.ipynb`, `new_bedford_cso_discharges*.csv` | The original CSO discharge analysis that motivated this detour (MassDEP CSO Data Portal, "Verified Data Report" events, 2022–2026). |
| `compare_precip.py` | Single-point comparison: IMERG vs. AORC vs. CSO-report gauge rainfall at New Bedford, 2025–2026. |
| `find_stations.py` | Pages through the CoCoRaHS API for one day nationwide, keeps stations inside a Buzzards Bay bounding box. |
| `pull_cocorahs.py` | Pulls full daily records for the candidate stations from the CoCoRaHS API. |
| `refilter_stations.py` | Re-tests candidate stations against the actual MassDEP watershed polygon (see below). |
| `buzzards_bay_watershed.geojson` | MassDEP's official "BUZZARDS BAY" major-basin polygon, pulled from MassGIS. |
| `compare_buzzards_bay.py` | First pass: CoCoRaHS vs. AORC across the watershed, daily totals. **Has a known timing bug — see below.** |
| `compare_buzzards_bay_v2.py` | Corrected version: re-buckets AORC into CoCoRaHS's actual 7am–7am Eastern Time accumulation window. |
| `plot_station_scatters.py` | Per-station scatter grid (superseded by the plotting built into `compare_buzzards_bay_v2.py`). |
| `*_comparison*.csv`, `*.png` | Outputs at each stage. |

## Timeline / what we found

**1. Single point, New Bedford (`compare_precip.py`).** IMERG and AORC daily
totals correlate at r=0.69 over 2025 (AORC has no 2026 data yet). Against
the sparse CSO-report gauge readings (only recorded on discharge-event
days), AORC tracked the gauge much better (r=0.63) than IMERG did (r=0.41) —
consistent with AORC being a bias-corrected radar/gauge blend rather than
satellite-only. Hit one real bug along the way: AORC's `APCP_surface` units
are `kg/m^2` (numerically equal to mm), not inches — an early version
mishandled that and produced a nonsense 56 in/day.

**2. Scaling up to the whole watershed.** The CoCoRaHS web API
(`api2.cocorahs.org`) turns out to have no working server-side geographic
filter — `subdiv1`/`subdiv2`/`country` params are silently ignored — but
per-station filtering by `stationField=StationNumber` does work. So the
approach became: page through one day of *national* data to build a station
roster, keep whatever falls in a rough bounding box, then pull each
station's full record individually.

**3. "Approximate bounding box" turned out to be wrong in an interesting
way.** The initial bbox pulled in Martha's Vineyard (a separate island
watershed) and two Taunton River towns (Somerset, Dighton — they drain to
Narragansett Bay, not Buzzards Bay), which were dropped by hand. Pulling
MassDEP's actual "Major Basins" GIS layer to do this properly turned up
something non-obvious: **MassDEP's official watershed boundary puts the
entire Cape Cod peninsula — including its Buzzards-Bay-facing west shore
(North Falmouth, Pocasset, West Falmouth) — under a separate "CAPE COD"
basin**, confirmed at both the major-basin and finer subbasin level. That's
an administrative/regulatory choice, not a hydrologic one: water off North
Falmouth physically drains into Buzzards Bay regardless of which basin
MassDEP books it under. The final station set (16 stations) combines the 6
stations in the official basin with the Cape Cod stations that geographically
face Buzzards Bay, excluding the Vineyard-Sound-facing side of Falmouth.

**4. The big one: a 24-hour bookkeeping mismatch.** `compare_buzzards_bay.py`
bucketed AORC into plain midnight-to-midnight UTC calendar days. CoCoRaHS
observers actually report a 24-hour total each morning at ~7am *local* time
— the accumulation since the previous morning's reading, not a calendar day.
Comparing those two conventions directly muddies every storm that straddles
either cutoff. `compare_buzzards_bay_v2.py` fixes this by re-bucketing AORC's
native hourly data into 7am-to-7am US/Eastern windows (DST-aware) before
matching to each CoCoRaHS report date. The effect was dramatic:

| | Naive calendar-day AORC | 7am–7am ET aligned AORC |
|---|---|---|
| Pooled daily correlation, all 16 stations | r = 0.60 | **r = 0.95** |
| Per-station range | r = 0.37 – 0.74 | r = 0.83 – 0.99 |

Most of what looked like "AORC vs. CoCoRaHS disagreement" in the first pass
was a time-bucketing artifact, not a real difference between the products.
Once aligned, AORC matches the volunteer gauge network almost exactly.

## Caveats / open threads

- The very first single-point comparison (`compare_precip.py`) likely has
  the same kind of timing mismatch against the CSO CSV's `rainfall_in`
  column — its reporting convention wasn't checked, so those r=0.41/0.63
  numbers are probably underestimates. Not yet fixed.
- AORC's Zarr store only goes through 2025; all 2026 comparisons are
  necessarily IMERG (or CoCoRaHS) only until NOAA publishes an update.
- The Buzzards Bay watershed station set is a judgment call (see point 3
  above), not an authoritative delineation — reasonable people could draw
  the Cape Cod/Buzzards Bay line differently.
