# Airborne lidar over the study AOIs

**Why.** In the drought-response trait models (`drought_response.md` §2),
most of the trait gain for mortality and resilience disappeared once NEON
2013 lidar structure was added. That check only covered the 7,040 NEON cells
with segmented lidar trees. This note inventories the other airborne lidar
over the AOIs, and records which sources are pre-drought, so the check can
be repeated elsewhere.

Checked 2026-09-28 against the NASA CMR, the LVIS all-flights trajectory
KMZ, the OpenTopography catalog, the USGS 3DEP elevation index and the NEON
AOP API. Coverage is the fraction of each AOI rectangle (and of its NLCD
2013 forest) inside the footprint, from `query_lidar_coverage.py`
(`hls_results/lidar_coverage/coverage_sources.csv`, `coverage_<aoi>.png`).
OpenTopography footprints are catalog polygons, some of them hulls.

## Summary by AOI

| AOI | Before 2015 | 2015–2017 | 2018–2020 (before the 2020–22 drought) | 2021 and later |
|---|---|---|---|---|
| `neon_soap_teak` | NEON 2013 trees (16% of AOI, 18% of forest); **LVIS Sep 2008 (46%)**; USFS Dinkey 2010 (33%) and 2012 (33%); NEON D17 prototype 2011–13 (37%); SSCZO 2010 (8%); NCALM 2005 (4%). Any pre-2015 source: **81% of AOI, 85% of forest** | **ASO composite**: Kings snow-off Oct 2015 (63%) plus 2016–17 San Joaquin flights | NEON AOP 2017, 2018, 2019; 3DEP Southern Sierra Jun–Jul 2020 (90%) | NEON 2021, 2023, 2024; 3DEP 2022 (100%) |
| `sierra_nf` | USFS Willow Creek Nov 2012 (10%) only | **ASO composite**: San Joaquin snow-off Oct 23–27 2016 (~86% combined) plus Jan–Aug 2017 snow-on flights | 3DEP Southern Sierra Jun–Jul 2020 (57%, before the Creek Fire); 3DEP Yosemite 2019 (0.1%) | 3DEP 2022 (100%) |
| `stanislaus` | none | USFS Power Fire, Nov 2014–Jun 2015 (43%); FEMA Alpine Aug 2017 (4%) | 3DEP Eldorado Oct 2019–Mar 2020 (54%) | 3DEP 2021–22 (100%) |

- **No 3DEP lidar predates the 2015–16 die-off in any AOI.** Pre-drought
  lidar comes from USFS Region 5, university (NCALM/CZO) and NEON flights.
- **3DEP covers every AOI completely in 2021–22**, after the second drought.
  The 2019–20 3DEP work units give structure just before the 2020–22
  drought over half of `sierra_nf` and `stanislaus` and most of NEON.

## NASA airborne lidar

**LVIS** (Land, Vegetation and Ice Sensor, NASA GSFC).
- **Sep 21–26 2008 "Sierra Nevada" flights** (LDS 1.03; four days, 12.8 M
  footprints). They cover 46% of `neon_soap_teak` with contiguous ~20 m
  footprints (4.4 M in the AOI) and none of `sierra_nf` or `stanislaus`.
  - Not in CMR/NSIDC: `https://lvis.gsfc.nasa.gov/data_sets/LDS_1.03/LVIS_US_CA_day{1-4}_2008_VECT_*.zip`
    (`.lge.1.03`: big-endian binary, 64-byte records: `lfid, shotnumber`
    uint32; `azimuth, incidence, range` float32; `time, glon, glat` float64;
    `zg, rh25, rh50, rh75, rh100` float32).
- Other LVIS over California is unusable here: 1999 (31% of NEON, too old),
  a snow-on IceBridge transit strip (Mar 2010, east edge of `sierra_nf` and
  NEON), and the Jul 2024 SARP flight (one strip). The 2006 Yosemite flights
  are not released. The GEDI calibration campaigns (2019, 2021) did not fly
  California. No LVIS line crosses `stanislaus`.

**ASO** (NASA-JPL Airborne Snow Observatory).
- ASO flies lidar for snow at about 1 pt/m². Ferraz et al. (2018,
  *Remote Sensing* 10:164) merged the 2014–2017 flights over the Kings and
  San Joaquin basins, removed snow returns and corrected flight offsets, and
  released the result as point clouds, a 5 m CHM and 10 m structure rasters
  (RH25–98, plant area index, foliage height diversity, height CV, crown
  ratio) (Ferraz et al. 2020, doi:10.5068/D16T06, Zenodo 3964981).
- Snow-off flights over the AOIs: Kings Oct 10 2015 (NEON), San Joaquin
  Oct 23–27 2016 (`sierra_nf`). The Kings Aug 2014 flight does not reach
  NEON. Snow-on flights from 2017 are merged in.
- **Caveat.** The composite dates from the middle of the 2015–16 die-off,
  so it partly records mortality: dead trees keep their height but lose
  foliage. As a structure control it is stricter than pre-drought
  structure.
- ASO's other public products over these basins (NSIDC `ASO_3M_PCDTM`) are
  bare-earth DTMs and snow depth, not vegetation.

**G-LiHT** (Goddard LiDAR/Hyperspectral/Thermal): the 2012 AMIGACarb
transects and Mar 2013 Teakettle flights cover ≤ 4% of any AOI.

**NASA CMS** "LiDAR-derived aboveground biomass and tree crowns for
California 2005–2014" (ORNL DAAC 1537, doi:10.3334/ORNLDAAC/1537) processes
the same USFS/NCALM collections listed above (Dinkey 2010/2012, Willow Creek
2012, Bull and Providence 2012), with per-tree crowns.

## GEDI

GEDI is too sparse for 90 m stand structure here.
- Tracks are 600 m apart, with footprints every 60 m along track.
- Over `sierra_nf`, about 300 raw shots/km² were collected in 2019–2023.
  After quality filtering that is roughly 75–120/km²: under one shot per
  90 m cell (most cells have none) and 5–9 per 270 m cell.
- Only April 2019 to mid-2020 precedes the 2020–22 drought, which gives
  about one shot per 270 m cell. Nothing precedes the 2012–16 drought.
- Height errors grow on slopes above about 30°.
- GEDI enters only through GEDI-calibrated wall-to-wall products such as
  GLAD forest height (already used as S).

## Structure layers built from these sources

`fetch_lidar_structure.py` puts ASO and LVIS onto the 30 m AOI grids
(`/Volumes/Earth04/ecopro/lidar/<aoi>_lidar.nc`); see its docstring for the
variables. The NEON 2013 tree summaries are in `env/<aoi>_env.nc`
(`aoi_env_layers.py`).
