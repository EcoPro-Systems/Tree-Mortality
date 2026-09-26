# HLS signal in USFS ADS mortality polygons (prototype)

**Question.** Is there a clear, consistent HLS (Harmonized Landsat Sentinel-2,
30 m) spectral signal associated with the human-annotated USFS Aerial
Detection Survey (ADS) tree-mortality polygons? If so, HLS could map mortality
directly instead of relying on coarse, noisy ADS labels. The paper model
(climate + topography → ADS severity on a 270 m grid) has near-zero
across-year skill.

This prototype uses three 30×30 km AOIs and the **raw ADS polygons**, not the
270 m rasterized product.

## Data

| Dataset | Location (`/Volumes/Earth04/ecopro/...`) | Notes |
|---|---|---|
| USFS R5 IDS geodatabase (all years) | `usfs_ids/CONUS_Region5_AllYears.gdb` | Downloaded 2026-09-25 from the [USFS FHP detection-survey page](https://www.fs.usda.gov/science-technology/data-tools-and-products/fhp-mapping-reporting/detection-surveys). Layers: damage areas, damage points, **surveyed areas**. Years 1999–2025. |
| Sierra subsets of the above | `usfs_ids/R5_{damage_areas_sierra,damage_points_sierra,surveyed_areas}_2012plus.gpkg` | ESRI:102039 |
| MTBS burned-area perimeters | `fire/mtbs_perimeter_data/` | Through 2025. Fires ≥1000 ac only (West), so smaller fires are missed. |
| NLCD 2013 land cover and tree canopy cover | `landcover/<aoi>_landcover.tif` | From the MRLC WCS, on the AOI grid. The pre-drought year avoids the mask depending on mortality. |
| HLS granule windows | `hls/<aoi>/<granule>.tif` | 7-band int16 (blue, green, red, nir, swir1, swir2, fmask). Day of year 150–300, 2013–2025. |
| Per-year labels | `hls_labels/<aoi>.nc`, `hls_labels/<aoi>_polygons.csv` | See Labels below. |
| Annual composites | `hls_composites/<aoi>_doy182-273.nc` | Median over Jul 1–Sep 30. |

The previously used gpkgs (`geom/ADS2012202_SierraNevada_2021_yerarToyear/`)
are a processed subset of the same source (see the findings below). The
analysis here uses the geodatabase directly.

## Findings about the ADS labels themselves

1. **No polygon does not mean no mortality.** Two separate issues apply.
   - **Coverage.** ADS does not fly everywhere every year. The geodatabase has
     a `SURVEYED_AREAS` layer with flight start and end dates. Survey extent in
     R5 dropped from about 500k km² per year (2012–2014) to about 200k km²
     (2015–2017), and 2020 was barely flown (COVID).
   - **Detection.** Inside surveyed areas, a missing polygon only means nothing
     was sketched. Small groups of dead trees (about 60% of dead trees
     occur in groups of three or fewer; Cheng et al. 2024) are routinely missed.

   We therefore use three classes: *mortality polygon*, *surveyed but no
   damage feature* (a weak negative), and *not surveyed* (excluded).
2. **Severity means different things before and after 2017.**
   - **Legacy surveys (2012–2016)** record dead trees per acre (`LEGACY_TPA`)
     and have *no* percent-affected value in the raw data.
   - **DMSM surveys (2017+)** record percent-affected classes
     (`PERCENT_AFFECTED`, `PERCENT_MID`) and no TPA.
   - The local gpkgs carry `PERCENT_AFFECTED_CODE` 1–5 for every legacy
     feature, but those codes are **TPA bins** (<1, 1–3, 3–10, 10–30, >30
     trees/acre), not percent of trees affected. From 2017 the codes are true
     percent classes, and `MORT_TPA` looks back-derived from them (it takes a
     few constant values per class).
   - So the severity "class" used as a target in the paper changes meaning
     between 2016 and 2017.
3. **Legacy polygons overlap heavily.** Separate polygons are drawn per host and
   agent over the same ground. In the `sierra_nf` AOI in 2015, the polygon
   areas sum to 899 km², but their union is 384 km². In DMSM years overlap is
   small (2018: 295 vs 266 km²). Summing `ACRES` therefore overstates legacy-era
   affected area by about 2×.
4. **Polygons are coarse relative to 30 m.** In `sierra_nf`, mortality polygons
   cover 43% of the AOI in 2015 and 66% in 2016.
5. **Point features.** From 2017, DMSM also records small mortality spots as
   *damage points* (with tree counts), stored in a separate layer. The local
   gpkgs include them for 2019 and 2021 as `BUFFERED_POINT` features.
6. **More years exist.** The geodatabase has 2022–2025 surveys, beyond the
   2012–2021 local gpkgs.
7. **2020 is not aerial survey data.** In the Sierra, 580 of 585 features in
   2020 have `DATA_SOURCE_NAME = REMOTE_SENSING`.

### Why ADS polygons should not align with 30 m imagery

This section draws on the geodatabase readme (`usfs_ids/IDS_FlatFiles_Readme.pdf`)
and the [GIS Handbook for Forest Health Detection Survey](https://www.fs.usda.gov/foresthealth/technology/docs/DMSM_Tutorial/story_content/external_files/GIS-Handbook-for-Forest-Health-Detection-Survey.pdf).

- **Polygons mark areas *with* damage, not damaged trees.**
  - The readme says: "Not all trees within the areas may have damage."
  - For DMSM, percent affected is the share of trees in the polygon that are
    damaged or recently dead. A Light (4–10%) polygon is more than 90% healthy
    trees by definition.
  - So even a perfectly placed polygon is mostly unchanged at 30 m. Polygon
    pixel AUC is bounded well below 1, and it should rise with severity. It
    does so only weakly, and mainly for the >50% class.
- **Location is sketched by eye from an aircraft.**
  - Polygons are drawn on a tablet (or on paper maps for legacy data) while
    flying, and are "oriented on base data."
  - The handbook states no positional accuracy. An independent assessment of
    USFS ADS ([Coleman et al. 2018, *Forest Ecology and Management*](https://www.sciencedirect.com/science/article/abs/pii/S0378112718311459))
    reported damage-location accuracy of about 68% within 50 m and about 79%
    within 500 m for bark beetles. These figures are taken from summaries of
    the paper; the full text has not been checked yet.
  - That matches our multi-scale result, where agreement with HLS grows from
    30 m to about 1 km.
- **Legacy data (all R5 Sierra features through 2016) come from paper and
  DASM sketch maps** that were later translated into the national database.
  - Some legacy "polygons" are **points buffered into polygons** to meet
    legacy standards.
  - These appear as the many polygons of about 0.7 acre (the median 2012
    feature is 0.7 ac, a circle of radius about 30 m). Their size and shape
    carry no information.
- **Surveyors choose the feature type by how well they can place it.**
  - Polygons are for damage that is "discrete and obvious from the air and easy
    to locate."
  - Points are for small clusters "where the extent … is less important than
    the location."
  - Grid cells (240–1920 m) are for diffuse damage that is "difficult to
    render precisely." None of our AOI features are grid cells.
- **Duplicated footprints ("pancakes").**
  - Multiple observations (different host or agent) on the same footprint are
    stored as duplicate geometries with the same `DAMAGE_AREA_ID`.
  - In our AOIs, 57% (2015) and 80% (2016) of features are flagged
    `OBSERVATION_COUNT = MULTIPLE`.
- **Timing.**
  - Features record "current year's" damage: trees with fading or red foliage
    during the June–October flights.
  - A tree killed in year Y−1 is often mapped in year Y. Red needles drop
    within about one to three years, and survey dates vary across the season.
  - So the spectral change can fall in Y−1, Y or Y+1 relative to the survey
    year. This is why single-year HLS differences flip sign.
- **Coverage.** The handbook says detection surveys "do not provide a full
  inventory of tree damage."

### Can inter-observer agreement be measured? Not from the public data

- **Legacy overlap is entirely duplicates.** In every AOI-year from 2012 to
  2016, the union of the mortality polygons equals the area of the
  de-duplicated footprints (unique `DAMAGE_AREA_ID`). So 100% of the overlap
  is identical-geometry "pancakes": the same footprint recorded once per host
  or agent. There are no independently drawn overlapping delineations.
- **Repeat flights never overlap.** Several AOI-years were flown twice in a
  season, e.g. `sierra_nf` 2017 on Aug 12 and Sep 27, and `lassen` 2017 on
  Jul 27 and Sep 30 (575 km² flown twice). Assigning DMSM polygons to flights
  by `CREATED_DATE`, the Jaccard overlap between the two flights' mortality
  polygons is **0 in all 11 flight pairs**.
  - In `sierra_nf` 2017, the first flight mapped 47.5% of the doubly flown
    area and the second 7.7%, but in different places.
  - So the second pass adds *new* features rather than re-mapping. DMSM
    lets a surveyor add an observation to another user's feature, which
    produces a pancake instead of a new polygon, and QA flags overlaps.
  - Published data are also reconciled: "records contested with different
    observations … must be resolved" (GIS Handbook).
- **What we can and can't conclude.** The published flat file is a single
  reconciled product, so it cannot measure how much independent observers
  agree. The readme lists `FEATURE_USER_ID` and `OBSERVATION_USER_ID`, but
  they are not in the public flat file. Measuring agreement would need the raw
  per-surveyor DMSM data from USFS R5 / FHAAST, or an independent reference
  (e.g. NAIP dead-tree maps).
- Analysis script: `src/proto_ads_repeat_flights.py`.

**Implication:** ADS is not pixel-level truth. Evaluate HLS against it only at
scales of at least 500 m to 1 km (or with a spatial tolerance), and use an
independent high-resolution reference for 30 m validation.

## AOIs

Each center is the 30 km window (3 km search step) with the most ADS
mortality acreage spread across 2014–2024 within each region (`config/hls_aois.yml`):

| AOI | Center (lon, lat) | UTM | HLS tiles | Mortality polygons | Dominant hosts / agents |
|---|---|---|---|---|---|
| `sierra_nf` | −119.375, 37.396 | 11N | 11SKB, 11SLB | 2039 | Mixed pine and fir. Fir engraver, western and mountain pine beetle. Peak 2015–17. Creek Fire 2020. |
| `stanislaus` | −120.165, 38.505 | 10N | 10SGH | 1646 | Red and white fir. Fir engraver. Mortality sustained through 2024. |
| `lassen` | −121.143, 40.598 | 10N | 10TFK, 10TFL | 1179 | White fir, ponderosa pine. Dixie Fire 2021. |

NLCD 2013 evergreen forest covers 64–73% of each AOI.

## Methods

Scripts are in `src/` and the config is `config/hls_aois.yml`.

1. **`fetch_hls_aoi.py`: windowed HLS reads.**
   - Searches CMR for HLSL30/HLSS30 v2.0 granules.
   - Reads only the AOI window from each COG over HTTPS (GDAL `/vsicurl/`,
     Earthdata cookie jar), instead of downloading full 110 km tiles. This
     takes about 1–3 s per granule and about 3 MB on disk.
   - Uses only tiles in the AOI's UTM zone, so no resampling is needed.
   - Skips granules with less than 1% clear pixels in the AOI.
2. **`fetch_landcover_aoi.py`:** NLCD land cover and TCC via WCS.
3. **`ads_labels_aoi.py`: per-year 30 m labels.**
   - A pixel is assigned to a polygon if its center falls inside the exact
     polygon geometry.
   - Produces: label (−1 not surveyed, 0 surveyed and clear, 1 mortality
     polygon, 2 mortality point within 45 m, 3 other damage), `poly_id` of
     the most severe covering polygon, `n_polys`, overlap-summed `tpa_sum` and
     `pct_sum`, and MTBS `burned`.
4. **`hls_annual_composites.py`: annual medians.**
   - Fmask screening (fill, cloud, adjacent, shadow, snow, water, high
     aerosol).
   - The same sensor and date seen on overlapping tiles is averaged into one
     observation.
   - Per-year Jul–Sep medians of the bands and of NDVI, NDMI, NBR,
     RGI (red/green) and EVI.
5. **`hls_flight_composites.py`: composites matched to the overflight date.**
   - Each pixel's survey day of year (`flight_doy` in the label files) comes
     from one of two sources:
     - the tablet `CREATED_DATE` of the covering DMSM mortality polygon
       (2017+), which is the day the polygon was drawn;
     - otherwise the latest surveyed-area `START_DATE`/`END_DATE` midpoint.
   - Legacy `CREATED_DATE`s are December digitization dates, so they are not
     used.
   - **Dates are missing** for 2016 in all AOIs, for 2015 in `sierra_nf`, and
     for 2020.
   - Flights range from early July (day of year about 192) to late October
     (about 302). Some areas were flown twice in a season.
   - For each pixel, the median of clear observations within ±20 days of its
     flight day is taken in four years: Y (`at`), Y−1 (`prev`), Y+1 (`next`)
     and 2013 (`base`). Every comparison is therefore at the same time of year.
   - In the signal and model scripts, pass
     `--composite-suffix _flight_w20.nc`. `d1` then means at−prev, `lead`
     means next−at, and `base` means at−2013.
6. **`proto_hls_mortality_signal.py`: signal analysis.**
   - Pixels are restricted to NLCD forest, not burned from 2012 through Y+1.
   - **Change features:** `d1` = X(Y)−X(Y−1), `d2` = X(Y)−X(Y−2),
     `lead` = X(Y+1)−X(Y), and `base` = X(Y)−X(2013).
   - **Comparisons:** pixel AUC of mortality-polygon pixels against
     surveyed-clear pixels, a 100–500 m ring, and pixels more than 1 km away.
     This is also done by host group.
   - **Severity:** Spearman correlation of the pixel change with severity.
   - **Polygon level:** the fraction of each polygon's pixels whose change
     exceeds the 5% false-positive threshold of the surveyed-clear
     distribution, compared with the ADS severity.

## Results

_In progress._

### Preliminary (`sierra_nf` only)

- **Pipeline sanity.** MTBS Creek Fire (2020) burned vs unburned pixels
  separate with ΔNBR AUC 0.95, so georegistration, grids and label
  rasterization are consistent.
- **HLS sees mortality; ADS outlines do not follow it.**
  - The QC maps (`hls_results/sierra_nf_only/qc_map_*.png`) show coherent
    patches of NDMI decline and red-green index increase, and red/brown crowns
    in true color.
  - These patches lie both inside and outside the ADS polygons.
- **Pixel level is weak.** AUC for polygon vs surveyed-no-damage pixels is
  about 0.4–0.6 for single-year changes (Jul–Sep composites).
- **Agreement grows with scale.** Spearman correlation of block ADS coverage
  with mean HLS change strengthens from 30 m to about 1 km in die-off years,
  up to about 0.4. This is consistent with the documented positional
  tolerance of ADS.
- **Severity barely tracks HLS change.** The fraction of changed pixels per
  polygon correlates with ADS severity at ρ ≈ 0.1. Only the DMSM >50% class
  stands out.
- **Single-year differences flip sign across years.** Stands move from green
  to red (RGI up) to gray (RGI down).
  - Flight-date-matched composites fix part of this. RGI d1 AUC is 0.64 in
    2019 and 2023, and 2024 goes from an inverted 0.33 to 0.52.
  - The pixel-level ceiling stays around 0.65.
- **Held-out-year model** (`sierra_nf` only):
  - Pixel AUC is 0.53–0.56 in legacy years and 0.54–0.71 from 2017 on.
  - 1 km block R² is mostly negative. The model cannot predict each year's
    overall ADS coverage level, which echoes the paper's near-zero
    across-year skill.

### Three AOIs (`hls_results/all/`, `hls_results/all_flight/`)

The three-AOI results confirm the `sierra_nf` picture.

- **Pixel level.** One-year ΔNDMI/ΔRGI AUC for ADS polygon vs
  surveyed-no-damage pixels is mostly 0.4–0.6 in every AOI. The sign is
  inconsistent across years (e.g. `lassen` 2023: RGI 0.71 but 2024: 0.30),
  with or without flight-date matching.
- **1 km scale.** Spearman of ADS coverage with ΔRGI reaches 0.4–0.5 in
  some AOI-years (`sierra_nf` 2019/2023, `lassen` 2023). It is near zero or
  negative in others.
- **Held-out model (HistGradientBoosting on the HLS trajectory):**

  | | Fixed Jul–Sep window | Flight-matched |
  |---|---|---|
  | Pixel AUC, leave-one-year-out | 0.61 (0.54–0.67) | 0.61 (0.55–0.71) |
  | Pixel AUC, leave-one-AOI-out | 0.59 | 0.59 |
  | 1 km block R², leave-one-year-out | ≤ 0.06 | ≤ 0.03 |
  | 1 km block Spearman, leave-one-year-out | 0.06–0.41 | 0.09–0.41 |

**Conclusion:** HLS carries little transferable information about *where
ADS draws polygons*. Given the documented ADS limitations above, this says
more about ADS as a 30 m label than about HLS.

### HLS vs the Cheng et al. 2020 NAIP dead-tree map (`hls_results/cheng2020/`)

**Setup.**
- **Composites** (`hls_naip_composites.py`): per-pixel medians of clear HLS
  observations within ±25 days of that pixel's 2020 NAIP acquisition date,
  for each year 2013–2025.
- **Nearest scene.** A Landsat scene falls within 1–9 days of each NAIP
  flight in almost all cases. The `sierra_nf` window ends Aug 29, before the
  Creek Fire.
- **Sensors.** One set uses Landsat only (L30). The other adds Sentinel-2
  (S30), restricted to the six Landsat-equivalent bands.
- **Aggregation.** 30 m pixels are averaged onto Cheng's native 100 m grid
  (EPSG:5072). Only NLCD forest not burned in 2012–2019 is used, and a cell
  needs at least 70% of its pixels valid. That leaves about 160k cells.
- **Targets.**
  - % dead canopy, which is a **fraction** of the hectare: 0.0126 = 1.26%.
    Checked: density × median crown area reproduces it (ratio 0.98).
  - Dead-tree density (trees/ha, bias-corrected).
  - Red-stage ratio.
- **Levels differ widely by AOI.** Median dead canopy is 2% in `sierra_nf`,
  0.5% in `stanislaus` and about 0.2% in `lassen`. Median density is 15, 7
  and 4 trees/ha.
- **Features.** Year-2020 values only (`l30_2020`), or the full history
  (`*_hist`): the 2020 values plus the change from 2013 for each year
  2014–2020. **No information after 2020 is used.**
- **Model.** HistGradientBoosting.

**Results: R² and Spearman ρ, 5-fold 3 km spatial-block CV, scored within
each AOI** (`model_cv_per_aoi.csv`):

| Target | Features | `sierra_nf` | `stanislaus` | `lassen` |
|---|---|---|---|---|
| % dead canopy | ADS coverage 2012–19 | −0.14 / 0.18 | −0.17 / 0.25 | −3.2 / 0.19 |
| % dead canopy | Landsat 2020 only | 0.00 / 0.35 | −0.62 / 0.21 | −0.13 / 0.21 |
| % dead canopy | **Landsat history** | **0.42 / 0.66** | **0.25 / 0.47** | −0.03 / 0.25 |
| % dead canopy | Landsat + S2 history | 0.43 / 0.66 | 0.24 / 0.47 | −0.01 / 0.24 |
| % dead canopy | Landsat history, trained within the AOI | 0.43 / 0.67 | 0.31 / 0.54 | 0.10 / 0.35 |
| Density | **Landsat history** | **0.43 / 0.68** | **0.25 / 0.51** | 0.12 / 0.37 |

- **Pooled R² overstates skill.** Pooled over AOIs, block CV gives R² 0.57
  (ρ 0.70), but that is inflated by the level differences between AOIs.
- **Leave-one-AOI-out fails.** R² is negative everywhere, and ρ is
  0.15–0.51.
- **Red-stage ratio is poorly predicted.** Pooled R² is 0.25.

**Interpretation.**
- **Where mortality is substantial, HLS finds it.** In `sierra_nf` and
  `stanislaus`, Landsat explains a useful share of the 100 m pattern of
  standing dead canopy (R² 0.25–0.43, ρ 0.5–0.7).
- **The multi-year trajectory is what carries the signal.** The strongest
  single features are NDMI/NBR change from 2013 to 2016–2019 (ρ ≈ −0.5 in
  `sierra_nf`, −0.35 in `stanislaus`). 2020 alone does little, because
  standing dead in 2020 is mostly trees killed in 2015–2018.
- **Sentinel-2 adds nothing** beyond Landsat for this target.
- **ADS adds nothing** either, neither alone nor with Landsat.
- **`lassen` is at the noise floor.** Median dead canopy is about 0.2%.
  HLS NDMI drops there mostly mark sharp-edged patches that look like
  timber-harvest units. Canopy is removed, which Cheng does not count as
  standing dead. A harvest/treatment mask (e.g. USFS FACTS) is needed.
- **Models calibrated in one region don't transfer to another** (negative
  leave-one-AOI-out R²). Training must span the range of conditions.
  The statewide Cheng raster makes that possible without new NAIP
  processing.

### Multi-year test against Hemming-Schroeder et al. 2023 at NEON SOAP/TEAK (`hls_results/neon_hs/`)

**Setup.**
- **New AOI.** `neon_soap_teak` (33×19 km) went through the same pipeline:
  1,416 HLS granules, labels and composites.
- **Reference.** HS gives the cumulative fraction of trees dead per 30 m
  pixel for 2013, 2017, 2018, 2019 and 2021, from lidar-tracked crowns plus
  a spectral live/dead classification. It is on the Landsat C2 grid and was
  resampled (area-weighted) onto the HLS lattice.
- **Pixels used.** NLCD forest, at least 3 trees per pixel, and not burned
  in 2012..Y.
- **Mean dead fraction** rises 6.7% (2013) → 32% (2017) → 34% → 36% →
  41% (2021).
- **Features.** "Year-relative" Landsat L30 features at the 2020 NAIP day
  of year for any target year Y: X(Y), X(Y)−X(2013), X(Y)−X(Y−k) for
  k = 1–3, and the running min/max of X−X(2013). One model can therefore be
  applied to any year.
- **CV.** The test year is held out **and** so are the test 3 km blocks, so
  no pixel's other years or neighbours are in training.

**Results** (`neon_loyo.csv`):

| Target | 30 m R² (ρ) | 90 m R² (ρ) | 270 m R² (ρ) |
|---|---|---|---|
| Cumulative dead fraction, held-out year 2017 / 2018 / 2019 / 2021 | 0.54 / 0.57 / 0.52 / 0.49 (ρ 0.66–0.70) | 0.61–0.67 (ρ 0.72–0.77) | 0.61–0.71 (ρ 0.68–0.81) |
| New mortality between HS years | < 0 (ρ −0.17 to 0.12) | < 0 | < 0 |

- **A Landsat model transfers across years for cumulative mortality.**
  Skill is roughly constant for held-out years 2017–2021 (R² ≈ 0.5 at 30 m,
  ≈ 0.65 at 90–270 m).
- **Year-to-year increments are not resolved.** HS increments are only
  about 2–5 percentage points per interval, below the combined noise of
  HLS and the reference.
- **Strongest single features:** NDMI/NBR/RGI change from 2013, and the
  running extremes (|ρ| ≈ 0.45–0.56 at 30 m). Single-year changes are weak.

**Transfer from the Cheng-trained model, and agreement between the
references** (100 m, `transfer.csv`):
- **Landsat model trained on Cheng 2020 in the other three AOIs:**
  - applied to NEON at Y=2020 vs Cheng 2020: ρ = 0.47, for an AOI never
    seen in training;
  - applied at Y = 2017/2018/2019/2021 vs HS for the same year:
    ρ = 0.31 / 0.27 / 0.36 / 0.17.
- **The two references disagree.** Cheng 2020 vs HS gives ρ = 0.17
  (HS 2019) and 0.05 (HS 2021) at 100 m, and ≈ 0 at 3 km blocks.
  - Over the same 3 km blocks, HS vs Landsat ΔNDMI is ρ = −0.80, while
    Cheng vs ΔNDMI is only −0.23.
  - Magnitudes differ too: the median HS dead fraction of trees in 2019 is
    34%, while the median Cheng dead canopy in 2020 is 1.1%.
- **Likely explanation (to be checked).** They measure different things.
  - HS tracks every lidar-segmented tree from 2013, so trees killed in
    2015–16 stay "dead" even after needle loss or falling.
  - Cheng detects dead crowns visible in 2020 NAIP, which mostly misses
    older gray or fallen snags.
  - HS spectral live/dead thresholds may also inflate dead fractions.
- **Consequence.** The *target definition* (cumulative dead since a
  baseline vs currently visible dead crowns) matters more than the choice
  of Landsat features. A regular-cadence product needs one consistent
  definition, and a multi-year reference that matches it.

## Next steps

### 1. Independent 30 m reference from NAIP dead-tree mapping (highest priority)

ADS cannot say whether HLS detects mortality at 30 m. Its polygons are
generalized, positionally loose (see above), severity-inconsistent across
eras, and reconciled into one product. We need an independent reference at or
below the HLS pixel.

**What is available (checked 2026-09-25):**

- **Cheng et al. 2024** ([Nat. Commun. 15:641](https://doi.org/10.1038/s41467-024-44991-z))
  mapped individual dead crowns statewide from 2020 NAIP, 91.4 million dead
  trees. They report about 60% of dead trees occur in groups of three or fewer
  per 30 m cell, which is exactly what ADS misses.
- **Their code release** ([figshare 10.6084/m9.figshare.23723388](https://doi.org/10.6084/m9.figshare.23723388),
  CC BY 4.0) is saved at `/Volumes/Earth04/ecopro/cheng2024/DeLfoRS_TreeMortality/`.
  - It contains training, prediction and post-processing code: an
    EfficientUNet with an ordinal watershed for instance segmentation, bias
    correction, red/gray-stage classification in HSV, and hectare-level
    zonal statistics. It also has a GEE NAIP download notebook.
  - **It does not include trained model weights or the statewide 2020
    dead-tree map.**
  - The bundled example data are one 2020 NAIP quarter-quad
    (`m_3711928_sw_11_060_20200805.tif`) and about 1 km² of hand-digitized
    dead-crown polygons (2,546 crowns, dated 2022) near Bass Lake. That is
    just west of `sierra_nf` and does not overlap any AOI.
  - The environment is Python 3.9, PyTorch 1.12 with CUDA 11.3, tested on an
    RTX 3090 under Linux. Training or inference at scale needs a GPU machine;
    this Mac is not suitable.
- **Update, second search (2026-09-25).** More is available than the code
  release suggests:
  - **2020 statewide Cheng products** (verified; [figshare 10.6084/m9.figshare.24845742](https://doi.org/10.6084/m9.figshare.24845742),
    CC BY 4.0, EPSG:5072). Downloading to
    `/Volumes/Earth04/ecopro/cheng2024/statewide_2020/`:
    - bias-corrected dead-tree density (trees/ha, 100 m);
    - % dead canopy area per ha (100 m);
    - median dead-crown size (100 m);
    - red-stage ratio (100 m);
    - mortality ratio (240 m);
    - crown eccentricity (500 m).

    The individual-crown vectors are not released.
  - **Pretrained weights** (link verified in the repo README; not
    downloaded). The newer
    [Cross-Resolution-Dead-Tree-Segmentation](https://github.com/YanCheng-go/Cross-Resolution-Dead-Tree-Segmentation)
    repo links a Google Drive folder containing `BestModel.pth` (about
    376 MB) and `config.json`.
    - It is a resolution-aware UNet (resnet50), RGB input, trained on
      California 2020 NAIP (60 cm) plus 20–40 cm European imagery.
    - This is a successor to the 2024 paper's model, not the exact model
      used for it.
    - **No license** is stated on the repo or the Drive folder.
  - **USGS "Dead Tree Detection Validation Data from Sequoia and Kings Canyon
    National Parks"** ([10.5066/P9GYXCPG](https://doi.org/10.5066/P9GYXCPG),
    CC0). Field tree points (live/dead, species) for 2016–2020 NAIP years.
    It is outside our AOIs. Reported by the search agent; not yet opened.
  - **Hemming-Schroeder et al. 2023**
    ([Zenodo 10.5281/zenodo.7938442](https://doi.org/10.5281/zenodo.7938442),
    verified). Crown perimeters for over 1 M Sierra Nevada trees, 2013–2021,
    from NEON lidar plus multispectral data (2.66 GB). The EcoPro paper
    already cites this work.
    - Downloaded and extracted to `/Volumes/Earth04/ecopro/hemming_schroeder2023/`.
    - It covers only two NEON sites: SOAP (−119.30 to −119.23, 37.00–37.06)
      and TEAK (−119.08 to −118.97, 36.95–37.09). It **does not overlap our
      AOIs**; SOAP is about 22 km south of `sierra_nf`.
    - It includes **30 m mortality rasters for 2013, 2017, 2018, 2019 and
      2021** (`data/deliverables/raster/{soap,teak}_mortality_<year>.tif`,
      EPSG:32611, the same UTM zone and resolution as our HLS grid), plus
      tree-level shapefiles.
    - This is the only multi-year 30 m reference found so far. It is the
      natural test of whether a Landsat model transfers across years: add a
      small `soap_teak` AOI and fetch its HLS.
- **NAIP coverage of our AOIs**, from the Planetary Computer STAC: all three
  AOIs have 2012 (1 m), 2014, 2016, 2018, 2020 and 2022 (0.6 m, 4-band RGBN).
  All are summer acquisitions (late June to mid September). The 2016 and 2018
  acquisitions bracket the die-off peak and fall near ADS flight dates.

**Plan:**

1. **Use the released 2020 map now, and the pretrained model next.**
   - **2020 map.** The 100 m dead-tree density, % dead canopy and red-stage
     ratio give an immediate reference year. Compare them with HLS change
     2019→2020 and 2020→2021 aggregated to 100 m, and with ADS 2020/2021 at
     100 m to 1 km.
   - **Pretrained model.** Run `BestModel.pth` on NAIP 2014–2022 for our
     AOIs to get multi-year 30 m references. Check the license with Yan Cheng
     (chengyan2017@gmail.com, per the README) first. Inference needs a GPU.
   - **Model validation.** Validate the model's output locally on a small
     photo-interpreted sample (step 3) before trusting it in years other than
     2020.
   - **Constraint:** the Creek Fire (Sep 2020) burned about 68% of
     `sierra_nf`, so 2020 validation there is limited to unburned pixels.
     `stanislaus` and `lassen` are mostly unaffected in 2020; Dixie (2021)
     affects `lassen` only from 2021.
2. **If no weights, retrain.** Hand-digitize dead crowns in a few NAIP
   quarter-quads per AOI and year (their labelling protocol is in the paper),
   starting from the bundled example labels. Train with
   `train.treehealth_ordinal_watershed` on a GPU machine. Their paper
   pretrained with knowledge distillation from a Rwanda tree model
   (`train/rwanda2cali_regression_map.py`), which also needs weights we do not
   have. Expect days of labelling plus GPU time.
3. **Cheaper fallback that doesn't need the model.** Draw a stratified random
   sample of about 300–500 HLS 30 m cells per AOI-year, stratified by HLS
   change (e.g. deciles of flight-matched ΔRGI and ΔNDMI) and by ADS label.
   Photo-interpret the number of red/gray crowns and the dead-canopy fraction
   in each cell from NAIP 2014/2016/2018/2020/2022.
   - This directly estimates HLS commission and omission at 30 m, and ADS
     accuracy against the same reference.
   - Use the NAIP pair bracketing each HLS change interval (e.g.
     2014→2016, 2016→2018).
4. **Build the reference layer.** Aggregate dead-crown detections to the HLS
   30 m grid (dead-crown count, dead-canopy fraction, red vs gray stage). Use
   the per-AOI grid from `fetch_hls_aoi.aoi_grid` so it aligns
   pixel-for-pixel with the composites and labels.
5. **Analyses:**
   - HLS Δ features against NAIP dead-canopy fraction at 30 m: regression,
     plus detection AUC at thresholds of 1, 3 and 5 dead crowns per cell.
   - ADS against NAIP at 30 m, 270 m and 1 km. This quantifies ADS omission,
     including scattered mortality, and commission.
   - Train the HLS model on NAIP-derived labels instead of ADS and repeat the
     leave-one-year-out and leave-one-AOI-out evaluation.
   - Stage timing: use the NAIP red/gray classification to learn which HLS
     lag (prev/at/next) captures each stage.

### 2. Other follow-ups

- **Flight-matched analysis on all three AOIs.** This is queued behind the
  fetch (`hls_results/all_flight/`). Also check the ±20-day window: try ±10
  and ±30 in the dense S30 years (2018+).
- **Multi-year trajectory features**, e.g. the maximum RGI or the minimum NDMI
  over Y−1..Y+1 relative to baseline. These would be robust to where a stand
  is in the red→gray sequence.
- **Harmonized severity target.** Legacy TPA and DMSM percent-affected are
  not comparable. Model them separately, or map both to an expected dead
  fraction with care.
- **Raw DMSM per-surveyor data.** Request data including
  `FEATURE_USER_ID`/`OBSERVATION_USER_ID` from USFS R5 / FHAAST to assess
  inter-observer agreement. The public flat file is reconciled and has no
  independent duplicates.
- **Add 2022–2024 ADS years** to any paper-model retraining. They are in the
  geodatabase but missing from the local gpkgs.
- **Smaller fires.** MTBS misses fires under 1,000 ac. Consider CAL FIRE FRAP
  perimeters; the site blocks scripted access, so use the Chrome plugin.
