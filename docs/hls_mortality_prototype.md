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

### Running the Cheng model locally on NAIP (`hls_results/cheng_naip_replication/`)

**Setup.**
- **Code.** `src/proto_cheng_naip_inference.py`, in the `deadtree` conda env:
  conda-forge PyTorch 2.10 with MPS, and segmentation-models-pytorch 0.3.4.
- **Weights.** The public cross-resolution weights
  (`cheng2024/cross_resolution_model/BestModel.pth`). This is the
  *successor* of the model behind the 2020 statewide map, whose weights are
  unpublished.
- **Reimplementation.** The network head and inference follow the repo
  (cloned at `cheng2024/Cross-Resolution-Dead-Tree-Segmentation`, commit
  fd8af12):
  - RGB/255 with ImageNet normalization;
  - pixel size 0.6 m as the scalar input;
  - 256 px patches with 10 px edges dropped;
  - ordinal energy levels, then watershed to get crown instances.
- **Checks.** The checkpoint loads with `strict=True`. Zoomed QC confirms
  detections are on gray crowns with no NIR response (dead). QC images are
  in `hls_results/cheng_naip_test/`.
- **Speed.** On an M1 Pro (MPS), one 0.6 m quarter-quad (about 12,500 ×
  10,200 px) takes about 50 s of inference plus about 12 s of watershed.
- **Test area.** Three 2020 NAIP quarter-quads in `sierra_nf` (Aug 3–4,
  2020), with Cheng map mean dead canopy of 3.1%, 2.4% and 1.5%. That is
  about 350k detected crowns. Crowns are aggregated by centroid (count) and
  area to Cheng's 100 m grid, using only cells fully covered by valid NAIP.

**Agreement with the published 100 m map:**

| | Dead-tree density | Dead-canopy fraction |
|---|---|---|
| 100 m, n = 12,966 cells (per quad) | r = 0.79 (0.73–0.85) | r = 0.81 (0.73–0.89) |
| 300 m, n = 1,173 | r = 0.88 | r = 0.91 |
| Mean ratio, ours ÷ map | 1.29 | 1.96 |

**Interpretation.**
- **The spatial pattern is reproduced well.** The successor model and
  simple aggregation give r ≈ 0.8 at 100 m and about 0.9 at 300 m.
- **The absolute level is higher.** We find about 1.3× more crowns, and
  crowns about 1.5× larger (median 12–16 m² vs about 9 m² in the map's
  crown-size layer), giving about 2× the dead-canopy fraction.
- **Our run is not bias-corrected.** The published density *is*
  bias-corrected, and the paper reports a 17–25% underestimation vs field
  data, so correction should raise counts. Our higher counts therefore come
  from the newer model, not from missing correction.
- **Use with other years.** To pair a model-derived multi-year reference with
  the 2020 map, calibrate linearly on 2020 (or use the model consistently
  across all years and ignore the published map's absolute level).
- **Practical implication.** Running the model on NAIP 2014–2022 for all
  AOIs is feasible on this laptop: about 1 min per quarter-quad, so the
  roughly 810 AOI quarter-quad-years take about 15 h, excluding download
  time. That gives a biennial, consistent "visible dead crowns" reference
  for training and validating a Landsat model across years.

### Cheng model vs independent tree-level references (`hls_results/cheng_vs_hs_labels/`, `hls_results/cheng_vs_seki/`)

**Data access.**
- NAIP was fetched as 384 m chips (640 px) around each reference tree or
  site (`src/fetch_naip_chips*.py`; see `naip_fetch.md`). Most came via
  Google Earth Engine, about 14× faster than the Planetary Computer.
- The model was run with `proto_cheng_naip_inference.py --by-year --no-cells`.
- Scoring scripts: `proto_cheng_vs_hs_labels.py` and `proto_cheng_vs_seki.py`.

**References.**
- **Hemming-Schroeder (HS) 2017 hand labels at NEON SOAP/TEAK.** 8,897
  lidar crowns photo-interpreted as live or dead from about 1 m NEON imagery
  (2,516 dead). Every tree is labeled in 1,572 sampled 30 m pixels. NAIP
  2016 and 2018 bracket the label date.
- **USGS SEKI field data (Das 2024).** GPS-located trees with field status:
  - 2020 transects (every tree > 40 cm): 290 dead, 962 live;
  - north 2016 points: 98 dead, 92 live;
  - roadside south 2020 points: 32 dead, 135 live;
  - 2019 crown polygons (TAOs): 56 dead, 298 live.

**Results** (detection = any model dead-crown pixel within the tolerance):

| Reference | Sampling | Dead detected | Live with a detection | AUC |
|---|---|---|---|---|
| HS hand labels, exact crown overlap (2016 / 2018 NAIP) | complete within sampled pixels | 44% / 45% | 7% / 8% | 0.69 |
| HS hand labels, within 2 m (high certainty only) | 〃 | 52–54% (60%) | 13–15% (16%) | 0.69–0.73 |
| SEKI 2020 transects, within 2 / 5 m | systematic, all trees > 40 cm | 47% / 65% | 6% / 20% | 0.72 |
| SEKI north 2016 points, within 2 / 5 m | hand-picked | 61% / 81% | 2% / 5% | 0.88 |
| SEKI south 2020 roadside, within 2 m | hand-picked | 97% | 0% | 0.93 |
| SEKI 2019 crown polygons, any hit (2018 / 2020 NAIP) | hand-picked TAOs | 80% / 64% | 21% / 18% | 0.85 / 0.76 |

At 30 m, the labeled dead fraction vs the model's dead-canopy fraction in
HS sample pixels gives ρ = 0.42–0.47.

**Interpretation.**
- **Sample design drives the scores.** Systematic or complete samples (SEKI
  transects, HS sample pixels) give AUC ≈ 0.70–0.72 and find about 55–65%
  of dead trees. They include small, partly occluded and ambiguous trees.
  Hand-picked samples favour conspicuous trees and give AUC 0.85–0.93.
- **The HS hand labels behave like field truth for this purpose.** Their
  agreement with the model matches the systematic field transects, so they
  are *not* obviously noisier. (An earlier note here suggested they were;
  the positional-tolerance test does not support that.)
- **Positional tolerance does not help.** Allowing 2–5 m finds more dead
  trees but flags proportionally more live ones, so AUC is flat. NAIP
  misregistration is not the main limit: the measured NAIP-vs-lidar offset
  at NEON is only about 1.3 m (median; p90 2.5 m), and correcting it
  changes the scores negligibly (see "Alignment of HS crowns with NAIP"
  below).
- **Dead-foliage vs bare trees.** Among dead trees on the SEKI transects,
  those still holding dead foliage (F1–F3) and bare ones (T0–T3) are
  detected at similar rates, 64–65% within 5 m.
- **Using the model as a reference for HLS.** It undercounts dead trees by
  roughly a third in dense mixed conifer, consistent with the 17–25%
  underestimation Cheng et al. report after bias correction. For a
  cumulative-mortality target, calibrate with the systematic references
  (HS pixels, SEKI transects) rather than treating model counts as
  absolute.

### Alignment of HS crowns with NAIP (`hls_results/hs_naip_alignment/`)

HS crown polygons overlaid on NAIP can look visibly offset from the crowns
in the image, which raises the question of whether the crown-level HS
scores above can be trusted. `proto_hs_naip_alignment.py` measures the
offset directly on the NEON chips (`naip_chips/neon_soap_teak/{2016,2018}`)
and the model energy in `hls_results/cheng_naip_neon/`.

**Checks.**
- **CRS.** With PROJ's network grids off (the default), EPSG:32611 → 26911
  is a 0.0 m shift in pyproj (WGS84 and NAD83 treated as identical). The
  true datum difference, about 1 m, is below the other effects here. With
  `PROJ_NETWORK=ON` (set by some conda activation scripts) pyproj applies a
  grid shift of about 0.8 m at NEON, which changes the per-chip offsets
  slightly; the numbers below are with the network off.
- **HS polygons vs HS lidar treetops.** The 2017 crown polygons are unbiased
  relative to their own treetops (`xlas2013`/`ylas2013`: median dx −0.08 m,
  dy 0.0 m; median distance 1.2 m), and their area matches `ca2017`.
- **NAIP vs lidar, rigid offset per chip** (`--step register`). NAIP NDVI
  is cross-correlated with a canopy mask of 1.01 M HS treetops (disks of
  `ca2013`) over ±15 m.
  - About 30% of chips have a reliable peak (peak-to-sidelobe ratio ≥ 6).
    There the offset is **median 1.3 m, p90 2.5 m, mostly about +1.2 m east
    and −0.6 m north** (NAIP relative to lidar), about 2–3 NAIP pixels.
  - It is consistent within each NAIP quarter-quad (sd 0.2–1.5 m).
  - Low-PSR chips give noisy offsets; 14–23 of them per year hit the ±15 m
    search limit.
- **Height dependence** (`--step height`). Short (5–15 m), medium
  (15–30 m) and tall (> 30 m) trees register within about 0.6 m of one
  another (p90 ≤ 1.3 m). **There is no meaningful relief displacement.**
- **Model dead mask vs HS crowns, stacked over all chips** (`--step
  stacked`). Model dead pixels are **4.6× (2016) and 5.5× (2018) enriched
  on HS dead crowns**, peaking at (+1.2, −0.6) m, the same offset as the
  NDVI check. Live crowns are depleted (0.37–0.50×). About half the excess
  lies within 5 m of the peak, roughly one dead-crown radius.

**Re-score with the offset corrected** (`--step rescore`;
`rescore_crowns.csv`, `rescore_pixels.csv`). `none` reproduces
`cheng_vs_hs_labels/crown_metrics.csv`; `chip` shifts each crown by its
chip's offset (the quad median where PSR < 6); `quad` shifts every crown by
the quad median. The same correction is available in
`proto_cheng_vs_hs_labels.py --shift chip|quad`.

| NAIP | Offset | AUC (all) | AUC (high certainty) | Dead detected | Live flagged | 30 m ρ |
|---|---|---|---|---|---|---|
| 2016 | none | 0.695 | 0.738 | 43.6% | 6.6% | 0.459 |
|  | per chip | 0.699 | 0.741 | 44.3% | 6.4% | 0.456 |
|  | per quad | 0.698 | 0.741 | 44.1% | 6.5% | 0.456 |
| 2018 | none | 0.699 | 0.721 | 45.4% | 7.5% | 0.465 |
|  | per chip | 0.706 | 0.726 | 46.0% | 6.8% | 0.468 |
|  | per quad | 0.704 | 0.725 | 45.8% | 7.0% | 0.468 |

**The change is negligible:** AUC rises by at most 0.006, dead detected by
under 1 percentage point, and the 30 m sample-pixel ρ changes by ≤ 0.004.

**Interpretation.**
- The systematic NAIP-vs-lidar misregistration at NEON is about 1–2 m.
  The crown-level HS results stand, and offsets of this size do not matter
  at aggregated (30 m and coarser) scales.
- The apparent misalignment when HS polygons are drawn on NAIP comes from
  several things together:
  - lidar-segmented crown perimeters do not match visible crown edges; NAIP
    crowns include a sunlit side and shadows, and HS treetops often fall in
    shadow;
  - many trees labelled dead in 2017 had shed foliage or fallen by 2018, so
    they appear in NAIP as gaps rather than gray crowns;
  - windows chosen to show dense dead clusters are where polygons overlap
    most.
- **Still open:** the same check for the SEKI field points (GPS error plus
  NAIP registration), and whether registration differs between SOAP and
  TEAK.

### Comparison with published accuracy of the Cheng et al. models

**Published results** for the original 2024 model ([Cheng et al. 2024, Nat.
Commun. 15:641](https://doi.org/10.1038/s41467-024-44991-z)):
- **Hand-digitized NAIP trees (about 3,000):** dead-tree IoU 0.53, count
  bias −3.6%, MAE 2.27 dead trees/ha.
- **Field point locations (2016–2020):** count underestimation 16.7–24.7%,
  i.e. roughly 75–83% of dead trees found.
- **Field plots (DBH > 40 cm, 2016 and 2018):** underestimation 5–20%, MAE
  2.2–2.9 trees/ha.
- **2020 model applied to 2016, 2018 and 2022:** underestimation 16.7–53.1%,
  highest in 2022.
- **Direct benchmark in the USGS release**
  (`usgs_seki_deadtree_validation/MCV2018NorthValidationNAIP.csv`, original
  model on 2018 NAIP, tree counted as detected if a predicted crown with a
  6 m buffer overlaps it):
  - **69% of 197 field-dead trees** detected;
  - by plot 17–89%;
  - by species: ABCO 59%, ABMA 58%, PILA 91%, CADE 100%.

**No published accuracy exists for the newer cross-resolution weights we
use.** Their only citation is the software release
([Zenodo 10.5281/zenodo.17234915](https://doi.org/10.5281/zenodo.17234915)).
Möhring et al. 2025 (ISPRS Open J.), co-authored by Cheng, describes a
different, global deadwood model.

**Side by side:**

| Test | Newer model (this work) | Original model (published) |
|---|---|---|
| Systematic transects, all trees > 40 cm | 65% of dead within 5 m, 72–83% within 10 m (2020) | 69% within a 6 m buffer (2018, same USGS transect survey) |
| Hand-picked field points | 81–97% within 5 m | about 75–83% (16.7–24.7% undercount) |
| Hand labels | 44–53% of dead crowns hit by exact overlap; AUC ≈ 0.7 (independent HS labels) | IoU 0.53 vs their own NAIP labels |
| Dead-tree density vs the 2020 map | +29% | map is bias-corrected upward |
| Years other than 2020 | no drop for 2016 or 2018 (NEON) | 17–53% undercount |

**Interpretation.**
- **Comparable performance.** On the same kind of field data, the newer
  model performs about like the original: roughly two-thirds of dead trees
  found in systematic plots, and 80–95% of conspicuous ones. It does not
  obviously degrade for 2016 or 2018.
- **The SEKI data is not independent of Cheng's work.** It was collected to
  validate the original NAIP model, so our SEKI scores re-test on the
  authors' own validation data. The HS hand labels are the independent
  check, and they give the lower end of the range.
- **Count bias understates per-tree error.** Omissions and commissions
  partly cancel in count bias. On the transects, 15–20% of live trees have a
  detection within 5 m, while about a third of dead trees are missed. That
  is why a −4% to −20% count bias coexists with 65–80% per-tree detection.
  For HLS training, calibrate model-derived references against the
  systematic samples rather than relying on the published bias figures.

### WDTS airborne imaging spectroscopy (`hls_results/wdts/`, `hls_results/master_quicklook/`)

**What WDTS is.** [WDTS](https://www.earthdata.nasa.gov/data/projects/wdts/data-access-tools)
is NASA's Western Diversity Time Series. It is ER-2 airborne imagery of fixed
California flight boxes, and before 2020 it was the HyspIRI campaign. It is
**not a mortality dataset** and has no dead-tree labels. So the question
tested here is whether airborne imaging spectroscopy adds skill over HLS
against our existing references.

Products checked (all at ORNL DAAC, with the same Earthdata `.netrc` auth):

| Product | Resolution, years | Notes |
|---|---|---|
| AVIRIS-C foliar-trait mosaics ([2403](https://doi.org/10.3334/ORNLDAAC/2403), `WDTS_AVIRIS-C_foliar_traits_2403`) | 30 m COG, 2013–2018, 2–4 dates/yr | **Used.** 14 PLSR traits (LMA, N, chlorophyll, …) with mean and sd, plus QC bands. The grid coincides with the HLS AOI grids |
| AVIRIS-C traits by flight line ([2454](https://doi.org/10.3334/ORNLDAAC/2454)) | 15 m ENVI | Not used |
| AVIRIS-C corrected reflectance ([2391](https://doi.org/10.3334/ORNLDAAC/2391)) | 15 m, 224 bands, ENVI BSQ | Not used. About 60 GB per line and ~16 TB over our AOIs; remote windowed reads do work |
| MASTER L1B/L2 ([1940](https://doi.org/10.3334/ORNLDAAC/1940), 1953, 2141, 2252, 2383, 2471) | ~50 m, 50 bands VNIR–TIR, 2020–2025 | Quick look only (Fall 2020) |

**Coverage.** Each AOI maps to a WDTS flight box:
- **`sierra_nf` and `neon_soap_teak`:** inside the Yosemite box, with 100% footprint on every early-summer date 2013–2018.
- **`stanislaus`:** in the Tahoe box, 15 dates.
- **SEKI:** only about 15–20%.
- **`lassen`:** none.

**Fetch.** `fetch_wdts_traits.py` pulls one early-summer date per year:
`20130612_v2, 20140603, 20150601_v2, 20160621, 20170607, 20180622`. The
`_v2` files are the radiometrically stabilized versions that the user guide
recommends for between-year use. Output goes to `wdts/<aoi>_traits.nc`.

**Mosaic structure.** The mosaics are aggregated from 15 m. Each QC band is
the fraction of 15 m subpixels that pass, and the trait mean is nodata
wherever no subpixel passes.
- In forest, 8–33% of pixels are masked, depending on year.
- Masked pixels have `QC_fc` ≈ 0, meaning green-vegetation cover below 0.5. That includes dead canopy.
- So the trait maps drop exactly the pixels that matter most. The QC fractions (`qc_all`, `qc_fc`, `qc_shadow`) are kept as features instead.

**Method.** `proto_hls_vs_wdts.py` compares three feature sets on identical
samples and folds: `hls`, `wdts`, and `both`.
- **HLS features:** `year_features` on the L30 composites matched to the 2020 NAIP date.
- **WDTS features:** the same year-relative construction (raw, change vs 2013, d1–d3, cmin/cmax) applied to the 14 trait means and the 3 QC fractions.
- **Samples:** NLCD forest, unburned through Y. The model is HGB.
- **Sanity check:** HLS-only skill on the NEON target reproduces `neon_hs/` (ρ 0.68 / 0.75 / 0.79 at 30 / 90 / 270 m).

Results:

| Target | CV | hls | wdts | both |
|---|---|---|---|---|
| NEON HS dead fraction 2017/18 (ρ) | leave-one-year-out + blocks, 30 m | 0.68 | 0.46 | 0.69 |
| | same, 270 m | 0.79 | 0.55 | 0.72 |
| Cheng 2020 % dead, 100 m (R², ρ) | 5-fold 3 km blocks | 0.45, 0.68 (HLS 2018) | 0.42, 0.65 (2018) | 0.49, 0.70 |
| | same, HLS at 2020 | 0.47, 0.69 | | 0.50, 0.71 |
| ADS polygon fraction 2014–18 (AUC) | leave-one-year-out + blocks, 90 m | 0.52 | 0.57 | 0.57 |
| | leave-one-AOI-out, 90 m | 0.76–0.78 | 0.76–0.81 | 0.76–0.81 |

- **NEON (the best reference) favors HLS.** WDTS alone is clearly worse,
  and adding it to HLS gives nothing at 30 m and hurts at 270 m.
- **Cheng 2020 gains a little.** WDTS alone is close to HLS at the same
  date (R² 0.42 vs 0.45). Adding it gains about 0.04 R² under block CV, but
  it is not consistently better under leave-one-AOI-out: `neon_soap_teak`
  R² drops from 0.34 to 0.28 when WDTS is added to HLS 2018.
- **ADS remains unpredictable across years with either source.**
  Leave-one-year-out AUC is 0.5–0.63. This matches the earlier finding that
  ADS is not a pixel-level target. WDTS is marginally higher, but in a range
  where nothing works.
- **Single traits are weak.** The best WDTS changes against NEON are sugar
  (ρ 0.46), NSC (0.43) and starch (−0.38). HLS ΔNDVI/ΔNDMI/ΔRGI reach
  |ρ| 0.61–0.64 at 90 m. Against Cheng, the WDTS traits give |ρ| ≤ 0.34,
  while HLS ΔNDMI gives −0.49.
- **Why the traits underperform.** The mosaics are not consistent between
  years. The ΔLMA 2013→2016 map (`maps_sierra_nf_2016.png`) is dominated by
  north–south flight-line striping of ±60 g/m², with no visible ADS
  pattern. The green-cover pass fraction falls scene-wide from 81% (2013) to
  56% (2017). Only `qc_fc` shows the large disturbance patches that HLS
  ΔNDMI shows. The user guide warns about this: only the Yosemite `_v2`
  dates are calibrated for trends, and that covers just 2 of our 6 years. On
  top of that, the PLSR traits are fitted to live foliage, and dead canopy is
  masked.

**MASTER quick look.** `proto_master_quicklook.py` uses flight 2190600 on
2020-10-15, lines 01–04 over NEON SOAP/TEAK. The Creek Fire burned 68% of
`sierra_nf` in 2020, so that AOI was not used. The lines were flown through
the fire's smoke ("> 70% clear").
- **Processing:** TOA reflectance and 11.3 µm brightness temperature, binned
  into Cheng 100 m cells over forest unburned 2012–2020, taking the
  nearest-nadir line where lines overlap.
- **Agreement with HLS:** MASTER agrees with HLS 2020 for the same quantity
  (SWIR1 ρ 0.86; NIR 0.58, which is smoke-affected).
- **Against the NEON lidar reference, within each site,** single-date MASTER
  NDMI/NBR (ρ −0.65 SOAP / −0.41 TEAK against HS 2019) match single-date HLS
  NDMI (−0.67 / −0.39). HLS change since 2013 does better at SOAP (−0.73).
  Thermal BT also tracks mortality (ρ +0.53 SOAP, +0.23 TEAK). Correlations
  pooled across both sites are inflated by the SOAP/TEAK contrast.
- **Against Cheng 2020 over the whole AOI (42.6k cells, block CV):** HLS
  R² 0.49, MASTER 0.18, HLS+MASTER 0.51.

**Conclusion.** For our purposes, WDTS is a feature source, not a reference.
Its trait products add at most a few hundredths of R² over HLS for Cheng
2020, and nothing for the NEON lidar reference. The main limits are the
inconsistent between-year calibration and the masking of non-green canopy.
MASTER single-date features are about equivalent to single-date HLS. It is
not worth extending the trait analysis to `stanislaus` or to fall dates.
Two things might still pay off:
- the 15 m reflectance itself (e.g. unmixing into green, non-photosynthetic
  and soil fractions, SWIR cellulose/lignin features), bought at the ~60 GB
  per line cost;
- the AVIRIS-3/-5 collections (2023+) for recent mortality.

### Predisposition: do pre-mortality indicators predict the fraction of trees that die? (`hls_results/predisposition/`)

**Question.** Can indicators measured *before* mortality predict the
**fraction of trees in a 30 m (or 90 m) cell that go on to die**, beyond
stand structure, topography and climate? We don't try to identify
individual trees; the tree-level data only supply the per-cell target.

**Data.** The Hemming-Schroeder tree-level release: about 1M lidar crowns at
SOAP/TEAK with a live/dead status per year (relative greenness in NEON
imagery) and per-tree covariates.

**Targets** (cells need at least 5 cohort trees; models are weighted by
that count):
- **Cohort A, the 2015–17 die-off:** the fraction of trees live in 2013 that
  were dead in both 2017 and 2018. Trees with inconsistent labels are
  dropped. 69.7k cells at 30 m; mean 0.33 (SOAP 0.49, TEAK 0.23).
- **Cohort B, the 2020–22 wave:** the fraction of trees live in 2018–19 that
  were dead in 2021. 57.2k cells; mean 0.12 (SOAP 0.35, TEAK 0.06).

**Features** (cell means over the 2013 lidar trees):
- **S, stand/site:**
  - tree count, mean / p90 / max 2013 height, fraction of trees over 30 m,
    crown area, fraction already dead in 2013;
  - tpa, neighbour distance, cover, elevation, slope, aspect, climate
    normals, granite, distance to rivers, site.
- **H:** HLS L30 NAIP-date composites.
- **W:** WDTS traits.
- **C:** canopy water from the WDTS 15 m reflectance (below).
- **M:** MASTER 2020-10-15.

Cohort A uses 2013, 2014 and Δ2014−2013. The 2015 (June) data, taken as the
die-off began, is added separately. Cohort B uses 2018–2020.

**Canopy water content from WDTS** (`fetch_wdts_cwc.py`, output
`wdts/neon_soap_teak_cwc.nc`):
- **Input:** the 2391 corrected reflectance (15 m, 224 bands). Only the AOI
  rows of about 45 bands (850–1270 nm, plus 660 nm) are read, with one HTTP
  range request per band (the files are BSQ). That takes about 30 s per
  flight line, versus a ~60 GB download per line.
- **Indicators:**
  - `ewt980`, `ewt1200`: equivalent water thickness from Beer–Lambert fits,
    ln R = c₀ + c₁λ − K_w(λ)·EWT, using PROSPECT-D water absorption, over the
    980 and 1200 nm liquid-water features (vapour-affected bands excluded);
  - `ndwi`: Gao 1996, 860/1240 nm;
  - `bd1200`: continuum-removed 1200 nm band depth.
- **Compositing:** 15 m pixels are aggregated 2 × 2 to the HLS grid. Each
  cell takes the nearest-nadir line from the date the trait mosaic used.
  Coverage is 99–100% except 2017 (78%). No seamlines are visible.
- **Sanity checks:**
  - the two EWT fits agree at ρ 0.98, and EWT vs HLS NDMI gives ρ 0.72–0.77;
  - median EWT980 over all AOI pixels is 0.155 cm (2013), 0.156 (2014),
    0.125 (2015) and 0.092 (2016), then recovers to 0.135 (2017) and 0.118
    (2018). Over NLCD forest pixels it is 0.184, 0.181, 0.148, 0.109, 0.145
    and 0.134 cm. (An earlier version of this note labelled the all-pixel
    series "forest".) This is the canopy water loss reported by Asner et al.
    2016.

**Results.** Tree-weighted R² under 1 km block CV, 30 m / 90 m cells:

| Features | Cohort A | Cohort B |
|---|---|---|
| S | 0.499 / 0.654 | 0.454 / 0.573 |
| H alone / W alone / C alone | 0.389 / 0.557, 0.349 / 0.557, 0.250 / 0.400 | 0.334 / 0.446, 0.293 / 0.437, 0.254 / 0.362 |
| S + C (2013–14; B: 2018) | 0.507 / 0.659 | 0.460 / 0.577 |
| S + C incl. 2015 (level + Δ2015−2013) | 0.517 / 0.675 | — |
| S + W (B: 2018) | 0.507 / 0.665 | 0.463 / 0.604 |
| S + W incl. 2015 | 0.519 / 0.680 | — |
| S + H (B: 2019–20) | 0.535 / 0.694 | 0.467 / 0.595 |
| S + H incl. 2015 | 0.560 / 0.712 | — |
| S + H + C incl. 2015 | 0.562 / 0.715 | 0.481 / 0.602 (S+H+C2018) |
| S + H + W + C + 2015 | 0.561 / 0.714 | — |
| S + M / S + H + W + C + M | — | 0.685 / 0.809, 0.691 / 0.813 |

Spearman ρ tracks R². For cohort A at 30 / 90 m it is 0.60 / 0.69 for S and
0.66 / 0.74 for the full set. Run-to-run noise is about ±0.003 R², from
HGB's random early-stopping split. The 2015 isolation rows are from
`leadtime_2015_cv.csv`. Leave-one-site-out R² is negative for every
feature set because base rates differ (SOAP 49% vs TEAK 23%). Absolute
levels do not transfer between sites; only rankings do.

- **Structure and site explain most of the predictable variance.** They
  give R² 0.50 at 30 m and 0.65 at 90 m.
  - Within site × 200 m elevation strata, mean tree height is the strongest
    single predictor (ρ 0.28 / 0.33). Stands of taller trees lose a larger
    fraction.
  - In cohort B the sign reverses (ρ −0.13 / −0.18). Tall stands had
    already lost their most susceptible trees.
- **Canopy water content adds little.**
  - It gains +0.008 R² at 30 m and +0.005 at 90 m over S with 2013–14 data,
    and +0.018 / +0.021 when June 2015 is included.
  - It adds nothing beyond HLS: S+H+H2015 gives 0.560 / 0.712, and adding
    CWC gives 0.562 / 0.715.
  - The single-feature signals match the literature:
    - higher pre-drought water content, i.e. denser, leafier canopies
      (EWT980 2013 ρ +0.15 / +0.23);
    - canopy water *loss* 2013→2015 (ρ −0.18 / −0.20). This is Brodrick &
      Asner's "progressive water stress".
  - But the early 2013→14 AVIRIS EWT change (ρ −0.08 / −0.10) is weaker than
    HLS ΔNDMI over the same interval (−0.20 / −0.29).
  - Caveat: our baseline is June 2013, the drought's second year. CAO's
    strongest results used 2011 as the baseline.
- **HLS is the most useful spectral source.** It adds +0.036 / +0.040 R²
  with 2013–14, and +0.061 / +0.058 with 2015. Its drying signal
  (ΔNDVI/ΔNDMI/ΔRGI 2013→14, within-strata |ρ| 0.20–0.23 at 30 m) is the
  strongest pre-mortality spectral indicator.
- **WDTS traits add +0.008 / +0.011** (+0.020 / +0.026 with 2015). The
  within-strata directions replicate Queally et al. 2025:
  - high LMA (ρ +0.24);
  - low N (−0.22);
  - high lignin, fiber and starch.
- **MASTER Oct 2020 is still best read as early detection.** Its gain in
  cohort B (+0.23 R²) comes from imagery taken after the 2020 beetle season,
  16 months after the 2019 status and 8 months before 2021.

**Answer.** At the pixel scale, pre-mortality spectral indicators add a
modest 0.04–0.06 R² over structure and site, and most of it comes from HLS.
- Canopy water from AVIRIS is physically meaningful and follows the expected
  trajectory, but it is largely redundant with HLS NDMI/SWIR at 30–90 m.
- A Sierra-wide predisposition map should therefore be built from:
  - stand structure (height, density; see `docs/stand_structure_datasets.md`);
  - climate and topography;
  - an HLS/Landsat drying signal, with a pre-2013 Landsat 5/7 baseline to
    capture the 2011→2013 onset.
- WDTS traits and CWC are secondary refinements at water-limited sites.

### Literature: remote-sensing indicators of predisposition to drought mortality

All citations below were checked against a publisher page or Crossref DOI
(2026-09). Numbers are from the papers' abstracts or text.

- **Canopy water content (CWC) is the best-supported pre-mortality spectral indicator.**
  - Asner et al. 2016 mapped progressive CWC loss 2011–2015 across
    California from CAO spectroscopy plus Landsat. More than 30% loss
    occurred on about 1 Mha.
  - Brodrick & Asner 2017 found that cumulative CWC loss 2011–2015 predicted
    2016 mortality. The 2014→15 loss predicted next-year mortality rising
    from about 1% to about 5%. The relationship is weak at low stress and
    varies by community.
  - Brodrick et al. 2019 extended CWC to 1990–2017 and mapped drought
    resistance.
  - Sapes et al. 2019 give the physiological basis: plant water content
    integrates hydraulic and carbon status.
  - Allen et al. 2026 caution that canopy structure can mask leaf drying, so
    single-date EWT is unreliable and time series are needed.
- **Foliar traits from WDTS.**
  - Queally et al. 2025 (GCB) used the WDTS 2014 AVIRIS-C traits at
    SOAP/TEAK against Stovall and Hemming-Schroeder mortality.
  - At SOAP, mortality rose with height, LMA, leaf sugars and CWC loss, and
    fell with N. At TEAK, elevation and climate dominated, with little role
    for traits. Site R² was 0.31–0.55.
  - Shen et al. 2025 discuss that study.
  - The WDTS trait-change paper (Zheng et al., "in preparation for PNAS")
    had not been published as of 2026-09.
- **Structure, site and climate.**
  - Stovall et al. 2019: mortality risk rose 1.26× per 10 m of tree height,
    amplified by VPD.
  - Stephenson & Das 2020: the height effect largely reflects pines
    dominating the tall classes.
  - Hemming-Schroeder et al. 2023: 25.4% of trees died 2013–2017. Height,
    elevation, density and distance to rivers were the predictors.
  - Young et al. 2017: dry and dense stands died disproportionately.
  - Restaino et al. 2019; Koontz et al. 2021: host size interacts with
    climatic water deficit.
  - Paz-Kagan et al. 2017: mortality was higher at low elevation, on SW
    aspects and on shallow soils.
  - Goulden & Bales 2019: deep (5–15 m) subsurface moisture depletion drove
    the die-off.
- **Beetles.**
  - Stephenson et al. 2019; Trugman et al. 2021: fir engraver acts as a
    "stress compounder" that kills already-stressed firs, while *Dendroctonus*
    beetles take the largest pines regardless of stress. Spectral stress
    indicators should therefore work better for fir.
  - Erbilgin et al. 2021; Adams et al. 2017: NSC depletion is inconsistent
    and follows beetle attack. Foliar NSC is a weak warning signal.
  - Fettig et al. 2019 give the scale of pine mortality.
- **Satellite early-warning signals.**
  - Liu et al. 2019: lag-1 autocorrelation of Landsat NDVI warned more than
    6 months ahead in 75% of cases. Skill decays with lead time and at fine
    scale (AUC 0.61–0.71 at 1/8°), and depends on species.
  - Byer & Jin 2017: higher pre-drought productivity predicted MODIS-scale
    mortality.
  - Rogers et al. 2018; Keen et al. 2022: multi-year early-warning signals
    in the boreal and in tree rings.
  - Forzieri et al. 2022; Smith et al. 2022: global resilience-loss
    indicators, not validated at stand scale.
  - Kunik et al. 2026: satellite solar-induced fluorescence (SIF) fell about
    2 years before beetle mortality.
  - Yang et al. 2021: thermal ET stress precedes mortality.
  - Ganz et al. 2025 (preprint): mortality forecasts barely beat
    spatial-autocorrelation baselines. Validate against naive
    neighbourhood models.
- **Detection, not prediction.** Tane et al. 2018 and Huesca et al. 2021
  map red-stage mortality with imaging spectroscopy. Coates et al. 2015
  relate AVIRIS green-vegetation fraction to MASTER surface temperature.

#### References

1. Adams HD et al. (2017) A multi-species synthesis of physiological mechanisms in drought-induced tree mortality. Nat Ecol Evol 1:1285–1291. https://doi.org/10.1038/s41559-017-0248-x
2. Allen J, Anderegg LDL, Roberts D, Trugman AT (2026) Detecting drought stress from the leaf to the landscape. New Phytol 251(6):3256–3270. https://doi.org/10.1111/nph.71450
3. Asner GP et al. (2016) Progressive forest canopy water loss during the 2012–2015 California drought. PNAS 113(2):E249–E255. https://doi.org/10.1073/pnas.1523397113
4. Brodrick PG, Asner GP (2017) Remotely sensed predictors of conifer tree mortality during severe drought. Environ Res Lett 12:115013. https://doi.org/10.1088/1748-9326/aa8f55
5. Brodrick PG, Anderegg LDL, Asner GP (2019) Forest drought resistance at large geographic scales. Geophys Res Lett 46(5):2752–2760. https://doi.org/10.1029/2018GL081108
6. Byer S, Jin Y (2017) Detecting drought-induced tree mortality in Sierra Nevada forests with time series of satellite data. Remote Sens 9(9):929. https://doi.org/10.3390/rs9090929
7. Coates AR, Dennison PE, Roberts DA, Roth KL (2015) Monitoring the impacts of severe drought on southern California chaparral species using hyperspectral and thermal infrared imagery. Remote Sens 7(11):14276–14291. https://doi.org/10.3390/rs71114276
8. Erbilgin N et al. (2021) Combined drought and bark beetle attacks deplete non-structural carbohydrates and promote death of mature pine trees. Plant Cell Environ 44(12). https://doi.org/10.1111/pce.14197
9. Fettig CJ, Mortenson LA, Bulaon BM, Foulk PB (2019) Tree mortality following drought in the central and southern Sierra Nevada. For Ecol Manage 432:164–178. https://doi.org/10.1016/j.foreco.2018.09.006
10. Forzieri G et al. (2022) Emerging signals of declining forest resilience under climate change. Nature 608:534–539. https://doi.org/10.1038/s41586-022-04959-9
11. Ganz K et al. (2025) Spatially explicit forest mortality forecasts are driven by autocorrelation, not ecological context. bioRxiv (preprint). https://doi.org/10.1101/2025.11.19.689366
12. Goulden ML, Bales RC (2019) California forest die-off linked to multi-year deep soil drying in 2012–2015 drought. Nat Geosci 12:632–637. https://doi.org/10.1038/s41561-019-0388-5
13. Hemming-Schroeder NM et al. (2023) Estimating individual tree mortality in the Sierra Nevada using lidar and multispectral reflectance data. JGR Biogeosci 128:e2022JG007234. https://doi.org/10.1029/2022JG007234
14. Huesca M et al. (2021) Detection of drought-induced blue oak mortality in the Sierra Nevada Mountains, California. Ecosphere 12(6):e03558. https://doi.org/10.1002/ecs2.3558
15. Keen RM et al. (2022) Changes in tree drought sensitivity provided early warning signals to the California drought and forest mortality event. Glob Change Biol 28(3):1119–1132. https://doi.org/10.1111/gcb.15973
16. Koontz MJ et al. (2021) Cross-scale interaction of host tree size and climatic water deficit governs bark beetle-induced tree mortality. Nat Commun 12:129. https://doi.org/10.1038/s41467-020-20455-y
17. Kunik L et al. (2026) Characterizing effects of tree mortality from wildfire and bark beetles using satellite observations of solar-induced chlorophyll fluorescence. Remote Sens Environ 344:115550. https://doi.org/10.1016/j.rse.2026.115550
18. Liu Y, Kumar M, Katul GG, Porporato A (2019) Reduced resilience as an early warning signal of forest mortality. Nat Clim Change 9:880–885. https://doi.org/10.1038/s41558-019-0583-9
19. Paz-Kagan T et al. (2017) What mediates tree mortality during drought in the southern Sierra Nevada? Ecol Appl 27(8):2443–2457. https://doi.org/10.1002/eap.1620
20. Queally N et al. (2025) Functional traits from imaging spectroscopy inform patterns of forest mortality during Sierra Nevada drought. Glob Change Biol 31(5):e70246. https://doi.org/10.1111/gcb.70246
21. Restaino C et al. (2019) Forest structure and climate mediate drought-induced tree mortality in forests of the Sierra Nevada, USA. Ecol Appl 29(4):e01902. https://doi.org/10.1002/eap.1902
22. Rogers BM et al. (2018) Detecting early warning signals of tree mortality in boreal North America using multiscale satellite data. Glob Change Biol 24(6):2284–2304. https://doi.org/10.1111/gcb.14107
23. Sapes G et al. (2019) Plant water content integrates hydraulics and carbon depletion to predict drought-induced seedling mortality. Tree Physiol 39(8):1300–1312. https://doi.org/10.1093/treephys/tpz062
24. Shafron E et al. (2025) WDTS: AVIRIS-Classic L2B Corrected and Georectified Surface Reflectance, 2013–2018. ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2391
25. Shen M, Dahlin K, Xu X, Butterfield Z (2025) Trait-based tree mortality risk assessment from the perspective of imaging spectroscopy. Glob Change Biol 31(7):e70337. https://doi.org/10.1111/gcb.70337
26. Smith T, Traxl D, Boers N (2022) Empirical evidence for recent global shifts in vegetation resilience. Nat Clim Change 12:477–484. https://doi.org/10.1038/s41558-022-01352-2
27. Stephenson NL, Das AJ (2020) Height-related changes in forest composition explain increasing tree mortality with height during an extreme drought. Nat Commun 11:3402. https://doi.org/10.1038/s41467-020-17213-5
28. Stephenson NL et al. (2019) Which trees die during drought? The key role of insect host-tree selection. J Ecol 107(5):2383–2401. https://doi.org/10.1111/1365-2745.13176
29. Stovall AEL, Shugart H, Yang X (2019) Tree height explains mortality risk during an intense drought. Nat Commun 10:4385. https://doi.org/10.1038/s41467-019-12380-6
30. Tane Z et al. (2018) A framework for detecting conifer mortality across an ecoregion using high spatial resolution spaceborne imaging spectroscopy. Remote Sens Environ 209:195–210. https://doi.org/10.1016/j.rse.2018.02.073
31. Trugman AT et al. (2021) Why is tree drought mortality so hard to predict? Trends Ecol Evol 36(6):520–532. https://doi.org/10.1016/j.tree.2021.02.001
32. Yang Y et al. (2021) Studying drought-induced forest mortality using high spatiotemporal resolution evapotranspiration data from thermal satellite imaging. Remote Sens Environ 265:112640. https://doi.org/10.1016/j.rse.2021.112640
33. Young DJN et al. (2017) Long-term climate and competition explain forest mortality patterns under extreme drought. Ecol Lett 20(1):78–86. https://doi.org/10.1111/ele.12711
34. Zheng T et al. (2025) WDTS: AVIRIS-Classic Derived Plant Trait Mosaics, 2013–2018. ORNL DAAC. https://doi.org/10.3334/ORNLDAAC/2403 (and per-flight-line traits, https://doi.org/10.3334/ORNLDAAC/2454)

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
   - **Multi-year test run.** The model was run on the same 1.2 km window
     (centre of quarter-quad `m_3711945_se`, the `cheng_naip_test` crop)
     cut from every `naip/sierra_nf/<year>` quarter-quad
     (`proto_cheng_naip_inference.py --by-year --no-cells --bounds ...`;
     about 25 s for all six years on MPS):

     | Year | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 |
     |---|---|---|---|---|---|---|
     | Dead crowns ha⁻¹ | 2.4 | 0.7 | 7.3 | 17.7 | 19.9 | 41.9 |
     | Dead canopy | 0.8% | 0.1% | 2.4% | 4.7% | 4.4% | 9.5% |

     - NAIP is 1 m in 2012 and 2014 and 0.6 m from 2016. The 2020 count
       (2,866 crowns) matches `cheng_naip_test`.
     - **The window burned completely in the Creek Fire** (ignition
       2020-09-05, after the 2020-08-03 flight): MTBS and CAL FIRE FRAP
       perimeters both cover all of it, and FACTS harvest polygons
       intersect it. The 2022 value is fire-killed trees plus salvage.
     - The 1 m years suggest a background (false-positive) rate of about
       1–2 dead crowns ha⁻¹.
   - **Masking for a multi-year reference.** Mask the NAIP-derived
     reference with MTBS plus CAL FIRE FRAP perimeters of all sizes (MTBS
     alone misses small fires) plus FACTS harvest and treatment polygons;
     this matters most for Creek (`sierra_nf`, from 2020) and Dixie
     (`lassen`, from 2021). Expect a background of about 1–2 dead crowns
     ha⁻¹ in the 1 m NAIP years (2012, 2014).
2. **(Superseded: public weights were found and run locally; see "Running the Cheng model locally" above.) If no weights, retrain.** Hand-digitize dead crowns in a few NAIP
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

### 2. Sierra-wide stand structure

Candidate pre-drought density and height layers for scaling the
predisposition model beyond NEON are documented in
`docs/stand_structure_datasets.md`. Of those, TreeMap 2014, LEMMA GNN 2012,
LANDFIRE 2014, GLAD height 2010/2015 and TCC 2012–13 are the pre-drought
candidates.

### 3. Other follow-ups

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
