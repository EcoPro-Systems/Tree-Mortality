# Drought response metrics, AVIRIS-C traits and trait dynamics

**Question.** Earlier tests (`hls_mortality_prototype.md`, "Predisposition")
found that AVIRIS-C traits and canopy water add only about +0.01–0.02 R² to
stand structure and site when the target is the per-cell **fraction of trees
that died**. These experiments give the traits a fairer test by addressing
the weaknesses that analysis identified:

| Weakness of the mortality test | What changes here |
|---|---|
| Mortality fraction is the noisiest response | Continuous Landsat response metrics: resistance, recovery, resilience, recovery time, stress response |
| Single-date traits | Trait and canopy-water **change** between June flights, 2013–2018 |
| 2013 is already drought year 2 | Landsat 5/7 baseline 2008–2011 |
| Between-year drift and view-angle striping | Cross-year-calibrated (`_v2`) 2013/2015 trait dates; per-line cross-track normalization; physics-based EWT |
| Dead canopy masked out of trait retrievals | Green-fraction QC kept as a feature |

It also inventories the airborne imaging spectroscopy flown over the study
boxes after 2018, which determines whether the 2020–2022 drought can be
analysed the same way.

## Data

| Dataset | Location (`/Volumes/Earth04/ecopro/...`) | Notes |
|---|---|---|
| Landsat C2 L2 summer and June composites | `landsat_composites/<aoi>_<variant>_doy{182-273,145-190}.nc` | From `fetch_landsat_c2_ee.py` (Earth Engine). 2008–2025. Variants: `c2` (L5/7/8/9), `c2l7` (Landsat 7 only, to 2022), `c2oli` (Landsat 8/9 only) |
| Env, terrain, structure, masks | `env/<aoi>_env.nc` | From `aoi_env_layers.py`. BCMv8 water-year CWD/AET/PET/PPT/Tmax and SPEI1–4 (2008–2024, nearest 270 m cell); local SRTM terrain indices (`topo/generated/`); GLAD height 2010, TCC 2010/2013, LANDFIRE 2014 EVH/EVC; NEON lidar tree summaries; NLCD 2013 forest; per-year fire (MTBS, CAL FIRE FRAP, CAL FIRE prescribed burns) and FACTS harvest/salvage, 2000–2025 |
| Fire and harvest polygons | `fire/disturbance/{frap_fires,rx_fires,facts_harvest}.gpkg`; `fire/disturbance_heldout/` for the held-out grids | From `fetch_disturbance_aois.py`: CAL FIRE FRAP perimeters (all sizes), CAL FIRE prescribed-fire perimeters, USFS FACTS timber-harvest activities (EDW) |
| Held-out site grids and masks | `config/heldout_aois.yml`; `env/{seki,yosemite_rest}_{env,mask}.nc`; `geom/seki_boundary.geojson` | From `fetch_heldout_masks.py` (2013 and 2018 trait-mosaic footprints, NPS SEQU/KICA boundary) and `aoi_env_layers.py --masks-only` (NLCD, terrain, fire and harvest only). Elevation strata: `config/heldout_strata.yml` (`proto_heldout_strata.py`); candidate variants in `hls_results/heldout_variants/`. Forest type: `env/{seki,yosemite_rest}_ftype.nc` (`fetch_forest_type.py ../config/heldout_aois.yml`) |
| AVIRIS flight-line inventory | `aviris_locator/AVIRIS-{C,NG}_flight_{table.csv,s.geojson}` | [ORNL DAAC 2140](https://doi.org/10.3334/ORNLDAAC/2140) flight tables (to Aug 2024) |
| WDTS traits and canopy water | `wdts/<aoi>_traits.nc`, `wdts/<aoi>_cwc.nc` | As before. `fetch_wdts_cwc.py` now also saves `nadir_dist`. `stanislaus` traits (Tahoe box, UTM 11) are warped onto the UTM 10 AOI grid with nearest neighbour; no canopy water there yet |
| Geology | `env/<aoi>_geology.nc`; source in `geology/` | From `fetch_geology.py`: USGS State Geologic Map Compilation, California (1:750,000), grouped into granitic, volcanic, metamorphic, surficial and other on the 30 m grids |
| Reflectance caches for emulated retrievals | `wdts/sim/<aoi>_refl<year>.nc`, `wdts/sim/<aoi>_refl_<yymmdd>_<product>.nc`, `wdts/sim/<aoi>_l8_<date>.nc` | From `proto_spaceborne_sim.py refl` (2013; 2018 for NEON and `sierra_nf`; 2013 for `stanislaus`), `proto_spaceborne_ts.py refl` (NEON and `sierra_nf`: 2013-05-03 and 2013-06-26 from ORNL DAAC 2391; 2018-06-22 and 2018-08-28 from ORNL DAAC 2154) and `fetch_landsat_c2_ee.py --scene` (2013-05-04, 05-20, 06-05, 06-21, 07-07, 07-23 (no clear pixels), 08-08; 2013-05-27 (no clear pixels), 06-12 and 06-28 Tahoe; 2018-06-19, 07-05, 07-21, 08-06, 08-22, 09-07) |
| Airborne lidar structure | `lidar/<aoi>_lidar.nc`; raw in `lidar/aso/`, `lidar/lvis2008/` | From `fetch_lidar_structure.py`: ASO 2014–17 composite (Ferraz et al. 2020) and LVIS Sep 2008 footprints on the 30 m grids. Coverage and sources: `lidar_coverage.md` |

## Airborne imaging spectroscopy over the study boxes, 2018–2025 (`hls_results/airborne_coverage/`)

`query_airborne_coverage.py` combines two sources:
- the AVIRIS Flight Line Locator tables, which list every line **flown**
  through Aug 2024;
- CMR granules of the public L2 reflectance collections (WDTS 2391,
  AVIRIS-Classic 2154, AVIRIS-NG 2110, AVIRIS-3 2357, AVIRIS-5 2484), which
  show which lines have **released reflectance**.

Coverage is the fraction of each AOI rectangle covered by the union of a
date's lines.

**Flown vs public L2 reflectance**, by AOI and date. Only dates covering at
least 30% of an AOI are shown; "–" means none.

| Year | `neon_soap_teak` | `sierra_nf` | `stanislaus` (Tahoe box) |
|---|---|---|---|
| 2018 | AVIRIS-C 06-22 (100% L2), 08-28 (100% L2) | AVIRIS-C 06-04 (47%), 06-22 (100%), 08-28 (91%) | AVIRIS-C 05-29 (89%), 06-08 (67%), 06-21 (100%) |
| 2019 | AVIRIS-C 08-02 (76%), 10-01 (100%) | AVIRIS-C 08-02 (34%), 08-13 (66%), 10-01 (100%) | – |
| 2020 | AVIRIS-C 10-15 flown 100%, **no L2**; 10-13 one line (49%, L2) | AVIRIS-C 09-24/25 and 10-15 flown 46–100%, **no L2**; 10-13 one line (32%, L2) | AVIRIS-C 09-24 flown 100%, **no L2** |
| 2021 | AVIRIS-C 03-29 flown 100%, **no L2** | AVIRIS-C 03-29 flown 100%, **no L2** | AVIRIS-C 03-30 flown 100%, **no L2** |
| 2022 | – | – | – |
| 2023 | AVIRIS-C 03-31 (33%) | AVIRIS-C 03-31 (99%) | AVIRIS-C 04-10 (89%) |
| 2024 | AVIRIS-C 06-04 (84%) | AVIRIS-C 06-04 (100%) | AVIRIS-C 06-21 (100%) |
| 2025 | AVIRIS-C and AVIRIS-5 07-17 (both 100%) | AVIRIS-C and AVIRIS-5 07-17 (both 100%) | AVIRIS-C and AVIRIS-5 07-15 (both 100%) |

- **The 2020–2025 WDTS spectrometer is AVIRIS-Classic**, flown with MASTER on
  the ER-2. No AVIRIS-NG or AVIRIS-3 line touches any AOI. CMR lists "WDTS"
  for 2020–25 only as MASTER collections, but AVIRIS-C L2 reflectance for
  2023–2025 is in the facility collection (2154).
  - The sensor is the same from 2013 through 2025, which removes most of the
    cross-sensor harmonization problem.
  - In July 2025, AVIRIS-5 flew the same boxes on the same days, which gives
    a same-day cross-sensor overlap.
- **The drought years are the gap.**
  - The only full-box flights in 2020–21 are post-season: Sep–Oct 2020 and
    late Mar 2021. Their L2 reflectance is not public; only one 2020-10-13
    line is.
  - Nothing was flown over any AOI in 2022.
  - The 2023 flights are early spring (Mar 31 / Apr 10). 2023 was a record
    snow year, so much of the box was likely snow-covered.
  - The first growing-season acquisitions after 2018 are June 2024 and July
    2025.
- **2018 pre-drought state:** all three AOIs have a full June 2018 AVIRIS-C
  acquisition with public L2 (`sierra_nf` and NEON 06-22, Tahoe 06-21). In
  the trait mosaics, 74–82% of forest pixels pass QC in 2018, against 86–90%
  in 2013.
- **Outputs:** `coverage_lines.csv` (every line × AOI, with overlap fraction
  and L2 collections), `coverage_dates.csv`, `trait_coverage.csv` and
  `coverage_map.png`.

## Methods

### Landsat composites

- **Masking:** QA_PIXEL (fill, dilated cloud, cirrus, cloud, shadow, snow,
  water), saturation, and haze/smoke: TM/ETM+ `SR_ATMOS_OPACITY` > 0.3 or a
  "high" OLI `SR_QA_AEROSOL` level. The haze mask matters. Without it, smoke
  from the 2015 Rough Fire next to SOAP/TEAK depressed Landsat 7 Jul–Sep NIRv
  at NEON by about 15%.
- **Cross-sensor bias:** TM/ETM+ are mapped to OLI with the Roy et al. (2016)
  coefficients. Even so, Landsat 7 forest NDMI stays about 0.015 below the
  L7+L8 mix in 2013–2019. The baseline (2008–2011) is L5/L7 only, so a mixed
  record would bias resistance by about a third of its spatial SD.
  - Cycle-1 metrics therefore use **Landsat 7 only** (`c2l7`), one sensor
    from baseline through recovery.
  - The 2017–2025 cycle uses **Landsat 8/9 only** (`c2oli`).
- **Check against HLS:** Landsat C2 Jul–Sep composites match HLS
  (`hls_composites/<aoi>_doy182-273.nc`) at ρ 0.96–0.99 per band and index
  in 2013–2014.

### Response metrics (`proto_response_metrics.py`)

Metrics are computed per cell at 30, 90 and 270 m, from Jul–Sep NDMI and NIRv.
Annual indices are first averaged to the cell.

| Metric | Definition |
|---|---|
| resistance | drought mean (2014–16) vs baseline (2008–11); `resistance_late` uses 2015–16 |
| recovery | post (2017–19) vs drought |
| resilience | post vs baseline |
| rectime | years after 2016 until the index returns within 1 SD of baseline (0 = never left; 4 = censored) |
| sens | stress response: per-cell OLS slope of the index anomaly on BCMv8 SPEI4, 2008–2019 |

- NIRv metrics are ratios. NDMI, which is near zero in open canopy, uses
  differences.
- **Noise floor ("placebo"):** the resistance formula applied inside the
  baseline, mean(2010–11) vs mean(2008–09), when there was no drought.
- **Cells:** forest (NLCD 2013 41/42/43) with Landsat data every year,
  excluding disturbance through the end of the period:
  - any fire 2000–2019 (MTBS, CAL FIRE FRAP of all sizes, or a CAL FIRE
    prescribed burn);
  - any FACTS harvest completed 2005–2019. "Natural Changes" records are
    mortality reports and are not counted. `--keep-salvage` keeps cells
    whose only harvests were salvage or sanitation cuts.
  - **Why this matters:** MTBS maps only fires ≥ 1000 ac. At NEON,
    prescribed burns (12% of forest pixels) and harvest (7%, mostly
    commercial thinning) produced one-year NDMI crashes that looked like the
    most "divergent" drought responses. The broader mask removes 14.9% of
    the MTBS-only sample at NEON and 2.6% at `sierra_nf`. The MTBS-only runs
    are kept in `hls_results/*_mtbs_only/` for comparison.
- **Feature blocks:**
  - **Env:** 1981–2010 CWD/PPT/Tmax normals; 2012–16 cumulative and 2014–16
    mean CWD anomaly; 2014–16 minimum SPEI4 and Tmax/PPT anomalies; 10 terrain
    indices.
  - **S:** GLAD height 2010, TCC 2010, LANDFIRE 2014 EVH/EVC.
  - **B1:** baseline NDVI and NIRv.
- **Models:** gradient boosting (`HistGradientBoostingRegressor`, fixed seed),
  out-of-fold predictions from 5-fold CV over 1 km spatial blocks. 95% CIs
  come from a block bootstrap (1 km and 5 km blocks) of the out-of-fold
  predictions. For R² *differences*, both models are resampled with the same
  blocks.

### Trait dynamics (`proto_trait_dynamics.py`)

- **Cross-track normalization.** Each year's trait and EWT maps are
  normalized per flight line: the value is regressed on a quadratic in
  across-track position plus a quadratic in elevation, and the across-track
  terms are removed. Along-track is the line footprint's first principal
  axis.
- **EWT metrics** (2013 base): resistance EWT2016/EWT2013; recovery
  mean(2017–18)/2016; resilience mean(2017–18)/2013.
- **Change features** for 2013→2015 (both `_v2`) and 2013→2014:
  - AVIRIS: ΔEWT, ΔLMA, ΔN, Δchlorophyll, Δgreen-fraction QC;
  - Landsat June: ΔNDMI, ΔNIRv, ΔNDVI.

### Nested trait models (`proto_response_traits.py`)

- **Ladder:** B1 | Env | Env+S | Env+S+T | Env+S+T+T14 | Env+S+W | Env+S+T+W |
  Env+S+Tres.
  - T: 2013 means of the 14 traits, the SD of LMA/N/chlorophyll, and the
    green-fraction QC. T14 adds 2014.
  - W: EWT 2013/2014.
  - Tres: each trait's out-of-fold residual from Env+S.
- **Conditional gains** within aridity terciles (CWD normal) and, at NEON,
  SOAP vs TEAK (the west and east halves of the AOI).
- SHAP importances and trait × CWD interaction strength.

### Cross-drought transfer (`proto_response_transfer.py`)

- **Cycle 1** (Landsat 7): baseline 2008–11, drought 2014–16, post 2017–19.
- **Cycle 2** (Landsat 8/9): baseline 2017–19, drought 2020–22, post
  2023–25.
- Env+S and B1 models are trained on one cycle and tested on the other, with
  drought-window features computed for each cycle. Cells are those never
  burned through 2025.

## Results

All numbers are from 90 m cells, with 95% block-bootstrap CIs over 1 km
blocks, unless noted. The 5 km-block CIs are wider but lead to the same
conclusions; both are in the CSVs. Cell counts are 44,079 at
`neon_soap_teak` and 58,676 at `sierra_nf`.

### 1. Same stress, different trajectories (`hls_results/response{,_sierra}/`)

**How much Env+S explains.** Out-of-fold R² for Env+S, with the 95% CI:

| Metric | NEON | `sierra_nf` |
|---|---|---|
| NDMI resistance | 0.725 [0.704, 0.744] | 0.554 [0.533, 0.575] |
| NDMI recovery | 0.683 [0.658, 0.705] | 0.529 [0.507, 0.548] |
| NDMI resilience | 0.634 [0.609, 0.655] | 0.604 [0.587, 0.619] |
| NDMI recovery time | 0.441 | 0.415 |
| NDMI stress response | 0.705 | 0.490 |
| NIRv resistance / recovery / resilience | 0.769 / 0.694 / 0.595 | 0.589 / 0.456 / 0.613 |

- **Structure matters on top of climate.** S adds +0.05 to +0.24 R² to Env
  for every metric at both AOIs; the CIs exclude 0 at 1 and 5 km blocks.
- **Baseline greenness (B1) adds little** on top of Env+S (+0.01 to +0.05).
  On its own it explains 0.07–0.41.

**Unexplained variance, by three measures:**

- **Within-bin share of variance.** With cells binned by CWD-anomaly quintile
  × 200 m elevation band × structure tercile, 36–59% of the variance is
  within bins at NEON and 67–80% at `sierra_nf`.
- **Noise-adjusted residual.** The Env+S residual variance minus the
  placebo noise, scaled for the number of years averaged, as a share of the
  total variance:

  | Metric | NEON | `sierra_nf` |
  |---|---|---|
  | NDMI resistance | 12% | 30% |
  | NDMI recovery | 17% | 39% |
  | NDMI resilience | 28% | 36% |

  The placebo includes real year-to-year variation, so these are lower
  bounds.
- **Spatial coherence.** The residuals are structured in space. The
  exponential-variogram nugget is 24–46% of the residual variance at NEON
  and 23–50% at `sierra_nf`, with practical ranges of 0.6–1.5 km. This is
  stand-scale structure, not pixel noise.

**NDMI is the better index for this analysis.** NIRv resistance's within-bin
SD is about equal to its placebo SD at NEON (0.057 vs 0.059), whereas NDMI
sits well above its placebo.

**Figures:**
- `bins_<aoi>_90m_ndmi_resistance_glad_h2010.png`: within-bin distributions
  against the placebo band.
- `pairs_<aoi>_90m_*.png`: matched-forcing neighbours (< 1 km, same bin) with
  divergent trajectories, and the Env+S residual map.
  - The example pairs exclude the 0.1–0.2% of cells with abrupt,
    stand-replacing NDMI crashes (one-year drop > 0.25, or > 0.2 below
    baseline by 2014, before the die-off peak).
  - These are disturbances missing from every fire and harvest record we
    have, including CAL FIRE Timber Harvest Plans. HLS confirms them
    independently.

### 2. Do traits add to Env+S? (`hls_results/response_traits{,_sierra}/`)

Gain in R² over Env+S, with the 95% CI:

| Target | NEON +T | NEON +Tres | `sierra_nf` +T | `sierra_nf` +Tres |
|---|---|---|---|---|
| NDMI resistance | +0.024 [0.020, 0.028] | +0.018 [0.013, 0.022] | +0.044 [0.038, 0.050] | +0.033 [0.028, 0.039] |
| NDMI resistance, 2015–16 only | +0.024 [0.019, 0.029] | +0.015 [0.010, 0.019] | +0.037 [0.033, 0.042] | +0.026 [0.022, 0.030] |
| NDMI recovery | **+0.065 [0.056, 0.074]** | +0.046 [0.038, 0.054] | **+0.089 [0.081, 0.097]** | +0.070 [0.063, 0.077] |
| NDMI resilience | +0.059 [0.050, 0.068] | +0.042 [0.034, 0.050] | +0.052 [0.046, 0.059] | +0.042 [0.037, 0.048] |
| NDMI recovery time | +0.044 [0.037, 0.053] | +0.031 [0.024, 0.039] | +0.036 [0.031, 0.042] | +0.027 [0.021, 0.032] |
| NDMI stress response | +0.022 [0.016, 0.028] | +0.014 [0.010, 0.019] | +0.055 [0.048, 0.062] | +0.042 [0.036, 0.048] |
| NIRv resistance | +0.020 [0.015, 0.025] | +0.011 [0.008, 0.015] | +0.040 [0.033, 0.047] | +0.031 [0.025, 0.037] |
| NIRv recovery | +0.046 [0.037, 0.056] | +0.033 [0.025, 0.042] | +0.073 [0.065, 0.084] | +0.060 [0.052, 0.070] |
| NIRv resilience | +0.043 [0.036, 0.049] | +0.024 [0.018, 0.030] | +0.044 [0.038, 0.051] | +0.033 [0.028, 0.038] |
| Lidar mortality fraction (NEON) | +0.083 [0.061, 0.107] | +0.041 [0.021, 0.065] | – | – |

- **Every CI excludes 0**, at both AOIs. S here is the wall-to-wall
  structure layers; with lidar structure (below), the mortality and
  resilience gains mostly disappear and the recovery gain remains.
  - That includes the residual traits: the part of each trait not
    predictable from climate, terrain and structure.
  - It also includes resistance measured only over 2015–16, which avoids
    any overlap with the 2014 flights.
- **Size of the gains.**
  - They are 2–10× what traits added to lidar structure and site for the
    mortality fraction in the earlier test (+0.008).
  - They are largest for **recovery**, where they account for about 40% of
    the noise-adjusted variance Env+S leaves unexplained.
- **Canopy water.** EWT adds +0.003 to +0.024 on top of Env+S, and little
  beyond the traits (Env+S+T+W vs Env+S+T: ≤ +0.011).
- **2014 traits add nothing beyond 2013** (Env+S+T+T14 vs Env+S+T ≈ 0).

**Directions.** They replicate at both AOIs, from residual traits within
aridity × elevation strata (partial dependence and ρ):
- **LMA:** higher LMA than Env+S predicts goes with worse recovery (ρ −0.17
  / −0.19) and, at NEON, more mortality (+0.13).
- **Nitrogen:** higher N than Env+S predicts goes with better recovery
  (+0.19 / +0.21) and less mortality (−0.12).
- **Canopy water:** wetter 2013 canopies lost more (EWT vs resistance
  −0.20 / −0.35).
- These match the NEON trait–mortality directions of Queally et al. 2025 and
  the earlier predisposition analysis.
- By SHAP, **2013 nitrogen is the top predictor of NDMI recovery at NEON**,
  and lignin and N are the top two at `sierra_nf`, ahead of every climate,
  terrain and structure variable.

**Where traits matter** (`response_traits_conditional.csv`):
- **Mortality, by site.** At NEON the trait gain is +0.19 [0.13, 0.27] at
  SOAP (lower, drier, pine/cedar) vs +0.04 [0.02, 0.06] at TEAK (higher,
  fir). This is the site contrast Queally et al. 2025 reported.
  Resilience follows the same pattern (+0.09 vs +0.04).
- **Other responses, by site.** Resistance and recovery gains are similar
  or larger at TEAK.
- **Aridity.** Across aridity terciles (1981–2010 CWD) there is no
  consistent "traits matter more where it is dry" pattern at either AOI.
  At `sierra_nf` the resistance gain is larger in the wettest tercile
  (+0.066 vs +0.027 in the driest).
- **Trait × CWD interactions** in SHAP are negligible. The context
  dependence looks like site or host composition, not aridity.

**Are the traits standing in for structure?** The S layers above are
wall-to-wall products (GLAD height, TCC, LANDFIRE), which describe stands
much less well than lidar. The check uses the 7,040 NEON cells with lidar
trees, adds the lidar summaries (tree count, mean/p90/max height, fraction
> 30 m, crown area, fraction dead in 2013), and recomputes the residual
traits against Env+S+Lidar (`response_traits/lidar_structure_check.csv`).

| Target | Lidar over Env+S | T over Env+S+Lidar | Tres over Env+S+Lidar |
|---|---|---|---|
| Lidar mortality fraction | +0.148 [0.119, 0.182] | +0.025 [0.013, 0.038] | +0.003 [−0.007, 0.011] |
| NDMI resilience | +0.056 [0.036, 0.074] | +0.015 [0.000, 0.032] | −0.010 [−0.024, 0.001] |
| NDMI recovery | +0.060 [0.036, 0.081] | **+0.064 [0.046, 0.082]** | **+0.038 [0.019, 0.057]** |

- **Mortality and resilience.** Most of the trait gain comes from traits
  standing in for the stand structure that lidar measures directly. Once
  lidar is in the model, the independent (residual) trait contribution is
  indistinguishable from 0, as in the earlier predisposition test.
- **Recovery.** The trait gain is unchanged by lidar structure, and most of
  it survives residualization.
- **Recovery is therefore the response where AVIRIS-C traits carry
  information that neither climate, terrain nor detailed structure
  provides.** Nitrogen and LMA lead, in the leaf-economics directions.

### 3. Trait dynamics vs Landsat change (`hls_results/trait_dynamics{,_sierra}/`)

**Canopy-water trajectories.** The medians over forest cells follow the
drought:

| | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|
| NEON | 0.184 | 0.181 | 0.148 | 0.109 | 0.145 | 0.134 |
| `sierra_nf` | 0.213 | 0.204 | 0.179 | 0.159 | 0.165 | 0.153 |

(cm). For all pixels at NEON the series is 0.155 → 0.092 → 0.135 cm; earlier
notes gave that all-pixel series as "forest".

- **By fate** (`trajectories_neon_soap_teak_90m.png`):
  - Cells that went on to lose more than half their 2013-live lidar trees
    started slightly drier, fell to about 0.05 cm by 2016, and did not
    recover.
  - Surviving cells were back at 2013 levels by 2017.
  - The trait panels pool SOAP and TEAK, so their fate differences mix site
    with fate. Post-2015 N/chlorophyll rises in dying cells are likely
    retrieval artifacts on red and grey canopy.
- **Agreement with Landsat.** EWT resilience vs the same-years Landsat June
  NDMI metric is ρ 0.78 (NEON) and 0.79 (`sierra_nf`). EWT resilience vs
  lidar mortality is ρ −0.47.

**Change features.** AVIRIS change (A: ΔEWT, ΔLMA, ΔN, Δchlorophyll,
Δgreen-fraction QC) vs Landsat June change (L: ΔNDMI, ΔNIRv, ΔNDVI), for
2013→2015. Gains in R²:

| Target | A − L (alone) | L+A − L | EnvS+A − EnvS | EnvS+L+A − EnvS+L |
|---|---|---|---|---|
| NEON NDMI recovery | **+0.118 [0.075, 0.162]** | +0.321 | +0.015 | +0.007 [0.004, 0.009] |
| NEON NDMI resilience | −0.018 (n.s.) | +0.164 | +0.039 | +0.013 [0.010, 0.017] |
| NEON lidar mortality | −0.115 | +0.124 | +0.048 | +0.022 [−0.007, 0.043] |
| `sierra_nf` NDMI recovery | **+0.150 [0.130, 0.170]** | +0.207 | +0.015 | +0.010 [0.007, 0.013] |
| `sierra_nf` NIRv recovery | **+0.076 [0.055, 0.097]** | +0.153 | +0.013 | +0.007 [0.004, 0.011] |
| `sierra_nf` NDMI resilience | +0.019 (n.s.) | +0.123 | +0.023 | +0.007 [0.005, 0.010] |

- **Head to head, AVIRIS change beats Landsat change for recovery** at both
  AOIs. Landsat change is equal or better for resilience and mortality.
- **The two are complementary.** L+A beats L by +0.10 to +0.32 everywhere.
- **On top of Env+S and Landsat change,** AVIRIS change still adds a small
  gain (+0.006 to +0.014) whose CI excludes 0 for every Landsat target. For
  the lidar mortality fraction it adds +0.017 to +0.022, with a CI touching
  0 (about 7,000 lidar cells).
- **Recovery failure** (bottom quintile of NDMI resilience), AUC at NEON:
  Env+S 0.912; +L 0.929; +L+A 0.932.
- **Cross-track normalization** helps the Landsat targets (A vs raw A: +0.03
  to +0.09) but hurts 2013→2015 lidar mortality (−0.054). Both versions are
  reported.
- **Caveat.** The Landsat targets share sensor noise with the Landsat change
  features, which favours L. The lidar target is the unbiased comparison.

### 4. Do environment-only models transfer between droughts? (`hls_results/response_transfer/`)

Setup: NEON, 90 m. 24,342 cells had no fire or harvest through 2025.
- Cycle 1: Landsat 7, 2008–2019.
- Cycle 2: Landsat 8/9, 2017–2025.

| Target | Env+S within C1 (R²) | within C2 | C1→C2 R² | C1→C2 ρ | C2→C1 ρ | C1 metric vs C2 metric (ρ) |
|---|---|---|---|---|---|---|
| NDMI resistance | 0.755 | 0.578 | −2.96 | 0.12 | 0.05 | 0.23 |
| NDMI recovery | 0.651 | 0.595 | −0.42 | 0.22 | 0.25 | 0.26 |
| NDMI resilience | 0.652 | 0.532 | −3.57 | **−0.39** | **−0.51** | **−0.41** |
| NIRv resistance | 0.799 | 0.628 | −2.67 | 0.12 | 0.11 | 0.19 |
| NIRv recovery | 0.704 | 0.638 | −0.27 | 0.41 | 0.30 | 0.24 |
| NIRv resilience | 0.644 | 0.660 | −3.04 | **−0.50** | **−0.53** | **−0.62** |

- **Within a drought, Env+S explains the spatial pattern well.** Across
  droughts it has no absolute skill (every transferred R² < 0) and little
  rank skill.
  - For resistance and recovery, ρ ≤ 0.41.
  - For resilience the ranking reverses.
  - The same holds for Env alone, B1 alone, and Env+S+B1.
- **The forcing is similar in both droughts, so it isn't the cause.** The
  cells' drought-window CWD anomalies are well correlated between the two
  droughts (ρ 0.73), and so are their baselines (ρ 0.84). The
  forcing-to-response relationship itself changed.
- **A legacy of the first drought explains the resilience reversal.**
  - Cells that lost most in 2014–16 enter the second drought with a lower
    2017–19 baseline: C1 resistance vs C2 baseline ρ −0.18.
  - The same cells then show high cycle-2 "resilience" (C1 resistance vs
    C2 resilience ρ −0.31), consistent with regrowth after the die-off.
  - Cycle-1 response is therefore a necessary predictor for the next
    drought.
- **Same at 270 m:** within-drought R² 0.65–0.85; transfer ρ 0.05–0.32
  for resistance and recovery, −0.43 to −0.72 for resilience.
- This extends, to continuous responses, the earlier finding that
  forcing-only mortality models fail across years.

### 5. Traits vs airborne lidar structure beyond the NEON trees (`hls_results/lidar_check/`)

§2's lidar check used only the 7,040 NEON cells with 2013 lidar trees. It
is now scripted (`proto_response_traits.py --structure`) and repeated with
two NASA airborne lidar sources (`lidar_coverage.md`,
`fetch_lidar_structure.py`):
- **LVIS**, Sep 2008: 46% of NEON, pre-drought, ~20 m footprints.
- **ASO composite** (Airborne Snow Observatory 2014–17, merged by Ferraz et
  al. 2020): 66% of NEON, 86% of `sierra_nf`. Its snow-off flights are Oct
  2015 (NEON) and Oct 2016 (`sierra_nf`), during the die-off.

Each run keeps the cells the source covers (≥ 70% of a 90 m cell), adds its
structure block L to Env+S, and residualizes the traits against Env+S+L.

**The structure layers agree with each other and with the NEON trees**
(`structure_agreement.csv`, 90 m, Spearman ρ):
- NEON 2013 tree height vs ASO: 0.83–0.93; vs LVIS: 0.84–0.88. ASO vs LVIS:
  0.90–0.94. The fraction of trees > 30 m agrees at 0.88–0.93.
- The wall-to-wall GLAD 2010 height reaches only ρ 0.56–0.67 against any
  lidar and reads about 10 m low in tall stands. This is the structure
  that traits could stand in for in §2.

**Does ASO already record the die-off?** (`structure_leakage.csv`). On the
NEON lidar-tree cells, ASO adds +0.012 [−0.001, 0.026] to the mortality
model once NEON 2013 structure is in, and LVIS 2008 adds +0.004. Both are
consistent with 0. ASO cover is unrelated to mortality (ρ ≈ 0); its link to
mortality is the same "taller stands died more" link as the 2013 trees.
ASO therefore behaves as a structure control, not as a mortality map.

**Trait gains over Env+S+L** (residual traits Tres in brackets; 95% CI,
1 km blocks):

| Target | NEON, NEON 2013 trees (7,040 cells) | NEON, LVIS 2008 (21,826) | NEON, ASO (29,004) | `sierra_nf`, ASO (48,731) |
|---|---|---|---|---|
| NDMI recovery | +0.064 [0.046, 0.082] (**+0.038** [0.019, 0.057]) | +0.072 [0.059, 0.086] (**+0.048** [0.036, 0.060]) | +0.056 [0.044, 0.067] (**+0.038** [0.029, 0.048]) | +0.037 [0.032, 0.042] (**+0.025** [0.020, 0.029]) |
| NIRv recovery | +0.072 (+0.047 [0.020, 0.077]) | +0.073 (+0.058 [0.042, 0.077]) | +0.072 (+0.044 [0.026, 0.060]) | +0.046 (+0.031 [0.026, 0.036]) |
| NDMI stress response | +0.030 (+0.028 [0.017, 0.044]) | +0.034 (+0.022 [0.013, 0.032]) | +0.033 (+0.024 [0.016, 0.034]) | +0.035 (+0.021 [0.016, 0.026]) |
| NDMI resistance | +0.019 (+0.003 [−0.011, 0.018]) | +0.027 (+0.013 [0.007, 0.020]) | +0.010 (+0.010 [0.004, 0.016]) | +0.022 (+0.012 [0.008, 0.016]) |
| NDMI resilience | +0.015 (−0.010 [−0.024, 0.001]) | +0.048 (+0.030 [0.022, 0.038]) | +0.027 (+0.015 [0.010, 0.020]) | +0.020 (+0.013 [0.009, 0.017]) |
| NDMI recovery time | +0.018 (+0.009, n.s.) | +0.018 (+0.009 [0.002, 0.017]) | +0.018 (+0.011 [0.005, 0.016]) | +0.021 (+0.009 [0.005, 0.014]) |
| Lidar mortality fraction | +0.025 (+0.003, n.s.) | +0.013 (−0.006, n.s.) | +0.026 (+0.021 [0.001, 0.041]) | – |

- **Recovery survives every lidar control, at both AOIs.** The residual-
  trait gain is +0.025 to +0.048 for NDMI recovery and +0.031 to +0.058 for
  NIRv recovery. At `sierra_nf` (the first check outside NEON) it is
  smaller than at NEON but its CI is far from 0.
- **Stress response also survives everywhere** (+0.021 to +0.028). §2 had
  not tested it against lidar.
- **Resistance, resilience and recovery time** keep only small residual
  gains (≤ +0.015, except LVIS resilience +0.030). They are distinguishable
  from 0 only with the large LVIS and ASO samples; against the NEON 2013
  trees they are not.
- **Mortality:** with the tree-level NEON structure or LVIS the residual
  gain is 0, as in §2.
- Lidar structure itself adds +0.02 to +0.09 over the wall-to-wall S for
  every Landsat target.

### 6. Is the recovery gain the traits themselves? (`hls_results/recovery_ablation/`)

`proto_response_traits.py --ablation`: gain of trait subsets over Env+S (and
over Env+S+NEON 2013 trees), on cell subsets that exclude likely artifacts,
for several recovery definitions. 90 m, 95% CIs.

| NDMI recovery, gain over | Full T | T without QC and SD bands | N + LMA only | Green-fraction QC only |
|---|---|---|---|---|
| Env+S, NEON (44,079 cells) | +0.065 | +0.057 | +0.026 [0.021, 0.031] | +0.009 |
| Env+S, `sierra_nf` (58,676) | +0.089 | +0.081 | +0.050 [0.044, 0.056] | +0.014 |
| Env+S+NEON 2013 trees (7,040) | +0.064 | +0.057 | **+0.045 [0.030, 0.061]** | +0.027 |

- **The QC and SD bands are not the source.** Without them the gain drops
  by 0.007–0.008. The green-fraction QC alone adds ≤ 0.014 over Env+S.
- **N and LMA alone carry much of it**: 40–56% of the full gain over Env+S,
  and 70% of it over NEON lidar structure.
- **Dead or grey canopy is not the source.** The gain is unchanged in
  green-dominated cells (2013 QC above its lower quartile: NEON +0.063,
  `sierra_nf` +0.089). It is +0.079 [0.031, 0.124] in the 1,187 NEON cells
  with no lidar-dead trees in 2013 (N + LMA over lidar there: +0.058).
- **Retrieval artifacts are not the source.** Dropping the cells whose N or
  chlorophyll rose after 2015 (top quintile of mean(2016–17) − 2015) keeps
  +0.052 (NEON) and +0.081 (`sierra_nf`).
- **The definition of recovery does not matter.** Post window 2017–18
  instead of 2017–19: NEON +0.063, `sierra_nf` +0.072. Recovery from
  2015–16 only: +0.063 / +0.087. The gains for recovery time (+0.036 to
  +0.047) and NIRv recovery are smaller.

### 7. Stability of slow traits, 2013 vs 2018 (`hls_results/trait_stability/`)

`proto_trait_stability.py`. The question is whether traits from the June
2018 flight (before the 2020–22 drought) can stand in for 2013 traits in
models trained on the first drought.
- **Cells:** undisturbed forest in the top tercile of NDMI resistance, with
  < 5% lidar-tree mortality at NEON (11,230 NEON and 19,559 `sierra_nf`
  cells at 90 m).
- **Method:** traits cross-track normalized per flight line as in §3.

**Spatial pattern (ρ vs 2013, stable cells, 90 m):**

| Trait | NEON 2014 / 2015 (`_v2`) / 2018 | `sierra_nf` 2014 / 2015 / 2018 |
|---|---|---|
| LMA | 0.85 / 0.90 / **0.89** | 0.82 / 0.84 / **0.81** |
| Nitrogen | 0.70 / 0.75 / **0.77** | 0.78 / 0.76 / **0.76** |
| Lignin | 0.65 / 0.73 / **0.76** | 0.67 / 0.80 / **0.75** |
| Chlorophyll | 0.64 / 0.47 / **0.68** | 0.65 / 0.67 / **0.66** |

- **The 2018 pattern is as close to 2013 as the next year's flights are.**
  Five years apart costs nothing beyond the flight-to-flight noise
  floor.
- **The level is not stable.** 2018 nitrogen sits +1.1 to +1.3 SD above
  2013 in stable cells, and chlorophyll −1.1 to −1.5 SD. LMA and lignin are
  within 0.5 SD. The 2015 `_v2` mosaic is within 0.2 SD of 2013 for every
  trait, so the offset is between-date calibration (2018 is not a
  cross-year-calibrated mosaic), not biology. Using 2018 traits in 2013
  models therefore needs per-date standardization (ranks or z-scores).
- **Directions carry over, weaker.** Within aridity × elevation strata the
  ρ of 2018 N with cycle-1 NDMI recovery is +0.22 to +0.25, against +0.37
  to +0.38 for 2013 N. LMA is −0.20 to −0.23 vs −0.37, and lignin −0.30 vs
  −0.39 to −0.41. The 2018 traits also reflect the die-off in between.

### 8. Replication in the Tahoe box (`stanislaus`; `hls_results/response{,_traits}_stanislaus/`)

Cycle-1 responses only (2008–2019), with the same masks and models. There is
no canopy water (W) or lidar here yet. The Tahoe 2013 mosaic (June 4) is not
cross-year calibrated.
- **Masks:** FACTS harvest covers 4.1% of forest (NEON 7.5%); 90% of forest
  is undisturbed through 2019. That leaves 56,275 cells at 90 m.
- **Env+S explains less** (90 m): NDMI resistance 0.35, recovery 0.29,
  resilience 0.30; NIRv 0.33–0.44. The drought signal is smaller relative
  to noise: resistance SD is 1.6× the placebo SD (0.026 vs 0.017), against
  1.9× at NEON (0.044 vs 0.023) and 2.0× at `sierra_nf` (0.030 vs 0.015).

**Traits add to every response:**

| Target | +T | +Tres |
|---|---|---|
| NDMI resistance | **+0.068** [0.054, 0.082] | +0.054 [0.039, 0.070] |
| NDMI recovery | +0.040 [0.025, 0.056] | +0.022 [0.008, 0.037] |
| NDMI resilience | +0.029 [0.019, 0.039] | +0.020 [0.010, 0.030] |
| NIRv resistance | +0.068 | +0.046 |
| NIRv recovery | +0.054 | +0.031 |

- **The largest gain here is resistance, not recovery.**
- **Directions only partly replicate.** Structural-carbon traits go with
  worse recovery, as in the Yosemite box:
  - residual cellulose ρ −0.12, lignin −0.11, fiber −0.08;
  - NEON and `sierra_nf`: −0.15 to −0.19.
- **The nitrogen and LMA directions do not replicate.** Residual N is ρ
  +0.02 and LMA +0.02 within strata (NEON and `sierra_nf`: N +0.19/+0.21,
  LMA −0.17/−0.19).
- The leaf-economics reading of the recovery gain therefore holds in the
  Yosemite box only. The structural-carbon reading holds in both boxes.
- **Possible reasons, not yet separated:**
  - forest type (Tahoe is fir-dominated);
  - calibration of the non-`_v2` Tahoe mosaics;
  - the weaker drought signal.

### 9. Why do trait directions differ between the two boxes? (`hls_results/trait_directions/`)

`proto_trait_directions.py` tests the three explanations of §8 on the same
cells and models. `fetch_forest_type.py` adds LANDFIRE 2014 EVT groups:
- pine / dry mixed conifer;
- mesic (white-fir) mixed conifer;
- red fir;
- subalpine;
- other.

The Tahoe box is 38% red fir, 33% mesic and 5% pine. NEON is 20%, 23% and
17%.

**How ρ is computed.** Within-stratum ρ of residual traits with NDMI
recovery, with 95% CIs from a 1 km block bootstrap. Residual traits are
cross-fitted on Env+S over all cells of an area. A forest-type subset is the
cells whose dominant group covers ≥ 50%.

**Summary by variant** (all cells):

| Area | Variant | N | LMA | Lignin | Cellulose |
|---|---|---|---|---|---|
| NEON | 2013 `_v2` (baseline) | +0.19 [0.16, 0.21] | −0.17 [−0.20, −0.14] | −0.17 | −0.15 |
| NEON | 2013 non-`_v2` | +0.17 [0.13, 0.20] | −0.15 [−0.18, −0.12] | −0.13 | −0.14 |
| NEON | 2014 / 2015 traits | +0.17 / +0.17 | −0.16 / −0.15 | −0.17 / −0.18 | −0.16 / −0.19 |
| NEON | per-line z-score | +0.18 | −0.17 | −0.18 | −0.16 |
| `sierra_nf` | 2013 `_v2` (baseline) | +0.22 [0.20, 0.23] | −0.19 [−0.21, −0.17] | −0.19 | −0.16 |
| `sierra_nf` | 2013 non-`_v2` | +0.20 [0.18, 0.21] | −0.18 [−0.20, −0.16] | −0.18 | −0.17 |
| `sierra_nf` | 2014 / 2015 traits | +0.20 / +0.18 | −0.18 / −0.16 | −0.21 / −0.19 | −0.19 / −0.20 |
| `sierra_nf` | per-line z-score | +0.22 | −0.21 | −0.20 | −0.17 |
| Tahoe | 2013 (baseline) | +0.03 [0.00, 0.05] | +0.02 [−0.01, 0.05] | −0.11 | −0.12 |
| Tahoe | 2014 / 2015 traits | −0.02 / +0.02 | +0.08 / +0.04 | −0.09 / −0.12 | −0.12 / −0.15 |
| Tahoe | per-line z-score | +0.03 | +0.03 | −0.10 | −0.11 |

**Calibration does not explain it.**
- In the Yosemite box, the non-`_v2` mosaic of the same date keeps the
  N/LMA directions: NEON N +0.17, LMA −0.15; `sierra_nf` +0.20, −0.18. So do
  the non-`_v2` 2014 mosaic and per-line z-scoring.
- In the Tahoe box, three independent acquisitions (2013, 2014, 2015) and
  per-line z-scoring all give N and LMA near 0.
- (A per-date z-score is a monotone transform of one mosaic, so it cannot
  change these results.)

**Forest type does not explain it.**
- In the Yosemite box, the direction holds in every group, and most
  strongly in red fir:
  - NEON red fir: N +0.20 [0.16, 0.24], LMA −0.27 [−0.31, −0.23];
  - `sierra_nf` red fir: N +0.22, LMA −0.19;
  - pine is the weakest group: NEON N +0.11, LMA −0.05; `sierra_nf` +0.17,
    −0.14.
- In the Tahoe box it appears only in the small pine group and in "other"
  (hardwood, shrub and riparian mixes):
  - pine (1,687 cells): N +0.11 [0.02, 0.19], LMA −0.14 [−0.23, −0.03];
    with 2015 traits +0.15 / −0.15;
  - mesic and red fir cells: N +0.01 to +0.02, LMA +0.03 to +0.04; with
    2014 traits, red fir LMA is +0.10.
- So Tahoe fir stands behave differently from Yosemite-box fir stands, not
  fir from pine.
- Forest-type fractions add ≤ +0.005 to Env+S. They leave the trait gain for
  recovery unchanged: NEON +0.059 → +0.058; Tahoe +0.032 → +0.027.

**Signal strength does not explain it.**
- Adding noise to the Yosemite-box recovery so that its signal share matches
  the Tahoe box (0.60 against 0.73–0.75; 20 draws) changes N and LMA only
  slightly:
  - NEON: N +0.15, LMA −0.13;
  - `sierra_nf`: N +0.19, LMA −0.16.
- Subsets with the lowest drought forcing (CWD anomaly terciles) have the
  same resistance-to-placebo SD ratio (1.90–2.07) and the same directions.

**Structural carbon replicates in every test.** Residual lignin, cellulose
and fiber go with worse recovery in both boxes and with every mosaic.
- Lignin does so in every forest-type subset (ρ −0.06 to −0.22).
- Cellulose weakens only in NEON's "other" group (−0.05).

**Reading.** The leaf-economics direction (high N, low LMA → better recovery)
is a property of the Yosemite box, where it holds in every forest type and
every mosaic. It is absent from the Tahoe fir stands. It is not explained by
calibration, by forest type as mapped by LANDFIRE, or by the weaker drought
signal. What differs between the boxes remains open. Candidates:
- the fir species mix (white vs red fir within LANDFIRE's classes);
- the mortality agents (fir engraver in the Tahoe box);
- the drought's timing in the northern Sierra.

§22 tests all three with ADS agent and host maps and per-cell timing; none
explains it.

### 10. Held-out sites for the 2020–22 drought

**Held-out sites.**
- These are kept aside for a forward test of models fixed on the other
  sites: `stanislaus` (the Tahoe box), SEKI, and the unburned Yosemite-box
  cells outside the NEON and `sierra_nf` AOIs.
- Their cycle-2 (2020–22) responses are not computed or examined.
- `response_common.HELD_OUT` and `check_cycle2()` enforce this for the
  AOIs: `proto_response_transfer.build_cycle` refuses cycle 2 for them.
- The two Yosemite-box sites have their own grids (§34):
  - `yosemite_rest`: the June 12 2013 and June 22 2018 trait-mosaic
    footprints, outside SEKI and outside the `neon_soap_teak` and
    `sierra_nf` grids;
  - `seki`: the same footprints inside the SEQU/KICA boundary.
- The Tahoe-box work in §8–9 and the canopy water and dynamics in §14 use
  cycle 1 only.

### 11. Do post-drought traits help predict the next drought? (`hls_results/forward_pilot/`)

`proto_forward_pilot.py`, on the NEON and `sierra_nf` AOIs only (the
pilot sites).
- **Targets:** cycle-2 responses (baseline 2017–19, drought 2020–22, post
  2023–25). Forest cells with no fire or harvest through 2025 are used;
  `sierra_nf` keeps 20,089 cells outside the Creek Fire and NEON 24,342.
- **Blocks:**
  - Leg: the cycle-1 responses;
  - T18: the June 2018 traits, cross-track normalized and z-scored per
    date;
  - T13: the 2013 traits, treated the same way.

**Within cycle 2 (1 km block CV):**

| Target | Env+S R² (NEON / `sierra_nf`) | +T18 | +T18 beyond Leg | +T18 beyond T13 |
|---|---|---|---|---|
| NDMI resistance | 0.58 / 0.45 | +0.072 / +0.053 | +0.023 / +0.015 | +0.029 / +0.029 |
| NDMI recovery | 0.60 / 0.49 | +0.065 / +0.105 | +0.035 / +0.019 | +0.005 (n.s.) / +0.018 |
| NDMI resilience | 0.53 / 0.58 | +0.039 / +0.048 | +0.018 / +0.015 | +0.014 / +0.020 |
| NIRv resistance | 0.63 / 0.45 | +0.107 / +0.130 | +0.021 / +0.033 | +0.045 / +0.087 |
| NIRv recovery | 0.64 / 0.48 | +0.063 / +0.075 | +0.032 / +0.028 | +0.020 / +0.035 |
| NIRv resilience | 0.66 / 0.53 | +0.095 / +0.123 | +0.017 / +0.041 | +0.055 / +0.080 |

Every gain except the NEON recovery T18-over-T13 step has a 95% CI above 0.
The cycle-1 legacy alone adds +0.06 to +0.15.

**The trait directions of the first drought carry over.** Within-stratum ρ
with cycle-2 NDMI recovery:

| Area | Traits | N | LMA | Lignin |
|---|---|---|---|---|
| NEON | 2018 | +0.30 | −0.31 | −0.23 |
| NEON | 2013 | +0.30 | −0.28 | −0.17 |
| `sierra_nf` | 2018 | +0.38 | −0.36 | −0.35 |

**Across cycles.** A model is fitted on cycle 1 with Env+S+T13 and applied
to cycle 2 with T18 swapped in (per-date z-scores).
- The level still does not transfer: R² < 0 everywhere, as in §4.
- At NEON, traits raise the rank skill for recovery: NDMI ρ 0.20 → 0.33;
  NIRv ρ 0.41 → 0.52. The 2013 traits in the same slot give 0.24 and 0.46.
- At `sierra_nf` the environment-only transfer already ranks cells in
  reverse (ρ −0.49 to −0.01, NDMI recovery ρ ≈ 0; why: §21). Traits do not rescue it: NDMI
  recovery ρ 0.10.
- The resilience ranking reverses at both AOIs with or without traits.
- Per-date percentile ranks instead of z-scores (`--rank`) give the same
  results (NEON NDMI recovery ρ 0.335, `sierra_nf` 0.09). The tree models
  are invariant to monotone transforms within a date, so only the 2018 swap
  could differ, and it does not.

**Reading.**
- Post-drought traits add skill for the next drought's responses beyond
  climate, structure and the first drought's legacy, whenever models are
  fitted within the new drought.
- Transferring a fitted model from one drought to the next is not yet
  reliable.

### 12. AVIRIS-Classic vs AVIRIS-5 on the same day (`hls_results/aviris5_bridge/`)

**Data and method** (`proto_aviris5_bridge.py`).
- Both instruments flew over NEON on July 17, 2025:
  - AVIRIS-C lines `f250717t01p00r08` and `r09` (ORNL DAAC 2154, 14.5 m);
  - AVIRIS-5 scenes `AV520250717t171949_006` and `AV520250717t173810_002`
    (ORNL DAAC 2484, 10.3 m, 424 bands).
- Both are area-averaged to 30 m, and AVIRIS-5 is convolved to the AVIRIS-C
  bands.
- They are compared on the 15,784 undisturbed forest 90 m cells that both
  cover.
- The 2025 AVIRIS-C L2 has isolated non-physical NIR values, so reflectance
  outside −0.05 to 1.5 is masked.

**Rank agrees; level does not.**
- EWT980: ρ 0.976. AVIRIS-5 is higher by 0.029 cm (median 0.243 against
  0.218 cm, +13%).
- NDVI: ρ 0.958, bias +0.049. NDWI: ρ 0.952, bias −0.055.
- Per band (median relative difference, median ρ):

  | Region | Relative difference | ρ |
  |---|---|---|
  | 450–700 nm | −4% | 0.80 |
  | 700–1300 nm | +18% | 0.70 |
  | 1450–1800 nm | +24% | 0.81 |
  | 2000–2400 nm | +4% | 0.83 |

- AVIRIS-5 reads brighter in the NIR and SWIR-1. Differences in view and
  illumination geometry, and in atmospheric correction (OE for AVIRIS-5),
  are likely contributors; this comparison cannot separate them.
- As with 2013 vs 2018 (§7), a model that carries traits or EWT from one
  sensor to the other needs per-date standardization, not raw levels.

### 13. How often does a spaceborne imaging spectrometer see these forests? (`hls_results/emit_coverage/`)

**Method** (`query_emit_coverage.py`).
- Every EMIT L2A reflectance granule over the Sierra Nevada from Aug 2022,
  read from CMR metadata: footprint, time, granule cloud cover and solar
  zenith.
- A scene is usable for a 90 m cell when:
  - its footprint covers the cell;
  - it is Jun–Sep;
  - SZA ≤ 50° (60° as a variant);
  - granule cloud < 20% (50% as a variant).
- Per-pixel masks are not read.

**Usable growing-season scenes per cell** (median; share of cells with at
least one), against AVIRIS growing-season flight dates with public L2:

| Year | NEON | `sierra_nf` | Tahoe box | AVIRIS dates (each box) |
|---|---|---|---|---|
| 2022 (from Aug) | 1 (100%) | 1 (53%) | 1 (100%) | 0 |
| 2023 | 3 (100%) | 1 (59%) | 0 (0%) | 0 |
| 2024 | 1 (78%) | 0 (0%; 1 at SZA ≤ 60°) | 3 (100%) | 1 |
| 2025 | 3 (100%) | 2 (100%) | 3 (100%) | 1 |
| 2026 | 3 (100%) | 2 (100%) | 1 (100%) | 0 |

- AVIRIS flew 1–3 growing-season dates per box per year in 2013–18 and one
  over the Yosemite box in 2019. It flew none in 2020–23, and one in 2024
  and in 2025.
- EMIT gives 1–3 usable views of most cells in most years since 2023, but
  some cell-years have none (`sierra_nf` 2024 at SZA ≤ 50°, Tahoe 2023).
- The ISS overpass drifts through the day, so the usable set depends on the
  SZA threshold: Jun 11, 2024 over `sierra_nf` is at SZA 53°.
- The scene closest in time to the July 2025 AVIRIS flights over the Tahoe
  box (Jul 31, 2025) covers only 3% of the `stanislaus` AOI.

### 14. Canopy water and trait dynamics in the Tahoe box (`wdts/stanislaus_cwc.nc`; `hls_results/{response_traits_stanislaus_w,trait_dynamics_stanislaus}/`)

**Data.**
- `fetch_wdts_cwc.py` now takes its dates per flight box. For the Tahoe
  box: 2013-06-04, 2014-06-02, 2015-06-08 (fallback 06-11), 2016-06-09,
  2017-06-20 and 2018-06-21.
- The easternmost Tahoe line on four dates is in UTM 11. Those lines are
  processed on their own grid and warped (area average) onto the UTM 10
  AOI grid.
- Coverage is 100% every year except 2016 (61%).
- None of the Tahoe mosaics is cross-year calibrated.

**The trajectory differs from the Yosemite box.** Forest EWT medians:

| 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|
| 0.217 | 0.252 | 0.168 | 0.241 | 0.237 | 0.201 cm |

- The Yosemite box reaches its minimum in 2016. The Tahoe box dips in 2015
  only, which fits the weaker drought signal there (§8) and the wet 2016
  winter in the northern Sierra.
- The EWT metrics also agree less with Landsat than in the Yosemite box:
  - EWT vs NDMI resilience: ρ 0.33 (`sierra_nf` 0.74);
  - EWT vs June Landsat NDMI resilience: ρ 0.58 (`sierra_nf` 0.79).

**The W block in the trait ladder** (gain over Env+S):

| Target | +T | +W | +Tres |
|---|---|---|---|
| NDMI resistance | +0.068 | +0.094 [0.056, 0.140] | +0.054 |
| NDMI resilience | +0.029 | +0.041 [0.024, 0.062] | +0.020 |
| NDMI recovery | +0.040 | +0.014 [0.006, 0.023] | +0.022 |
| NIRv resistance | +0.068 | +0.036 | +0.046 |
| NIRv recovery | +0.054 | +0.018 | +0.031 |

- W is EWT for June 2013 and June 2014, and June 2014 is inside the drought
  window. So the large W gain for resistance is partly the response
  measured directly. For recovery, W adds little, as in the Yosemite box.

### 15. Structure from spectra, and models without lidar (`hls_results/structure_from_spectra/`)

`proto_structure_from_spectra.py`, on each lidar source's cells.

**1. How well is lidar structure predicted?** Out-of-fold boosting with
1 km blocks, Spearman ρ:

| Area, source | Metric | GLAD 2010 alone | Wall-to-wall + Landsat baseline | AVIRIS traits + EWT | Both |
|---|---|---|---|---|---|
| NEON, ASO | canopy height (rh98) | 0.67 | 0.82 | 0.80 | 0.86 |
| NEON, ASO | cover > 5 m | 0.62 | 0.77 | 0.81 | 0.84 |
| NEON, LVIS 2008 | RH100 | 0.59 | 0.78 | 0.76 | 0.83 |
| NEON, 2013 trees | height p90 | 0.56 | 0.68 | 0.77 | 0.79 |
| NEON, 2013 trees | tree count | 0.25 | 0.48 | 0.62 | 0.65 |
| `sierra_nf`, ASO | canopy height (rh98) | 0.66 | 0.75 | 0.79 | 0.83 |
| `sierra_nf`, ASO | cover > 5 m | 0.63 | 0.78 | 0.85 | 0.87 |

- The spectra, on their own or combined, reach ρ 0.79–0.88 for height and
  cover. That approaches lidar-to-lidar agreement (0.83–0.94).
- GLAD alone reaches 0.56–0.71.
- Tree count is the hardest metric (0.65).

**2. Models with no lidar.** Ŝ is the out-of-fold prediction of the source's
structure metrics from both input sets.

| Area, source | Target | Env+S | Env+S+T | Env+S+Ŝ+T | Env+S+L+T (reference) | Skill kept |
|---|---|---|---|---|---|---|
| NEON, ASO | NDMI recovery | 0.506 | 0.605 | 0.612 | 0.648 | 0.94 |
| NEON, LVIS | NDMI recovery | 0.498 | 0.614 | 0.622 | 0.648 | 0.96 |
| NEON, 2013 trees | NDMI recovery | 0.466 | 0.568 | 0.569 | 0.590 | 0.96 |
| `sierra_nf`, ASO | NDMI recovery | 0.518 | 0.605 | 0.606 | 0.643 | 0.94 |

- Over all five targets and four source/area pairs, models with no lidar
  keep 0.92–1.00 of the lidar-reference R². The loss is −0.050 to +0.002
  R², and its CI excludes 0 for most targets.
- Ŝ adds almost nothing once T is in the model (+0.002 median, ≤ +0.009).
  Real lidar adds +0.004 to +0.055 (median +0.028).
- So the structure the spectra recover is already carried by T. What lidar
  adds is information that neither the spectra nor the wall-to-wall layers
  hold.
- Ŝ comes from the same spectra as T. This measures what a model without
  lidar can do. It does not show that traits carry information beyond
  structure; that is §5.

### 16. Within-cell trait diversity (`hls_results/trait_diversity/`)

`proto_trait_diversity.py`. Mean traits (Tmean: 14 means plus the
green-fraction QC) are compared with a diversity block D:
- the per-pixel PLSR SD bands of LMA, N and chlorophyll;
- the spatial SD of N and LMA over the 30 m pixels of a 90 m cell;
- functional dispersion (FDis) on standardized N + LMA, and on
  N + LMA + EWT.

**Gains** (NDMI recovery / resilience; ranges over all five targets in
brackets):

| Area, base | Tmean | D alone | D beyond Tmean |
|---|---|---|---|
| NEON, Env+S | +0.059 / +0.051 | +0.016 / +0.015 | +0.005 / +0.006 [+0.005 to +0.008] |
| NEON, Env+S+L (LVIS) | +0.068 / +0.044 | +0.020 / +0.010 | +0.007 / +0.003 [+0.003 to +0.011] |
| NEON, Env+S+L (ASO) | +0.050 / +0.025 | +0.014 / +0.005 | +0.006 / +0.002 [+0.001 to +0.016] |
| `sierra_nf`, Env+S | +0.086 / +0.050 | +0.032 / +0.016 | +0.003 / +0.002 [+0.002 to +0.007] |
| `sierra_nf`, Env+S+L (ASO) | +0.034 / +0.018 | +0.015 / +0.008 | +0.005 / +0.001 [+0.001 to +0.005] |

**Directions** (within-stratum ρ with NDMI recovery, NEON / `sierra_nf`):
- Spatial diversity goes with **better** recovery:
  - FDis(N, LMA): +0.10 / +0.12;
  - pixel SD of LMA: +0.11 / +0.15;
  - pixel SD of N: +0.09 / +0.10.
- The PLSR SD bands go with **worse** recovery (chlorophyll −0.14 / −0.16,
  N −0.10 / −0.07). These are retrieval uncertainty, not diversity, and are
  likely to track canopy condition.

**Reading.**
- Diversity carries some information on its own (up to +0.036 for NIRv
  recovery over LVIS).
- Beyond mean traits it adds ≤ +0.016, and usually ≤ +0.008. Mixed stands
  recover slightly better, but diversity is a small refinement, not a
  separate driver.

### 17. Spatial-neighbourhood baseline (`hls_results/neighbour_baseline/`)

**Method.** `proto_response_traits.py --neighbour`.
- B0 is the mean response of the training cells within 2 km of a cell,
  excluding the cell's own 1 km block.
- It is recomputed in every CV fold (`response_common.neighbour_mean`), so
  no label from a test block is used.
- This is the spatial-autocorrelation baseline: can traits beat simply
  borrowing the neighbours' response?

**Results** (R²; trait gain over Env+S+B0 with 95% CI):

| Area | Target | B0 | Env+S | Env+S+B0 | +T | +Tres |
|---|---|---|---|---|---|---|
| NEON | NDMI recovery | 0.44 | 0.68 | 0.69 | +0.060 [0.051, 0.068] | +0.044 [0.036, 0.051] |
| `sierra_nf` | NDMI recovery | 0.27 | 0.53 | 0.53 | +0.090 [0.082, 0.098] | +0.072 [0.065, 0.079] |
| Tahoe | NDMI recovery | 0.15 | 0.29 | 0.31 | +0.035 [0.020, 0.049] | +0.019 [0.009, 0.031] |
| NEON | NDMI resistance | 0.59 | 0.73 | 0.75 | +0.017 | +0.011 |
| `sierra_nf` | NDMI resistance | 0.25 | 0.55 | 0.58 | +0.042 | +0.033 |
| Tahoe | NDMI resistance | 0.16 | 0.35 | 0.37 | +0.066 | +0.060 |
| NEON | NDMI resilience | 0.32 | 0.63 | 0.65 | +0.054 | +0.039 |
| `sierra_nf` | NDMI resilience | 0.26 | 0.60 | 0.61 | +0.053 | +0.042 |
| NEON | stress response | 0.60 | 0.71 | 0.73 | +0.015 | +0.008 |
| `sierra_nf` | stress response | 0.35 | 0.49 | 0.51 | +0.051 | +0.039 |
| NEON | NIRv recovery | 0.63 | 0.69 | 0.73 | +0.035 | +0.025 |
| `sierra_nf` | NIRv recovery | 0.32 | 0.46 | 0.47 | +0.073 | +0.061 |

**Reading.**
- The neighbourhood alone is a strong baseline at NEON (B0 R² 0.32–0.71),
  weaker elsewhere (0.02–0.37).
- Once Env+S is in the model, it adds only +0.00 to +0.04.
- Trait and residual-trait gains over Env+S+B0 are close to the gains over
  Env+S. Their CIs exclude 0 for every Landsat response except Tahoe
  stress response.
- So the trait signal is not spatial autocorrelation of the response
  passing through the traits.

### 18. Does the trait signal survive spaceborne-like sampling? (`wdts/sim/`, `hls_results/spaceborne_sim/`)

**Data and emulator** (`proto_spaceborne_sim.py`, NEON).
- The 2013-06-12 `_v2` 2391 reflectance, all bands, on the 30 m grid.
- Each pixel comes from the line the 2403 trait mosaic used there.
- The 2403 PLSR coefficients are not public, so each configuration gets its
  own emulator: PLSR from log reflectance to the 2403 2013 trait map, fitted
  and applied out of fold over 1 km blocks.
- The simulated files have no SD or QC bands, so the reference for gains is
  the native configuration. It reproduces the real-map results: +0.063 for
  NDMI recovery over Env+S (real maps +0.065); +0.049 for Tres over LVIS
  (real +0.048).

**Configurations:**
- **native:** AVIRIS-C bands, 30 m.
- **emit:** EMIT band centres and FWHM; 60 m; noise from the per-band
  reflectance uncertainty of an EMIT scene over the Sierra (2025-08-04,
  vegetated pixels).
- **sbg_hi / sbg_lo:** 10 nm bands, 30 m, the EMIT noise level or half of
  it. This brackets a 30 m spaceborne imaging spectrometer.
- **oli:** Landsat 8 OLI reflective bands (boxcar passbands), 30 m, no EWT.
  The multispectral control.

**Emulator out-of-fold R² against the 2403 maps:**

| Trait | native | emit | sbg_hi | sbg_lo | oli |
|---|---|---|---|---|---|
| Nitrogen | 0.81 | 0.61 | 0.70 | 0.73 | 0.53 |
| LMA | 0.95 | 0.86 | 0.92 | 0.94 | 0.73 |
| Lignin | 0.81 | 0.52 | 0.63 | 0.71 | 0.46 |
| Cellulose | 0.79 | 0.48 | 0.55 | 0.62 | 0.27 |

On fully QC-passing pixels the native emulator reaches 0.88–0.89 for N and
lignin.

**Trait gains** (Tres = residual traits; L = lidar structure):

| Target | Gain | native | emit | sbg_hi | sbg_lo | oli |
|---|---|---|---|---|---|---|
| NDMI recovery | T over Env+S | +0.063 | +0.051 | +0.052 | +0.060 | +0.044 |
| NDMI recovery | Tres over Env+S+L (ASO) | +0.039 | +0.033 | +0.037 | +0.038 | +0.025 |
| NDMI recovery | Tres over Env+S+L (LVIS) | +0.049 | +0.042 | +0.047 | +0.050 | +0.039 |
| NIRv recovery | Tres over Env+S+L (ASO / LVIS) | +0.041 / +0.054 | +0.048 / +0.056 | +0.047 / +0.052 | +0.046 / +0.053 | +0.042 / +0.054 |
| Stress response | Tres over Env+S+L (ASO / LVIS) | +0.020 / +0.022 | +0.025 / +0.030 | +0.024 / +0.028 | +0.022 / +0.029 | +0.019 / +0.022 |
| NDMI resistance | Tres over Env+S+L (ASO / LVIS) | +0.011 / +0.016 | +0.009 / +0.013 | +0.008 / +0.012 | +0.010 / +0.014 | +0.003 / +0.013 |

All entries have 95% CIs above 0 except Landsat-band resistance over ASO.

**Reading.**
- **Spaceborne-like sampling keeps most of the recovery signal.** Of the
  native NDMI-recovery gain beyond lidar, the EMIT-like configuration keeps
  85–86% and the 30 m configurations 95–102%. Stress-response gains are
  unchanged or slightly larger.
- **Landsat bands keep much of it too.** Trait proxies emulated from the
  seven OLI bands keep 64–80% of the NDMI-recovery gain over lidar, and
  nearly all of the NIRv-recovery and stress-response gains.
- Full spectra beat Landsat bands for NDMI recovery (ASO +0.033 to +0.038
  against +0.025; LVIS +0.042 to +0.050 against +0.039), but the CIs
  overlap. Much of the recovery-relevant information in a single June scene
  is broadband.
- **Caveats:**
  - one area (NEON) and one date;
  - emulated rather than published retrievals, so absolute values are
    uncertain;
  - the noise model is per-band and spatially independent;
  - the OLI proxies come from the same AVIRIS scene, so they carry none of
    Landsat's real calibration or atmospheric differences. §20 replaces
    them with a real Landsat 8 scene and a paired test.

### 19. Scale sensitivity: 30, 90 and 270 m (`hls_results/scale_sensitivity/`)

**Method.**
- The trait ladder and the lidar check are rerun at 30 m (single pixels;
  metrics in `response_30m/` and `response_sierra_30m/`) and at 270 m.
- Blocks stay 1 km.
- The 30 m ladders are fitted on a 200,000-cell sample. The 30 m lidar
  checks use every covered cell (215,000–495,000).
- The NEON 2013 tree source exists at 90 m only.

**Trait gains over Env+S** (30 / 90 / 270 m):

| Area | Target | Env+S R² | +T | +Tres |
|---|---|---|---|---|
| NEON | NDMI recovery | 0.61 / 0.68 / 0.77 | +0.050 / +0.065 / +0.065 | +0.041 / +0.046 / +0.037 |
| NEON | NDMI resilience | 0.58 / 0.63 / 0.70 | +0.034 / +0.059 / +0.070 | +0.028 / +0.042 / +0.042 |
| NEON | NDMI resistance | 0.66 / 0.73 / 0.80 | +0.023 / +0.024 / +0.027 | +0.018 / +0.018 / +0.007 |
| NEON | stress response | 0.64 / 0.71 / 0.80 | +0.021 / +0.022 / +0.022 | +0.018 / +0.014 / +0.006 |
| NEON | NIRv recovery | 0.64 / 0.69 / 0.76 | +0.043 / +0.046 / +0.055 | +0.034 / +0.033 / +0.036 |
| `sierra_nf` | NDMI recovery | 0.44 / 0.53 / 0.64 | +0.068 / +0.089 / +0.073 | +0.059 / +0.070 / +0.054 |
| `sierra_nf` | NDMI resilience | 0.54 / 0.60 / 0.69 | +0.037 / +0.052 / +0.042 | +0.031 / +0.042 / +0.023 |
| `sierra_nf` | NDMI resistance | 0.48 / 0.55 / 0.61 | +0.046 / +0.044 / +0.044 | +0.038 / +0.033 / +0.023 |
| `sierra_nf` | stress response | 0.41 / 0.49 / 0.60 | +0.044 / +0.055 / +0.065 | +0.037 / +0.042 / +0.037 |
| `sierra_nf` | NIRv recovery | 0.38 / 0.46 / 0.56 | +0.054 / +0.073 / +0.078 | +0.048 / +0.060 / +0.060 |

**Residual traits over Env+S+L** (30 / 90 / 270 m; 95% CI at 270 m):

| Area, source | NDMI recovery | NIRv recovery | Stress response | NDMI resistance |
|---|---|---|---|---|
| NEON, ASO | +0.040 / +0.038 / +0.030 [0.014, 0.046] | +0.045 / +0.044 / +0.063 | +0.030 / +0.024 / +0.010 (n.s.) | +0.019 / +0.010 / +0.005 (n.s.) |
| NEON, LVIS | +0.045 / +0.048 / +0.030 [0.010, 0.051] | +0.052 / +0.058 / +0.074 | +0.026 / +0.022 / −0.001 (n.s.) | +0.024 / +0.013 / +0.009 (n.s.) |
| `sierra_nf`, ASO | +0.030 / +0.025 / +0.024 [0.015, 0.033] | +0.030 / +0.031 / +0.031 | +0.022 / +0.021 / +0.012 | +0.023 / +0.012 / −0.000 (n.s.) |

**Reading.**
- **The recovery result does not depend on the 90 m grid.** Trait and
  residual-trait gains for NDMI and NIRv recovery are of the same size at
  30, 90 and 270 m, and survive lidar structure at every scale.
- At 30 m the gains over Env+S are a little smaller (more pixel noise in
  both traits and responses).
- Stress response and resistance keep their residual gains over lidar at
  30 and 90 m. At 270 m, with only 2,300–5,200 cells, resistance loses them
  everywhere and stress response at NEON. `sierra_nf` keeps +0.012 [0.003,
  0.021].

### 20. Do full spectra add to multispectral bands? A paired test with real Landsat (`hls_results/spaceborne_paired/`)

§18 left open whether full-spectrum (VSWIR) inputs beat Landsat bands. Its
separate CIs overlapped, and its Landsat-band control came from the AVIRIS
scene itself. This section closes both gaps.
- **Paired test:**
  - `proto_response_traits.py --save-preds` writes each configuration's
    out-of-fold predictions;
  - `proto_spaceborne_sim.py compare` joins them on the cells and
    bootstraps the *difference* in gain on the same resampled blocks;
  - the base models use no traits and agree across configurations
    (to 1e-5), so the difference in gain is the difference in R² of
    the trait models.
- **Second area:** `sierra_nf` (2013-06-12 `_v2` lines), over ASO (48,731
  cells), alongside NEON over ASO and LVIS.
- **Real Landsat:**
  - Landsat 8 Collection 2 surface reflectance, LC08 042034 of 2013-06-21,
    nine days after the AVIRIS flight (`fetch_landsat_c2_ee.py --scene`);
  - clear over 98% of both AOIs;
  - the 2013-06-05 scene is 66% clear at NEON;
  - the same-day 2013-06-12 path-43 scene covers 4% of `sierra_nf`.
- **Two new configurations:**
  - **l8:** the seven OLI bands through the same out-of-fold emulator to
    the 2403 maps;
  - **l8raw:** the bands, NDVI, NDMI, NBR and NIRv themselves as the trait
    block (`--raw-block`), with no airborne training at all.
- Runs restricted to NDMI and NIRv recovery and stress response reproduce
  the §18 gains exactly.

**Emulator out-of-fold R² against the 2403 maps, `sierra_nf` (NEON in
§18):**

| Trait | native | emit | sbg_hi | sbg_lo | oli | l8 (NEON l8) |
|---|---|---|---|---|---|---|
| Nitrogen | 0.79 | 0.57 | 0.67 | 0.70 | 0.51 | 0.46 (0.51) |
| LMA | 0.92 | 0.79 | 0.87 | 0.90 | 0.65 | 0.59 (0.72) |
| Lignin | 0.82 | 0.56 | 0.65 | 0.73 | 0.48 | 0.38 (0.36) |
| Cellulose | 0.75 | 0.44 | 0.52 | 0.59 | 0.25 | 0.31 (0.38) |

Real Landsat reproduces the airborne trait maps a little less well than the
simulated OLI bands do.

**NDMI recovery gains** (Tres over Env+S+L; T over Env+S):

| Configuration | NEON, ASO | NEON, LVIS | `sierra_nf`, ASO | NEON, over Env+S | `sierra_nf`, over Env+S |
|---|---|---|---|---|---|
| native | +0.039 | +0.049 | +0.023 | +0.063 | +0.086 |
| emit | +0.033 | +0.042 | +0.017 | +0.051 | +0.068 |
| sbg_hi | +0.037 | +0.047 | +0.019 | +0.052 | +0.070 |
| sbg_lo | +0.038 | +0.050 | +0.021 | +0.060 | +0.080 |
| oli (simulated) | +0.025 | +0.039 | +0.013 | +0.044 | +0.063 |
| **l8** (real Landsat, emulated) | +0.035 | +0.046 | **+0.027** | +0.054 | +0.083 |
| **l8raw** (real Landsat, no emulator) | +0.029 | +0.041 | +0.023 | +0.046 | +0.060 |

**Paired differences, NDMI recovery** (95% CI over 1 km blocks):

| Difference | NEON, ASO | NEON, LVIS | `sierra_nf`, ASO | NEON, over Env+S | `sierra_nf`, over Env+S |
|---|---|---|---|---|---|
| native − oli | **+0.014** [0.008, 0.020] | **+0.010** [0.002, 0.019] | **+0.010** [0.007, 0.014] | **+0.019** [0.014, 0.024] | **+0.023** [0.019, 0.027] |
| emit − oli | **+0.009** [0.003, 0.015] | +0.003 [−0.003, 0.010] | **+0.004** [0.001, 0.007] | **+0.007** [0.003, 0.011] | **+0.005** [0.000, 0.009] |
| sbg_lo − oli | **+0.014** [0.008, 0.020] | **+0.012** [0.005, 0.020] | **+0.008** [0.005, 0.011] | **+0.016** [0.011, 0.021] | **+0.017** [0.013, 0.021] |
| native − l8 | +0.004 [−0.004, 0.011] | +0.002 [−0.006, 0.010] | −0.004 [−0.007, 0.000] | **+0.009** [0.003, 0.016] | +0.003 [−0.002, 0.008] |
| emit − l8 | −0.002 [−0.008, 0.004] | −0.005 [−0.012, 0.003] | **−0.010** [−0.013, −0.007] | −0.003 [−0.008, 0.003] | **−0.015** [−0.021, −0.010] |
| sbg_lo − l8 | +0.003 [−0.004, 0.010] | +0.004 [−0.005, 0.013] | **−0.006** [−0.010, −0.002] | +0.006 [−0.000, 0.013] | −0.003 [−0.008, 0.002] |
| native − l8raw | **+0.010** [0.003, 0.018] | +0.008 [−0.001, 0.017] | +0.000 [−0.004, 0.004] | **+0.018** [0.011, 0.025] | **+0.026** [0.020, 0.033] |

- **5 km blocks** widen the intervals slightly. native − l8 over Env+S at
  NEON becomes [−0.000, 0.019].
- **NIRv recovery:** the VSWIR configurations do no better than simulated
  OLI at NEON. At `sierra_nf` they beat it by +0.007 to +0.010, but l8 beats
  every VSWIR configuration by +0.009 to +0.011.
- **Stress response:** every VSWIR configuration does *worse* than real
  Landsat (−0.009 to −0.018 vs l8; −0.015 to −0.025 vs l8raw, at both
  areas).

**Reading.**
- **Against simulated OLI, full spectra win** for NDMI recovery, at both
  areas and over every lidar source: +0.010 to +0.014 for native AVIRIS,
  +0.008 to +0.014 for the 30 m 10 nm configuration. The EMIT-like
  configuration wins narrowly (+0.003 to +0.009, one CI touching 0).
- **Against a real Landsat 8 scene, they do not.**
  - Emulated from real Landsat, the trait block gives NDMI-recovery gains
    over lidar equal to native AVIRIS at NEON, and slightly larger at
    `sierra_nf` (CI touching 0).
  - Every spaceborne-like VSWIR configuration is equal to or below it.
- The real Landsat bands beat the simulated ones by +0.008 to +0.020.
  Simulated OLI is therefore not a fair multispectral control: real Landsat
  carries information the AVIRIS-convolved bands do not.
  - A second date: June 21 rather than June 12.
  - Landsat's own geometry and registration, shared with the Landsat-based
    response metrics.
  - A view geometry free of AVIRIS cross-track effects.
- **What full spectra still add.** Without lidar, native AVIRIS traits beat
  the raw Landsat block by +0.018 (NEON) and +0.026 (`sierra_nf`), and
  emulated Landsat at NEON by +0.009. The advantage is in the retrievals at
  airborne quality, against Landsat with no airborne training. It does not
  survive spaceborne-like noise and sampling, or comparison with
  airborne-trained Landsat proxies.
- **Recovery-relevant trait information is largely multispectral** once
  lidar structure is in the model. Airborne trait maps make it usable by
  training Landsat proxies: l8 beats l8raw by +0.004 to +0.007 for NDMI
  recovery over lidar, and by +0.009 to +0.023 over Env+S.
- **Caveats:**
  - one June date per area;
  - emulated rather than published retrievals;
  - the Landsat responses share sensor and registration with the l8 inputs,
    which may favour them;
  - the 2013 scene falls inside the 2012–16 drought, so it also carries
    early stress (relevant to the stress-response result).

### 21. Why does the cross-drought transfer rank `sierra_nf` cells in reverse? (`hls_results/transfer_diagnostics/`)

`proto_transfer_diagnostics.py` refits the §11 transfers (cycle-1 model,
Env+S or Env+S+T13, applied to cycle 2 with T18 swapped in) on the same
cells. It then asks where the rank skill is lost. It reproduces §11:
`sierra_nf` NDMI recovery ρ −0.01 (Env+S) and +0.10 (with traits); NEON
+0.20 and +0.33.

**Covariates per 90 m cell:**
- distance to the nearest 2020–21 fire perimeter (FRAP; mostly the Creek
  Fire);
- distance to the nearest FACTS harvest unit completed in 2016–25;
- the share of the cell mapped in an ADS mortality polygon in 2015–17;
- the cell's cycle-1 response;
- the change in drought forcing;
- elevation;
- climatic CWD.

ρ CIs come from a 1 km block bootstrap.

**Not the fire edge, and not treatments.**
- 28% of the `sierra_nf` cells lie within 2 km of a 2020–21 fire.
- Dropping them makes the transfer *worse*: NDMI recovery ρ −0.04 at
  ≥ 2 km and −0.10 [−0.21, 0.01] at ≥ 5 km. NIRv recovery goes from −0.23
  to −0.29 and −0.37.
- Cells nearest the fire rank best (0–2 km: +0.09).
- Recent treatments are within 1 km of only 2% of the cells, and dropping
  them changes nothing.

**Not cycle-1 damage as ADS maps it.**
- 97% of the `sierra_nf` cells have some pixel inside a 2015–17 mortality
  polygon, so the ADS extent says little there.
- Across its quartiles the transfer ρ shows no trend (+0.04 to +0.24).

**The elevation gradient of the response flips between the droughts.**
Spearman ρ of elevation with each response, in cycle 1 and cycle 2, on the
same cells:

| Response | `sierra_nf` cycle 1 → 2 | NEON cycle 1 → 2 |
|---|---|---|
| NDMI resistance | +0.40 → **−0.59** | +0.73 → +0.23 |
| NDMI recovery | +0.35 → **−0.44** | −0.20 → −0.69 |
| NIRv recovery | +0.09 → **−0.40** | −0.40 → −0.70 |
| NDMI resilience | +0.50 → **−0.62** | +0.50 → **−0.37** |

- At `sierra_nf`, low-elevation cells responded worse in 2012–16 and
  better in 2020–22, for every response.
- Climatic temperature flips with it (recovery −0.35 → +0.42), and the
  canopy height and cover associations lose or flip their sign.
- The local drought forcing does not flip: ρ of the drought CWD anomaly
  with recovery is +0.23 in cycle 1 and +0.20 in cycle 2.
- At NEON only resilience flips, which is the resilience reversal that §4
  and §11 report at both AOIs.

**Within elevation quartiles the transfer ranks cells correctly.**

| `sierra_nf`, within elevation quartile | q1 | q2 | q3 | q4 |
|---|---|---|---|---|
| NDMI recovery, Env+S | +0.24 | +0.23 | +0.20 | +0.29 |
| NDMI recovery, Env+S+T | +0.20 | +0.21 | +0.36 | +0.50 |
| NDMI resistance, Env+S | −0.05 | −0.09 | −0.03 | +0.04 |
| NIRv recovery, Env+S | −0.06 | −0.14 | −0.03 | −0.09 |

All CIs on the NDMI-recovery entries exclude 0. Compare the pooled values:
NDMI recovery −0.01, resistance −0.38, NIRv recovery −0.23.
- With traits, the within-band ranking improves most at high elevation
  (+0.36 and +0.50 in the top two quartiles).
- Dropping the quarter of cells with the weakest cycle-1 recovery also
  helps: NDMI recovery +0.06 (Env+S) and +0.22 (with traits).

**Reading.**
- The reverse ranking at `sierra_nf` is a between-elevation effect. The
  first drought hit the lower slopes hardest and the second hit the upper
  slopes hardest, so a model that learned the first gradient ranks the
  second backwards.
- Within elevation bands the cycle-1 model, and its traits, still rank
  cycle-2 cells in the right order.
- Why the gradient flipped is not settled here. Candidates are the upslope
  shift of the 2020–22 drought's impact, the first drought's losses
  lowering the low-elevation baseline, and snow years in the post window
  (2023).
- **For any forward test:** report rank skill within elevation (or
  climate) strata as well as pooled, or include an explicit term for how
  the response's elevation gradient changes between droughts.

### 22. Do mortality agent, host or drought timing explain the Tahoe-box directions? (`hls_results/trait_directions_groups/`)

§9 ruled out calibration, LANDFIRE forest type and signal strength as
reasons why residual N and LMA track NDMI recovery in the Yosemite box but
not in the Tahoe box. This section tests three further candidates.
`proto_trait_directions.py --group` splits the cells, using cycle 1 only
and the same residual traits and within-stratum ρ as §9.

**Groups:**
- **agent:** the dominant mortality agent that the Aerial Detection Survey
  mapped in 2014–17. Classes are fir engraver, pine beetles (mountain,
  western and Jeffrey pine beetle) and none. A class counts as dominant
  when it covers ≥ 25% of the cell's pixels.
- **host:** the dominant host of that mapped mortality: white fir,
  California red fir or pine.
- **cwd_late:** terciles of 2016's share of the cell's 2012–16 CWD anomaly.
  The box-mean anomaly peaks in 2014 in the Yosemite box and in 2016 in the
  Tahoe box.
- **ewt_min:** the year of the cell's lowest June EWT in 2014–16. Most
  cells reach it in 2016 in the Yosemite box and in 2015 in the Tahoe box;
  20,000 Tahoe cells lack 2016 coverage.

**Residual N / LMA vs NDMI recovery** (within-stratum ρ; 95% CIs from 1 km
blocks; groups of ≥ 1,000 cells):

| Group | NEON | `sierra_nf` | Tahoe box |
|---|---|---|---|
| all cells | +0.19 / −0.17 | +0.22 / −0.19 | +0.03 / +0.02 |
| agent: fir engraver | +0.19 / −0.21 | +0.21 / −0.19 | +0.03 / +0.02 |
| agent: pine beetles | +0.18 / −0.14 | +0.22 / −0.20 | +0.01 / +0.06 |
| agent: none mapped | +0.16 / −0.16 | +0.20 / −0.12 | +0.04 / +0.02 |
| host: white fir | +0.04 / −0.16 (1,635 cells) | +0.26 / −0.26 | +0.05 / −0.03 |
| host: red fir | +0.26 / −0.28 | +0.19 / −0.20 | −0.01 / **+0.06** |
| host: pine | +0.18 / −0.15 | +0.21 / −0.17 | +0.03 / +0.04 |
| CWD peak late (top tercile) | +0.23 / −0.27 | +0.27 / −0.25 | +0.03 / +0.04 |
| CWD peak early (bottom tercile) | +0.18 / −0.14 | +0.18 / −0.15 | +0.02 / +0.02 |
| EWT minimum in 2015 | +0.17 / −0.14 | +0.27 / −0.22 | +0.04 / +0.01 |
| EWT minimum in 2016 | +0.15 / −0.13 | +0.17 / −0.15 | −0.03 / +0.03 |

**Structural carbon** (residual lignin and cellulose → worse recovery)
holds in every group in both boxes. Lignin is −0.06 to −0.15 in the Tahoe
box and −0.10 to −0.22 in the Yosemite box. The one exception is the 1,195
Tahoe cells whose EWT minimum fell in 2014 (not significant).

**Reading.**
- **None of the three candidates explains the difference.** In the Tahoe
  box, N and LMA stay near 0 in every agent, host and timing group,
  including fir-engraver and red-fir cells and cells that are dry late in
  the drought.
- In the Yosemite box the leaf-economics direction holds in all those
  groups. That includes the cells whose timing matches the Tahoe box (EWT
  minimum in 2015: N +0.17 and +0.27).
- The only Tahoe signal is a weak N direction under white-fir mortality
  (+0.05 [0.02, 0.09]). Red-fir cells lean the other way for LMA (+0.06
  [0.02, 0.12]).
- **What separates the two boxes is a property of the box, not of these
  stand attributes.** Untested candidates:
  - **substrate and soils:** volcanic mudflow and andesitic soils are
    common in the northern Sierra, granitic soils in the Yosemite box; the
    link between foliar N and growth could differ with nutrient supply;
  - species mixes that ADS host codes do not resolve;
  - retrieval behaviour specific to the Tahoe box that the §9 calibration
    checks would not catch.
- The structural-carbon direction is the one that generalizes across
  boxes, agents, hosts and drought timing.

### 23. Spaceborne-like VSWIR vs Landsat without lidar or local airborne training (`hls_results/spaceborne_only/`)

§20 compared spaceborne-like VSWIR with Landsat trait proxies trained on the
AVIRIS maps of the same place and date (l8). Beyond the airborne boxes there
are no such maps, and no wall-to-wall lidar. This section uses Landsat
controls that need no airborne training, and models with no lidar.

**Method** (`proto_spaceborne_only.py compare`).
- Blocks:
  - the emulated 2013 VSWIR retrievals of §18/§20 (native, emit, sbg_hi,
    sbg_lo) and l8;
  - **l8raw:** the 2013-06-21 Landsat 8 scene's bands and indices (§20);
  - **l8multi (new):** the same variables from the June (DOY 145–190) and
    Jul–Sep (DOY 182–273) 2013 Landsat 8 composites. Time series are what
    make multispectral trait proxies work (e.g. Liu et al. 2024, Sentinel-2).
- All blocks go into one cell table, so every model shares the cells and
  folds. Differences in R² are paired by construction.
- Over Env+S, no lidar: Env+S | Env+S+T<block>.
- **Spaceborne stack:** Env+S+Ŝ+T, where Ŝ is the lidar structure predicted
  out of fold from the *same* block (with wall-to-wall structure and
  Landsat baseline greenness, as in §15). It is scored against
  Env+S+L+T(native): lidar plus airborne retrievals.
- The Jul–Sep 2013 composite is one of the twelve years the stress-response
  slope is fitted on, so l8multi is partly circular for that target.
  Recovery (2017–19 vs 2014–16) does not use 2013.

**NEON, over Env+S** (44,079 cells; ΔR² with paired 95% CIs, 1 km blocks):

| Block | NDMI recovery | − l8raw | − l8multi | NIRv recovery − l8multi | Stress response − l8multi |
|---|---|---|---|---|---|
| native | +0.068 | **+0.022** | **+0.007** [0.001, 0.014] | −0.006 (n.s.) | **−0.009** |
| emit | +0.055 | **+0.009** | −0.005 [−0.011, 0.001] | **−0.008** | **−0.007** |
| sbg_hi | +0.058 | **+0.012** | −0.003 (n.s.) | **−0.008** | **−0.008** |
| sbg_lo | +0.064 | **+0.019** | +0.004 [−0.002, 0.011] | −0.005 (n.s.) | **−0.007** |
| l8 | +0.054 | **+0.009** | **−0.006** | **−0.012** | **−0.008** |
| l8raw | +0.046 | – | **−0.015** | **−0.017** | **−0.008** |
| l8multi | **+0.060** | **+0.015** | – | – | – |

Bold: CI excludes 0. Over 5 km blocks the intervals widen, and native −
l8multi becomes [−0.002, 0.020].

**`sierra_nf`, over Env+S** (58,676 cells):

| Block | NDMI recovery | − l8raw | − l8multi | NIRv recovery − l8multi | Stress response − l8multi |
|---|---|---|---|---|---|
| native | +0.094 | **+0.034** | **+0.019** [0.014, 0.026] | +0.005 (n.s.) | **−0.013** |
| emit | +0.072 | **+0.012** | −0.002 [−0.008, 0.003] | **−0.009** | **−0.015** |
| sbg_hi | +0.074 | **+0.014** | −0.001 (n.s.) | **−0.006** | **−0.020** |
| sbg_lo | +0.084 | **+0.024** | **+0.009** [0.004, 0.015] | −0.001 (n.s.) | **−0.017** |
| l8 | +0.083 | **+0.023** | **+0.008** | −0.001 (n.s.) | **−0.016** |
| l8raw | +0.060 | – | **−0.014** | **−0.018** | **−0.010** |
| l8multi | +0.075 | **+0.014** | – | – | – |

**Spaceborne stack, NEON over ASO** (29,004 cells; R², and paired
differences for NDMI recovery):

| Inputs | NDMI recovery | NIRv recovery | Stress response | NDMI recovery − l8multi stack |
|---|---|---|---|---|
| Env+S | 0.506 | 0.402 | 0.509 | |
| Env+S+L+T(native), reference | 0.652 | 0.540 | 0.589 | |
| native | 0.615 (0.94) | 0.504 | 0.565 | **+0.011** [0.001, 0.020] |
| emit | 0.604 (0.93) | 0.506 | 0.567 | −0.000 [−0.009, 0.009] |
| sbg_hi | 0.607 (0.93) | 0.504 | 0.566 | +0.003 (n.s.) |
| sbg_lo | 0.619 (0.95) | 0.510 | 0.570 | **+0.015** [0.005, 0.024] |
| l8raw | 0.590 (0.91) | 0.498 | 0.569 | **−0.014** |
| l8multi | 0.604 (0.93) | 0.516 | 0.586 | – |

In brackets: share of the reference R² kept.

Over LVIS 2008 (21,826 cells; reference 0.656 for NDMI recovery) the stacks
keep 0.88 (l8raw) to 0.95 (native) of the reference. Against the l8multi
stack: native **+0.025**, sbg_lo **+0.022** [0.010, 0.035], sbg_hi +0.010
[−0.001, 0.022], emit +0.002 (n.s.).

Over ASO at `sierra_nf` (48,731 cells; reference 0.643 for NDMI recovery)
the stacks keep 0.90 (l8raw) to 0.95 (native): emit 0.92, sbg_hi 0.92,
sbg_lo 0.94, l8multi 0.93. Against the l8multi stack: native **+0.014**,
sbg_lo +0.006 [−0.000, 0.012], sbg_hi −0.004 (n.s.), emit **−0.009**. For
stress response every VSWIR stack is below both Landsat stacks.

**Reading.**
- **Spaceborne-like VSWIR beats a single Landsat scene** for NDMI recovery
  at both areas (EMIT-like +0.009 / +0.012, 30 m +0.012 to +0.024) and for
  NIRv recovery, with no airborne training on either side.
- **It does not beat a multi-date Landsat block.**
  - For NDMI recovery the EMIT-like and the noisier 30 m configuration tie
    with it at both areas (−0.005 to −0.001). The low-noise 30 m
    configuration is ahead at `sierra_nf` (+0.009) but not at NEON
    (+0.004, n.s.).
  - For NIRv recovery and stress response, l8multi equals or beats every
    VSWIR block.
  - Native AVIRIS beats it at both areas (+0.007, +0.019).
- **Without lidar, the full stack keeps 88–95% of the lidar-plus-AVIRIS
  reference for NDMI recovery, whatever the inputs.** The multi-date
  Landsat stack keeps as much as the EMIT-like one (0.91–0.93) and beats
  it at `sierra_nf` (−0.009). The low-noise 30 m stack is ahead of it at
  NEON over both lidar sources (+0.015, +0.022), not at `sierra_nf`
  (+0.006, n.s.).
- A second June scene and a summer composite give Landsat what one scene
  lacks. Most of the recovery-relevant information that spaceborne-like
  VSWIR carries, multi-date multispectral data carry too.

### 24. Is structural carbon a VSWIR signal? (`hls_results/spaceborne_only/carbon_<aoi>/`)

The structural-carbon direction (high lignin, cellulose and fiber → worse
recovery) is the one that replicates in both boxes (§8, §9, §22). Is it
something spaceborne VSWIR resolves and Landsat does not? Broadband "trait"
proxies may carry canopy structure and greenness rather than chemistry
(Knyazikhin et al. 2013).

**Method** (`proto_spaceborne_only.py carbon`). The same NEON cells and blocks
as §23.
1. **Blocks of single trait groups** over Env+S, per emulated configuration:
   - SC: lignin, cellulose, fiber;
   - NL: nitrogen, LMA.
2. **The same blocks on top of Landsat alone** (Env+S+l8raw, Env+S+l8multi),
   paired against the same traits emulated from real Landsat (l8). l8 is
   the negative control: its traits are functions of the scene's bands.
3. **Residual-trait directions** (within-stratum ρ, 1 km block-bootstrap
   CIs, as in §22), with each trait cross-fitted on Env+S, and on
   Env+S+l8raw (what the trait carries beyond the Landsat bands).

**NDMI recovery, NEON** (ΔR², 95% CI):

| SC or NL from | SC over Env+S | SC over Env+S+l8raw (− the l8 SC) | SC over Env+S+l8multi (− the l8 SC) | NL over Env+S+l8multi (− the l8 NL) |
|---|---|---|---|---|
| native | +0.032 | +0.016 (**+0.013**) | +0.014 (**+0.009**) | +0.006 (+0.002, n.s.) |
| emit | +0.024 | +0.010 (**+0.006**) | +0.008 (**+0.004**) | +0.004 (+0.001, n.s.) |
| sbg_lo | +0.032 | +0.013 (**+0.010**) | +0.012 (**+0.007**) | +0.004 (+0.001, n.s.) |
| l8 | +0.038 | +0.004 | +0.004 | +0.004 |

For NIRv recovery the VSWIR SC blocks also add over the Landsat blocks
(+0.007 to +0.010), but their lead over the l8 SC block is not
distinguishable from 0 (+0.004 to +0.006).

**NDMI recovery, `sierra_nf`** (58,676 cells; same columns):

| SC or NL from | SC over Env+S | SC over Env+S+l8raw (− the l8 SC) | SC over Env+S+l8multi (− the l8 SC) | NL over Env+S+l8multi (− the l8 NL) |
|---|---|---|---|---|
| native | +0.045 | +0.016 (**+0.005**) | +0.011 (**+0.004**) | +0.012 (−0.002, n.s.) |
| emit | +0.031 | +0.009 (−0.002, n.s.) | +0.004 (−0.003, n.s.) | +0.009 (**−0.005**) |
| sbg_lo | +0.040 | +0.012 (+0.001, n.s.) | +0.007 (+0.000, n.s.) | +0.009 (**−0.005**) |
| l8 | +0.052 | +0.011 | +0.007 | +0.014 |

**Residual directions vs NDMI recovery** (within-stratum ρ):

| Traits from | Residual on | Lignin | Cellulose | Fiber | N | LMA |
|---|---|---|---|---|---|---|
| maps | Env+S | −0.17 | −0.15 | −0.19 | +0.19 | −0.17 |
| emit | Env+S | −0.14 [−0.17, −0.11] | −0.13 [−0.16, −0.11] | −0.17 | +0.18 | −0.15 |
| sbg_lo | Env+S | −0.16 | −0.16 | −0.20 | +0.19 | −0.15 |
| l8 | Env+S | −0.19 | **−0.02** [−0.04, 0.01] | −0.15 | +0.23 | −0.19 |
| maps | Env+S+l8raw | −0.11 | −0.12 | −0.13 | +0.09 | −0.06 |
| emit | Env+S+l8raw | −0.09 [−0.11, −0.06] | −0.11 [−0.13, −0.08] | −0.11 | +0.07 | −0.04 |
| sbg_lo | Env+S+l8raw | −0.10 | −0.13 | −0.13 | +0.08 | −0.04 |
| l8 | Env+S+l8raw | −0.07 | **−0.03** | −0.06 | +0.08 | −0.06 |

At `sierra_nf`:

| Traits from | Residual on | Lignin | Cellulose | Fiber | N | LMA |
|---|---|---|---|---|---|---|
| maps | Env+S | −0.18 | −0.16 | −0.18 | +0.21 | −0.19 |
| emit | Env+S | −0.15 [−0.17, −0.14] | −0.12 [−0.14, −0.11] | −0.16 | +0.20 | −0.17 |
| sbg_lo | Env+S | −0.18 | −0.16 | −0.19 | +0.21 | −0.18 |
| l8 | Env+S | −0.15 | **−0.03** [−0.05, −0.01] | −0.13 | +0.24 | −0.21 |
| maps | Env+S+l8raw | −0.13 | −0.12 | −0.13 | +0.13 | −0.11 |
| emit | Env+S+l8raw | −0.10 [−0.11, −0.08] | −0.08 [−0.10, −0.07] | −0.10 | +0.11 | −0.09 |
| sbg_lo | Env+S+l8raw | −0.12 | −0.10 | −0.12 | +0.12 | −0.09 |
| l8 | Env+S+l8raw | **−0.06** | **−0.02** | **−0.06** | +0.12 | −0.10 |

**Reading.**
- **The structural-carbon *direction* is a VSWIR signal, at both areas.**
  - Emulated from spaceborne-like spectra, structural carbon keeps the
    directions of the maps (EMIT-like lignin −0.14 / −0.15, cellulose
    −0.13 / −0.12 at NEON / `sierra_nf`, CIs excluding 0).
  - Emulated from Landsat, cellulose loses its direction (−0.02 / −0.03).
  - Beyond the Landsat bands, VSWIR lignin and cellulose still go with
    worse recovery (−0.08 to −0.13). Landsat "lignin" and "cellulose" keep
    only −0.02 to −0.07.
- **The added *skill* is VSWIR-specific at NEON only.**
  - At NEON, VSWIR structural carbon adds more on top of either Landsat
    block than the l8 version does (+0.004 to +0.013).
  - At `sierra_nf` it adds no more than the l8 version (−0.003 to +0.001);
    only native AVIRIS does (+0.004 to +0.005).
- **Leaf economics is not a VSWIR signal.**
  - N and LMA from VSWIR add no more beyond the multi-date Landsat block
    than the Landsat-emulated N and LMA do (`sierra_nf`: less, −0.005).
  - Their directions beyond the Landsat bands are the same for every
    source.
- **Alone, the Landsat-emulated blocks add the most** (SC +0.038 / +0.052;
  NL +0.036 / +0.068). The broadband proxy carries skill, but as greenness
  and structure rather than chemistry: its cellulose has no direction.

### 25. Do trait retrievals carry over to another area or date? (`hls_results/retrieval_transfer/`)

A Landsat trait proxy is trained on AVIRIS maps of the same place and date.
Does a spaceborne-like VSWIR retrieval need that too, or does it hold where
it was not trained?

**Method** (`proto_retrieval_transfer.py`).
- **Emulators** (as in §18): PLSR (25 components; rerun with 10, below)
  from log inputs to the 2403 maps, for native, emit and sbg_lo (VSWIR), and two Landsat proxies:
  - l8: one Landsat 8 scene near the flight;
  - l8multi: the June and Jul–Sep composites of the year.
  Noise draws are seeded per domain.
- **Transfers** (fit on the source, apply to the target):

  | Leg | Source → target | Responses scored |
  |---|---|---|
  | Across areas | NEON 2013 ↔ `sierra_nf` 2013 (same flight day and Landsat scene) | cycle 1 |
  | Across boxes | Yosemite box 2013 (both AOIs) → Tahoe box 2013 (June 4 flight, non-`_v2`, Landsat path 43, 2013-06-28) | cycle 1 only |
  | Across dates | NEON 2013 → NEON 2018 and `sierra_nf` 2013 → 2018 (June 22 flight, non-`_v2`, Landsat 2018-06-19) | cycle 2 (pilot AOIs) |

- New inputs:
  - the June 2018 reflectance for NEON and `sierra_nf`, and the Tahoe
    2013 reflectance (`proto_spaceborne_sim.py refl --year`);
  - Landsat 8 scenes 2018-06-19 and (Tahoe) 2013-06-28. The 2013-06-12
    Tahoe scene is only 48% clear.
- Pixels: those whose spectra come from the trait mosaic's own line, with
  every configuration's inputs present.
  - The Tahoe 2013 lines have no 1323 and 1333 nm data. Those bands leave
    the usable set there.
- **In place:** the same configuration's emulator fitted out of fold over
  1 km blocks within the target.
- **Scores** on the 90 m cells with responses, with paired 1 km block
  bootstraps:
  - agreement with the target's AVIRIS maps (ρ);
  - the NDMI-recovery gain over Env+S with in-place vs transferred traits;
  - the difference in loss (VSWIR − Landsat proxy; negative means VSWIR
    loses less);
  - transferred VSWIR against Landsat alone (l8raw, l8multi as raw blocks).
    In the first four 25-component legs the raw l8multi contrast replaced
    the one against the transferred l8multi emulator, which shared its
    name; later runs keep both (`*_raw` contrasts).

**NDMI-recovery gain over Env+S, in place → transferred:**

| Target (cells) | native | emit | sbg_lo | l8 | l8multi | l8raw (raw) | l8multi (raw) |
|---|---|---|---|---|---|---|---|
| NEON 2013 ← `sierra_nf` (43,247) | +0.061 → +0.058 | +0.048 → +0.048 | +0.056 → +0.057 | +0.049 → +0.051 | +0.060 → +0.058 | +0.042 | +0.057 |
| `sierra_nf` 2013 ← NEON (58,239) | +0.085 → +0.088 | +0.064 → +0.070 | +0.078 → +0.080 | +0.084 → +0.080 | +0.088 → +0.087 | +0.061 | +0.075 |
| Tahoe 2013 ← Yosemite box (55,106) | +0.037 → +0.022 | +0.039 → +0.030 | +0.046 → +0.031 | +0.042 → +0.031 | +0.044 → +0.040 | +0.036 | +0.047 |
| NEON 2018 ← NEON 2013 (23,251; cycle 2) | +0.063 → +0.061 | +0.055 → +0.056 | +0.062 → +0.062 | +0.070 → +0.073 | +0.080 → +0.081 | +0.077 | +0.087 |
| `sierra_nf` 2018 ← 2013 (20,809; cycle 2) | +0.098 → +0.081 | +0.086 → +0.071 | +0.092 → +0.084 | +0.099 → +0.103 | +0.097 → +0.105 | +0.108 | +0.114 |

**Paired contrasts, NDMI recovery** (95% CI):

| Contrast | NEON ← `sierra_nf` | `sierra_nf` ← NEON | Tahoe ← Yosemite | NEON 2018 ← 2013 | `sierra_nf` 2018 ← 2013 |
|---|---|---|---|---|---|
| loss, emit − loss, l8 | +0.003 (n.s.) | **−0.009** | −0.002 (n.s.) | +0.002 (n.s.) | **+0.019** |
| loss, emit − loss, l8multi | −0.002 (n.s.) | **−0.006** | +0.005 (n.s.) | −0.001 (n.s.) | **+0.022** |
| loss, sbg_lo − loss, l8multi | −0.003 (n.s.) | −0.003 (n.s.) | +0.010 [−0.001, 0.022] | +0.001 (n.s.) | **+0.015** |
| transferred emit − l8multi raw block | **−0.009** | −0.005 [−0.011, 0.000] | **−0.017** | **−0.031** | **−0.043** |
| transferred sbg_lo − l8multi raw block | −0.001 (n.s.) | +0.005 (n.s.) | **−0.016** | **−0.025** | **−0.030** |
| transferred sbg_lo − l8raw | **+0.014** | **+0.019** | −0.004 (n.s.) | **−0.015** | **−0.023** |

**Agreement with the target's AVIRIS maps** (ρ, transferred; native / emit /
sbg_lo / l8 / l8multi), and the paired difference in loss of ρ (emit − l8):

| Target | Nitrogen | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| NEON 2013 | 0.90 / 0.89 / 0.92 / 0.76 / 0.83 (**−0.03**) | 0.99 / 0.97 / 0.98 / 0.87 / 0.90 (**−0.02**) | 0.76 / 0.74 / 0.83 / 0.55 / 0.61 (+0.02) | 0.82 / 0.81 / 0.85 / 0.72 / 0.71 (**+0.03**) |
| `sierra_nf` 2013 | 0.89 / 0.85 / 0.89 / 0.67 / 0.71 (**−0.02**) | 0.95 / 0.91 / 0.94 / 0.75 / 0.76 (+0.01) | 0.83 / 0.76 / 0.82 / 0.59 / 0.63 (**+0.07**) | 0.71 / 0.75 / 0.80 / 0.64 / 0.67 (**+0.06**) |
| Tahoe 2013 | 0.72 / 0.85 / 0.76 / 0.67 / 0.73 (**−0.03**) | 0.55 / 0.69 / 0.84 / 0.71 / 0.76 (**+0.19**) | 0.27 / 0.75 / 0.63 / 0.68 / 0.71 (**+0.08**) | 0.07 / 0.77 / 0.67 / 0.68 / 0.65 (**+0.07**) |
| NEON 2018 | 0.89 / 0.91 / 0.93 / 0.80 / 0.81 (**−0.02**) | 0.96 / 0.94 / 0.96 / 0.84 / 0.87 (**−0.03**) | 0.89 / 0.48 / 0.65 / 0.42 / 0.50 (**+0.15**) | 0.87 / 0.63 / 0.69 / 0.63 / 0.62 (**+0.11**) |

**`sierra_nf` 2018, map agreement** (ρ, transferred; native / emit / sbg_lo /
l8 / l8multi): N 0.90 / 0.85 / 0.90 / 0.74 / 0.75; LMA 0.95 / 0.89 / 0.94 /
0.79 / 0.82; lignin 0.88 / 0.76 / 0.87 / 0.63 / 0.66; cellulose 0.85 / 0.78 /
0.85 / 0.77 / 0.73.

**Rerun with 10-component emulators** (`hls_results/retrieval_transfer_nc10/`;
§26 found 10 components carry across sensors and 25 do not).

| Contrast, NDMI recovery | NEON ← `sierra_nf` | `sierra_nf` ← NEON | Tahoe ← Yosemite | NEON 2018 ← 2013 | `sierra_nf` 2018 ← 2013 |
|---|---|---|---|---|---|
| loss, emit − loss, l8 | +0.004 (n.s.) | **−0.004** | −0.011 [−0.024, 0.003] | −0.001 (n.s.) | **+0.013** |
| loss, emit − loss, l8multi | −0.002 (n.s.) | −0.001 (n.s.) | −0.002 (n.s.) | −0.003 (n.s.) | **+0.013** |
| loss, sbg_lo − loss, l8multi | −0.003 (n.s.) | −0.001 (n.s.) | **+0.011** | −0.001 (n.s.) | +0.008 (n.s.) |
| transferred emit − l8multi raw block | **−0.011** | **−0.010** | −0.012 [−0.026, 0.002] | **−0.029** | **−0.043** |
| transferred sbg_lo − l8multi raw block | −0.005 (n.s.) | −0.000 (n.s.) | **−0.022** | **−0.027** | **−0.032** |
| transferred sbg_lo − l8raw | **+0.010** | **+0.014** | −0.011 (n.s.) | **−0.017** | **−0.025** |

Map agreement after transfer with 10 components (ρ; native / emit / sbg_lo /
l8 / l8multi):

| Target | Nitrogen | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| NEON 2013 | 0.93 / 0.89 / 0.91 / 0.76 / 0.83 | 0.99 / 0.97 / 0.98 / 0.87 / 0.90 | 0.81 / 0.71 / 0.80 / 0.55 / 0.61 | 0.81 / 0.79 / 0.82 / 0.72 / 0.72 |
| `sierra_nf` 2013 | 0.85 / 0.81 / 0.85 / 0.67 / 0.71 | 0.93 / 0.90 / 0.92 / 0.75 / 0.77 | 0.83 / 0.75 / 0.80 / 0.59 / 0.63 | 0.83 / 0.74 / 0.78 / 0.64 / 0.68 |
| Tahoe 2013 | 0.47 / 0.78 / 0.71 / 0.67 / 0.73 | 0.83 / 0.66 / 0.93 / 0.71 / 0.76 | **0.86** / 0.82 / 0.88 / 0.68 / 0.71 | **0.75** / 0.78 / 0.76 / 0.68 / 0.65 |
| NEON 2018 | 0.93 / 0.91 / 0.93 / 0.80 / 0.81 | 0.96 / 0.94 / 0.96 / 0.84 / 0.87 | 0.73 / 0.40 / 0.55 / 0.42 / 0.49 | 0.75 / 0.61 / 0.67 / 0.63 / 0.61 |
| `sierra_nf` 2018 | 0.87 / 0.85 / 0.89 / 0.74 / 0.75 | 0.94 / 0.87 / 0.92 / 0.79 / 0.82 | 0.83 / 0.75 / 0.84 / 0.63 / 0.66 | 0.61 / 0.77 / 0.83 / 0.77 / 0.73 |

**Reading.**
- **Transferred VSWIR retrievals do not lose less recovery skill than
  transferred Landsat proxies,** with 25 or with 10 components.
  - Across the Yosemite box transfer costs almost nothing for anyone
    (losses within ±0.005).
  - To the Tahoe box every configuration loses some; the VSWIR − Landsat
    differences in loss include 0 (one exception each way).
  - From 2013 to 2018, nobody loses at NEON. At `sierra_nf` VSWIR loses
    *more* than the Landsat proxies (+0.008 to +0.022), which gain.
- **Landsat alone, with no airborne training, matches or beats transferred
  VSWIR outside the training area and date.** The multi-date block is
  ahead of transferred EMIT-like traits in the Tahoe box (+0.012 to +0.017)
  and in 2018 (+0.029 to +0.043, both AOIs). Transferred VSWIR beats a
  single Landsat scene only within the Yosemite box in 2013.
- **Map agreement after transfer:**
  - VSWIR keeps higher absolute agreement than the Landsat proxies for N,
    LMA and (with 10 components) lignin in most legs, e.g. Tahoe lignin
    0.82–0.88 against 0.68–0.71.
  - That advantage in agreement does not turn into recovery skill.
  - EMIT-like lignin transferred to NEON 2018 is the weak spot (0.40–0.48).
- **Emulator size.** With 25 components, native AVIRIS collapses in the
  Tahoe box (lignin 0.27, cellulose 0.07). With 10 components it holds
  (0.86, 0.75), as §26 found across sensors. The recovery-skill conclusions
  do not change.
- In 2018 the Landsat blocks are especially strong (+0.087 to +0.114 for the
  multi-date block). They share sensor, registration and the Jul–Sep
  compositing with the Landsat 8/9 cycle-2 responses, which may favour them
  (as in §20).
- **Caveats:** emulated, not published, retrievals; one transfer per leg;
  the 2018 and Tahoe mosaics are not cross-year calibrated, which affects
  every configuration's agreement with the maps.

### 26. The AVIRIS-5 bridge at a second area, and for traits (`hls_results/aviris5_bridge/`)

§12 compared AVIRIS-Classic and AVIRIS-5 on the same day at NEON only, and
for reflectance and EWT only. This section repeats it over `sierra_nf` and
adds traits at both areas.

**Data** (`proto_aviris5_bridge.py`).
- Both instruments flew `sierra_nf` on July 17, 2025:
  - AVIRIS-C lines `f250717t01p00r09` to `r13` (ORNL DAAC 2154);
  - AVIRIS-5 scenes `AV520250717t175725_003/_004`, `t181601_004/_005` and
    `t183638_003` (ORNL DAAC 2484). These are the five of the eleven that
    cover the most of the AOI.
- The method is the same as §12: area averaging to 30 m, AVIRIS-5 convolved
  to the AVIRIS-C bands, and a nearest-nadir mosaic per sensor. Lines are
  now folded into the mosaic one at a time, which bounds memory.
- 17,497 undisturbed forest 90 m cells are covered by both sensors (NEON:
  15,784). Rerunning NEON reproduces §12 exactly.

**Reflectance and canopy water.** NEON values are from §12.

| | NEON | `sierra_nf` |
|---|---|---|
| EWT980 ρ (AVIRIS-5 − AVIRIS-C) | 0.976 (+0.029 cm, +13%) | **0.903** (+0.021 cm, +9%) |
| NDVI ρ (bias) | 0.958 (+0.049) | 0.931 (+0.039) |
| NDWI ρ (bias) | 0.952 (−0.055) | 0.916 (−0.057) |
| 450–700 nm: relative difference, ρ | −4%, 0.80 | −4%, 0.71 |
| 700–1300 nm | +18%, 0.70 | +11%, 0.70 |
| 1450–1800 nm | +24%, 0.81 | +19%, 0.78 |
| 2000–2400 nm | +3%, 0.83 | −0%, 0.83 |

- The same pattern holds at the second area: AVIRIS-5 reads brighter in
  the NIR and SWIR-1, and agrees in rank.
- Rank agreement is somewhat lower at `sierra_nf`, which has more lines, a
  wider spread of view angles, and steeper terrain. EWT ρ is 0.90, below
  NEON's 0.98.

**Traits.**
- *Emulator.* The 2403 PLSR coefficients are not public, so one emulator
  per AOI is fitted, as in §18. It is a PLSR from log reflectance to the
  2403 trait maps, fitted on the June 2018 AVIRIS-C reflectance
  (`wdts/sim/<aoi>_refl2018.nc`, the most recent year with both spectra
  and a trait map; only pixels from the line the map used).
- *Application.* The emulator is applied unchanged to both 2025 mosaics,
  each interpolated onto the 2018 band centres. Both 2025 inputs go
  through the same emulator, so their agreement measures how consistently
  the two sensors support one retrieval. It does not measure the
  retrieval's accuracy.
- *Input variants.*
  - **raw:** each mosaic as delivered.
  - **matched:** each mosaic's per-band log reflectance is rescaled to the
    mean and SD of the 2018 training pixels. This is a scene-level
    calibration that removes per-band level and gain offsets.
- *Model size.* Emulators with 5, 10 and 25 PLSR components (25 is the §18
  setting). Out-of-fold R² in 2018 (1 km blocks) is given for reference.

Spearman ρ between AVIRIS-C and AVIRIS-5 on the same 90 m cells (raw inputs;
matched in brackets):

| Trait | Components | NEON | `sierra_nf` |
|---|---|---|---|
| Nitrogen | 5 | 0.79 (0.80) | 0.72 (0.72) |
| | 10 | **0.89** (0.85) | **0.73** (0.67) |
| | 25 | 0.39 (0.18) | 0.28 (0.07) |
| LMA | 5 | 0.77 (0.77) | 0.73 (0.74) |
| | 10 | **0.90** (0.87) | **0.76** (0.69) |
| | 25 | 0.79 (0.59) | 0.67 (0.45) |
| Lignin | 5 | 0.64 (0.60) | 0.57 (0.59) |
| | 10 | **0.77** (0.78) | **0.66** (0.66) |
| | 25 | 0.68 (0.60) | 0.56 (0.69) |
| Cellulose | 5 | 0.65 (0.71) | 0.63 (0.61) |
| | 10 | **0.68** (0.53) | **0.57** (0.67) |
| | 25 | 0.53 (0.54) | 0.25 (0.48) |

Emulator out-of-fold R² in 2018 (5 / 10 / 25 components):
- NEON: N 0.70 / 0.71 / 0.77; LMA 0.77 / 0.77 / 0.84; lignin 0.59 / 0.62 /
  0.77; cellulose 0.38 / 0.45 / 0.61.
- `sierra_nf`: N 0.66 / 0.71 / 0.74; LMA 0.69 / 0.76 / 0.81; lignin 0.68 /
  0.77 / 0.83; cellulose 0.51 / 0.58 / 0.64.

Each 2025 sensor against the 2018 map (ρ, 10 components, raw;
AVIRIS-C / AVIRIS-5):

| Trait | NEON | `sierra_nf` |
|---|---|---|
| Nitrogen | 0.86 / 0.89 | 0.71 / 0.74 |
| LMA | 0.90 / 0.92 | 0.73 / 0.78 |
| Lignin | 0.61 / 0.66 | 0.54 / 0.55 |
| Cellulose | 0.47 / 0.66 | 0.43 / 0.62 |

**Level offsets** (median AVIRIS-5 − AVIRIS-C, in SD of the AVIRIS-C values):
- *raw inputs:* 0.0–2.4 SD with 5 components; 4–9 SD with 10; 7–29 SD with
  25 (N at `sierra_nf` −29 SD).
- *matched inputs:* within 0.5 SD with 5 and 10 components; up to 1.9 SD
  with 25.
- The emulator turns the brighter AVIRIS-5 NIR and SWIR-1 into level
  offsets that grow with model size.

**Reading.**
- **The reflectance and EWT bridge replicates at a second area.** Rank
  agrees (EWT ρ 0.90–0.98, NDVI 0.93–0.96); level does not (AVIRIS-5
  +9–13% EWT, +11–24% NIR/SWIR-1). Per-date standardization (§7) applies to
  cross-sensor use too.
- **Traits bridge in rank with a moderately sized retrieval.**
  - With 10 components, AVIRIS-C and AVIRIS-5 agree at ρ 0.89–0.90 for N
    and LMA at NEON, and 0.73–0.76 at `sierra_nf`. Lignin is 0.66–0.77 and
    cellulose 0.57–0.68.
  - That is within the range of the 2013 → 2018 stability of the AVIRIS-C
    maps themselves (§7: ρ 0.65–0.90 for N, LMA and lignin).
  - AVIRIS-5 tracks the 2018 map at least as well as 2025 AVIRIS-C does.
- **The 25-component emulator does not bridge.** N agreement falls to ρ
  0.28–0.39, and level offsets reach 17–29 SD.
  - The extra components fit fine spectral features that differ between
    the two processing chains: OE atmospheric correction for AVIRIS-5, and
    the 2154 L2 for AVIRIS-C 2025, against the 2391 L2 the emulator was
    trained on.
  - Matching each band's level and gain does not fix this. It is a
    spectral-shape difference, not a per-band calibration offset.
- **Implication for carrying retrievals across sensors:**
  - choose retrieval complexity by cross-sensor (or cross-date)
    consistency, not only by in-sample fit;
  - standardize per date;
  - a 25-component emulator fitted to one processing chain is not
    portable.
- **Caveats:**
  - emulated, not published, retrievals; the real 2403 coefficients may
    behave differently;
  - one day; five of eleven AVIRIS-5 scenes at `sierra_nf`;
  - ρ is reported without bootstrap CIs;
  - geometry (view and illumination differences between the two aircraft
    lines), atmospheric correction and calibration cannot be separated
    here.

**Not run** (planned, stopped at a pause request):
- block-bootstrap CIs on the trait ρ;
- choosing the component count by cross-validated cross-sensor agreement;
- the remaining ten traits (only N, LMA, lignin and cellulose were
  compared);
- adding the other six `sierra_nf` AVIRIS-5 scenes;
- caching the 2025 30 m mosaics, so that variants need not re-read the
  AVIRIS-C lines;
- repeating the trait bridge with the real 2403 coefficients, if they are
  shared.

### 27. Does substrate explain the Tahoe-box directions? (`hls_results/trait_directions_substrate/`)

§9 and §22 ruled out calibration, forest type, signal strength, mortality
agent, host and drought timing as reasons why residual N and LMA track NDMI
recovery in the Yosemite box but not in the Tahoe box. A remaining candidate
is substrate. The northern Sierra carries Tertiary andesitic volcanic and
mudflow deposits, while the Yosemite box is mostly batholith granite. Soil
parent material could change how foliar N and LMA relate to growth and
recovery.

**Data** (`fetch_geology.py`):
- USGS State Geologic Map Compilation (SGMC), California polygons
  (`https://mrdata.usgs.gov/geology/state/shp/CA.zip`). These carry the
  1:750,000 Geologic Map of California units.
- The generalized lithology is grouped into granitic, volcanic,
  metamorphic, surficial (mainly Quaternary glacial deposits) and other,
  and rasterized onto the 30 m grids (`env/<aoi>_geology.nc`).
- At this scale, units smaller than about a kilometre are generalized.

| Substrate (share of AOI) | NEON | `sierra_nf` | Tahoe box |
|---|---|---|---|
| Granitic (Mesozoic granodiorite/quartz monzonite, Sierra Nevada batholith) | 82% | 87% | 69% |
| Volcanic | 0% | 3% (Tertiary flows, Mesozoic metavolcanics) | **28%** (Tertiary andesitic pyroclastic and mudflow deposits) |
| Metamorphic | 9% | 4% | 1% |
| Surficial (glacial) | 7% | 4% | 1% |

All three AOIs share the same granitic unit, so the boxes can be compared
on the same substrate.

**Method.**
- `proto_trait_directions.py --group substrate`: the cells' dominant
  substrate (≥ 50% of the cell's pixels).
- Residual traits, strata and block-bootstrap CIs as in §22, for groups of
  ≥ 1,000 cells.
- A ladder Env+S | Env+S+G | Env+S+T | Env+S+G+T for NDMI recovery, with G
  the substrate fractions, and the trait gain within each substrate.

**Residual traits vs NDMI recovery** (within-stratum ρ, 95% CI):

| AOI, substrate (cells) | N | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| NEON, all (43,301) | +0.19 [0.16, 0.21] | −0.17 [−0.20, −0.14] | −0.17 [−0.19, −0.14] | −0.15 [−0.18, −0.13] |
| NEON, granitic (36,491) | +0.19 [0.16, 0.22] | −0.16 [−0.19, −0.13] | −0.16 [−0.20, −0.13] | −0.16 [−0.19, −0.13] |
| NEON, metamorphic (3,572) | +0.15 [0.07, 0.24] | −0.23 [−0.31, −0.16] | −0.26 [−0.32, −0.17] | −0.23 [−0.28, −0.13] |
| NEON, surficial (3,215) | +0.21 [0.14, 0.28] | −0.29 [−0.35, −0.22] | −0.16 [−0.23, −0.09] | −0.02 (n.s.) |
| `sierra_nf`, all (58,264) | +0.21 [0.20, 0.23] | −0.19 [−0.21, −0.17] | −0.18 [−0.20, −0.17] | −0.16 [−0.18, −0.15] |
| `sierra_nf`, granitic (51,464) | +0.21 [0.19, 0.23] | −0.18 [−0.20, −0.16] | −0.18 [−0.20, −0.16] | −0.17 [−0.19, −0.15] |
| `sierra_nf`, **volcanic** (2,653) | **+0.25** [0.18, 0.33] | **−0.29** [−0.36, −0.23] | −0.24 [−0.31, −0.18] | −0.05 (n.s.) |
| `sierra_nf`, metamorphic (2,030) | +0.26 [0.18, 0.34] | −0.23 [−0.33, −0.14] | −0.21 [−0.29, −0.12] | −0.11 [−0.20, −0.04] |
| `sierra_nf`, surficial (2,045) | +0.20 [0.13, 0.26] | −0.25 [−0.34, −0.15] | −0.22 [−0.30, −0.11] | −0.05 (n.s.) |
| Tahoe box, all (55,105) | +0.02 [0.00, 0.05] | +0.02 (n.s.) | −0.11 [−0.13, −0.09] | −0.12 [−0.14, −0.10] |
| Tahoe box, **granitic** (35,689) | **+0.02** (n.s.) | **+0.03** [0.00, 0.06] | −0.09 [−0.11, −0.07] | −0.10 [−0.12, −0.07] |
| Tahoe box, volcanic (17,804) | +0.04 [0.01, 0.08] | −0.01 (n.s.) | −0.16 [−0.19, −0.12] | −0.15 [−0.18, −0.11] |

NIRv recovery gives the same picture. In the Tahoe box, N is −0.05 and LMA
+0.10 on granitic cells, and 0.00 and +0.04 on volcanic cells.

**Ladder** (NDMI recovery, ΔR² with 95% CI):

| AOI | +G over Env+S | +T over Env+S | +T over Env+S+G | +T within granitic | +T within volcanic |
|---|---|---|---|---|---|
| NEON | +0.003 [0.001, 0.005] | +0.059 | +0.058 | +0.061 | – |
| `sierra_nf` | +0.001 (n.s.) | +0.086 | +0.085 | +0.085 | +0.114 [0.081, 0.150] |
| Tahoe box | +0.008 (n.s.) | +0.032 | +0.025 | +0.036 [0.023, 0.054] | +0.002 [−0.023, 0.024] |

**Reading.**
- **Substrate does not explain the difference.**
  - On the *same* granitic unit, residual N and LMA track recovery in the
    Yosemite box (N +0.19 to +0.21, LMA −0.16 to −0.18) and not in the Tahoe
    box (N +0.02, LMA +0.03).
  - The Yosemite-box volcanic cells (`sierra_nf`, 2,653) show the direction
    at least as strongly as its granitic cells.
  - Volcanic substrate therefore neither removes the direction where it
    exists nor accounts for its absence in the Tahoe box.
- **Structural carbon holds on every substrate in both boxes**, and in the
  Tahoe box it is strongest on volcanic cells (lignin −0.16, cellulose
  −0.15).
- **Substrate adds almost nothing to Env+S** (≤ +0.008) and leaves the
  trait gain nearly unchanged.
  - The one substrate contrast is in the Tahoe box: the trait gain for NDMI
    recovery is +0.036 on granitic cells and about 0 on volcanic cells.
  - So on Tahoe volcanic substrate, traits add no recovery skill beyond
    Env+S, even though the structural-carbon direction is present there.
- With §9 and §22, every stand and site attribute tested so far (forest
  type, mortality agent, host, drought timing, substrate) leaves the
  Yosemite/Tahoe N–LMA contrast intact. What remains is a property of the
  box as a whole: species mixes below the resolution of LANDFIRE and ADS,
  or retrieval behaviour specific to the Tahoe acquisitions.
- **Caveat:** at 1:750,000 the map generalizes small volcanic caps and
  contacts. A finer test would use SSURGO parent material.

### 28. Does a VSWIR time series beat a Landsat time series of the same dates? (`hls_results/spaceborne_ts/`)

§23 set one spaceborne-like VSWIR date against Landsat. A spaceborne
imaging spectrometer delivers a time series, as Landsat does, so this
section compares time series with time series: the same number of dates in
the same window on both sides, with no lidar and no airborne maps on the
Landsat side.

**Data.**

| | VSWIR (AVIRIS-C) | Landsat 8 (path 42) |
|---|---|---|
| 2013 (first drought; cycle-1 responses) | May 3 (2391, non-`_v2`), **Jun 12** (`_v2`), Jun 26 (`_v2`; one non-`_v2` line at `sierra_nf`) | nearest: May 4, Jun 5, Jun 21 (NEON clear 76%, 66%, 98%); all-clear check: May 20, Jun 21, Jul 7 (≥ 97%) |
| 2018 (cycle-2 responses, pilot cells) | **Jun 22**, Aug 28, both from ORNL DAAC 2154 | Jun 19, Aug 22 (≥ 97%) |

- Path 43 reaches at most 15% of either AOI, so every Landsat scene is
  path 42. Each VSWIR date is matched to its nearest distinct Landsat
  scene.
- Aug 28 2018 exists only in 2154 (AVIRIS-Classic L2, orthocorrected
  14–15 m; no topographic or BRDF correction). Those are long lines flown
  22–23° off north. To keep the 2018 series on one processing chain,
  Jun 22 is read from 2154 too (the line the 2018 trait map used, where
  lines overlap). Rotated lines are read with HTTP range requests over the
  rows that cross the AOI. Each 30 m pixel is the mean of the source pixels
  whose centres fall in it. Against the nearest Landsat scene, NDVI is best
  aligned at zero shift at NEON and within one 30 m pixel elsewhere.
- **Aug 28 crosses only half of `sierra_nf`**, and that half is mostly the
  side the cycle-2 fire mask removes (6,074 of 205,549 pilot pixels). The
  2018 leg is therefore NEON only.

**Method** (`proto_spaceborne_ts.py`).
- `refl` caches each date. `emulate` fits one PLSR emulator per trait (10
  components; §25, §26) on the first date in bold, against that year's 2403
  map. It is fitted out of fold over 1 km blocks, and each fold's model
  predicts every date's pixels in its held-out blocks.
  - Later dates go through the same models: native spectra are interpolated
    onto the first date's band centres; emit and sbg_lo take each date's own
    band responses.
  - Noise is drawn separately for each date. Bands missing from any line
    (1323 and 1333 nm in the non-`_v2` lines) are interpolated across.
  - EWT is computed per date. Traits and EWT are cross-track normalized
    along each date's own lines.
- First-date emulator R² against the maps (native / emit / sbg_lo):

  | Trait | 2013 NEON | 2013 `sierra_nf` | 2018 NEON (2154 spectra) |
  |---|---|---|---|
  | N | 0.75 / 0.62 / 0.72 | 0.69 / 0.52 / 0.65 | 0.63 / 0.62 / 0.61 |
  | LMA | 0.94 / 0.86 / 0.93 | 0.90 / 0.76 / 0.88 | 0.78 / 0.78 / 0.77 |
  | Lignin | 0.75 / 0.50 / 0.67 | 0.78 / 0.52 / 0.68 | 0.43 / 0.38 / 0.38 |

  The 2018 emulators are weaker. The 2403 maps come from the topographically
  corrected 2391 spectra, but these inputs are the uncorrected 2154 ones.
- `compare` puts every block in one cell table, so all contrasts are paired.
  Bootstrap over 1 km blocks, with 5 km as a check. Over Env+S, no lidar:
  - **V:\<c\>**: all dates' traits and EWT, plus their change (last −
    first);
  - **L**: the matched Landsat scenes' bands, NDVI, NDMI, NBR and NIRv,
    plus the same change;
  - **V1, L1**: the first VSWIR date and its nearest scene;
  - **l8multi**: the year's June and Jul–Sep composites;
  - **M1**: the first date's trait maps.
- Cells must have every date on both sides.

**2013, nearest Landsat dates** (NEON 28,672 cells, `sierra_nf` 47,543;
NDMI recovery over Env+S 0.682 / 0.505; ΔR², paired, bold where the 95% CI
excludes 0):

| NDMI recovery, NEON / `sierra_nf` | native | emit | sbg_lo |
|---|---|---|---|
| V (time series) over Env+S | +0.086 / +0.097 | +0.070 / +0.086 | +0.082 / +0.085 |
| **V − L (balanced)** | **+0.030** / +0.006 [−0.002, 0.014] | **+0.014** [0.006, 0.023] / −0.006 [−0.014, 0.002] | **+0.026** / −0.006 [−0.014, 0.002] |
| L+V − L (VSWIR beyond Landsat) | **+0.045** / **+0.028** | **+0.036** / **+0.023** | **+0.043** / **+0.021** |
| V1 − L1 (one date each) | **+0.041** / **+0.037** | **+0.022** / **+0.015** | **+0.030** / **+0.025** |
| V − V1 (what the extra dates buy VSWIR) | **+0.014** / **+0.004** | **+0.018** / **+0.016** | **+0.021** / **+0.005** |
| V − l8multi (unbalanced) | **+0.027** / **+0.020** | **+0.012** / **+0.009** | **+0.023** / **+0.008** |

L over Env+S is +0.056 / +0.092, and L − L1 is **+0.026** / **+0.036**: the
extra dates buy Landsat more than VSWIR at `sierra_nf`.

- NIRv recovery, V − L: NEON **+0.010** to **+0.014**; `sierra_nf`
  **−0.008** (native) to **−0.017** (emit).
- Stress response, V − L: NEON +0.001 to +0.006 (n.s.); `sierra_nf`
  **−0.008** to **−0.013**. This target is partly circular for the 2013
  Landsat dates.
- With 5 km blocks, NEON V − L for NDMI recovery keeps its sign and
  significance. At `sierra_nf` every V − L stays n.s.

**2013, all-clear Landsat check** (May 20, Jun 21, Jul 7; NEON 43,883 cells,
`sierra_nf` 58,437):

| NDMI recovery, V − L | native | emit | sbg_lo |
|---|---|---|---|
| NEON | **+0.013** | +0.004 [−0.003, 0.010] | **+0.012** |
| `sierra_nf` | +0.000 [−0.006, 0.007] | **−0.009** [−0.015, −0.002] | **−0.009** [−0.016, −0.002] |
| L+V − L (NEON / `sierra_nf`) | **+0.032** / **+0.020** | **+0.025** / **+0.017** | **+0.030** / **+0.015** |
| V − l8multi (NEON / `sierra_nf`) | **+0.023** / **+0.016** | **+0.014** / **+0.007** | **+0.021** / **+0.007** |

NIRv recovery, V − L: NEON −0.002 to +0.000 (n.s.); `sierra_nf` **−0.009** to
**−0.016**.

**2018, NEON** (cycle-2 responses; 22,649 pilot cells; Env+S 0.596):

| NDMI recovery | native | emit | sbg_lo |
|---|---|---|---|
| V over Env+S | +0.086 | +0.066 | +0.068 |
| **V − L (balanced)** | −0.007 [−0.018, 0.007] | **−0.028** | **−0.025** |
| L+V − L | **+0.018** | **+0.014** | **+0.009** |
| V1 − L1 | −0.006 (n.s.) | **−0.018** | **−0.016** |

L over Env+S is +0.093, and l8multi +0.085. For NIRv recovery V − L is
**−0.012** to **−0.029**. The 2018 Landsat scenes fall in the cycle-2
baseline years (2017–19), with the same sensor and processing as the
responses, which favours Landsat.

**Reading.**
- **A VSWIR time series does not consistently beat a Landsat time series
  of the same dates.**
  - At NEON in 2013 the 30 m (sbg_lo) and native series lead (+0.012 to
    +0.030) with either set of Landsat dates. The EMIT-like series leads
    against the nearest, partly cloudy scenes (+0.014) and not against the
    clear ones (+0.004, n.s.).
  - At `sierra_nf` the series tie (nearest dates) or Landsat leads by 0.009
    (clear dates).
  - In 2018 (NEON) Landsat leads every spaceborne-like configuration
    (−0.025 to −0.028). The weaker emulators on uncorrected spectra and
    the shared baseline years both work against VSWIR there.
- **What extra dates buy:** a second and third date add more to Landsat
  (L − L1 +0.017 to +0.036) than to VSWIR at `sierra_nf` (+0.004 to
  +0.016). The single-date lead of VSWIR (§23, and V1 − L1 here) shrinks
  or disappears once both sides have the same dates.
- **VSWIR adds to Landsat in every case:** with both series in the model,
  VSWIR adds +0.009 to +0.045 for NDMI recovery (CIs exclude 0 at both
  areas, both years, every configuration, both sets of Landsat dates).
  So does the trait map itself (L+M1 − L +0.011 to +0.035).
- Against full-season Landsat composites (two composites, unbalanced) the
  VSWIR series leads in 2013 at both areas (+0.007 to +0.027), but not in
  2018.
- **Caveats:**
  - emulated retrievals fitted to one date's maps;
  - the 2013 VSWIR dates span only May–June, and Jun 12 and Jun 26 are two
    weeks apart;
  - the 2018 series is one area, and its spectra lack topographic and BRDF
    correction;
  - the Landsat inputs share sensor and processing with the Landsat
    responses;
  - per-band noise is spatially independent.

### 29. Do 2018 traits add to 2018 Landsat for the next drought? (`hls_results/forward_pilot_landsat/`)

In §25 a single June 2018 Landsat scene gave a larger cycle-2 recovery gain
than in-place 2018 AVIRIS traits did (unpaired). This section runs the
paired test on the §11 forward-pilot cells: does T18 (the June 2018 trait
maps, z-scored per date) add to 2018 Landsat reflectance?

**Method** (`proto_forward_pilot.py --landsat 20180619 --landsat 20180822
--landsat-composites`).
- **L1:** the Jun 19 2018 Landsat 8 scene (three days before the flight):
  bands, NDVI, NDMI, NBR, NIRv.
- **L2:** the Jun 19 and Aug 22 scenes and their change.
- **LM:** the 2018 June and Jul–Sep composites.
- Cells with every Landsat input: NEON 24,270 of 24,342; `sierra_nf` 20,017
  of 20,089.
- The same 1 km block folds and paired bootstrap as §11. The runs without
  `--landsat` are unchanged.
- The cycle-2 responses are Landsat 8/9 with a 2017–19 baseline, so every
  2018 Landsat block shares sensor, processing and baseline years with
  them.

**NDMI recovery** (Env+S 0.595 / 0.482; ΔR², paired 95% CI; NEON /
`sierra_nf`):

| Contrast | NEON | `sierra_nf` |
|---|---|---|
| L1 over Env+S | **+0.063** | **+0.116** |
| T18 − L1, over Env+S (one date each) | +0.001 [−0.011, 0.017] | −0.008 [−0.021, 0.006] |
| T18 − L1, over Env+S+Leg | −0.004 [−0.014, 0.009] | −0.004 [−0.012, 0.006] |
| T18 beyond Env+S+L1 | **+0.019** | **+0.020** |
| L1 beyond Env+S+Leg | **+0.029** | **+0.025** |
| **T18 beyond Env+S+Leg+L1** | +0.000 [−0.011, 0.009] | **+0.009** [0.003, 0.015] |
| T18 beyond Env+S+Leg+L2 | +0.004 [−0.005, 0.010] | **+0.010** |
| T18 beyond Env+S+Leg+LM | +0.007 [−0.000, 0.013] | **+0.011** |
| T18 − T13, beyond Env+S+Leg+L1 | −0.004 (n.s.) | +0.001 (n.s.) |

**Other responses, T18 beyond Env+S+Leg+L1:**
- NIRv recovery **+0.015** / **+0.021** (T18 − T13 there **+0.004** /
  **+0.007**);
- NDMI resistance **+0.011** / **+0.007**;
- NDMI resilience +0.006 (n.s.) / **+0.005**.

Over Env+S+Leg the Landsat scene and T18 tie for every response, except
`sierra_nf` resilience, where Landsat leads (−0.007).

**Directions beyond the Landsat bands** (within-stratum ρ of residual 2018
traits with cycle-2 NDMI recovery; residualized on Env+S, then on
Env+S+L1):

| | N | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| NEON | +0.23 → **+0.11** | −0.24 → **−0.12** | −0.17 → **−0.09** | −0.07 → **−0.05** |
| `sierra_nf` | +0.23 → **+0.12** | −0.23 → **−0.12** | −0.23 → **−0.12** | −0.15 → **−0.07** |

Every CI excludes 0. NIRv recovery is similar (N +0.15 / +0.16, lignin −0.12
/ −0.11 beyond L1).

**Reading.**
- **For NDMI recovery in the second drought, one Landsat scene matches the
  2018 AVIRIS traits.** Date for date they tie, with or without legacy.
  Beyond Env+S, legacy and the Landsat scene, the traits add nothing at
  NEON and +0.009 at `sierra_nf`. The traits' within-drought gain over
  legacy alone in §11 (+0.035 / +0.019) is mostly Landsat-accessible.
- **For NIRv recovery and resistance the traits do add beyond Landsat** at
  both areas (+0.007 to +0.021). A second scene or the full-season
  composites do not remove that.
- **Post-drought traits are not what adds:** beyond Landsat, the 2018 traits
  do no better than the 2013 ones for NDMI recovery, and only slightly
  better for NIRv recovery.
- **The trait directions survive the Landsat bands at about half
  strength**, for leaf economics and structural carbon at both areas. Half
  of each direction is shared with the Landsat reflectance; the rest is
  not, even where the traits add little skill. (§24 found that, in 2013,
  only the structural-carbon part is specific to VSWIR.)
- **Caveat:** the shared baseline years and processing favour Landsat for
  every cycle-2 response. This is the most favourable setting for the
  Landsat control.

### 30. Is the Tahoe-box leaf-economics gap a retrieval effect? (`hls_results/trait_directions_retrieval/`)

§9, §22 and §27 ruled out calibration, forest type, mortality agent, host,
drought timing and substrate. One candidate remained: retrieval behaviour
specific to the Tahoe acquisitions. A hint came from §25, where an emulator
trained on Yosemite-box 2013 maps agreed with the Tahoe N map at only ρ 0.47
(10 components). If the Tahoe N and LMA maps miss a signal that a
Yosemite-trained retrieval sees, the directions should reappear with that
retrieval.

**Method** (`proto_trait_directions.py --trait-cells <§25 output>
--trait-source …`).
- The 2013 traits are replaced by the §25 emulated retrievals (cross-track
  normalized cell means):
  - `in`: the configuration's emulator fitted out of fold on the AOI's own
    maps;
  - `transfer`: the emulator fitted elsewhere. For the Tahoe box that is the
    pooled NEON + `sierra_nf` emulator; NEON and `sierra_nf` (the positive
    controls) each use the other's.
- Native and EMIT-like configurations, with 10 and 25 PLSR components.
- The maps are rerun on the same cells. Otherwise everything is as in §9:
  residuals on Env+S, within-stratum ρ with NDMI recovery, and 1 km
  block-bootstrap CIs.

**Residual traits vs NDMI recovery** (10 components; 25 in brackets where
run):

| Traits from | N | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| Tahoe box, maps (55,106 cells) | +0.02 (n.s.) | +0.03 (n.s.) | −0.11 | −0.12 |
| Tahoe, native in place | −0.01 (+0.01) n.s. | **+0.07** (+0.04) | −0.08 (−0.12) | −0.13 (−0.14) |
| **Tahoe, native Yosemite-trained** | +0.02 (+0.01) n.s. | +0.01 (+0.02) n.s. | −0.07 (−0.07) | −0.07 (−0.05) |
| Tahoe, EMIT-like in place | **−0.04** (−0.01 n.s.) | **+0.07** (+0.06) | −0.06 (−0.09) | −0.13 (−0.16) |
| **Tahoe, EMIT-like Yosemite-trained** | −0.01 (+0.01) n.s. | **+0.06** (+0.05) | −0.11 (−0.11) | −0.15 (−0.18) |
| NEON, maps (43,247) | +0.19 | −0.18 | −0.17 | −0.16 |
| NEON, native from `sierra_nf` | +0.20 | −0.17 | −0.16 | −0.18 |
| NEON, EMIT-like from `sierra_nf` | +0.18 | −0.16 | −0.14 | −0.13 |
| `sierra_nf`, maps (58,239) | +0.21 | −0.19 | −0.19 | −0.16 |
| `sierra_nf`, native from NEON | +0.21 | −0.18 | −0.19 | −0.18 |
| `sierra_nf`, EMIT-like from NEON | +0.19 | −0.17 | −0.16 | −0.11 |

Bold, and every structural-carbon entry: the CI excludes 0.

**Reading.**
- **The gap is not a retrieval effect.** With the Yosemite-trained
  retrieval applied to Tahoe spectra, residual N and LMA still do not track
  recovery (N −0.01 to +0.02; LMA +0.01 to +0.06, the wrong sign where it
  differs from 0). The in-place emulators agree with the maps.
- **The positive controls work.** A retrieval moved between NEON and
  `sierra_nf` reproduces the maps' N and LMA directions (N +0.18 to +0.21,
  LMA −0.16 to −0.18). So a transferred retrieval can carry the direction
  where it exists.
- **Structural carbon holds with every retrieval in the Tahoe box** (lignin
  −0.06 to −0.12, cellulose −0.05 to −0.18).
- With §9, §22 and §27, the Yosemite/Tahoe leaf-economics contrast is not
  calibration, forest type, mortality agent, host, drought timing,
  substrate or the retrieval. It stands as a regional difference in how
  leaf economics relates to recovery.
- **Caveats:** emulated retrievals; the Tahoe 2013 lines lack the 1323 and
  1333 nm bands; one acquisition per box.

### 31. Do VSWIR dates add more to a Landsat series than as many extra Landsat dates? (`hls_results/spaceborne_ts/`, `hls_results/forward_pilot_landsat/`)

In §28, a VSWIR time series added +0.009 to +0.045 for NDMI recovery on top
of the matched Landsat series. That model also has twice as many dates as
the Landsat series alone, so any extra dates might add as much. This section
adds the same number of further Landsat dates instead (L′) and compares the
two additions on the same cells and folds.

**Data.** Further Landsat 8 path-42 scenes (`fetch_landsat_c2_ee.py
--scene`), as many as the VSWIR dates:

| | L (as in §28) | L′ (extra dates) |
|---|---|---|
| 2013, nearest set | May 4, Jun 5, Jun 21 | May 20, Jul 7, Aug 8 |
| 2013, all-clear set | May 20, Jun 21, Jul 7 | May 4, Jun 5, Aug 8 |
| 2018, NEON | Jun 19, Aug 22 | Jul 5, Sep 7 |

- Jul 23 2013 has no clear pixels over either AOI after masking, so Aug 8
  (NEON 98% clear, `sierra_nf` 81%) takes its place. That puts one L′ date
  six weeks after the last VSWIR date (Jun 26); a later-season view favours
  Landsat.
- Jul 21 and Aug 6 2018 are 49–88% clear (Ferguson Fire smoke), so the 2018
  L′ uses Jul 5 and Sep 7 (≥ 97%).
- The two 2013 runs use the same six dates, split differently between L
  and L′, on the same cells (NEON 28,664, `sierra_nf` 39,750; fewer than in
  §28 because Aug 8 is only 81% clear at `sierra_nf`).

**Method** (`proto_spaceborne_ts.py compare --landsat-extra`). L′ is built
like L: each scene's bands, NDVI, NDMI, NBR and NIRv, plus its own last −
first change. The emulated VSWIR blocks are those of §28. Over Env+S, no
lidar, paired (1 km block bootstrap, 5 km as a check):
- **L+V − L+L′** (balanced: the same number of dates added to the same L);
- L+L′ − L and L+V − L (what each addition buys on its own);
- L+V − L+l8multi (against the year's June and Jul–Sep composites);
- L+L′+V − L+L′ (does VSWIR still add on top of the denser series?).

**2013, NDMI recovery** (NEON Env+S 0.682, `sierra_nf` 0.520; bold where
the 95% CI excludes 0):

| ΔR², NEON / `sierra_nf` | nearest set as L | all-clear set as L |
|---|---|---|
| L+L′ − L | **+0.023** / **+0.011** | **+0.014** / **+0.010** |
| L+V − L, native | **+0.046** / **+0.030** | **+0.034** / **+0.027** |
| L+V − L, emit | **+0.035** / **+0.024** | **+0.024** / **+0.023** |
| L+V − L, sbg_lo | **+0.043** / **+0.023** | **+0.030** / **+0.021** |
| **L+V − L+L′, native** | **+0.023** / **+0.019** | **+0.021** / **+0.017** |
| **L+V − L+L′, emit** | **+0.012** [0.005, 0.019] / **+0.013** [0.007, 0.018] | **+0.011** [0.005, 0.016] / **+0.013** [0.008, 0.018] |
| **L+V − L+L′, sbg_lo** | **+0.020** / **+0.012** | **+0.017** / **+0.011** |
| L+V − L+l8multi, emit / sbg_lo | **+0.023** / **+0.031** (NEON); **+0.015** / **+0.015** (`sierra_nf`) | **+0.016** / **+0.022**; **+0.014** / **+0.012** |
| L+L′+V − L+L′, emit / sbg_lo | **+0.025** / **+0.030**; **+0.021** / **+0.020** | **+0.017** / **+0.022**; **+0.018** / **+0.016** |

- With 5 km blocks every L+V − L+L′ entry keeps its sign. The NEON emit
  entries widen to [0.000, 0.024] (nearest) and [0.001, 0.022] (all-clear);
  every other entry still excludes 0.
- **NIRv recovery**, L+V − L+L′: NEON nearest +0.007 for every
  configuration (n.s.), all-clear **+0.011** to **+0.013**; `sierra_nf`
  **+0.006** to **+0.008** with either set (emit and sbg_lo lower bounds at
  0.000–0.001). L+L′+V − L+L′ is **+0.012** to **+0.019** everywhere.

**2018, NEON** (cycle-2 responses; 22,628 pilot cells; Env+S 0.603):

| ΔR² | NDMI recovery | NIRv recovery |
|---|---|---|
| L+L′ − L | **+0.016** | **+0.034** |
| L+V − L, native / emit / sbg_lo | **+0.020** / +0.008 (n.s.) / +0.006 (n.s.) | **+0.023** / **+0.012** / **+0.015** |
| **L+V − L+L′, native** | +0.004 [−0.008, 0.015] | **−0.011** |
| **L+V − L+L′, emit** | **−0.008** [−0.016, −0.000] | **−0.022** |
| **L+V − L+L′, sbg_lo** | **−0.010** [−0.022, −0.001] | **−0.019** |
| L+V − L+l8multi, native / emit / sbg_lo | **+0.020** / **+0.008** / +0.006 (n.s.) | **+0.021** / **+0.010** / **+0.013** |
| L+L′+V − L+L′, native / emit / sbg_lo | **+0.013** / **+0.009** / +0.006 [0.000, 0.013] | **+0.011** / +0.002 (n.s.) / **+0.005** |

The 2018 composites add nothing to L (−0.000 / +0.002), while two more
scenes add +0.016 / +0.034.

**Fold-assignment sensitivity** (`--fold-seed`). The block bootstrap holds
the CV folds fixed. Dropping 21 of §28's 22,649 cells for the extra dates
moved the 2018 Env+S R² from 0.596 to 0.603 and emit L+V − L from +0.014 to
+0.008, because GroupKFold's size-balanced assignment of blocks to folds
changes. So the key contrasts were rerun with blocks shuffled into folds
under three seeds (NDMI recovery, NEON):

| L+V − L+L′ | default folds | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| 2013 (nearest set), emit | **+0.012** | **+0.012** | **+0.015** | **+0.011** |
| 2018, emit | **−0.008** | −0.003 | −0.006 | **−0.013** |
| 2018, sbg_lo | **−0.010** | −0.005 | **−0.008** | **−0.016** |

Fold assignment moves these contrasts by up to ±0.006, more than the
bootstrap CIs suggest.

**Cycle-2 analogue** (`proto_forward_pilot.py --landsat 20180619 --landsat
<second scene> --landsat-composites`, the §29 cells): the 2018 traits (T18)
against one more Landsat scene (L2: the second scene and the change) or the
composites (LM), each added beyond Env+S+Leg+L1. Second scene Aug 22 (as in
§29) or Jul 5 (the next clear path-42 date after Jun 19).

| ΔR², NEON / `sierra_nf` | T18 − Aug 22 | T18 − Jul 5 | T18 − composites (Aug 22 cells) |
|---|---|---|---|
| NDMI resistance | **−0.014** / **−0.026** | −0.003 / +0.000 | +0.000 / −0.006 |
| NDMI recovery | −0.004 / +0.005 (n.s.) | +0.010 [−0.001, 0.024] / −0.003 | −0.000 / **+0.011** |
| NDMI resilience | +0.000 / **−0.007** | +0.000 / **−0.008** | +0.004 / −0.001 |
| NIRv resistance | −0.005 / +0.002 (n.s.) | **+0.004** / +0.004 | **+0.005** / **+0.011** |
| NIRv recovery | +0.002 / **+0.007** | **+0.007** / **−0.009** | **+0.010** / **+0.022** |
| NIRv resilience | +0.002 / **+0.015** | +0.001 / +0.005 | **+0.008** / **+0.022** |

The Aug 22 run reproduces §29 exactly (T18 beyond Env+S+Leg+L1, NDMI
recovery: +0.000 / +0.009). The Jul 5 run, on 24,290 / 20,048 cells (20 / 31
more than the Aug 22 run, which loses a few cloudy ones), gives **+0.014**
[0.006, 0.024] / **+0.012** for the same contrast, with Env+S R² 0.569
against 0.595 at NEON. That is the fold-assignment effect above. Rerunning the
Aug 22 control on the §29 cells with blocks shuffled into folds
(`proto_forward_pilot.py --fold-seed`, three seeds):

| ΔR², NEON / `sierra_nf` | default folds (§29) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| NDMI recovery, T18 beyond Env+S+Leg+L1 | +0.000 / **+0.009** | **+0.010** / **+0.016** | **+0.011** / **+0.008** | **+0.013** / **+0.008** |
| NDMI recovery, T18 − Aug 22 | −0.004 / +0.005 | +0.005 / **+0.012** | +0.001 / +0.003 | +0.007 [0.000, 0.015] / +0.002 |
| NIRv recovery, T18 beyond Env+S+Leg+L1 | **+0.015** / **+0.021** | **+0.013** / **+0.018** | **+0.017** / **+0.021** | **+0.015** / **+0.016** |
| NDMI resistance, T18 − Aug 22 | **−0.014** / **−0.026** | **−0.023** / **−0.023** | **−0.017** / **−0.027** | **−0.017** / **−0.028** |

§29's +0.000 at NEON is the outlier among four fold assignments.

**Reading.**
- **In the first drought, a VSWIR series adds more to a Landsat series than
  as many extra Landsat dates do.** For NDMI recovery this holds at both
  areas, with either split of the six Landsat dates, for every
  configuration (emit +0.011 to +0.013, sbg_lo +0.011 to +0.020), and
  under every fold assignment tried. The extra dates include a later-season
  view that the VSWIR series lacks. For NIRv recovery the margin is smaller
  (+0.006 to +0.013) and not significant at NEON with the nearest set.
- **In 2018 it does not hold.** At NEON two more Landsat dates match or beat
  the spaceborne-like VSWIR series (emit −0.003 to −0.013, sbg_lo −0.005 to
  −0.016 across fold assignments; NIRv recovery −0.019 to −0.022). Native
  AVIRIS ties. The 2018 caveats of §28 apply: weaker emulators on
  uncorrected spectra, and Landsat inputs in the responses' baseline years.
- **VSWIR is not redundant with a denser Landsat series:** on top of L+L′
  it still adds +0.016 to +0.032 (2013) and up to +0.013 (2018) for NDMI
  recovery.
- **Cycle 2, one scene each:** a second Landsat scene matches the 2018
  traits for every NDMI response (or beats them: NDMI resistance with Aug
  22, resilience at `sierra_nf`). For the NIRv responses the result depends
  on the second scene's date (the traits beat Aug 22 and trail Jul 5 for
  NIRv recovery at `sierra_nf`); against the composites the traits add for
  every NIRv response at both areas.
- **§29's NEON headline is fold-sensitive:** T18 beyond Env+S+Leg+L1 for
  NDMI recovery is +0.000 with §29's folds and +0.010 to +0.013 under three
  shuffled fold assignments (+0.014 on the Jul 5 cells); `sierra_nf` +0.008
  to +0.016. So beyond one June 2018 Landsat scene the 2018 traits add about
  +0.01 for cycle-2 NDMI recovery at both areas, but no more than a second
  scene does.
- **Caveats:** emulated retrievals; the 2013 VSWIR dates span May–June
  while L′ reaches August; the 2018 leg is one area; fold-assignment
  variability of ±0.006 (and up to 0.014 for the cycle-2 contrast) on top
  of the bootstrap CIs.

### 32. Does a more flexible model close the gap or absorb the trait gain? (`hls_results/model_capacity/`)

Every R² gain so far comes from one untuned gradient-boosting model
(`response_common.hgb`: 300 iterations, learning rate 0.05). This section
asks whether a more flexible learner gets the same skill from climate,
terrain and structure alone, absorbs the trait gain, or makes the
cross-drought transfer work.

**Learners** (`response_common.make_model`):
- **hgb**, the default model of every other section;
- **hgb_tuned**, gradient boosting with leaves (15, 63), minimum leaf size
  (20, 100) and the number of iterations (≤ 500 at learning rate 0.1) chosen
  on an inner split holding out 20% of the training fold's 1 km blocks,
  then refit on the whole training fold;
- **mlp**, a multilayer perceptron (256-128-64, adam) on standardized,
  median-imputed inputs with missing-value indicators and a standardized
  target, early-stopped on the same kind of inner block split.

Deep tabular models (TabPFN, FT-Transformer) are not tested: the `ecopro`
env has no torch. So this tests model capacity on these inputs, not deep
learning in general.

**Method** (`proto_model_capacity.py`). Each feature set is fitted with every
learner inside one ladder (`<set>|<learner>`), so every difference, within a
learner or between learners, is paired on the same cells, 1 km folds and
bootstrap draws (5 km as a check). Residual traits (Tres) are cross-fitted
with hgb, as in the other sections, for every learner.
- `within`: cycle 1 (§2), Env+S | Env+S+T | Env+S+Tres; with `--structure
  aso`, Env+S | Env+S+L | Env+S+L+T | Env+S+L+Tres on the ASO cells (§5).
  NDMI recovery and stress response.
- `transfer`: environment-only models fitted on cycle 1 and applied to cycle
  2 (§4, §21), pooled and within elevation quartiles.
- `forward`: within cycle 2 (§11), Env+S | +Leg | +T18 | +Leg+T18.

hgb reproduces the earlier numbers: +0.065 / +0.089 for Env+S+T (§2),
+0.038 / +0.025 for Tres over ASO (§5), +0.035 / +0.019 for T18 beyond Leg
(§11).

**Cycle 1, no lidar** (NEON 44,079 cells, `sierra_nf` 58,676; stress
response 36,646 / 55,511). ΔR², bold where the 95% CI excludes 0:

| NEON / `sierra_nf` | hgb | hgb_tuned | mlp |
|---|---|---|---|
| NDMI recovery, Env+S R² | 0.683 / 0.529 | 0.693 / 0.544 | 0.664 / 0.502 |
| Env+S, − hgb | – | **+0.010** / **+0.016** | **−0.019** / **−0.027** |
| +T | **+0.065** / **+0.089** | **+0.070** / **+0.091** | **+0.076** / **+0.110** |
| +Tres | **+0.046** / **+0.070** | **+0.052** / **+0.072** | **+0.052** / **+0.076** |
| Env+S+T\|hgb − Env+S of this learner | – | **+0.055** / **+0.073** | **+0.084** / **+0.115** |
| Env+S+Tres\|hgb − Env+S of this learner | – | **+0.036** / **+0.054** | **+0.065** / **+0.097** |
| Stress response, Env+S R² | 0.705 / 0.490 | 0.719 / 0.528 | 0.692 / 0.460 |
| Env+S, − hgb | – | **+0.013** / **+0.038** | **−0.014** / **−0.030** |
| +T | **+0.022** / **+0.055** | **+0.029** / **+0.068** | **+0.029** / **+0.091** |
| +Tres | **+0.014** / **+0.042** | **+0.017** / **+0.052** | +0.005 / **+0.057** |
| Env+S+T\|hgb − Env+S of this learner | – | **+0.009** / **+0.017** | **+0.036** / **+0.085** |
| Env+S+Tres\|hgb − Env+S of this learner | – | +0.001 / +0.004 | **+0.028** / **+0.072** |

**Cycle 1, over ASO lidar** (NEON 29,004 cells, `sierra_nf` 48,731; gains over
Env+S+L):

| NEON / `sierra_nf` | hgb | hgb_tuned | mlp |
|---|---|---|---|
| NDMI recovery, Env+S+L − hgb | – | **+0.012** / **+0.020** | **−0.031** / −0.007 |
| +T | **+0.056** / **+0.037** | **+0.059** / **+0.038** | **+0.066** / **+0.043** |
| +Tres | **+0.038** / **+0.025** | **+0.041** / **+0.027** | **+0.035** / **+0.012** |
| Env+S+L+Tres\|hgb − Env+S+L of this learner | – | **+0.027** / +0.005 | **+0.070** / **+0.032** |
| Stress response, Env+S+L − hgb | – | **+0.010** / **+0.040** | −0.003 / +0.001 |
| +T | **+0.033** / **+0.035** | **+0.043** / **+0.043** | **+0.052** / **+0.045** |
| +Tres | **+0.024** / **+0.021** | **+0.037** / **+0.031** | **+0.025** / +0.011 [−0.000, 0.022] |
| Env+S+L+Tres\|hgb − Env+S+L of this learner | – | **+0.014** / **−0.019** | **+0.027** / **+0.020** |

**Cross-drought transfer** (environment only, cycle 1 → cycle 2; NEON 24,342
cells, `sierra_nf` 20,089; pooled R² and Spearman ρ):

| NEON / `sierra_nf` | hgb | hgb_tuned | mlp |
|---|---|---|---|
| NDMI recovery R² | −0.42 / −2.68 | −0.45 / −3.06 | −4.70 / −13.9 |
| NDMI recovery ρ | +0.22 / −0.07 | +0.24 / −0.08 | −0.34 / −0.38 |
| NIRv recovery ρ | +0.41 / −0.22 | +0.44 / −0.26 | +0.41 / −0.38 |
| NDMI resilience ρ | −0.39 / −0.36 | −0.33 / −0.37 | −0.29 / −0.55 |
| NDMI recovery ρ within elevation quartiles | +0.15, −0.14, +0.26, +0.27 / +0.18, +0.17, +0.19, +0.29 | +0.16, −0.17, +0.25, +0.31 / +0.21, +0.19, +0.16, +0.28 | −0.01, −0.20, +0.16, +0.16 / −0.24, +0.18, −0.05, −0.01 |

**Within cycle 2** (§11 cells). Leg and T18 gains are much the same under
every learner:

| NEON / `sierra_nf` | hgb | hgb_tuned | mlp |
|---|---|---|---|
| NDMI recovery, Env+S − hgb | – | +0.001 / −0.003 | **−0.056** / **−0.060** |
| +Leg | **+0.063** / **+0.151** | **+0.066** / **+0.160** | **+0.084** / **+0.162** |
| +T18 | **+0.065** / **+0.105** | **+0.064** / **+0.115** | **+0.082** / **+0.133** |
| +T18 beyond Leg | **+0.035** / **+0.019** | **+0.029** / **+0.020** | **+0.031** / **+0.032** |
| NIRv recovery, +T18 beyond Leg | **+0.032** / **+0.028** | **+0.039** / **+0.030** | **+0.033** / **+0.033** |
| NDMI resilience, +T18 beyond Leg | **+0.018** / **+0.015** | **+0.030** / **+0.020** | **+0.037** / **+0.018** |

Tuning changes cycle-2 Env+S by −0.003 to +0.007.

**Reading.**
- **For recovery the shortfall is missing information, not model
  capacity.**
  - Tuned boosting lifts Env+S by only +0.010 / +0.016 for NDMI recovery
    (≤ 0.02), and by nothing in cycle 2.
  - The MLP is worse than the default model everywhere.
  - Every trait and residual-trait gain excludes 0 under every learner, at
    both areas, without and with lidar. Traits on the default model still
    beat the tuned learner without them (+0.055 / +0.073; residual traits
    +0.036 / +0.054).
  - No learner makes the environment-only transfer work: level fails, the
    resilience ranking reverses, and the `sierra_nf` recovery ranking stays
    near 0 pooled with the same within-elevation skill (§21).
- **For stress response a better learner does help, at `sierra_nf`.**
  Tuning adds +0.038 to Env+S (+0.040 over lidar), and there the residual
  traits on the default model no longer beat the tuned learner without them
  (+0.004 without lidar; −0.019 over lidar). Within the tuned learner the
  trait gains are larger (+Tres +0.052 / +0.031 over lidar), so the gain
  survives; stress-response gains should be quoted with the tuned learner.
- **The MLP** gives the largest trait gains (+T), because it extracts least
  from Env+S. Its residual-trait gains are weakest: n.s. for NEON stress
  response and borderline over lidar at `sierra_nf`, but it is the weakest
  learner on every response.
- **Caveats:** a small tuning grid; one inner split per fold; Tres
  residualized with hgb for every learner; no deep tabular models; two runs
  ran concurrently, which affects timing, not results.

### 33. Does the Tahoe-box structural-carbon direction survive a Landsat scene? (`hls_results/spaceborne_only/carbon_stanislaus/`)

In the Yosemite box, residual lignin and cellulose go with worse recovery
beyond the Landsat bands, in cycle 1 (§24) and cycle 2 (§29). The Tahoe
box has the weakest drought signal of the three areas and shows only the
structural-carbon half of the trait directions (§8–9, §22, §27, §30). This
section asks whether that half also survives a Landsat scene there.
Cycle 1 only.

**Data.** The June 4 2013 Tahoe trait mosaic (non-`_v2`, cross-track
normalized) and Landsat 8 path-43 scenes near it (`fetch_landsat_c2_ee.py
--scene`):

| Scene | Days from the flight | Clear share of the AOI |
|---|---|---|
| 2013-05-27 | −8 | 0% (no clear pixels) |
| 2013-06-12 | +8 | 48% |
| **2013-06-28** | +24 | **99%** |

No path-42 scene covers the box on 2013-06-05. Jun 28 is the nearest
usable scene. Jun 12 is a near-date check: cell means use only clear
pixels, so about half of its cells carry no Landsat information.

**Method** (`proto_spaceborne_only.py carbon --resid-landsat l8raw
--resid-landsat l8multi --skip-ladder`).
- Traits are residualized out of fold on three bases: Env+S; Env+S + the
  scene's bands and indices (l8raw); and Env+S + the 2013 June and Jul–Sep
  composites (l8multi).
- The statistic is the within-stratum ρ (aridity × 200 m elevation) with
  cycle-1 recovery, with 1 km block-bootstrap CIs.
- 55,105 cells.
- The run is repeated with blocks shuffled into folds (`--fold-seed` 1–3)
  beside the default assignment.

**NDMI recovery**, default folds (range over the four fold assignments in
brackets; bold where every CI excludes 0):

| Residual trait | on Env+S | + Jun 28 scene | + 2013 composites | + Jun 12 scene |
|---|---|---|---|---|
| **Lignin** | **−0.109** (−0.105 to −0.109) | **−0.090** (−0.086 to −0.090) | **−0.084** (−0.078 to −0.084) | **−0.108** |
| **Cellulose** | **−0.116** (−0.116 to −0.119) | **−0.095** (−0.095 to −0.099) | **−0.092** (−0.092 to −0.095) | **−0.117** |
| Fiber | **−0.081** | **−0.067** | **−0.063** | |
| Nitrogen | +0.025 (+0.021 to +0.025; CI touches 0 under two) | +0.016 (n.s.) | +0.016 (n.s.) | +0.017 (n.s.) |
| LMA | +0.023 (n.s.) | **+0.028** (+0.028 to +0.033; wrong sign, lower bounds +0.001 to +0.007) | **+0.033** (wrong sign) | +0.025 (n.s.) |

- **The structural-carbon direction survives the Landsat bands in the
  Tahoe box.** It survives under every fold assignment, with either Landsat
  base. Beyond the scene the box keeps 83% of the lignin ρ and 82% of the
  cellulose ρ.
- In the Yosemite box, cycle 1 (§24, the same model, the Jun 21 scene
  nine days after the flight), the fractions kept are:
  - NEON: lignin −0.166 → −0.108 (65%), cellulose −0.154 → −0.125 (81%);
  - `sierra_nf`: lignin −0.185 → −0.132 (71%), cellulose −0.163 → −0.122
    (75%).

  The Tahoe direction starts weaker but loses proportionally less.
- The Jun 28 scene is 24 days after the flight, so it is a weaker control
  than the nine-day Yosemite-box scenes. The two composites, which span June
  to September, remove a little more (lignin −0.084, cellulose −0.092).
- N keeps no direction. LMA has a weak wrong-sign one (+0.03), as in §30.
- **NIRv recovery** is weaker throughout:
  - lignin −0.046 on Env+S, −0.033 (−0.029 to −0.033, CIs exclude 0)
    beyond the scene, and −0.016 (n.s. under every assignment) beyond the
    composites;
  - cellulose −0.062, −0.045 and −0.030 (−0.030 to −0.034), every CI
    excluding 0.

**Reading.** In cycle 1 the Tahoe box keeps the lignin and cellulose →
slower NDMI recovery direction once a Landsat scene, or the season's
composites, are in the residual base. It has the same sign as in the
Yosemite box and holds under every fold assignment tried. A second-drought
test of the direction beyond Landsat therefore has cycle-1 support on all
three areas. For NIRv recovery, only cellulose keeps the direction beyond
the composites.

### 34. Held-out sites in the Yosemite box, and elevation strata fixed in advance (`config/heldout_aois.yml`, `config/heldout_strata.yml`)

§10 holds out the Tahoe box, SEKI and the rest of the Yosemite box for the
second drought. Only the Tahoe box (`stanislaus`) had a grid. This section
defines the other two. It also fixes the elevation strata in which
cross-drought rank skill will be reported on all three (§21). The strata
are computed from elevation alone, before any of their 2020–22 responses
exist.

**Grids and masks** (`fetch_heldout_masks.py`, `config/heldout_aois.yml`).
- Both sites lie on the 30 m UTM 11 lattice of the Yosemite-box trait
  mosaics.
- The footprint is `flight_id` > 0 in both the June 12 2013 (`_v2`) and
  June 22 2018 mosaics, the dates of the 2013 and 2018 traits. The two
  footprints cover 16,626 and 17,623 km²; their intersection is 16,489 km².
- `yosemite_rest` is the whole mosaic extent (117 × 190 km). It keeps the
  footprint outside SEKI and outside the `neon_soap_teak` and `sierra_nf`
  grids: 14,915 km².
- `seki` is a 15 × 31 km box. It keeps the footprint inside the NPS
  SEQU/KICA boundary: 47 km², in western Kings Canyon around Grant Grove.
  No other trait-mosaic box reaches the parks.
- The grids live in their own config file, so scripts that loop over
  every AOI of `hls_aois.yml` do not pick them up.
- `aoi_env_layers.py --masks-only` builds NLCD 2013, terrain and the
  per-year fire and harvest layers for them, but not climate or structure,
  which a 117 × 190 km 30 m grid would not hold in memory. Their fire and
  harvest polygons are in `fire/disturbance_heldout/`.
- `yosemite_rest` is added to `response_common.HELD_OUT`.

**Elevation strata** (`proto_heldout_strata.py`).
- Cells: 90 m cells (≥ 70% valid) of NLCD 2013 forest with no fire or
  harvest through 2025, inside the site mask.
- Elevation quartile cutpoints are computed separately for each site. No
  Landsat response or composite enters.

| Site | Cells | Cutpoints (m) | Elevation range (m) |
|---|---|---|---|
| `stanislaus` (Tahoe box) | 52,456 | 1,733 / 1,976 / 2,178 | 931–2,749 |
| `yosemite_rest` | 361,464 | 966 / 2,334 / 2,718 | 181–3,700 |
| `seki` | 389 | 1,916 / 2,007 / 2,235 | 1,692–2,357 |

For comparison, the pilot AOIs on the same definition: NEON 26,672 cells,
1,178 / 1,904 / 2,306 m; `sierra_nf` 20,809 cells, 1,771 / 2,143 /
2,389 m.

- **SEKI is too small to test on its own.** Fire and harvest remove 55% of its
  footprint forest pixels by 2019 and 90% by 2025 (47,538 → 4,764 pixels),
  leaving 389 cells. The forward-pilot minimum is 5,000. SEKI can enter
  only pooled with `yosemite_rest`, or through trait data from outside the
  Yosemite box.
- **`yosemite_rest` is large and spans more elevation than the pilots.**
  It has 7.6× the cells of the two pilot AOIs together, and its lowest quartile (< 966 m) is
  foothill forest and woodland below the mixed-conifer zone of the pilots.

### 35. A second-drought test averaged over fixed fold splits, dry run on the pilot sites (`hls_results/forward_pilot_rule/`)

§31 found that reassigning 1 km blocks to folds can move a single
contrast by more than its bootstrap CI shows. A test fixed in advance for
the held-out sites therefore should not rest on one fold assignment. This
section implements such a test and runs it on the two pilot sites, on the
§11 forward-pilot cells, exactly as the held-out sites will get it.

**Rule** (`proto_forward_pilot.py --rule`; `response_common.FOLD_SEEDS`,
`ladder_splits`, `bootstrap_r2_splits`;
`proto_trait_directions.rho_within_splits`).
- Each contrast is fitted under five fixed fold assignments: 1 km blocks
  shuffled into five folds with seeds 11–15.
- One set of 1 km block-bootstrap draws serves every assignment. Each draw
  averages the paired ΔR² (or the within-stratum ρ) over the five, so the
  CI is that of the average.
- A contrast passes if the CI of the average excludes 0. Per-split values
  and CIs are kept beside it.
- With one assignment the code reproduces `bootstrap_r2` and
  `rho_within_boot` exactly.
- Learner: the tuned gradient boosting of §32, on both arms of every skill
  contrast and in the transfer. Trait residuals use the default learner.

The test has three parts. NDMI recovery is primary; NIRv recovery and NDMI
resistance are reported beside it.
- **Skill:** Env+S → +Leg (the cycle-1 legacy), and Env+S+Leg → +T18 (the
  June 2018 traits, z-scored per date).
- **Directions:** within-stratum ρ of 2018 N, LMA, lignin and cellulose
  with cycle-2 recovery. They are reported as standardized, and
  residualized on four bases: Env+S; Env+S+L1; Env+S+Leg; Env+S+Leg+L1.
  L1 is the Jun 19 2018 Landsat 8 scene, three days before the flight.
- **Transfer:** a cycle-1 model (Env+S, or Env+S+T13) applied to cycle 2
  with T18 swapped in. Rank ρ is computed pooled and within elevation
  quartiles. The quartile cutpoints come from elevation alone on the forest
  cells undisturbed through 2025 (`proto_heldout_strata.py`, the §34
  procedure), not from the response cells. The quartiles are therefore not
  equal in size on the analysis cells (NEON 4,480–6,659 cells).

**Skill** (NEON 24,301 cells / `sierra_nf` 20,055; average over the five
splits, 95% CI, per-split range):

| ΔR² | NDMI recovery | NIRv recovery | NDMI resistance |
|---|---|---|---|
| +Leg over Env+S, NEON | **+0.072** [0.057, 0.094] (0.065–0.078) | **+0.071** | **+0.097** |
| +Leg over Env+S, `sierra_nf` | **+0.157** [0.135, 0.179] (0.150–0.163) | **+0.103** | **+0.086** |
| **+T18 over Env+S+Leg, NEON** | **+0.032** [0.024, 0.041] (0.029–0.039) | **+0.034** [0.027, 0.042] | **+0.024** [0.018, 0.031] |
| **+T18 over Env+S+Leg, `sierra_nf`** | **+0.021** [0.015, 0.028] (0.016–0.027) | **+0.029** [0.022, 0.037] | **+0.022** [0.015, 0.030] |

Env+S R² for NDMI recovery is 0.593 (NEON) and 0.486 (`sierra_nf`). Every
per-split CI excludes 0 as well; the lowest per-split lower bound is
+0.008. These gains match §11 under the default learner and folds (T18
beyond Leg +0.035 / +0.019).

**Directions**, NDMI recovery (average over the five splits; bold where
the CI of the average excludes 0):

| Residual ρ, NEON / `sierra_nf` | Nitrogen | LMA | Lignin | Cellulose |
|---|---|---|---|---|
| standardized, not residualized | **+0.30** / **+0.38** | **−0.31** / **−0.36** | **−0.23** / **−0.35** | **−0.12** / **−0.24** |
| on Env+S | **+0.23** / **+0.23** | **−0.24** / **−0.23** | **−0.16** / **−0.23** | **−0.06** / **−0.16** |
| on Env+S+L1 | **+0.12** / **+0.12** | **−0.12** / **−0.13** | **−0.09** / **−0.12** | **−0.05** / **−0.08** |
| on Env+S+Leg | **+0.17** / **+0.15** | **−0.18** / **−0.15** | **−0.10** / **−0.14** | −0.02 [−0.05, 0.00] / **−0.08** |
| on Env+S+Leg+L1 | **+0.10** / **+0.09** | **−0.11** / **−0.10** | **−0.07** [−0.09, −0.04] / **−0.09** | −0.02 [−0.05, 0.00] / **−0.06** [−0.08, −0.02] |

- Per-split ranges are within ±0.01 of the average.
- Every direction keeps its cycle-1 sign under every base at both sites,
  and every CI excludes 0 except one case: **NEON cellulose once the
  legacy is in the base** (−0.015 to −0.031 across splits; the CI
  touches 0 under every split).
- Lignin passes under all four bases at both sites (−0.066 [−0.093, −0.042]
  and −0.089 [−0.112, −0.061] on the strictest).
- NIRv recovery: every direction, cellulose included, passes under every
  base at NEON (cellulose −0.041 on Env+S+Leg+L1).
- A direction test that requires both lignin and cellulose beyond legacy
  and Landsat would fail at NEON. One on lignin alone, or on Env+S+L1 (no
  legacy), passes at both sites.

**Transfer** (rank ρ with 1 km block-bootstrap CIs; the tuned learner):

| Rank ρ, Env+S → Env+S+T | NEON, NDMI recovery | NEON, NIRv recovery | `sierra_nf`, NDMI recovery | `sierra_nf`, NIRv recovery |
|---|---|---|---|---|
| pooled | +0.26 → +0.27 | +0.34 → **+0.54** | +0.00 → **+0.12** | −0.22 → −0.16 |
| elevation q1 | +0.25 → +0.31 | +0.10 → +0.27 | +0.16 → +0.15 | −0.10 → −0.13 |
| q2 | −0.08 → −0.03 | −0.09 → +0.07 | +0.21 → +0.22 | −0.06 → −0.05 |
| q3 | +0.24 → +0.34 | −0.09 → +0.06 | +0.24 → +0.37 | +0.00 → +0.08 |
| q4 | +0.22 → +0.35 | −0.04 → +0.14 | +0.27 → **+0.51** | −0.12 → −0.05 |

- At `sierra_nf` the §21 pattern reappears with the tuned learner and the
  elevation-only cutpoints. NDMI recovery ranks at about 0 pooled and at
  +0.16 to +0.27 within every quartile; the traits raise the upper two
  quartiles to +0.37 and +0.51.
- NIRv recovery does not transfer at `sierra_nf`, pooled or within
  quartiles (as in §21).
- At NEON the tuned learner transfers the environment better than the
  default one (pooled NDMI recovery +0.26, against +0.20 in §11). Pooled,
  the traits then add little for NDMI recovery (+0.27; §11: +0.20 → +0.33).
  Within three of the four quartiles they still add +0.06 to +0.13.
  NEON's second quartile ranks about 0 with or without traits.

**Reading.** Under the averaged-over-splits rule, the pilot sites pass the
skill part for every target: the 2018 traits add +0.021 to +0.034 beyond
climate, structure and the first drought's legacy, and every split agrees.
The directions keep their signs. Their size depends on the residual base:
legacy or a Landsat scene each remove a quarter to
two thirds of ρ on Env+S, and together more. Under the strictest base cellulose fails at NEON. The exact
direction statistic therefore decides whether NEON would pass, and needs to
be fixed before any held-out response is computed. Transfer rank skill
within elevation strata reproduces §21 for NDMI recovery. It is a
diagnostic (the transfer has no fitted cycle-2 baseline), not a pass/fail
test.

### 36. Which small contrasts survive reassigning blocks to folds? (`hls_results/fold_robustness/`)

The block bootstrap holds the five CV folds fixed. §31 showed that
shuffling 1 km blocks into folds differently can move a single contrast by
up to ±0.006, and §29's NEON control by 0.013, more than its CI shows. This
section reruns the earlier contrasts that sit within about 0.015 of 0. Each
gets three shuffled fold assignments (`--fold-seed` 1–3) beside the
default, on the same cells. A contrast is called **robust** if its CI
excludes 0, with the same sign, under all four assignments.
`collect_fold_seeds.py` writes one table per family to
`hls_results/fold_robustness/` (default / seed 1 / 2 / 3, range, robust).
The large gains, such as traits over Env+S and recovery residual gains
over lidar, were not rerun.

Runs, on NEON and `sierra_nf`, 90 m:
- `proto_spaceborne_only.py compare --no-stack` and `carbon`, NDMI
  recovery;
- `proto_trait_dynamics.py --interval d1513`;
- `proto_forward_pilot.py` (main ladder);
- `proto_structure_from_spectra.py`;
- `proto_response_traits.py --structure` (NDMI resistance, resilience and
  stress response) and `--neighbour` (NDMI resistance, recovery,
  resilience and stress response).

Each was run with `--fold-seed` and `--tag _fold<s>_<aoi>`.

**Results** (ΔR² range over the four assignments; bold where robust):

| Contrast (section) | NEON | `sierra_nf` |
|---|---|---|
| VSWIR − Landsat scene, date for date, EMIT-like (§23) | **+0.009 to +0.012** | **+0.008 to +0.012** |
| same, 30 m (sbg_lo) / native | **+0.019 to +0.021** / **+0.022 to +0.025** | **+0.020 to +0.024** / **+0.032 to +0.035** |
| EMIT-like − two Landsat composites (§23) | −0.003 to −0.005 (tie) | −0.002 to −0.005 (tie) |
| structural carbon beyond the Landsat scene, EMIT-like / 30 m (§24) | **+0.009 to +0.010** / **+0.013 to +0.015** | **+0.005 to +0.009** / **+0.008 to +0.012** |
| same, beyond Landsat-emulated structural carbon (§24) | **+0.006** / **+0.009 to +0.011** | −0.002 to −0.004 / −0.001 to +0.002 (n.s.) |
| AVIRIS change on top of Env+S + Landsat change, NDMI recovery (§3) | **+0.006 to +0.008** | **+0.009 to +0.011** |
| same, NDMI resilience / NIRv recovery / NIRv resilience | **+0.010 to +0.013** / **+0.009 to +0.012** / **+0.006 to +0.008** | **+0.007 to +0.010** / **+0.007 to +0.009** / **+0.006 to +0.007** |
| same, lidar mortality fraction | +0.021 to +0.030 (CI touches 0) | – |
| residual traits over lidar, NDMI resistance: 2013 trees / LVIS / ASO (§5) | +0.003 to +0.011 / **+0.013 to +0.017** / +0.005 to +0.010 (CI touches 0 under two) | ASO **+0.010 to +0.014** |
| same, NDMI resilience | −0.004 to −0.010 / **+0.028 to +0.033** / **+0.015 to +0.018** | **+0.011 to +0.015** |
| same, NDMI stress response | **+0.008 to +0.028** (CI touches 0 under three) / **+0.019 to +0.024** / **+0.016 to +0.024** | **+0.018 to +0.021** |
| neighbour baseline B0: +Tres over Env+S+B0, resistance / recovery / resilience / stress response (§17) | **+0.010 to +0.012** / **+0.042 to +0.044** / **+0.036 to +0.040** / **+0.008 to +0.011** | **+0.030 to +0.035** / **+0.070 to +0.073** / **+0.041 to +0.044** / **+0.037 to +0.039** |
| 2018 traits beyond legacy, cycle 2: NDMI recovery / resistance / resilience (§11) | **+0.025 to +0.035** / **+0.015 to +0.023** / **+0.014 to +0.025** | **+0.016 to +0.024** / **+0.013 to +0.020** / **+0.010 to +0.017** |
| same, NIRv recovery / resistance / resilience | **+0.028 to +0.033** / **+0.014 to +0.021** / **+0.015 to +0.019** | **+0.026 to +0.030** / **+0.028 to +0.042** / **+0.038 to +0.041** |
| skill kept without lidar, R²(Env+S+Ŝ+T) / R²(Env+S+L+T) (§15) | 0.94–1.01 (ASO 0.94–0.96, LVIS 0.95–1.01, 2013 trees 0.95–1.01) | ASO 0.92–0.96 |

Notes on the table:
- In the stress-response row the 2013-trees entry is not robust: the bold
  marks only the LVIS and ASO entries.
- Medians of the spread (max − min over the four assignments) per family:
  - 0.002–0.003 for the head-to-head, structural-carbon and
    trait-dynamics contrasts and the neighbour gains;
  - 0.003–0.005 for the LVIS and ASO lidar checks;
  - 0.006–0.008 for the cycle-2 gains, the no-lidar ladders and the
    7,040-cell 2013-tree lidar check.

  The largest spread among the quoted contrasts is 0.020: the stress
  response over the 2013 trees.

**Reading.**
- **Robust:**
  - the date-for-date VSWIR lead over a Landsat scene, at both areas and
    for every configuration;
  - the structural-carbon skill beyond the Landsat scene and composites;
  - the AVIRIS-change add-on to Landsat change for every recovery and
    resilience target;
  - every neighbour-baseline gain;
  - every cycle-2 gain of the 2018 traits beyond legacy;
  - with LVIS and ASO, the residual-trait gains over lidar for resilience
    and stress response, and for resistance except ASO at NEON;
  - the no-lidar skill kept (0.92–1.01).
- **Fold-sensitive:**
  - NEON stress response over the 2013 lidar trees: +0.028 with the default
    folds, +0.008 to +0.015 otherwise;
  - NDMI resistance over ASO at NEON: +0.005 to +0.010;
  - the trait-change gain for the lidar mortality fraction.

  These were already the weakest entries of §5 and §3. The stress-response
  and resistance gains over lidar rest on LVIS and on ASO at `sierra_nf`.
- **Unchanged ties and nulls:** the EMIT-like vs two-composite tie, the
  `sierra_nf` structural-carbon margin over Landsat-emulated carbon, and the
  2013-tree resistance and resilience gains stay ties or nulls under every
  assignment.

### 37. A structural-carbon score under the averaged rule (`hls_results/forward_pilot_rule/`)

In §35, lignin keeps its direction with cycle-2 recovery on every residual
base at both pilot sites. Cellulose loses its CI at NEON once the
cycle-1 legacy is in the base. A direction test on both traits would then
depend on which trait is asked. This section adds one statistic for
the pair: a structural-carbon score.

**Score** (`proto_forward_pilot.py --rule --rule-directions-only`).
- The mean of z-scored lignin and cellulose. On the residualized versions,
  the z-scores are of the two residuals, separately for each base and fold
  assignment. The expected sign is negative.
- Its within-stratum ρ, the averaged-over-splits CI and the per-split rows
  come from the same code and the same block draws as each trait in §35.
- `--rule-directions-only` skips the skill and transfer parts. The
  per-trait rows reproduce `rule_directions_*.csv` exactly (200 of 200
  rows at each site).

**Directions** (within-stratum ρ with cycle-2 recovery, average over the
five splits, 95% CI):

| Residual ρ of the score | NDMI recovery, NEON | NDMI recovery, `sierra_nf` | NIRv recovery, NEON | NIRv recovery, `sierra_nf` |
|---|---|---|---|---|
| standardized, not residualized | **−0.18** [−0.22, −0.14] | **−0.31** [−0.34, −0.27] | **−0.14** | **−0.09** |
| on Env+S | **−0.12** [−0.15, −0.09] | **−0.21** [−0.23, −0.18] | **−0.11** | **−0.12** |
| on Env+S+L1 | **−0.07** [−0.10, −0.05] | **−0.11** [−0.13, −0.07] | **−0.09** | **−0.08** |
| on Env+S+Leg | **−0.07** [−0.09, −0.04] | **−0.12** [−0.14, −0.09] | **−0.08** | **−0.09** |
| on Env+S+Leg+L1 | **−0.05** [−0.07, −0.02] | **−0.08** [−0.10, −0.05] | **−0.07** [−0.10, −0.05] | **−0.07** [−0.09, −0.04] |

On the strictest base (Env+S+Leg+L1), NDMI recovery:

| ρ (95% CI) | NEON | `sierra_nf` |
|---|---|---|
| lignin | −0.066 [−0.093, −0.042] | −0.089 [−0.112, −0.061] |
| cellulose | −0.022 [−0.053, 0.003] | −0.055 [−0.081, −0.024] |
| score | **−0.046** [−0.074, −0.019] | **−0.077** [−0.102, −0.047] |

- **The score passes on every base at both sites, for both recovery
  targets.** Every per-split CI excludes 0 as well. The weakest is NEON
  NDMI recovery on the strictest base: per-split ρ −0.037 to −0.051, upper
  bounds −0.009 to −0.024.
- It sits between its two traits: weaker than lignin alone and stronger
  than cellulose. At NEON the strictest-base margin from 0 (upper bound
  −0.019) is smaller than lignin's (−0.042).

**Reading.** As a single statistic for structural carbon, the score
passes where cellulose alone fails, at NEON beyond legacy. It keeps both
traits in the test. Lignin alone keeps the wider margin.

### 38. Candidate definitions of the held-out remainder (`hls_results/heldout_variants/`)

§34 left two questions about the Yosemite-box held-out sites.
- `seki` is too small to test alone.
- `yosemite_rest` reaches down into foothill woodland, below the mixed
  conifer of the pilot sites.

This section measures what pooling and restricting would do. It uses only
elevation, forest type and the disturbance masks; no 2020–22 response of
these areas is computed. The cutpoints in `config/heldout_strata.yml` are
unchanged. The candidates are written to
`hls_results/heldout_variants/strata_candidates.yml`.

**Forest type.**
- `fetch_forest_type.py` now also covers the two held-out grids (LANDFIRE
  2014 EVT, the groups of §9).
- `yosemite_rest` is 10% red fir, 8% mesic mixed conifer and 7% dry
  pine/mixed conifer. Foothill pine woodland is 10%, chaparral 8% and oak
  woodland 5%.
- A cell counts as conifer if at least 50% of its valid pixels are in the
  pine, mesic, red-fir or subalpine groups. Subalpine meadow is excluded.

**Options** of `proto_heldout_strata.py`:
- `--conifer`;
- `--min-elevation` and `--max-elevation`;
- `--pool` (add another area's cells);
- `--key` (write under its own name).

**Variants** (90 m cells of undisturbed NLCD forest, as in §34):

| Variant | Cells | Cutpoints (m) | Elevation range (m) |
|---|---|---|---|
| `yosemite_rest` (as fixed in §34) | 361,464 | 966 / 2,334 / 2,718 | 181–3,700 |
| + `seki` | 361,853 | 966 / 2,333 / 2,718 | 181–3,700 |
| conifer | 234,713 | 2,244 / 2,591 / 2,813 | 346–3,597 |
| conifer + `seki` | 235,057 | 2,242 / 2,591 / 2,813 | 346–3,597 |
| floor 1,178 m (NEON's lowest cutpoint) | 252,675 | 2,271 / 2,590 / 2,810 | 1,178–3,700 |
| conifer + floor 1,178 m | 225,302 | 2,308 / 2,610 / 2,822 | 1,178–3,597 |
| conifer + ceiling 3,042 m (NEON's highest cell) | 223,346 | 2,211 / 2,566 / 2,783 | 346–3,042 |
| conifer + floor + ceiling | 213,935 | 2,285 / 2,587 / 2,792 | 1,178–3,042 |
| `seki`, conifer | 344 | 1,915 / 2,008 / 2,247 | 1,692–2,357 |

The same conifer filter on the other areas (undisturbed forest cells):

| Area | All | Conifer | Kept | Conifer cutpoints (m) |
|---|---|---|---|---|
| NEON | 26,672 | 18,856 | 71% | 1,766 / 2,163 / 2,425 |
| `sierra_nf` | 20,809 | 18,777 | 90% | 1,832 / 2,173 / 2,403 |
| `stanislaus` (Tahoe box) | 52,456 | 49,442 | 94% | 1,738 / 1,977 / 2,181 |

- **Pooling `seki` changes nothing that matters.** It adds 389 cells (344
  conifer) and moves no cutpoint by more than 2 m.
- **Most of the foothill cells go with either restriction.** A conifer
  filter and a floor at 1,178 m each remove about a third of
  `yosemite_rest`. Each raises the lowest cutpoint from 966 m to about
  2,250 m.
- **The remainder is mostly higher than the pilot sites, even restricted.**
  - The conifer `yosemite_rest` cells have a median of 2,591 m. The
    conifer cells of NEON and `sierra_nf` have medians of 2,163 and
    2,173 m.
  - 65% (NEON) and 66% (`sierra_nf`) of the conifer remainder lie above
    the pilot sites' upper conifer cutpoint (2,425 / 2,403 m). Only 5%
    and 12% lie above their highest cell.
  - So the remainder is mostly upper montane and subalpine forest (red fir
    and lodgepole pine), where the pilots are mostly mixed conifer. A
    ceiling at the pilots' highest cell trims only 11,367 cells.
- **The filter would not touch the pilot sites equally.** It drops 29% of
  NEON's cells but 10% of `sierra_nf`'s and 6% of the Tahoe box's. A
  conifer-only definition of the held-out sites therefore needs the dry
  run of §35 repeated on conifer pilot cells
  (`proto_forward_pilot.py --rule --conifer-only`; below).

**The §35 rule on conifer pilot cells** (`--rule --conifer-only`;
elevation strata recomputed from the conifer cells; files
`rule_*_conifer_<aoi>.csv`). NEON keeps 18,643 cells and `sierra_nf`
18,317.

| ΔR², average over five splits (95% CI) | All cells | Conifer cells |
|---|---|---|
| +Leg over Env+S, NEON, NDMI recovery | +0.072 | +0.061 [0.048, 0.076] |
| **+T18 over Env+S+Leg, NEON**: NDMI recovery / NIRv recovery / NDMI resistance | +0.032 / +0.034 / +0.024 | **+0.037** [0.024, 0.055] / **+0.034** / **+0.040** |
| +Leg over Env+S, `sierra_nf`, NDMI recovery | +0.157 | +0.149 [0.128, 0.172] |
| **+T18 over Env+S+Leg, `sierra_nf`**: NDMI recovery / NIRv recovery / NDMI resistance | +0.021 / +0.029 / +0.022 | **+0.026** [0.019, 0.034] / **+0.031** / **+0.020** |

| Residual ρ on Env+S+Leg+L1, NDMI recovery | All cells, NEON / `sierra_nf` | Conifer cells, NEON / `sierra_nf` |
|---|---|---|
| nitrogen | **+0.10** / **+0.09** | **+0.10** / **+0.09** |
| LMA | **−0.11** / **−0.10** | **−0.11** / **−0.09** |
| lignin | **−0.07** / **−0.09** | **−0.09** / **−0.09** |
| cellulose | −0.02 [−0.05, 0.00] / **−0.06** | **−0.05** [−0.08, −0.02] / **−0.06** |
| score (§37) | **−0.05** / **−0.08** | **−0.07** / **−0.08** |

- **On conifer cells every direction passes on every base at both
  sites**, for NDMI and NIRv recovery. That includes NEON cellulose beyond
  legacy, which fails on all cells. The cells that the filter removes at
  NEON (oak and foothill pine woodland, chaparral and riparian, 29%) are
  where cellulose loses its direction.
- The skill part passes as before. The 2018 traits add +0.020 to +0.040
  beyond legacy, every split agreeing, as much as on all cells or more.
- **Transfer** (rank ρ of NDMI recovery, Env+S → Env+S+T):
  - NEON pooled goes from +0.26 → +0.27 on all cells to −0.09 → +0.09 on
    conifer cells.
  - Within the conifer quartiles NEON gives −0.05 → +0.04, +0.16 → +0.34,
    +0.27 → +0.43 and +0.38 → +0.52.
  - `sierra_nf` pooled goes from +0.00 → +0.12 to −0.11 → +0.05.
  - Within its quartiles `sierra_nf` gives +0.25 → +0.13, +0.16 → +0.22,
    +0.20 → +0.37 and +0.30 → +0.46.
  - As in §21 and §35, the transfer ranks within elevation bands better
    than pooled. On conifer cells the traits raise every quartile except
    the lowest.

**Computation.** Independent fits (fold assignments × feature sets in
`ladder_splits`, and the trait residuals of `--rule`) now run in worker
processes, `response_common.pmap`, when `ECOPRO_JOBS` > 1. Each worker
uses one OpenMP thread. On tables of this size a gradient-boosting fit is
faster on one thread than on ten, and gives the same predictions.
Parallel and serial runs give identical ladders, predictions and
residuals; the first NEON conifer contrasts match a serial run digit for
digit. A full `--rule` run on one site takes about 15 minutes with eight
workers, against 3–3.5 hours before.

**Reading.**
- Restricting the held-out remainder to conifer removes its foothill
  woodland and costs the pilot test nothing.
- On conifer cells the pilots pass every part of the rule. That includes
  both structural-carbon traits beyond legacy and the Landsat scene.
- What a restriction does not change is that the remainder sits higher
  than the pilots: two thirds above their upper conifer quartile.

## Summary

1. **Coverage.**
   - The 2013–2025 AVIRIS record over the Yosemite and Tahoe boxes is one
     sensor, AVIRIS-Classic, with same-day AVIRIS-5 in July 2025.
   - The 2020–2022 drought itself is poorly covered: late-season 2020 and
     early-spring 2021 flights with no public L2 reflectance, and nothing
     in 2022.
   - Growing-season acquisitions resume in June 2024.
2. **Continuous responses have headroom.** Climate, terrain and structure
   leave 12–39% of the response variance unexplained after allowing for
   noise. The residual is spatially coherent at about 1 km: stands under the
   same forcing follow different trajectories.
3. **Traits add to climate and structure for every response, at both
   AOIs** (+0.02 to +0.09 R²).
   - **For recovery the gain is robust to lidar structure.** Residual
     traits add +0.025 to +0.048 over NEON 2013 trees, LVIS 2008 and the
     ASO composite, at NEON and `sierra_nf` (§5). Stress response also
     survives (+0.021 to +0.028).
   - It is not the green-fraction QC, dead canopy or post-2015 retrieval
     artifacts, and it holds for other recovery definitions. N and LMA
     alone give 70% of it over lidar structure at NEON (§6).
   - In the Yosemite box it follows leaf economics: high N and low LMA
     recover better. In the Tahoe box only the structural-carbon direction
     (high lignin, cellulose and fiber → worse recovery) replicates, and
     resistance gains most (§8).
   - The leaf-economics difference is not calibration, forest type or the
     weaker drought signal (§9): Yosemite-box fir stands show it, and Tahoe
     fir stands do not. Structural carbon replicates in every test.
   - For mortality, the gain is traits standing in for structure. For
     resistance and resilience, small residual gains (≤ +0.015) remain
     with the larger lidar samples.
4. **Trait change vs Landsat change.** For recovery, AVIRIS change beats
   Landsat change head to head. For every response the two are
   complementary. On top of climate, structure and Landsat change, AVIRIS
   change adds about +0.01.
5. **Site dependence.** Traits matter far more for mortality at SOAP than at
   TEAK. They do not matter more in drier cells.
6. **Environment-only models do not transfer between droughts**, and the
   resilience ranking reverses. That points to legacies of the first drought.
7. **Lidar.** Pre-drought airborne lidar exists for most of NEON (LVIS 2008
   and USFS/NCALM/NEON flights) but only 10% of `sierra_nf` and none of
   `stanislaus`. The ASO 2014–17 composite covers both Yosemite-box AOIs
   and does not detectably encode the die-off. GEDI is too sparse at 90 m
   (`lidar_coverage.md`).
8. **Slow traits keep their spatial pattern 2013 → 2018** (ρ 0.76–0.89 for
   N, LMA and lignin in stable cells) but not their level (2018 N +1.2 SD).
   Applying 2013-trained models to 2018 traits needs per-date
   standardization (§7).
9. **Post-drought traits help with the next drought** (§11). Within
   2020–22, the 2018 traits add +0.015 to +0.041 beyond the first drought's
   legacy at NEON and `sierra_nf`, and the N/LMA/lignin directions carry
   over. Transferring a model fitted on 2012–16 is still unreliable.
10. **Not lidar, not autocorrelation, not scale.**
    - Models with no lidar keep 92–100% of the lidar-reference skill (§15).
      Spectra recover lidar height and cover at ρ 0.79–0.88.
    - Trait gains survive a spatial-neighbourhood baseline in every area
      (§17).
    - They hold at 30, 90 and 270 m (§19).
11. **Spaceborne-like sampling keeps most of the recovery signal** (§18):
    85% for EMIT-like spectra and 95–102% for 30 m imaging-spectrometer-like
    spectra, over lidar.
    - In a paired test (§20), full spectra beat *simulated* Landsat bands
      for NDMI recovery (+0.004 to +0.014 over lidar).
    - They do not beat a real Landsat 8 scene. Trait proxies emulated from
      it match native AVIRIS over lidar at NEON and `sierra_nf`, and beat
      the EMIT-like configuration.
    - Without lidar, native AVIRIS traits still beat untrained Landsat
      bands (+0.018 to +0.026).
12. **Diversity is a small refinement** (≤ +0.016 beyond mean traits; §16).
    AVIRIS-5 and AVIRIS-C agree in rank but not level (§12). EMIT gives 1–3
    growing-season views per cell per year (§13).
13. **The reversed cross-drought ranking at `sierra_nf` is an elevation
    effect** (§21).
    - The elevation gradient of every response flips between the two
      droughts.
    - Within elevation quartiles the transferred model ranks cells
      correctly: NDMI recovery ρ +0.20 to +0.29, and +0.20 to +0.50 with
      traits.
    - The Creek Fire edge and recent treatments do not explain it.
14. **The Tahoe-box leaf-economics gap is not mortality agent, host or
    drought timing** (§22).
    - In the Tahoe box, N and LMA stay near 0 in every ADS agent, host and
      timing group.
    - In the Yosemite box the direction holds in all of them.
    - Structural carbon holds everywhere.
15. **Without lidar or local airborne training, spaceborne-like VSWIR beats
    one Landsat scene but not a multi-date Landsat block** (§23, NEON and
    `sierra_nf`).
    - NDMI recovery over Env+S: EMIT-like +0.009 / +0.012 and 30 m +0.012
      to +0.024 over the single scene. Against June + Jul–Sep composites
      EMIT-like ties at both areas; the low-noise 30 m configuration is
      ahead only at `sierra_nf` (+0.009). Native AVIRIS is ahead at both
      (+0.007, +0.019).
    - The no-lidar stack keeps 88–95% of the lidar-plus-AVIRIS reference
      whatever the inputs; multi-date Landsat keeps as much as EMIT-like.
      Only the low-noise 30 m configuration stays ahead of it (+0.015 to
      +0.022).
16. **The structural-carbon direction is the VSWIR-specific part of the
    trait signal** (§24). At both areas spaceborne-like spectra keep the
    lignin and cellulose directions, also beyond the Landsat bands, while
    Landsat-emulated cellulose has none. The extra recovery skill on top
    of Landsat is VSWIR-specific at NEON only. Leaf economics adds nothing
    beyond multi-date Landsat.
17. **Transferred VSWIR retrievals do not hold up better than transferred
    Landsat proxies** (§25), with 25- or 10-component emulators. Losses
    in recovery skill are alike, or larger for VSWIR (`sierra_nf` 2018).
    In the Tahoe box and in 2018 an untrained multi-date Landsat block
    matches or beats transferred VSWIR (by up to +0.043). VSWIR keeps
    higher map agreement after transfer, which does not become recovery
    skill. 10 components fix the 25-component collapse of transferred
    native structural carbon in the Tahoe box.
18. **The AVIRIS-5 bridge replicates at `sierra_nf`** (EWT ρ 0.90, AVIRIS-5
    +9%) **and extends to traits** with a 10-component retrieval (N and LMA
    ρ 0.73–0.90), not with a 25-component one (N 0.28–0.39) (§26).
19. **Substrate does not explain the Tahoe-box leaf-economics gap** (§27).
    On the same granitic unit, N and LMA track recovery in the Yosemite box
    and not in the Tahoe box; Yosemite-box volcanic cells keep the
    direction. Structural carbon holds on every substrate.
20. **A spaceborne-like VSWIR time series does not consistently beat a
    Landsat time series of the same dates** (§28, no lidar).
    - 2013: at NEON the 30 m and native series lead (+0.012 to +0.030),
      and the EMIT-like series leads only against the partly cloudy
      nearest scenes. At `sierra_nf` they tie, or Landsat leads (−0.009).
    - 2018 (NEON only; Aug 28 misses the `sierra_nf` pilot cells): Landsat
      leads (−0.025 to −0.028).
    - With both series in the model, VSWIR adds +0.009 to +0.045 for NDMI
      recovery in every case.
21. **For cycle-2 NDMI recovery one June 2018 Landsat scene matches the 2018
    AVIRIS traits** (§29). Beyond Env+S, legacy and the scene, the traits
    add +0.000 (NEON) and +0.009 (`sierra_nf`); under three shuffled fold
    assignments +0.010 to +0.013 and +0.008 to +0.016 (§31). For NIRv
    recovery and resistance they add +0.007 to +0.021. The N, LMA, lignin and
    cellulose directions survive the Landsat bands at about half strength.
22. **The Tahoe-box leaf-economics gap is not a retrieval effect** (§30). A
    Yosemite-trained retrieval applied to Tahoe spectra gives N and LMA no
    direction, while the same transfer between NEON and `sierra_nf` keeps
    it. Structural carbon holds with every retrieval.
23. **In the first drought, a VSWIR series adds more to a Landsat series
    than as many extra Landsat dates do** (§31). For NDMI recovery: EMIT-like
    +0.011 to +0.013, 30 m +0.011 to +0.020, at both areas, with either split
    of six Landsat dates, under every fold assignment tried. In 2018 (NEON)
    two more Landsat dates match or beat it (EMIT-like −0.003 to −0.013).
    On top of the denser Landsat series VSWIR still adds. In cycle 2 a second
    Landsat scene matches the 2018 traits for NDMI responses. Fold assignment
    moves these contrasts by up to ±0.006, and §29's NEON +0.000 by up to
    +0.013.
24. **The recovery shortfall is information, not model capacity** (§32).
    Tuned boosting adds ≤ +0.016 to Env+S for NDMI recovery, an MLP does
    worse, every trait and residual-trait gain survives under every learner
    (with and without lidar), and no learner makes the cross-drought
    transfer work. For stress response at `sierra_nf` tuning adds +0.038,
    and the trait gains are larger within the tuned learner.
25. **The Tahoe-box structural-carbon direction survives a Landsat scene**
    (§33). Cycle 1, NDMI recovery: residual lignin −0.109 → −0.090 and
    cellulose −0.116 → −0.095 beyond the nearest clear 2013 scene (−0.084 /
    −0.092 beyond the June and Jul–Sep composites). Every CI excludes 0
    under every fold assignment. That keeps more than 80% of the ρ, against
    65–81% in the Yosemite box.
26. **Held-out sites in the Yosemite box** (§34). `yosemite_rest` (361,464
    undisturbed forest cells) and `seki` (389 cells, too few on their own)
    are defined on the trait-mosaic footprints. Elevation quartile cutpoints
    for them and the Tahoe box are fixed in `config/heldout_strata.yml` from
    elevation alone.
27. **The second-drought test, averaged over five fixed fold splits,
    passes on the pilot sites** (§35). With the tuned learner, the 2018
    traits add +0.021 to +0.034 beyond Env+S + legacy (NDMI and NIRv
    recovery, NDMI resistance; both sites; every split agrees). N, LMA and
    lignin keep their directions beyond legacy and the 2018 Landsat scene at
    both sites. Cellulose does not beyond legacy at NEON. Within elevation
    quartiles the `sierra_nf` transfer ranks NDMI recovery correctly (§21).
28. **Most small contrasts survive reassigning blocks to folds** (§36).
    Three shuffled fold assignments beside the default leave robust:
    - the date-for-date VSWIR lead over a Landsat scene;
    - the structural-carbon skill beyond Landsat;
    - the AVIRIS-change add-on;
    - every neighbour-baseline gain;
    - every cycle-2 gain of the 2018 traits beyond legacy;
    - the LVIS and ASO residual gains for resilience and stress response;
    - the no-lidar skill kept (0.92–1.01).

    Fold-sensitive: stress response over the 2013 lidar trees at NEON
    (+0.008 to +0.028) and resistance over ASO at NEON (+0.005 to
    +0.010).
29. **A structural-carbon score passes where cellulose alone does not**
    (§37). The mean of z-scored lignin and cellulose keeps its negative
    direction with cycle-2 recovery on every residual base at both pilot
    sites, for NDMI and NIRv recovery, under every split. Beyond legacy and
    the 2018 Landsat scene, NDMI recovery gives −0.046 [−0.074, −0.019]
    (NEON) and −0.077 (`sierra_nf`). Lignin alone gives −0.066 and −0.089.
30. **Candidate definitions of the held-out remainder** (§38).
    - Pooling `seki` changes no cutpoint by more than 2 m.
    - A conifer filter or a 1,178 m floor removes about a third of
      `yosemite_rest` (the foothill woodland).
    - Either way, two thirds of what remains lies above the pilot sites'
      upper elevation quartile.
    - On conifer pilot cells (29% fewer at NEON) the rule passes every part
      at both sites. That includes NEON cellulose beyond legacy (−0.05),
      which fails on all cells.

## Code

`fetch_landsat_c2_ee.py` (`--scene`), `fetch_disturbance_aois.py`, `aoi_env_layers.py` (`--masks-only`, `--disturbance-dir`),
`fetch_heldout_masks.py`, `proto_heldout_strata.py` (`--conifer`,
`--min-elevation`, `--max-elevation`, `--pool`, `--key`), `collect_fold_seeds.py`,
`query_airborne_coverage.py`,
`response_common.py`, `proto_response_metrics.py`, `proto_trait_dynamics.py`
(`--interval`, `--fold-seed`, `--tag`),
`proto_response_traits.py` (`--structure`, `--ablation`, `--neighbour`,
`--traits`/`--cwc`, `--trait-year`, `--line-z`, `--raw-block`,
`--save-preds`, `--gains-only`, `--fold-seed`, `--tag`),
`proto_response_transfer.py`, `query_lidar_coverage.py`,
`fetch_lidar_structure.py`, `proto_lidar_validation.py`,
`proto_trait_stability.py`, `fetch_wdts_cwc.py`, `fetch_wdts_traits.py`
(`--date`, `--suffix`), `fetch_forest_type.py`,
`proto_trait_directions.py` (`--group`, incl. `substrate`; `--trait-cells`/`--trait-source`;
`rho_within_splits`),
`proto_forward_pilot.py` (`--landsat`, `--landsat-composites`, `--tag`,
`--fold-seed`, `--rule` with `--rule-scene`, `--learner`, `--resid-learner`,
`--rule-directions-only`, `--conifer-only`; `ECOPRO_JOBS` worker processes
via `response_common.pmap`),
`proto_transfer_diagnostics.py`,
`proto_structure_from_spectra.py` (`--fold-seed`, `--tag`), `proto_trait_diversity.py`,
`proto_spaceborne_sim.py` (`refl` with `--year`, `simulate` with
`--l8`/`--seed`, `compare`), `proto_spaceborne_only.py` (`compare` with
`--no-stack`, `carbon` with `--resid-landsat` and `--skip-ladder`; both
with `--fold-seed` and `--tag`), `proto_spaceborne_ts.py` (`refl`, `emulate`, `compare` with `--landsat-extra` and `--fold-seed`),
`proto_retrieval_transfer.py` (`emulate`, `score`),
`proto_model_capacity.py` (`within`, `transfer`, `forward`; learners in
`response_common.make_model`),
`proto_aviris5_bridge.py` (`--traits`, `--n-comp`), `fetch_geology.py`,
`query_emit_coverage.py`. Run from `src/` in
the `ecopro` env. Earth Engine uses the Cloud project `ecopro-509818`.
`shap` is installed with pip. Run concurrent model jobs with
`OMP_NUM_THREADS` set so that their threads do not exceed the cores (e.g. 3
jobs × 3 threads on 10 cores); oversubscribing slowed runs 5–10×.
