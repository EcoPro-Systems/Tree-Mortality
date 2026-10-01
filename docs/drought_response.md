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
| Fire and harvest polygons | `fire/disturbance/{frap_fires,rx_fires,facts_harvest}.gpkg` | From `fetch_disturbance_aois.py`: CAL FIRE FRAP perimeters (all sizes), CAL FIRE prescribed-fire perimeters, USFS FACTS timber-harvest activities (EDW) |
| AVIRIS flight-line inventory | `aviris_locator/AVIRIS-{C,NG}_flight_{table.csv,s.geojson}` | [ORNL DAAC 2140](https://doi.org/10.3334/ORNLDAAC/2140) flight tables (to Aug 2024) |
| WDTS traits and canopy water | `wdts/<aoi>_traits.nc`, `wdts/<aoi>_cwc.nc` | As before. `fetch_wdts_cwc.py` now also saves `nadir_dist`. `stanislaus` traits (Tahoe box, UTM 11) are warped onto the UTM 10 AOI grid with nearest neighbour; no canopy water there yet |
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

## Code

`fetch_landsat_c2_ee.py` (`--scene`), `fetch_disturbance_aois.py`, `aoi_env_layers.py`,
`query_airborne_coverage.py`,
`response_common.py`, `proto_response_metrics.py`, `proto_trait_dynamics.py`,
`proto_response_traits.py` (`--structure`, `--ablation`, `--neighbour`,
`--traits`/`--cwc`, `--trait-year`, `--line-z`, `--raw-block`,
`--save-preds`, `--gains-only`),
`proto_response_transfer.py`, `query_lidar_coverage.py`,
`fetch_lidar_structure.py`, `proto_lidar_validation.py`,
`proto_trait_stability.py`, `fetch_wdts_cwc.py`, `fetch_wdts_traits.py`
(`--date`, `--suffix`), `fetch_forest_type.py`,
`proto_trait_directions.py` (`--group`), `proto_forward_pilot.py`,
`proto_transfer_diagnostics.py`,
`proto_structure_from_spectra.py`, `proto_trait_diversity.py`,
`proto_spaceborne_sim.py` (`refl`, `simulate` with `--l8`/`--seed`,
`compare`), `proto_aviris5_bridge.py`,
`query_emit_coverage.py`. Run from `src/` in
the `ecopro` env. Earth Engine uses the Cloud project `ecopro-509818`.
`shap` is installed with pip. Run concurrent model jobs with
`OMP_NUM_THREADS` set so that their threads do not exceed the cores (e.g. 3
jobs × 3 threads on 10 cores); oversubscribing slowed runs 5–10×.
