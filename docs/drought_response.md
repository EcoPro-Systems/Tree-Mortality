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
| WDTS traits and canopy water | `wdts/<aoi>_traits.nc`, `wdts/<aoi>_cwc.nc` | As before. `fetch_wdts_cwc.py` now also saves `nadir_dist` |

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
   - **For recovery the gain is robust to lidar structure:** +0.064 at
     NEON, with residual traits +0.038.
   - It follows leaf economics: high N and low LMA recover better.
   - For mortality and resilience, most of the gain is traits standing in
     for structure.
4. **Trait change vs Landsat change.** For recovery, AVIRIS change beats
   Landsat change head to head. For every response the two are
   complementary. On top of climate, structure and Landsat change, AVIRIS
   change adds about +0.01.
5. **Site dependence.** Traits matter far more for mortality at SOAP than at
   TEAK. They do not matter more in drier cells.
6. **Environment-only models do not transfer between droughts**, and the
   resilience ranking reverses. That points to legacies of the first drought.

## Code

`fetch_landsat_c2_ee.py`, `fetch_disturbance_aois.py`, `aoi_env_layers.py`,
`query_airborne_coverage.py`,
`response_common.py`, `proto_response_metrics.py`, `proto_trait_dynamics.py`,
`proto_response_traits.py`, `proto_response_transfer.py`. Run from `src/` in
the `ecopro` env. Earth Engine uses the Cloud project `ecopro-509818`.
`shap` is installed with pip.
