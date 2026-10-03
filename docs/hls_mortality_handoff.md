# Handoff: HLS vs USFS ADS mortality prototype

Last updated 2026-09-25 22:10 PDT. Findings, methods and next steps are in
[`hls_mortality_prototype.md`](hls_mortality_prototype.md). This file covers
where things are and how to resume.

## State at handoff

| Step | `sierra_nf` | `stanislaus` | `lassen` |
|---|---|---|---|
| HLS fetch (2013–2025, DOY 150–300) | done (862 + 37 cloudy) | done (613 + 14 cloudy) | **running** at handoff |
| NLCD land cover + canopy | done | done | done |
| ADS labels, survey dates, MTBS | done | done | done |
| Jul–Sep composites | done | queued | queued |
| Flight-matched composites (±20 d) | done | queued | queued |
| Signal analysis, fixed window | done (single AOI) | queued (all AOIs) | queued |
| Signal analysis, flight-matched | done (single AOI) | queued (all AOIs) | queued |
| Held-out model | done (single AOI, fixed window) | queued | queued |

**Queued runs (Claude Code background shells in this session):**

1. `lassen` fetch → `hls_annual_composites.py -a stanislaus -a lassen` →
   `proto_hls_mortality_signal.py` → `proto_hls_mortality_model.py`, writing
   to `hls_results/all/` (log: `pipeline.log`).
2. After (1) writes `model_cv.csv`: `hls_flight_composites.py -a stanislaus
   -a lassen`, then signal and model with `--composite-suffix _flight_w20.nc`,
   writing to `hls_results/all_flight/`.

If the session ended before these finished, re-run them manually with the
commands below. Every step is restartable.

**Added since the table above:**
- A fourth AOI, `neon_soap_teak` (33×19 km around NEON SOAP/TEAK). It is
  fully processed: 1,416 HLS granules, labels, and all composite types.
- NAIP-2020-date-matched composites for all four AOIs
  (`hls_composites/<aoi>_naip2020_{L30,L30S30}_w25.nc`).
- Cheng 2020 comparison: `hls_results/cheng2020/` (3 AOIs) and
  `hls_results/cheng2020_4aoi/` (4 AOIs).
- Hemming-Schroeder multi-year test: `hls_results/neon_hs/`.
- All the main fixed-window and flight-matched ADS runs finished
  (`hls_results/all/`, `hls_results/all_flight/`).
- **The NAIP download is paused** (Planetary Computer throttling). `sierra_nf` is
  complete; the rest is outstanding. See [`naip_fetch.md`](naip_fetch.md) for status and
  how to resume on another machine.

```sh
python hls_naip_composites.py $CFG $E/hls $E/hls_composites -s L30          # and -s L30 -s S30
python proto_hls_vs_cheng.py $CFG $E/hls_results/cheng2020_4aoi
python proto_hls_vs_hs.py    $CFG $E/hls_results/neon_hs
```

- **Cheng model run locally on NAIP.** 3 quarter-quads in `sierra_nf`,
  about 1 min each on MPS. Results are in
  `hls_results/cheng_naip_replication/`. Use the `deadtree` conda env:
  ```sh
  ~/anaconda3/envs/deadtree/bin/python proto_cheng_naip_inference.py <outdir> <naip.tif> [...] --device mps
  ```
  It was built with `conda create -n deadtree -c conda-forge python=3.10
  rasterio scikit-image scipy pandas pyproj tqdm click pyyaml pytorch
  torchvision` plus `pip install segmentation-models-pytorch==0.3.4`.
  - Install PyTorch from conda-forge, not pip: the pip wheel's bundled
    libomp crashes alongside conda-forge's ("OMP: Error #15").
  - The env has no matplotlib; plot with the `ecopro` env.

- **Tree-level validation of the Cheng model.** Against the NEON hand
  labels and the SEKI field data (NAIP chips in `naip_chips/`, results in
  `hls_results/cheng_vs_{hs_labels,seki}/`). Earth Engine project:
  `ecopro-509818`.

## Environment

- Conda env `ecopro` was created from `environment.yml`; `pypdf` was added
  with pip. Run scripts from `src/`, e.g.
  `~/anaconda3/envs/ecopro/bin/python <script> ...`.
- HLS reads use Earthdata credentials from `~/.netrc`
  (`urs.earthdata.nasa.gov`).

## Pipeline commands

Run from `src/`. `CFG=../config/hls_aois.yml` and `E=/Volumes/Earth04/ecopro`.

```sh
python fetch_hls_aoi.py $CFG $E/hls -j 4                # resumable; skips done/cloudy granules
python fetch_landcover_aoi.py $CFG $E/landcover
python ads_labels_aoi.py $CFG $E/hls_labels
python hls_annual_composites.py $CFG $E/hls $E/hls_composites            # Jul–Sep medians
python hls_flight_composites.py $CFG $E/hls $E/hls_composites -w 20      # overflight-matched
python proto_hls_mortality_signal.py $CFG $E/hls_results/all
python proto_hls_mortality_signal.py $CFG $E/hls_results/all_flight --composite-suffix _flight_w20.nc
python proto_hls_mortality_model.py  $CFG $E/hls_results/all
python proto_hls_mortality_model.py  $CFG $E/hls_results/all_flight --composite-suffix _flight_w20.nc
python fetch_naip_aoi.py $CFG $E/naip -j 12            # resumable; -y YEAR / -a AOI to restrict
python proto_ads_repeat_flights.py      # repeat-flight / duplicate-overlap check (prints tables)
```

To restrict a run to some AOIs: the fetch, composite and signal scripts take
`-a <aoi>`. The model script reads every AOI in the config, so pass a copy of
the config with the others removed. Running a single AOI in the model script
skips leave-one-AOI-out.

## Data (all under `/Volumes/Earth04/ecopro/`)

| Path | Contents |
|---|---|
| `usfs_ids/CONUS_Region5_AllYears.gdb`, `IDS_FlatFiles_Readme.pdf` | Raw USFS R5 IDS download (damage areas, points, surveyed areas; 1999–2025) |
| `usfs_ids/R5_*_2012plus.gpkg` | Sierra extracts used by the scripts |
| `fire/mtbs_perimeter_data/` | MTBS perimeters (to 2025) |
| `landcover/<aoi>_landcover.tif` | NLCD 2013 class + TCC on the AOI grid |
| `hls/<aoi>/*.tif` | Per-granule AOI windows (7 bands incl. Fmask). `*.cloudy` marks skipped granules. `_manifest.json` caches the CMR search. Logs: `hls/fetch*.log` |
| `hls_labels/<aoi>.nc`, `<aoi>_polygons.csv` | Per-year 30 m labels, severity, burned, flight DOY |
| `hls_composites/<aoi>_doy182-273.nc`, `<aoi>_flight_w20.nc` | Composites |
| `hls_results/{sierra_nf_only,sierra_nf_flight,all,all_flight}/` | Metrics CSVs, PDFs, PNGs, QC maps |
| `cheng2024/DeLfoRS_TreeMortality/` | Cheng et al. 2024 code release (no model weights) |
| `cheng2024/statewide_2020/` | Cheng et al. 2020 statewide dead-tree products (100–500 m GeoTIFFs, EPSG:5072, CC BY 4.0) |
| `cheng2024/cross_resolution_model/` | `BestModel.pth` and `config.json` from the [Cross-Resolution-Dead-Tree-Segmentation](https://github.com/YanCheng-go/Cross-Resolution-Dead-Tree-Segmentation) README's Drive link. ResNet-50 UNet (`unet_with_scalar`), RGB, 256 px patches, about 33M parameters. The checkpoint also holds the Adam optimizer state. **No license stated** |
| `usgs_seki_deadtree_validation/` | Das 2024 USGS release ([10.5066/P9GYXCPG](https://doi.org/10.5066/P9GYXCPG), CC0): field tree points for NAIP dead-tree validation in Sequoia & Kings Canyon NPs (2016/2019/2020), plus FGDC metadata |
| `hemming_schroeder2023/` | [Zenodo 7938442](https://doi.org/10.5281/zenodo.7938442), extracted (9 GB). Crown perimeters and 30 m mortality rasters 2013/2017/2018/2019/2021 for NEON SOAP and TEAK only (EPSG:32611). Does not overlap our AOIs |
| `naip/<aoi>/<year>/<item>.tif` (+ `.json`) | NAIP quarter-quads (4-band COG as published) intersecting each AOI, 2012–2022 biennial, from the Planetary Computer. Paused: `sierra_nf` has all 6 years; `stanislaus` has 27 of 42 for 2022; the rest is missing. See `naip_fetch.md`. `_items.json` holds the STAC manifest |
| `wdts/<aoi>_traits.nc` | WDTS AVIRIS-Classic foliar-trait mosaics ([ORNL DAAC 2403](https://doi.org/10.3334/ORNLDAAC/2403)) on the AOI grid: 14 trait means/sd and QC fractions, one early-summer date per year 2013–2018, for `sierra_nf` and `neon_soap_teak` (Yosemite flight box). From `fetch_wdts_traits.py` |
| `master/MASTER_WDTS_SeptOct_2020/` | MASTER L1B HDF4 lines 01–04 of flight 2190600 (2020-10-15) over NEON SOAP/TEAK ([ORNL DAAC 1940](https://doi.org/10.3334/ORNLDAAC/1940)) |
| `wdts/neon_soap_teak_cwc.nc` | Canopy water indicators (EWT at 980/1200 nm, NDWI, 1200 nm band depth) from the WDTS 15 m reflectance ([ORNL 2391](https://doi.org/10.3334/ORNLDAAC/2391)), 30 m, 2013–2018. From `fetch_wdts_cwc.py`. `wdts/aux/prospect_d_spectra.txt` holds the PROSPECT-D absorption coefficients |
| `stand_structure/` | Small samples of TreeMap 2014/2016/2020, LANDFIRE 2014/2016/2022, GLAD height and the Meta CHM, used to document candidate stand-structure layers (`docs/stand_structure_datasets.md`) |
| `hls_results/predisposition/` | Pixel-level predisposition results at NEON (cell tables, CV, univariate) |
| `landsat_composites/<aoi>_{c2,c2l7,c2oli}_doy{182-273,145-190}.nc` | Landsat C2 L2 summer and June composites, 2008–2025, on the AOI grids (Earth Engine). `c2l7` = Landsat 7 only, `c2oli` = Landsat 8/9 only. From `fetch_landsat_c2_ee.py` |
| `env/<aoi>_env.nc` | BCMv8 climate (2008–2024), local SRTM terrain indices, GLAD/TCC/LANDFIRE structure, NEON lidar tree summaries, NLCD forest, and per-year fire (MTBS, FRAP, prescribed) and FACTS harvest/salvage masks. From `aoi_env_layers.py` |
| `fire/disturbance/{frap_fires,rx_fires,facts_harvest}.gpkg` | CAL FIRE FRAP and prescribed-fire perimeters and USFS FACTS timber harvests over the AOIs. From `fetch_disturbance_aois.py` |
| `aviris_locator/` | AVIRIS Flight Line Locator tables and GeoJSON ([ORNL DAAC 2140](https://doi.org/10.3334/ORNLDAAC/2140)) |
| `wdts/sierra_nf_cwc.nc` | Canopy water indicators for `sierra_nf` (as for NEON), now with `nadir_dist` |
| `hls_results/airborne_coverage/` | AVIRIS-C/NG/3/5 flight-line coverage of the AOIs, 2013–2025 (`query_airborne_coverage.py`) |
| `hls_results/response{,_sierra}/`, `response_traits{,_sierra}/`, `trait_dynamics{,_sierra}/`, `response_transfer/` | Drought response metrics, nested trait models, trait dynamics and cross-drought transfer (`docs/drought_response.md`). `*_mtbs_only/` are earlier runs with the MTBS-only fire mask |
| `wdts/stanislaus_traits.nc` | Tahoe-box trait mosaics for `stanislaus` (UTM 11 mosaics warped to the UTM 10 grid, nearest neighbour) |
| `lidar/aso/`, `lidar/lvis2008/`, `lidar/<aoi>_lidar.nc` | ASO 2014–17 structure composite (Ferraz et al. 2020, [Zenodo 3964981](https://doi.org/10.5281/zenodo.3964981): CHM, 10 m metrics, tile boundaries) and LVIS 2008 LDS 1.03 files; structure on the AOI grids from `fetch_lidar_structure.py` (`docs/lidar_coverage.md`) |
| `hls_results/lidar_coverage/`, `lidar_check/`, `recovery_ablation/`, `trait_stability/`, `response{,_traits}_stanislaus/` | Lidar coverage, trait-vs-lidar checks and structure validation, recovery ablations, 2013–2018 trait stability, Tahoe-box replication (`docs/drought_response.md` §5–8) |

Code: `src/fetch_hls_aoi.py`, `fetch_landcover_aoi.py`, `ads_labels_aoi.py`,
`hls_annual_composites.py`, `hls_flight_composites.py`,
`proto_hls_mortality_signal.py`, `proto_hls_mortality_model.py`,
`proto_ads_repeat_flights.py`, `fetch_naip_aoi.py`, `hls_naip_composites.py`, `proto_hls_vs_cheng.py`, `proto_hls_vs_hs.py`, `proto_cheng_naip_inference.py`, `fetch_naip_chips.py`, `fetch_naip_chips_ee.py`, `proto_cheng_vs_hs_labels.py`, `proto_cheng_vs_seki.py`, `fetch_wdts_traits.py`, `proto_hls_vs_wdts.py`, `proto_master_quicklook.py`, `proto_predisposition_neon.py`, `fetch_wdts_cwc.py`, `fetch_landsat_c2_ee.py`, `fetch_disturbance_aois.py`, `aoi_env_layers.py`, `query_airborne_coverage.py`, `response_common.py`, `proto_response_metrics.py`, `proto_trait_dynamics.py`, `proto_response_traits.py`, `proto_response_transfer.py`, `query_lidar_coverage.py`, `fetch_lidar_structure.py`, `proto_lidar_validation.py`, `proto_trait_stability.py`. Config: `config/hls_aois.yml`.

## Gotchas

- **LP DAAC throttling.** At `-j 6`, about 36% of windowed reads failed with
  "not recognized as being in a supported file format". GDAL got a non-TIFF
  body. `read_window` now retries with backoff, and `-j 4` has had 0 errors.
  Don't raise the parallelism.
- **URS lockout.** Authentication happens once, serially, before the parallel
  reads. Don't parallelize logins. The cookie jar is `hls/.cookies.txt`.
- **UTM zones.** Only tiles in each AOI's UTM zone are used, so no resampling
  is needed. `stanislaus` is in zone 10 (tile 10SGH), which drops the zone-11
  duplicates.
- **ADS quirks the scripts handle:**
  - Legacy severity is TPA and DMSM severity is percent affected.
  - Legacy `CREATED_DATE` is a digitization date and is ignored.
  - Duplicate "pancake" geometries: the label script uses the most severe
    polygon for `poly_id` and sums TPA/percent across overlaps.
  - 2020 features come from remote sensing, not ADS.
- **No flight dates** for 2016 (all AOIs), 2015 (`sierra_nf`) or 2020. These
  years are absent from the flight-matched composites.
- **Figures.** `.gitignore` ignores `*.png`, `*.pdf` and `figs`. Result
  figures stay on Earth04. If some should go in the repo, add a `.gitignore`
  exception, e.g. `!docs/figures/*.png`.
- **Cheng model compute.** The repo was tested on RTX 3090 (24 GB) and H100
  with CUDA 12 (`env_cuda12.yml`: pytorch-cuda 12.1, smp 0.3.4 dev).
  - Inference on 256 px RGB patches needs only a few GB of VRAM. Any 8 GB+
    NVIDIA GPU works; an A100 (40 or 80 GB) is ample.
  - Retraining at the config's batch size of 98 needs a large GPU (A100/H100
    class).
  - NAIP is 4-band RGBN, but the model takes RGB only, so drop band 4.
- **NAIP download speed.** The Planetary Computer throttled to about 0.6 MB/s
  total after about 80 GB, so the download is paused. See
  [`naip_fetch.md`](naip_fetch.md).
- **Uncommitted work.** Nothing from this session is committed yet: all new
  `src/`, `config/hls_aois.yml` and `docs/` files.

## Open decisions for the next person

1. Contact Yan Cheng for trained weights or the 2020 statewide dead-tree map,
   or commit to the photo-interpretation fallback. See Next steps in the
   prototype doc.
2. Decide whether an HLS mortality product is validated against NAIP
   (30 m) or ADS (at 1 km or more). Pixel-level ADS agreement is capped by
   ADS positional accuracy and polygon generalization.
3. Whether to extend AOIs or years once a 30 m reference exists.
