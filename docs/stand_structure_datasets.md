# Stand density and tree height datasets for the Sierra Nevada

This is a survey of wall-to-wall stand-structure layers that could serve as
predictors of the fraction of trees dying per 30 m pixel. It was researched
on 2026-09-27; no analysis has been done with these data yet.

- **VERIFIED** means the landing page, catalog entry or service was opened.
- **sampled** means a small subset was also downloaded and inspected. Samples
  are in `/Volumes/Earth04/ecopro/stand_structure/` (~383 MB).
- **Sample window:** 30 × 30 km in EPSG:5070 around lon −119.4, lat 37.4
  (Sierra NF / Shaver Lake), x −2046225 to −2016225, y 1831935 to 1861935.

**Key constraint.** The 2015–2017 die-off needs *pre-drought* structure
(about 2012–2014). Most modern products, including LANDFIRE 2016+, TreeMap
2016+, the CFO/ETH/Meta/CTrees canopy-height maps, GEDI and most 3DEP lidar,
postdate the mortality and partly encode it. They are only safe for
validation or for later-period models such as the 2020–22 wave.

## Comparison

| Dataset | Density variables | Height / cover | Epoch | Resolution | Derivation | Access |
|---|---|---|---|---|---|---|
| **TreeMap 2014** (USFS RMRS, RDS-2019-0026) | **Yes, from the tree table:** TPA, BA, QMD, SDI computed from per-tree DIA, HT, TPA_UNADJ | per-tree HT | circa 2014 target layers, but **513 of 2,636 CA plots are INVYR 2015–16** (checked) | 30 m, CONUS | Random-forest imputation of FIA plots onto LF 2014 | 4.8 GB zip of a plot-ID raster plus `Tree_table_CONUS.txt` |
| **TreeMap 2016** (RDS-2021-0074) | **Yes:** TPA_LIVE, TPA_DEAD, BALIVE, SDIPCT_RMRS, QMD_RMRS | STANDHT, CANOPYPCT (continuous) | circa 2016, includes 2015–16 disturbance | 30 m | same | Per-attribute COG in stored zips (windowed `/vsizip//vsicurl/` reads work); GEE `USFS/GTAC/TreeMap/v2016` |
| **TreeMap 2020 / 2022 / 2023** (RDS-2025-0031, -0032, RDS-2026-0038) | **Yes:** TPA_LIVE/DEAD, BALIVE, SDIsum, QMD | STANDHT, CANOPYPCT (**binned**: 5 and 9 distinct values in the window, checked) | post-drought | 30 m | same, onto LF 2020+ | Raster gateway; GEE `projects/gtac-data-publish/assets/TreeMap/Product_Version/2026-1` |
| **LEMMA GNN** (Oregon State) | **Yes:** BA, TPH, QMD, … | cover, height | **2012 imagery** (also 2017, 2021; annual 1985–2012/17) | 30 m, CA/OR/WA | GNN imputation with Landsat + climate | `lemmadownload.forestry.oregonstate.edu` was **down for maintenance** (2026-09-25); 2012 mosaic is 930 MB |
| **FHTET/NIDRM species BA and SDI** | **Yes:** BA, SDI per species and total | – | circa 2002 | 240 m | Models on FIA + Landsat 1985–2005 | IIPP ImageServer; Ag Data Commons (CC-BY-4.0) |
| **Sierra RRK / ACCEL (F3)** (USFS R5) | **Yes:** TPA, BA, SDI, QMD, large-tree TPA | cover | 2019, updated to 2021 | 300 m public | FIA + FVS + FastEmap | View-only map service; gateway "temporarily unavailable" |
| **FIA BIGMAP 2018** | stocking %, size class | height, age | 2014–2018 | 30 m | kNN imputation with Landsat 8 | Raster gateway; ImageServer |
| **LANDFIRE LF 2014 (1.4.0)** EVH, EVC | No | **classes** only | LF 2014; undisturbed areas come from the ~2001 base map | 30 m | Plot-trained decision trees + disturbance updates | **GEE only** (`LANDFIRE/Vegetation/EVH/v1_4_0`, `.../EVC/v1_4_0`); LF 2014 CH/CC/CBD via the Help Desk |
| **LANDFIRE LF 2016 Remap → LF 2025** EVH, EVC, CH, CC, CBD, CBH | No | EVH/EVC continuous (100 + m or %); CH/CC binned | post | 30 m | Remap used lidar | LFPS API; ArcGIS ImageServers |
| **GLAD Forest height 2000/05/10/15/20** (Potapov et al. 2022) | No | continuous height (≥ 3 m) | **2010, 2015** | 30 m, global | GEDI-calibrated Landsat model, hindcast | GEE `projects/glad/GLCLU2020/Forest_height_{year}` |
| **USFS/NLCD Tree Canopy Cover** | No | continuous cover + SE | **annual 1985–2025** | 30 m | Landsat/S2 + FIA photo plots | GEE `USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5`; MRLC. Already on disk for the AOIs (NLCD 2013 TCC in `landcover/`) |
| **Planet Forest Carbon Diligence** | No | height, cover | **annual from 2013** | 30 m | Lidar-trained deep learning | Commercial (Subscriptions API) |
| **California Forest Observatory** | No (ladder fuel, layers) | height (MAE ~2 m), cover | 2020 confirmed (2016–19 claimed) | 10 m, CA | U-Net on S1/S2 trained on lidar | Free account + `cfo` API, non-commercial |
| **ETH Global Canopy Height 2020** (Lang et al. 2023) | No | height + sd | 2020 | 10 m | GEDI + S2 CNN | GEE `users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1` |
| **Meta/WRI CHM v1/v2** (Tolan et al. 2024) | proxy only: `count5m` (segments > 5 m per cell) | 1 m CHM; downsampled avg/p95/cover | CA imagery 2018–20 (v2: 2015–20) | 0.6 m | ViT on Maxar, trained on ALS + GEDI | AWS `s3://dataforgood-fb-data/forests/` (CC-BY-4.0) |
| **CTrees CA height 2020** (Wagner et al. 2024) | No | 0.6 m CHM | 2020 | 0.6 m, CA | U-Net on NAIP 2020 | AWS `ctrees-tree-height-ca-2020` |
| **GEDI L2A/L2B/L4A/L4B** | No | RH metrics, cover, PAI, AGBD | 2019–2023 | 25 m footprints; 1 km grids | Spaceborne lidar | GEE `LARSE/GEDI/...`; ORNL DAAC |
| **USGS 3DEP lidar** | via tree segmentation | CHM | **≤ 2014: ~3% of the Sierra Nevada Conservancy area**; 2015–18: ~25%; ≥ 2019: ~52% | 4–40 pts/m² | ALS | AWS EPT (`usgs-lidar-public`), TNM |
| **Other pre-drought lidar** | via segmentation | CHM | NEON SOAP/TEAK/SJER 2013; NCALM N. Sierra 2012 (437 km²); Tahoe Basin 2010; **NASA CMS CA 2005–2014** (53 areas, per-tree crowns; doi:10.3334/ORNLDAAC/1537) | 1 m-ish | ALS | NEON API; OpenTopography; ORNL DAAC |
| **CALVEG (R5 EVeg)** | DBH size classes, 10% cover classes | cover classes | imagery 1995–2016 by zone | polygons, 1 ha MMU | Landsat classification | R5 GIS / ArcGIS Hub |
| *Local:* Hemming-Schroeder 2023 | tree counts | 2013 height per tree | 2013 | trees, SOAP/TEAK only | NEON ALS | `hemming_schroeder2023/` |

## Notes

**TreeMap** (VERIFIED: https://research.fs.usda.gov/firelab/products/dataandtools/treemap-tree-level-model-united-states-forests, https://data.fs.usda.gov/geodata/rastergateway/treemap/index.php, RDS catalog pages above).
- **Grid and nodata:** EPSG:5070, 30 m, origin (−2361585, 3177435). Float nodata is 3.4028235e+38; STANDHT nodata is 65535; CANOPYPCT nodata is 255.
- **Units:** BA ft²/ac; TPA per acre (live > 1″ DBH, dead ≥ 5″); STANDHT ft; QMD inches.
- **2016 metadata accuracy:** at 2,749 independent FIA plots, cover was within 10% at 60.9% of plots, height within 5 m at 73.0%, and forest type matched at 51.8%.
- **In the sample window:**
  - 2016 QMD_RMRS and SDIPCT_RMRS are valid on only ~13% of forested pixels; TPA and BA are complete.
  - Agreement between versions is weak: TPA_LIVE 2016 vs 2020 r = 0.20, BALIVE r = 0.58.
- **TreeMap 2014** ships as a plot-ID raster (`national_c2014_tree_list.tif`, 62,758 plots) plus the tree table (`tl_id, CN, INVYR, STATUSCD, SPCD, DIA, HT, CR, TPA_UNADJ`). You build TPA, BA, QMD and SDI yourself. Its metadata says plots were pulled from FIA in December 2012, but the table contains INVYR up to 2016. Flag or re-impute pixels assigned to 2015–16 plots before using it as a pre-drought predictor.

**LANDFIRE** (VERIFIED: https://landfire.gov/vegetation/evh, https://www.landfire.gov/data/comparison-table, https://lfps.usgs.gov/arcgis/rest/services).
- **Versions** (with the disturbance years they cover): LF 2001, 2008, 2010, 2012 (2011–12), 2014 (2013–14), 2016 Remap (2015–16), 2020 (2017–20), 2022, 2023, 2024, 2025.
- **LF 2014:** EVC uses classes 101–109 (10% cover bins). In the window, EVH is dominated by codes 110 and 111. These are probably the 10–25 m and 25–50 m forest height classes, but confirm with the LF 1.4.0 code table before use.
- **LF 2016 onward:** EVH/EVC are continuous (tree = 100 + value in m or %).
- **LF 2022:** shows large blocks of identical values in disturbed areas (drought, Creek Fire).
- **Epoch:** LF 1.x vegetation in undisturbed areas can be ~13 years stale.

**LEMMA GNN** (VERIFIED: https://lemma.forestry.oregonstate.edu/data/structure-maps). This is the precedent: Young et al. 2017 (doi:10.1111/ele.12711) used GNN 30 m basal area as the competition predictor and found GNN trees-per-hectare much less reliable than basal area. The download site was under maintenance when checked.

**GLAD forest height** (VERIFIED: https://glad.umd.edu/dataset/GLCLUC2020; GEE listing confirms 2005, 2010 and 2015). In the window, woody ≥ 3 m covers ~86%, the median height is 18 m (2010, 2015), and the maximum is 36–38 m. It saturates in tall mixed conifer.

**Pre-drought lidar.** 3DEP boundaries (https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson) intersected with the Sierra Nevada Conservancy boundary give only CA_CalaverasTuolumne_2011 and CA_PlacerCo_2012 before 2015. NASA CMS "LiDAR-derived AGB for California 2005–2014" (Xu et al. 2018, https://daac.ornl.gov/CMS/guides/CMS_LiDAR_AGB_California.html) includes per-tree crowns for 53 survey areas and is the best pre-drought structure source for calibration or validation beyond NEON.

**Other entries** (VERIFIED at the URLs in the table):
- Planet FCD: https://docs.planet.com/data/planetary-variables/forest-carbon-diligence/
- CFO: https://forestobservatory.com/about.html
- Meta CHM: https://registry.opendata.aws/dataforgood-fb-forests/
- ETH: https://langnico.github.io/globalcanopyheight/
- CTrees: https://registry.opendata.aws/ctrees-california-vhr-tree-height/
- GEDI L4B: https://daac.ornl.gov/GEDI/guides/GEDI_L4B_Gridded_Biomass_V2_1.html
- Favrichon et al. 2024 (10 m CA height, data on request): https://doi.org/10.3389/frsen.2024.1459524
- RRK/ACCEL map service: https://apps.fs.usda.gov/fsgisx02/rest/services/r05/r05_SNV_ACCEL_01/MapServer
- FHTET BA: https://imagery.geoplatform.gov/iipp/rest/services/Vegetation/USFS_EDW_FHP_TreeSpeciesMetrics_BasalArea/ImageServer
- BIGMAP: https://data.fs.usda.gov/geodata/rastergateway/bigmap/index.php
- CALVEG: https://www.fs.usda.gov/r6/reo/monitoring/downloads/reports/CALVEG_paper.htm

## Recommendation for pre-drought predictors (Sierra-wide)

1. **Density:**
   - **TreeMap 2014**, with live TPA, BA, QMD and SDI built from the tree table, after flagging 2015–16 plots.
   - **LEMMA GNN 2012 basal area** as the second source once its site is back.
   - FHTET circa-2002 BA/SDI (240 m) as a coarse fallback.
2. **Height:** LF 2014 EVH classes plus **GLAD Forest_height_2010/2015** (continuous, saturates around 30–35 m). Calibrate or validate with the NEON 2013, CMS 2005–2014, NCALM 2012 and Tahoe 2010 lidar. Planet FCD 2013 is the best continuous height if a commercial license is possible.
3. **Cover:** annual USFS/NLCD TCC 2012–2013.
4. **Aggregate imputed layers (TreeMap, GNN) to ≥ 90 m before use.** Pixel-level imputation is noisy (plot-level cover within 10% at only 61% of plots; r = 0.2 between TreeMap versions for TPA).

## Local samples (`/Volumes/Earth04/ecopro/stand_structure/`)

- `treemap/`:
  - `TreeMap2016_{TPA_LIVE,TPA_DEAD,BALIVE,SDIPCT_RMRS,QMD_RMRS,STANDHT,CANOPYPCT}_sample.tif`
  - `TreeMap2020_{TPA_LIVE,TPA_DEAD,BALIVE,SDIsum,QMD,STANDHT,CANOPYPCT,TM_ID}_sample.tif`
  - metadata XML
  - `treemap2014_meta/Tree_table_CONUS.txt` (302 MB), `TL_CN_Lookup.txt`, VAT
- `landfire/`: `LF140_{EVH,EVC}_sample.tif` (GEE), `LF2016_{EVH,EVC,CH,CC,CBD}`, `LF2022_{EVH,EVC,CH}`
- `glad/GLAD_forest_height_{2010,2015,2020}_sample.tif`
- `meta_chm/`: a 3 km CA v6 CHM sample (0.6 m, uint16 cm), downsampled `{avg,p95,cover5m,count5m}` samples, tile index GeoJSONs
