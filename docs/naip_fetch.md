# Fetching NAIP for the HLS mortality AOIs (resume guide)

NAIP 4-band (RGB + NIR) quarter-quad images over the four AOIs in
`config/hls_aois.yml` are needed to build multi-year 30 m dead-tree
references, e.g. by running the Cheng et al. model (see
`hls_mortality_prototype.md`, Next steps). The download was **paused on
2026-09-26** because the Microsoft Planetary Computer throttled this machine
to about 0.04 MB/s per connection after about 80 GB. The local link still
measured 13 MB/s to other hosts. This guide covers how to finish it
elsewhere.

## What exists and what is missing

Images are stored as published (cloud-optimized GeoTIFF, UTM/NAD83, 0.6 m;
2012 is 1 m) at `/Volumes/Earth04/ecopro/naip/<aoi>/<year>/<item_id>.tif`,
each with its STAC item `<item_id>.json`. `<aoi>/_items.json` holds the full
search result, including footprints and acquisition dates.

| AOI | 2012 | 2014 | 2016 | 2018 | 2020 | 2022 |
|---|---|---|---|---|---|---|
| `sierra_nf` | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 | 30/30 |
| `stanislaus` | 0/42 | 0/42 | 0/42 | 0/42 | 0/42 | 27/42 |
| `lassen` | 0/35 | 0/35 | 0/35 | 0/35 | 0/35 | 0/35 |
| `neon_soap_teak` | 0/28 | 0/28 | 0/28 | 0/28 | 0/28 | 0/28 |

- Files are about 0.2 GB each for 2012/2014 and about 0.45–0.53 GB for
  2016–2022.
- **Remaining:** 603 files, about 230 GB. A priority subset of 2016, 2018
  and 2020 for the three unfinished AOIs is about 145 GB.
- 2016 and 2018 bracket the die-off peak, and 2020 is the Cheng et al.
  reference year.

## Resume with the existing script (Planetary Computer)

`src/fetch_naip_aoi.py` searches the Planetary Computer STAC API
(collection `naip`) for items intersecting each AOI's bounding box. It gets
a SAS token from `https://planetarycomputer.microsoft.com/api/sas/v1/token/naip`
and refreshes it automatically, then streams each image to a `.part` file
that is renamed once the byte count matches. No account or credentials are
needed.

**Behaviour:**
- **Completed files are skipped:** any existing `<item_id>.tif` is not
  re-downloaded.
- **Interrupted `.part` files restart from zero.**
- **Every run re-queries STAC and rewrites** `_items.json` and the per-item
  JSON.

1. **Environment.** Only a few packages are needed:

   ```sh
   git clone <this repo>; cd ecopro
   conda create -n naip -c conda-forge python=3.10 requests tqdm click pyyaml numpy pyproj rasterio
   conda activate naip          # or use the full `ecopro` env from environment.yml
   ```

   The script imports `aoi_grid`/`aoi_lonlat_bbox` from `src/fetch_hls_aoi.py`
   and `load_config` from `src/util.py`, so run it from `src/`.

2. **Dry run.** This checks the item counts and writes the manifests:

   ```sh
   cd src
   python fetch_naip_aoi.py ../config/hls_aois.yml /path/to/naip --dry-run
   ```

3. **Fetch.** Restrict by AOI (`-a`) and year (`-y`), both repeatable. Use
   modest parallelism:

   ```sh
   # priority subset first
   python fetch_naip_aoi.py ../config/hls_aois.yml /path/to/naip -j 4 \
       -a stanislaus -a lassen -a neon_soap_teak -y 2016 -y 2018 -y 2020
   # then the rest
   python fetch_naip_aoi.py ../config/hls_aois.yml /path/to/naip -j 4
   ```

   To avoid re-downloading what already exists, point `/path/to/naip` at
   `/Volumes/Earth04/ecopro/naip` if that disk is mounted. Otherwise copy
   the finished directories over, or restrict with `-a`/`-y` and merge
   later. Keep the `<aoi>/<year>/` layout.

4. **Monitor.** Progress goes to stderr (tqdm). A per-AOI summary line
   `[aoi] {'ok': n, 'skipped': n, 'error': n}, X GB` is printed at the end.
   Re-run the same command to retry any errors.

**Throttling.** At 12 parallel connections the Planetary Computer delivered
about 4 MB/s total at first, then about 0.6 MB/s after about 80 GB.
- Start with `-j 4` from a machine or network that hasn't been used for
  bulk downloads.
- Check a single-stream rate before starting a long run: signed URL plus
  `curl -r 0-52428799 -w '%{speed_download}'`. For reference, 0.5 MB/s
  per stream was the healthy rate here.

## Faster: windowed chips, preferably via Google Earth Engine

For analyses that only need NAIP around specific locations (validation
trees, labeled sites), fetch **chips** instead of whole quarter-quads. Two
scripts take the same arguments and write identical chips (same window and
native 0.6 m grid) to `<outdir>/<year>/<item_id>__r<row>_c<col>.tif`:

- `src/fetch_naip_chips.py`: windowed COG reads from the Planetary Computer.
  No account is needed. `--local-dir` reuses fully downloaded items.
- `src/fetch_naip_chips_ee.py`: Earth Engine `USDA/NAIP/DOQQ` via
  `ee.data.computePixels` on the high-volume endpoint.

```sh
python fetch_naip_chips_ee.py <aoi>/_items.json <outdir> --project ecopro-509818 \
    -t trees.shp:2016,2018 -j 8          # --inner 512 --margin 64 (px) by default
```

**Speed test** (2026-09-26, 60 chips of 640×640 px × 4 bands, 8 threads,
from this Mac):

| Source | Time | Rate |
|---|---|---|
| Earth Engine | 11 s | 5.6 chips/s (about 9 MB/s raw) |
| Planetary Computer | 151 s | 0.4 chips/s |

- **Pixel agreement.** 99.8% of pixel values are identical on average.
  Differences occur only where quarter-quads overlap: EE mosaics the
  collection, while PC reads one item.
- **The Planetary Computer bottleneck is transfer, not auth.** A Planetary
  Computer subscription key only raises SAS-token rate limits, not transfer
  speed, and NAIP is hosted in Azure West Europe (`naipeuwest`).

**Earth Engine setup:**
- Cloud project `ecopro-509818` (name "ecopro"), registered for
  noncommercial Earth Engine. Community tier: 150 EECU-hours per month, no
  billing account.
- Authenticate once per machine:
  `earthengine authenticate --auth_mode=localhost`. The default mode
  requires a working gcloud.
- The Google Cloud CLI (586.0.0) is installed via Homebrew
  (`brew upgrade --cask gcloud-cli`).

For **whole quarter-quads**, EE export to Google Drive/Cloud Storage or AWS
(below) are the options; the Planetary Computer is slow from here.

## Alternative source: AWS Open Data (requester pays)

NAIP is also in the AWS Open Data registry as a **requester-pays** S3
bucket. It needs AWS credentials, and you pay the egress, roughly
$0.09/GB, so about $20 for the remaining set.

**The bucket layout below has not been verified in this project.** Check it
with a listing before scripting:

```sh
aws s3 ls --request-payer requester s3://naip-analytic/ca/2020/
# expected pattern (to confirm): s3://naip-analytic/ca/<year>/<res>/rgbir_cog/<quad>/m_<...>.tif
```

The filenames match the Planetary Computer item IDs (drop the `ca_`
prefix, e.g. `ca_m_3812023_se_10_060_20220811` → `m_3812023_se_10_060_20220811.tif`).
So `_items.json` from a `--dry-run` gives the exact list of files needed per
AOI and year.

## Afterwards

- **Check file sizes.** Every `<item_id>.tif` should open with rasterio and
  have 4 bands, uint8, 0.6 m (2012: 1 m).
- **Copy to Earth04.** Place the files at `/Volumes/Earth04/ecopro/naip/<aoi>/<year>/`
  so paths in the other docs stay valid.
- **Using them with the Cheng model.** The model expects RGB only (drop
  band 4, NIR). See `hls_mortality_handoff.md` for the model files and GPU
  notes.
