#!/usr/bin/env python
"""
Run the Cheng et al. cross-resolution dead-tree segmentation model on NAIP
quarter-quads and aggregate the detected dead crowns to the 100 m grid of
the Cheng et al. (2024) statewide 2020 maps, to test whether the published
maps can be approximately reproduced.

Model: https://github.com/YanCheng-go/Cross-Resolution-Dead-Tree-Segmentation
(`BestModel.pth` + `config.json` from the README's pretrained-model link).
This is the successor of the model used for the 2024 statewide maps, whose
weights are not public, so agreement is expected to be approximate.

The network and inference follow the repository code
(src/modelling/models/unet_with_scalar.py, train/ordinal_watershed.py
AttachConv, predict/ordinal_watershed.py):
  - input: RGB (NAIP bands 1-3) / 255, ImageNet mean/std normalization;
    pixels with all bands 0 are nodata
  - scalar input: pixel size in meters (0.6 for NAIP 2020)
  - 256 px patches, 10 px border discarded (stride 236)
  - energy level = number of leading positive ordinal outputs (0-5)
  - instances = watershed(-energy, connectivity=2, mask=energy > 0)

Runs on Apple MPS if available, else CPU.

Outputs per quad in <outputdir>: <quad>_energy.tif (uint8), <quad>_crowns.csv
(centroid, area per instance), and a combined cells_100m.csv comparing
per-cell dead-crown density and dead-canopy fraction with the Cheng maps.
"""
import os
import json
import time
import click
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from tqdm import tqdm
from pathlib import Path
from pyproj import Transformer
from scipy import ndimage
from rasterio.windows import Window
from skimage.segmentation import watershed

PATCH = 256
EDGE = 10
STRIDE = PATCH - 2 * EDGE
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CHENG = {
    'pct_dead': 'NAIP_2020_CA_owatershed_v8_area_100m_proj_max_clip_mask_percentperha.tif',
    'density': 'NAIP_2020_CA_owatershed_v8_density_100m_proj_clip_mask_biascorrected.tif',
}


class AttachConv(nn.Module):
    """Sobel + ordinal energy head (train/ordinal_watershed.py)"""

    def __init__(self, input_dim, n_energy_bins, ordinal_connect):
        super().__init__()
        self.sobel_layer = nn.Conv2d(input_dim, 2, 1, bias=False)
        self.energy_layer = nn.Conv2d(input_dim, n_energy_bins, 1)
        self.ordinal_connect = ordinal_connect

    def forward(self, x):
        energy = self.energy_layer(x)
        if self.ordinal_connect:
            energy = energy.cumsum(1)
        return self.sobel_layer(x), energy


class WatershedUNetWithScalar(nn.Module):
    """UNetWithScalar (src/modelling/models/unet_with_scalar.py) with the
    AttachConv head; encoder weights come from the checkpoint"""

    def __init__(self, in_channels, backbone, n_energy_bins,
                 ordinal_connect, scalar_counts=1):
        super().__init__()
        base = smp.Unet(backbone, encoder_weights=None,
                        in_channels=in_channels, classes=1)
        self.encoder, self.decoder = base.encoder, base.decoder
        f = 128
        self.scalar_ = nn.Sequential(
            nn.Conv2d(scalar_counts, f, 1), nn.BatchNorm2d(f), nn.ReLU(),
            nn.Conv2d(f, f, 1), nn.BatchNorm2d(f), nn.ReLU(),
        )
        self.scalar_in = nn.Conv2d(f, in_channels, 1)
        self.scalar_bottle = nn.Conv2d(f, self.encoder.out_channels[-1], 1)
        self.scalar_out = nn.Conv2d(f, 16, 1)
        self.last_conv = AttachConv(16, n_energy_bins, ordinal_connect)

    def forward(self, x, scalar):
        s = self.scalar_(scalar.reshape(-1, 1, 1, 1).to(x.device))
        x = self.scalar_in(s) + x
        feats = self.encoder(x)
        feats[-1] = self.scalar_bottle(s) + feats[-1]
        out = self.scalar_out(s) + self.decoder(*feats)
        return self.last_conv(out)


def load_model(model_dir, device):
    cfg = json.load(open(model_dir / 'config.json'))
    assert cfg['model_type'] == 'unet_with_scalar'
    model = WatershedUNetWithScalar(cfg['in_channels'], cfg['backbone'],
                                    cfg['n_energy_bins'],
                                    cfg['ordinal_connect'])
    st = torch.load(model_dir / 'BestModel.pth', map_location='cpu',
                    weights_only=False)
    model.load_state_dict(st['net_params'], strict=True)
    return model.eval().to(device), cfg


@torch.no_grad()
def predict_energy(model, path, device, batch_size, max_value):
    """Energy level (0-5) for every pixel of a NAIP image"""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with rasterio.open(path) as ds:
        H, W = ds.shape
        res = round(ds.transform.a, 2)
        energy = np.zeros((H, W), dtype=np.uint8)
        origins = [(r, c) for r in range(-EDGE, H - EDGE, STRIDE)
                   for c in range(-EDGE, W - EDGE, STRIDE)]
        scalar = torch.tensor([res], dtype=torch.float32, device=device)
        for i in tqdm(range(0, len(origins), batch_size), leave=False,
                      desc=path.stem):
            batch = origins[i:i + batch_size]
            x = np.stack([
                ds.read([1, 2, 3], window=Window(c, r, PATCH, PATCH),
                        boundless=True, fill_value=0)
                for r, c in batch
            ]).astype(np.float32)
            nodata = torch.from_numpy((x == 0).all(1, keepdims=True))
            xt = (torch.from_numpy(x) / max_value - mean) / std
            xt = torch.where(nodata, torch.zeros_like(xt), xt)
            _, e = model(xt.to(device), scalar)
            e = e.cpu()
            e = e.masked_fill(nodata, 0)
            lev = (e > 0).float().cumprod(1).sum(1).numpy().astype(np.uint8)
            for (r, c), lv in zip(batch, lev):
                r0, c0 = r + EDGE, c + EDGE
                h = min(STRIDE, H - r0)
                w = min(STRIDE, W - c0)
                energy[r0:r0 + h, c0:c0 + w] = lv[EDGE:EDGE + h, EDGE:EDGE + w]
        profile = ds.profile
    return energy, profile, res


def crowns(energy, transform, res):
    mask = energy > 0
    labels = watershed(-energy.astype(np.int16), connectivity=2, mask=mask)
    n = labels.max()
    if n == 0:
        return pd.DataFrame(columns=['x', 'y', 'area_m2', 'max_energy'])
    idx = np.arange(1, n + 1)
    area = ndimage.sum(mask, labels, idx) * res * res
    cy, cx = np.array(ndimage.center_of_mass(mask, labels, idx)).T
    emax = ndimage.maximum(energy, labels, idx)
    x, y = transform * (cx + 0.5, cy + 0.5)
    return pd.DataFrame({'x': x, 'y': y, 'area_m2': area, 'max_energy': emax})


def cell_table(quad, crowns_df, energy, profile, cheng_dir):
    """Per Cheng 100 m cell: model dead-crown count/area vs Cheng values,
    for cells fully covered by valid NAIP pixels"""
    crs = profile['crs']
    tr = Transformer.from_crs(crs, 'EPSG:5072', always_xy=True)
    with rasterio.open(cheng_dir / CHENG['pct_dead']) as ds:
        inv = ~ds.transform
        t5072 = ds.transform
    # Coverage: sample valid NAIP pixels every 5 px (3 m) and count per cell
    T = profile['transform']
    H, W = energy.shape
    rr, cc = np.mgrid[2:H:5, 2:W:5]
    with rasterio.open(profile['path']) as src:
        ok = (src.read(1)[2:H:5, 2:W:5] != 0).ravel()
    xs, ys = T * (cc + 0.5, rr + 0.5)
    ex, ny = tr.transform(xs.ravel()[ok], ys.ravel()[ok])
    col, row = inv * (np.asarray(ex), np.asarray(ny))
    key = np.floor(row).astype(np.int64) * 100000 + np.floor(col).astype(np.int64)
    cov = pd.Series(key).value_counts()
    expected = (100 / (5 * T.a)) ** 2
    full = cov[cov >= 0.97 * expected].index

    ex, ny = tr.transform(crowns_df.x.values, crowns_df.y.values)
    col, row = inv * (np.asarray(ex), np.asarray(ny))
    crowns_df = crowns_df.assign(
        key=np.floor(row).astype(np.int64) * 100000
        + np.floor(col).astype(np.int64))
    g = crowns_df.groupby('key').agg(n=('area_m2', 'size'),
                                     area=('area_m2', 'sum'))
    df = pd.DataFrame(index=full)
    df['model_density'] = g.n.reindex(full).fillna(0).values
    df['model_dead_frac'] = (g.area.reindex(full).fillna(0) / 1e4).values
    df['row'], df['col'] = df.index // 100000, df.index % 100000
    for k, f in CHENG.items():
        with rasterio.open(cheng_dir / f) as ds:
            r0, r1 = df.row.min(), df.row.max() + 1
            c0, c1 = df.col.min(), df.col.max() + 1
            a = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(float)
            a[(a < -1000) | ~np.isfinite(a)] = np.nan
            df[f'cheng_{k}'] = a[df.row - r0, df.col - c0]
    df['quad'] = quad
    return df.reset_index(drop=True)


@click.command()
@click.argument('outputdir', type=click.Path(path_type=Path))
@click.argument('quads', nargs=-1, type=click.Path(path_type=Path,
                                                   exists=True))
@click.option('--no-cells', is_flag=True,
              help='Skip the Cheng 100 m cell comparison (e.g., small chips)')
@click.option('--by-year', is_flag=True,
              help='Write outputs to <outputdir>/<NAIP year>/; the year is '
                   'taken from the parent directory name of each input')
@click.option('--model-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/cheng2024/cross_resolution_model'))
@click.option('--cheng-dir', type=click.Path(path_type=Path), default=(
    '/Volumes/Earth04/ecopro/cheng2024/statewide_2020'))
@click.option('--device', default='auto',
              type=click.Choice(['auto', 'mps', 'cpu']))
@click.option('--batch-size', default=16, show_default=True)
@click.option('--crop', default=0, show_default=True,
              help='If > 0, only process a centered crop of this many px')
def main(outputdir, quads, no_cells, by_year, model_dir, cheng_dir, device,
         batch_size, crop):

    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    os.makedirs(outputdir, exist_ok=True)
    model, cfg = load_model(model_dir, device)
    click.echo(f'model loaded on {device}; {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params')

    # Directories expand to the GeoTIFFs they contain
    files = []
    for q in quads:
        files += sorted(q.glob('*.tif')) if q.is_dir() else [q]
    tables = []
    for q in tqdm(files, desc='images', disable=len(files) < 20):
        odir = outputdir / q.parent.name if by_year else outputdir
        os.makedirs(odir, exist_ok=True)
        if (odir / f'{q.stem}_energy.tif').exists():
            continue
        src = q
        if crop:
            with rasterio.open(q) as ds:
                r0, c0 = (ds.height - crop) // 2, (ds.width - crop) // 2
                win = Window(c0, r0, crop, crop)
                prof = ds.profile.copy()
                prof.update(height=crop, width=crop,
                            transform=ds.window_transform(win))
                src = outputdir / f'{q.stem}_crop{crop}.tif'
                with rasterio.open(src, 'w', **prof) as dst:
                    dst.write(ds.read(window=win))
        t0 = time.time()
        energy, profile, res = predict_energy(model, src, device, batch_size,
                                              cfg['max_value'])
        t_pred = time.time() - t0
        cdf = crowns(energy, profile['transform'], res)
        t_ws = time.time() - t0 - t_pred
        prof = profile.copy()
        prof.update(count=1, dtype='uint8', compress='deflate', nodata=None,
                    photometric=None)
        with rasterio.open(odir / f'{q.stem}_energy.tif', 'w',
                           **prof) as dst:
            dst.write(energy, 1)
        cdf.to_csv(odir / f'{q.stem}_crowns.csv', index=False)
        if no_cells:
            continue
        profile['path'] = str(src)
        t = cell_table(q.stem, cdf, energy, profile, cheng_dir)
        tables.append(t)
        click.echo(f'[{q.stem}] {energy.shape} px: predict {t_pred:.0f}s, '
                   f'watershed {t_ws:.0f}s; {len(cdf)} crowns '
                   f'(median {cdf.area_m2.median() if len(cdf) else 0:.1f} m2); '
                   f'{len(t)} full 100 m cells')
    if tables:
        pd.concat(tables).to_csv(outputdir / 'cells_100m.csv', index=False)


if __name__ == '__main__':
    main()
