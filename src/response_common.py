"""
Shared helpers for the drought-response experiments (response metrics,
nested trait models, trait dynamics): AOI grids, cell tables at several
scales, spatial-block CV with gradient boosting, paired block-bootstrap
confidence intervals on R² differences, and residual semivariograms.
"""
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

from util import load_config
from fetch_hls_aoi import aoi_grid
from proto_hls_vs_hs import block_mean

E = Path('/Volumes/Earth04/ecopro')
CONFIG = Path(__file__).resolve().parent.parent / 'config/hls_aois.yml'
RES = 30
N_SPLITS = 5
MIN_VALID = 0.7  # fraction of 30 m pixels valid for a coarser cell


def aoi_info(name, configfile=CONFIG):
    cfg = load_config(configfile)
    aoi = cfg['aois'][name]
    transform, shape = aoi_grid(aoi, cfg['size_m'], cfg['resolution'])
    return transform, shape, aoi['epsg']


def open_env(name):
    return xr.open_dataset(E / 'env' / f'{name}_env.nc')


HARVEST_FROM = 2005


def undisturbed(env, last_year, first_year=2000, keep_salvage=False):
    """NLCD 2013 forest with no fire (MTBS, CAL FIRE FRAP or prescribed
    burn) from first_year and no FACTS harvest from HARVEST_FROM, through
    last_year. keep_salvage keeps cells whose only harvests were salvage or
    sanitation cuts (usually of drought-killed trees)."""
    def years(v, y0):
        return env[v].sel(burn_year=slice(y0, last_year)).values > 0
    fire = (years('burned', first_year) | years('frap', first_year)
            | years('rx', first_year)).any(0)
    cut = years('harvest', HARVEST_FROM)
    if keep_salvage:
        cut &= ~years('salvage', HARVEST_FROM)
    return (env.forest.values == 1) & ~fire & ~cut.any(0)


def cell_table(layers, valid, k, min_valid=MIN_VALID,
               block_m=(1000, 5000)):
    """Cells of k x k pixels: block means of each 2D layer over valid pixels.

    Cells need >= min_valid of their pixels valid. Returns one row per cell
    with cell row/col, the valid-pixel count and block ids for each size in
    block_m (columns block<size>)."""
    _, n = block_mean(np.zeros(valid.shape, np.float32), valid, k)
    keep = n >= min_valid * k * k
    df = pd.DataFrame({name: block_mean(a, valid, k)[0][keep]
                       for name, a in layers.items()})
    rr, cc = np.nonzero(keep)
    df['cell_row'], df['cell_col'], df['n_px'] = rr, cc, n[keep]
    for b in block_m:
        bs = max(1, int(round(b / (RES * k))))
        df[f'block{b}'] = (rr // bs) * 10000 + (cc // bs)
    return df


def hgb(seed=0):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         random_state=seed)


def wr2(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w)
    y, p = np.asarray(y), np.asarray(p)
    mu = np.average(y, weights=w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - mu) ** 2)


def oof_predict(df, cols, target, blocks, weight=None, seed=0,
                n_splits=N_SPLITS):
    """Out-of-fold predictions with GroupKFold over spatial blocks"""
    p = np.full(len(df), np.nan)
    X, y = df[cols].values, df[target].values
    w = None if weight is None else df[weight].values
    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=df[blocks]):
        m = hgb(seed)
        m.fit(X[tr], y[tr], sample_weight=None if w is None else w[tr])
        p[te] = m.predict(X[te])
    return p


def block_sums(y, preds, w, blocks):
    """Per-block sufficient statistics for weighted R² of several
    prediction vectors"""
    codes, inv = np.unique(blocks, return_inverse=True)
    nb = len(codes)
    s = {'w': np.bincount(inv, w, nb),
         'wy': np.bincount(inv, w * y, nb),
         'wy2': np.bincount(inv, w * y * y, nb)}
    for k, p in preds.items():
        s[k] = np.bincount(inv, w * (y - p) ** 2, nb)
    return s, nb


def bootstrap_r2(y, preds, blocks, w=None, n_boot=1000, seed=0,
                 pairs=()):
    """Block-bootstrap R² for each prediction and R² differences.

    preds: {name: oof predictions}; pairs: [(a, b)] giving R²(b) - R²(a).
    Blocks are resampled with replacement. Returns {name or 'b-a':
    (estimate, lo95, hi95)}."""
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    s, nb = block_sums(y, preds, w, blocks)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nb, size=(n_boot, nb))
    m = np.stack([np.bincount(d, minlength=nb) for d in draws]).astype(float)

    def r2(mult, key):
        sw, swy, swy2 = mult @ s['w'], mult @ s['wy'], mult @ s['wy2']
        sst = swy2 - swy ** 2 / sw
        return 1 - (mult @ s[key]) / sst

    one = np.ones((1, nb))
    out = {}
    for k in preds:
        b = r2(m, k)
        out[k] = (r2(one, k)[0], *np.percentile(b, [2.5, 97.5]))
    for a, bname in pairs:
        d = r2(m, bname) - r2(m, a)
        out[f'{bname}-{a}'] = (r2(one, bname)[0] - r2(one, a)[0],
                               *np.percentile(d, [2.5, 97.5]))
    return out


def ladder(df, target, fsets, blocks, weight=None, seed=0, n_boot=1000,
           pairs=None, extra_blocks=()):
    """Fit each feature set, then R² with CIs and the R² gain of each step.

    fsets: ordered {name: [cols]}. pairs defaults to consecutive steps.
    extra_blocks: further block columns to bootstrap over (e.g. 5 km).
    Returns (rows, preds)."""
    d = df[df[target].notna()]
    preds = {name: oof_predict(d, cols, target, blocks, weight, seed)
             for name, cols in fsets.items()}
    names = list(fsets)
    pairs = pairs or list(zip(names[:-1], names[1:]))
    w = None if weight is None else d[weight].values
    rows = []
    for bcol in (blocks,) + tuple(extra_blocks):
        bs = bootstrap_r2(d[target].values, preds, d[bcol].values, w,
                          n_boot, seed, pairs)
        for name in names:
            est, lo, hi = bs[name]
            rows.append(dict(target=target, features=name, compare='',
                             boot_blocks=bcol, n=len(d), r2=est,
                             lo=lo, hi=hi,
                             rho=spearmanr(d[target], preds[name])[0]))
        for a, b in pairs:
            est, lo, hi = bs[f'{b}-{a}']
            rows.append(dict(target=target, features=b, compare=a,
                             boot_blocks=bcol, n=len(d), r2=est, lo=lo,
                             hi=hi, rho=np.nan))
    return rows, pd.DataFrame(preds, index=d.index)


def crossfit_residuals(df, target, cols, blocks, seed=0):
    """target minus its out-of-fold prediction from cols (no leakage)"""
    ok = df[target].notna()
    r = pd.Series(np.nan, index=df.index)
    d = df[ok]
    r[ok] = d[target].values - oof_predict(d, cols, target, blocks,
                                           seed=seed)
    return r


def within_strata_rho(d, cols, target, strata, min_n=100):
    """Spearman ρ pooled and averaged within strata (weighted by size)"""
    rows = []
    for c in cols:
        ok = d[c].notna() & d[target].notna()
        if ok.sum() < 200:
            continue
        within, wts = [], []
        for _, g in d[ok].groupby(strata[ok]):
            if len(g) >= min_n and g[target].std() > 0 and g[c].std() > 0:
                within.append(spearmanr(g[c], g[target])[0])
                wts.append(len(g))
        rows.append(dict(feature=c, target=target, n=int(ok.sum()),
                         rho=spearmanr(d[c][ok], d[target][ok])[0],
                         rho_within=(np.average(within, weights=wts)
                                     if within else np.nan),
                         n_strata=len(within)))
    return rows


def semivariogram(x, y, v, max_dist=5000, n_bins=25, n_sample=4000,
                  seed=0):
    """Empirical semivariogram of v at points (x, y) in meters"""
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isfinite(v))
    idx = rng.choice(ok, min(n_sample, len(ok)), replace=False)
    x, y, v = x[idx], y[idx], v[idx]
    edges = np.linspace(0, max_dist, n_bins + 1)
    num, cnt = np.zeros(n_bins), np.zeros(n_bins)
    for i in range(len(v) - 1):
        d = np.hypot(x[i + 1:] - x[i], y[i + 1:] - y[i])
        g = 0.5 * (v[i + 1:] - v[i]) ** 2
        b = np.digitize(d, edges) - 1
        m = (b >= 0) & (b < n_bins)
        num += np.bincount(b[m], g[m], n_bins)
        cnt += np.bincount(b[m], minlength=n_bins)
    with np.errstate(invalid='ignore'):
        gamma = num / cnt
    return 0.5 * (edges[1:] + edges[:-1]), gamma, cnt, np.var(v)


def fit_exponential(h, gamma, var):
    """Nugget, partial sill and practical range (3a) of an exponential
    model; returns NaNs if the fit fails"""
    ok = np.isfinite(gamma)
    g = gamma / var  # fit in units of the variance (metrics span 1e-5-1)

    def model(h, n, s, a):
        return n + s * (1 - np.exp(-h / a))
    try:
        (n, s, a), _ = curve_fit(model, h[ok], g[ok], p0=(0.3, 0.7, 500),
                                 bounds=([0, 0, 1], [2, 2, 1e5]))
    except RuntimeError:
        return dict(nugget=np.nan, psill=np.nan, range_m=np.nan)
    return dict(nugget=n * var, psill=s * var, range_m=3 * a)
