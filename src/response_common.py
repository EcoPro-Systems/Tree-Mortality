"""
Shared helpers for the drought-response experiments (response metrics,
nested trait models, trait dynamics): AOI grids, cell tables at several
scales, spatial-block CV with gradient boosting, paired block-bootstrap
confidence intervals on R² differences, and residual semivariograms.
"""
import copy
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from util import load_config
from fetch_hls_aoi import aoi_grid
from proto_hls_vs_hs import block_mean

E = Path('/Volumes/Earth04/ecopro')
CONFIG = Path(__file__).resolve().parent.parent / 'config/hls_aois.yml'
RES = 30
N_SPLITS = 5
MIN_VALID = 0.7  # fraction of 30 m pixels valid for a coarser cell
# None: GroupKFold's deterministic, size-balanced assignment of blocks to
# folds (every experiment so far); an int shuffles blocks into folds with
# that seed (fold-assignment sensitivity checks)
FOLD_SEED = None
# Fixed fold assignments over which a contrast is averaged when the rule
# "mean over several fold splits" is used (bootstrap_r2_splits); distinct
# from the seeds of the sensitivity reruns (1-3)
FOLD_SEEDS = (11, 12, 13, 14, 15)
GLOBAL = 'global'  # fold_seed default: use the module's FOLD_SEED
# AOIs held out for the 2020-22 drought: their cycle-2 responses stay
# unexamined until models fixed on the other AOIs are applied to them
HELD_OUT = {'stanislaus', 'seki', 'yosemite_rest'}


def check_cycle2(aoi, allow=False):
    """Refuse to build 2020-22 responses for a held-out AOI"""
    if aoi in HELD_OUT and not allow:
        raise RuntimeError(f'{aoi} is held out for the 2020-22 drought; '
                           'its cycle-2 responses are not computed')


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


def inner_split(n, groups, frac=0.2, seed=0):
    """(train, validation) masks holding out about frac of the groups
    (spatial blocks); a random split of rows without groups"""
    rng = np.random.default_rng(seed)
    if groups is None:
        val = rng.random(n) < frac
    else:
        g = np.unique(groups)
        val = np.isin(groups, rng.choice(g, max(1, int(round(frac * len(g)))),
                                         replace=False))
    return ~val, val


class TunedHGB:
    """Gradient boosting tuned on an inner block split: leaves and
    minimum leaf size from a small grid, the number of iterations (up to
    max_iter at learning rate 0.1) from staged predictions on the held-out
    blocks; then refit on all training cells"""
    GRID = [dict(max_leaf_nodes=nl, min_samples_leaf=ms)
            for nl in (15, 63) for ms in (20, 100)]

    def __init__(self, seed=0, max_iter=500, learning_rate=0.1):
        self.seed, self.max_iter, self.lr = seed, max_iter, learning_rate

    def _model(self, n_iter, **kw):
        return HistGradientBoostingRegressor(
            max_iter=n_iter, learning_rate=self.lr, early_stopping=False,
            random_state=self.seed, **kw)

    def fit(self, X, y, sample_weight=None, groups=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        w = None if sample_weight is None else np.asarray(sample_weight)
        tr, va = inner_split(len(y), groups, seed=self.seed)
        wv = None if w is None else w[va]
        best = (np.inf, None, None)
        for kw in self.GRID:
            m = self._model(self.max_iter, **kw).fit(
                X[tr], y[tr], sample_weight=None if w is None else w[tr])
            for i, p in enumerate(m.staged_predict(X[va])):
                mse = np.average((y[va] - p) ** 2, weights=wv)
                if mse < best[0]:
                    best = (mse, kw, i + 1)
        self.params_ = dict(best[1], n_iter=best[2])
        self.model_ = self._model(best[2], **best[1]).fit(
            X, y, sample_weight=w)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, float))


class BlockMLP:
    """Multilayer perceptron on standardized inputs (median-imputed, with
    missing-value indicators) and a standardized target; the number of
    epochs is chosen by early stopping on an inner block split, and the
    best epoch's weights are kept"""
    def __init__(self, seed=0, hidden=(256, 128, 64), alpha=1e-3,
                 batch_size=256, max_epochs=200, patience=10):
        self.seed, self.hidden, self.alpha = seed, hidden, alpha
        self.batch_size, self.max_epochs = batch_size, max_epochs
        self.patience = patience

    def _prep(self, X):
        X = np.asarray(X, float)
        miss = np.isnan(X[:, self.miss_cols_])
        X = np.where(np.isnan(X), self.med_, X)
        return np.column_stack([(X - self.mu_) / self.sd_, miss])

    def fit(self, X, y, sample_weight=None, groups=None):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.miss_cols_ = np.flatnonzero(np.isnan(X).any(0))
        self.med_ = np.nanmedian(X, 0)
        Xi = np.where(np.isnan(X), self.med_, X)
        self.mu_, self.sd_ = Xi.mean(0), Xi.std(0)
        self.sd_[self.sd_ == 0] = 1
        self.ymu_, self.ysd_ = y.mean(), y.std()
        Z, t = self._prep(X), (y - self.ymu_) / self.ysd_
        tr, va = inner_split(len(y), groups, seed=self.seed)
        m = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha,
                         batch_size=self.batch_size, random_state=self.seed)
        best, best_m, wait = np.inf, None, 0
        for epoch in range(self.max_epochs):
            m.partial_fit(Z[tr], t[tr])
            mse = np.mean((t[va] - m.predict(Z[va])) ** 2)
            if mse < best - 1e-5:
                best, best_m, wait = mse, copy.deepcopy(m), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.model_, self.n_epochs_ = best_m, epoch + 1 - wait
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X)) * self.ysd_ + self.ymu_


LEARNERS = ('hgb', 'hgb_tuned', 'mlp')


def make_model(learner='hgb', seed=0):
    """hgb: the default model of every experiment; hgb_tuned and mlp take
    groups= (spatial blocks of the training cells) in fit"""
    if learner == 'hgb':
        return hgb(seed)
    if learner == 'hgb_tuned':
        return TunedHGB(seed)
    if learner == 'mlp':
        return BlockMLP(seed)
    raise ValueError(f'unknown learner {learner}')


def fit_model(learner, X, y, sample_weight=None, groups=None, seed=0):
    m = make_model(learner, seed)
    if learner == 'hgb':
        return m.fit(X, y, sample_weight=sample_weight)
    return m.fit(X, y, sample_weight=sample_weight, groups=groups)


def wr2(y, p, w=None):
    w = np.ones(len(y)) if w is None else np.asarray(w)
    y, p = np.asarray(y), np.asarray(p)
    mu = np.average(y, weights=w)
    return 1 - np.sum(w * (y - p) ** 2) / np.sum(w * (y - mu) ** 2)


def oof_predict(df, cols, target, blocks, weight=None, seed=0,
                n_splits=N_SPLITS, fold_features=None, learner='hgb',
                fold_seed=GLOBAL):
    """Out-of-fold predictions with GroupKFold over spatial blocks.

    fold_seed: None for GroupKFold's deterministic assignment, an int to
    shuffle blocks into folds; defaults to the module's FOLD_SEED.

    fold_features(df, train_mask) -> (n, k) array: extra features that
    must be recomputed in each fold from the training cells only (e.g.
    neighbour_mean). learner: see make_model; tuned learners split the
    training blocks again for their own validation."""
    p = np.full(len(df), np.nan)
    X, y = df[cols].values, df[target].values
    g = df[blocks].values
    w = None if weight is None else df[weight].values
    fs = FOLD_SEED if fold_seed == GLOBAL else fold_seed
    kf = (GroupKFold(n_splits=n_splits) if fs is None else
          GroupKFold(n_splits=n_splits, shuffle=True, random_state=fs))
    for tr, te in kf.split(X, groups=g):
        Xf = X
        if fold_features is not None:
            train = np.zeros(len(df), bool)
            train[tr] = True
            Xf = np.column_stack([X, fold_features(df, train)])
        m = fit_model(learner, Xf[tr], y[tr],
                      None if w is None else w[tr], g[tr], seed)
        p[te] = m.predict(Xf[te])
    return p


def neighbour_mean(df, target, train, radius_m, cell_m, block='block1000',
                   min_n=5):
    """Mean target of the training cells within radius_m of each cell,
    leaving out the cell's own spatial block (a spatial-neighbourhood
    baseline that uses no training label from the test block).

    Disk sums come from an FFT convolution on the cell grid; the own-block
    sum is then subtracted, which is exact while the block fits inside the
    disk (1 km blocks, radius >= 1.5 km)."""
    from scipy.signal import fftconvolve
    rows, cols = df.cell_row.values, df.cell_col.values
    y = df[target].values
    tm = train & np.isfinite(y)
    shape = (rows.max() + 1, cols.max() + 1)
    s, n = np.zeros(shape), np.zeros(shape)
    s[rows[tm], cols[tm]] = y[tm]
    n[rows[tm], cols[tm]] = 1
    r = int(radius_m // cell_m)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (yy ** 2 + xx ** 2 <= r * r).astype(float)
    S = fftconvolve(s, k, 'same')[rows, cols]
    N = np.round(fftconvolve(n, k, 'same'))[rows, cols]
    b = df[block].values
    bs = pd.Series(np.where(tm, y, 0.0)).groupby(b).transform('sum').values
    bn = pd.Series(tm.astype(float)).groupby(b).transform('sum').values
    N = N - bn
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(N >= min_n, (S - bs) / N, np.nan)


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


def block_draws(nb, n_boot, seed=0):
    """(n_boot, nb) multiplicities of nb blocks resampled with
    replacement"""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, nb, size=(n_boot, nb))
    return np.stack([np.bincount(d, minlength=nb)
                     for d in draws]).astype(float)


def r2_draws(s, mult, key):
    """Weighted R² of prediction key under block multiplicities mult"""
    sw, swy, swy2 = mult @ s['w'], mult @ s['wy'], mult @ s['wy2']
    sst = swy2 - swy ** 2 / sw
    return 1 - (mult @ s[key]) / sst


def bootstrap_r2(y, preds, blocks, w=None, n_boot=1000, seed=0,
                 pairs=()):
    """Block-bootstrap R² for each prediction and R² differences.

    preds: {name: oof predictions}; pairs: [(a, b)] giving R²(b) - R²(a).
    Blocks are resampled with replacement. Returns {name or 'b-a':
    (estimate, lo95, hi95)}."""
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    s, nb = block_sums(y, preds, w, blocks)
    m = block_draws(nb, n_boot, seed)
    r2 = lambda mult, key: r2_draws(s, mult, key)
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


def bootstrap_r2_splits(y, preds, blocks, w=None, n_boot=1000, seed=0,
                        pairs=()):
    """Block-bootstrap R² and R² differences averaged over fold splits.

    preds: {split: {name: oof predictions}}, every split on the same cells.
    One set of block draws serves every split; each draw averages the R²
    (or the paired difference) over the splits, so the CI is that of the
    average. Returns {name or 'b-a': dict(est, lo, hi, per_split={split:
    (est, lo, hi)})}."""
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    sums = {k: block_sums(y, p, w, blocks) for k, p in preds.items()}
    nb = next(iter(sums.values()))[1]
    m = block_draws(nb, n_boot, seed)
    one = np.ones((1, nb))
    names = list(next(iter(preds.values())))
    stats = {n: (lambda mult, n=n: {k: r2_draws(s, mult, n)
                                    for k, (s, _) in sums.items()})
             for n in names}
    stats.update({f'{b}-{a}': (lambda mult, a=a, b=b: {
        k: r2_draws(s, mult, b) - r2_draws(s, mult, a)
        for k, (s, _) in sums.items()}) for a, b in pairs})
    out = {}
    for key, f in stats.items():
        pt, bt = f(one), f(m)
        per = {k: (pt[k][0], *np.percentile(bt[k], [2.5, 97.5]))
               for k in preds}
        avg = np.mean([bt[k] for k in preds], 0)
        out[key] = dict(est=float(np.mean([pt[k][0] for k in preds])),
                        lo=np.percentile(avg, 2.5),
                        hi=np.percentile(avg, 97.5), per_split=per)
    return out


def ladder_splits(df, target, fsets, blocks, fold_seeds=FOLD_SEEDS,
                  weight=None, seed=0, n_boot=1000, pairs=None,
                  learner='hgb'):
    """ladder() repeated over fixed fold assignments, with every R² and
    gain averaged over them (bootstrap_r2_splits). Rows carry
    fold_seed='mean' for the average and the seed for each split.
    Returns (rows, {seed: preds DataFrame})."""
    d = df[df[target].notna()]
    lr = learner if isinstance(learner, dict) else \
        {name: learner for name in fsets}
    preds = {fs: {name: oof_predict(d, cols, target, blocks, weight, seed,
                                    learner=lr.get(name, 'hgb'),
                                    fold_seed=fs)
                  for name, cols in fsets.items()}
             for fs in fold_seeds}
    names = list(fsets)
    pairs = pairs or list(zip(names[:-1], names[1:]))
    w = None if weight is None else d[weight].values
    bs = bootstrap_r2_splits(d[target].values, preds, d[blocks].values, w,
                             n_boot, seed, pairs)
    rows = []
    for key, (b, a) in [(n, (n, '')) for n in names] + \
            [(f'{b}-{a}', (b, a)) for a, b in pairs]:
        x = bs[key]
        base = dict(target=target, features=b, compare=a, boot_blocks=blocks,
                    n=len(d))
        rows.append(dict(base, fold_seed='mean', r2=x['est'], lo=x['lo'],
                         hi=x['hi']))
        for fs, (est, lo, hi) in x['per_split'].items():
            rows.append(dict(base, fold_seed=fs, r2=est, lo=lo, hi=hi))
    return rows, {fs: pd.DataFrame(p, index=d.index)
                  for fs, p in preds.items()}


def ladder(df, target, fsets, blocks, weight=None, seed=0, n_boot=1000,
           pairs=None, extra_blocks=(), fold_features=None, learner='hgb'):
    """Fit each feature set, then R² with CIs and the R² gain of each step.

    fsets: ordered {name: [cols]}. pairs defaults to consecutive steps.
    extra_blocks: further block columns to bootstrap over (e.g. 5 km).
    fold_features: {name: callable} of per-fold features (oof_predict) for
    the named sets, which may then have no columns of their own.
    learner: one learner for every set, or {name: learner} (default hgb).
    Returns (rows, preds)."""
    d = df[df[target].notna()]
    ff = fold_features or {}
    lr = learner if isinstance(learner, dict) else \
        {name: learner for name in fsets}
    preds = {name: oof_predict(d, cols, target, blocks, weight, seed,
                               fold_features=ff.get(name),
                               learner=lr.get(name, 'hgb'))
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


def crossfit_residuals(df, target, cols, blocks, seed=0, learner='hgb',
                       fold_seed=GLOBAL):
    """target minus its out-of-fold prediction from cols (no leakage)"""
    ok = df[target].notna()
    r = pd.Series(np.nan, index=df.index)
    d = df[ok]
    r[ok] = d[target].values - oof_predict(d, cols, target, blocks,
                                           seed=seed, learner=learner,
                                           fold_seed=fold_seed)
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
