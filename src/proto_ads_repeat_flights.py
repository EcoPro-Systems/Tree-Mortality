#!/usr/bin/env python
"""
Check whether ADS overlaps carry independent delineations: (1) share of
legacy overlap that is identical-geometry duplicates ("pancakes"), and (2)
agreement between DMSM mortality polygons drawn on repeat flights over the
same ground in one season (flights identified from surveyed-area dates and
feature CREATED_DATE). Run from src/:  python proto_ads_repeat_flights.py
"""
import geopandas as gpd, pandas as pd, numpy as np, warnings
from shapely.geometry import box
from rasterio.features import rasterize
from util import load_config
from fetch_hls_aoi import aoi_grid
warnings.filterwarnings("ignore")
c = load_config("../config/hls_aois.yml")
D = "/Volumes/Earth04/ecopro/usfs_ids/"
sa_all = gpd.read_file(D + "R5_surveyed_areas_2012plus.gpkg")
da_all = gpd.read_file(D + "R5_damage_areas_sierra_2012plus.gpkg", where="DAMAGE_TYPE = 'Mortality'")

def campaigns(dates, gap=3):
    """Cluster sorted unique dates into campaigns separated by > gap days"""
    u = sorted(set(dates)); lab = {}; k = 0
    for i, d in enumerate(u):
        if i and (d - u[i-1]).days > gap: k += 1
        lab[d] = k
    return lab

rows, pancake = [], []
for name, a in c["aois"].items():
    crs = f"EPSG:{a['epsg']}"
    tr, sh = aoi_grid(a, c["size_m"], c["resolution"])
    b = box(tr.c, tr.f - sh[0] * tr.a, tr.c + sh[1] * tr.a, tr.f)
    bb = gpd.GeoSeries([b], crs=crs).to_crs(sa_all.crs).iloc[0]
    sa = sa_all[sa_all.intersects(bb)].to_crs(crs).clip(b)
    da = da_all[da_all.intersects(bb)].to_crs(crs).clip(b)
    # 1. How much of the legacy overlap is exact duplicate geometry ("pancakes")?
    for y, g in da[da.SURVEY_YEAR <= 2016].groupby("SURVEY_YEAR"):
        n_ids = g.DAMAGE_AREA_ID.nunique()
        sum_area = g.area.sum()
        dedup = g.drop_duplicates("DAMAGE_AREA_ID").area.sum()
        union = g.union_all().area
        pancake.append(dict(aoi=name, year=y, features=len(g), footprints=n_ids,
                            sum_km2=sum_area/1e6, dedup_km2=dedup/1e6, union_km2=union/1e6))
    # 2. Repeat flights (DMSM years, feature CREATED_DATE = drawing date)
    for y in range(2017, 2025):
        s = sa[(sa.SURVEY_YEAR == y)].copy()
        s["d"] = pd.to_datetime(s.START_DATE, errors="coerce").dt.tz_localize(None).dt.normalize()
        s = s[s.d.notna()]
        m = da[da.SURVEY_YEAR == y].copy()
        m["d"] = pd.to_datetime(m.CREATED_DATE, errors="coerce").dt.tz_localize(None).dt.normalize()
        if s.empty: continue
        camp = campaigns(list(s.d) + list(m.d.dropna()))
        s["camp"] = s.d.map(camp); m["camp"] = m.d.map(camp)
        cs = sorted(s.camp.unique())
        if len(cs) < 2: continue
        cov = {k: rasterize(s[s.camp == k].geometry, out_shape=sh, transform=tr, fill=0, dtype="uint8") for k in cs}
        mort, sev = {}, {}
        for k in cs:
            mk = m[m.camp == k]
            mort[k] = (rasterize(mk.geometry, out_shape=sh, transform=tr, fill=0, dtype="uint8")
                       if len(mk) else np.zeros(sh, np.uint8))
            mk = mk.sort_values("PERCENT_MID")
            sev[k] = (rasterize(zip(mk.geometry, mk.PERCENT_MID.fillna(0)), out_shape=sh, transform=tr, fill=0, dtype="float32")
                      if len(mk) else np.zeros(sh, np.float32))
        dates = {k: sorted({d.date() for d in s[s.camp == k].d}) for k in cs}
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                A, B = cs[i], cs[j]
                both = (cov[A] == 1) & (cov[B] == 1)
                if both.sum() < 1000: continue
                pa, pb = mort[A][both] == 1, mort[B][both] == 1
                n = both.sum(); p_a, p_b = pa.mean(), pb.mean()
                po = (pa == pb).mean(); pe = p_a * p_b + (1 - p_a) * (1 - p_b)
                bothpos = pa & pb
                sa_, sb_ = sev[A][both][bothpos], sev[B][both][bothpos]
                rows.append(dict(
                    aoi=name, year=y, flight_A=f"{dates[A][0]:%m-%d}", flight_B=f"{dates[B][0]:%m-%d}",
                    overlap_km2=round(n * 900 / 1e6, 1), mort_A=round(p_a, 3), mort_B=round(p_b, 3),
                    jaccard=round(bothpos.sum() / max((pa | pb).sum(), 1), 3),
                    kappa=round((po - pe) / (1 - pe), 3) if pe < 1 else np.nan,
                    sev_same_class=round((sa_ == sb_).mean(), 2) if len(sa_) else np.nan,
                ))
pd.set_option("display.width", 200)
p = pd.DataFrame(pancake)
p["dup_share_of_overlap"] = ((p.sum_km2 - p.dedup_km2) / (p.sum_km2 - p.union_km2)).round(2)
print("== Legacy overlap: identical-geometry duplicates vs distinct overlapping polygons")
print(p.round(1).to_string(index=False))
print("\n== Repeat flights over the same ground (DMSM 2017+): agreement of mortality polygons")
print(pd.DataFrame(rows).to_string(index=False))
