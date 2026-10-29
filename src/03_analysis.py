"""Acquisition profile, aggregation rules, count-controlled evaluation, view-specific models,
view combinations (89 patients with all views) and missing views. Usage: python 03_analysis.py scores.pkl results.json"""
import sys, json, pickle, itertools, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import V, profile_X, boot_ci

def main(spath, out):
    Z = pickle.load(open(spath, 'rb')); h, P, IMG = Z['h'], Z['P'], Z['img']; yP = P.y.values; pids = P.index.values; SEEDS = range(5); O = {}
    pat = lambda s, mask=None, how='mean': (pd.DataFrame(dict(p=h.pid.values, s=s))[mask if mask is not None else slice(None)]).groupby('p').s.agg(how)
    # acquisition profile only
    Xa = profile_X(P); au = []; first = None
    for seed in SEEDS:
        oof = np.zeros(len(yP))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(Xa, yP):
            oof[te] = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xa[tr], yP[tr]).predict_proba(Xa[te])[:, 1]
        au.append(roc_auc_score(yP, oof)); first = oof if first is None else first
    O['profile'] = dict(auc=float(np.mean(au)), sd=float(np.std(au)), ci=boot_ci(yP, first))
    O['n_images_only'] = dict(auc=float(roc_auc_score(yP, P.n)), ci=boot_ci(yP, P.n.values))
    O['n_views_only'] = dict(auc=float(roc_auc_score(yP, P.nviews)), ci=boot_ci(yP, P.nviews.values))
    # aggregation rules
    O['aggregation'] = {}
    for feat in ('R18', 'Desc'):
        r = {k: [] for k in ('mean', 'max', 'sum', 'mean+profile')}
        for seed in SEEDS:
            po = IMG[(feat, 'pooled', seed)]
            for how in ('mean', 'max', 'sum'): r[how].append(roc_auc_score(yP, pat(po, None, how).loc[pids]))
            g = np.clip(pat(po).loc[pids].values, 1e-6, 1 - 1e-6); Xc = np.c_[np.log(g / (1 - g)), profile_X(P)]; oof = np.zeros(len(yP))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=200 + seed).split(Xc, yP):
                oof[te] = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xc[tr], yP[tr]).predict_proba(Xc[te])[:, 1]
            r['mean+profile'].append(roc_auc_score(yP, oof))
        O['aggregation'][feat] = {k: [float(np.mean(x)), float(np.std(x))] for k, x in r.items()}
    # fixed number of images per patient
    rng = np.random.default_rng(7); O['count_controlled'] = {}
    for feat in ('R18', 'Desc'):
        res = {k: [] for k in ('one 4C image', 'one image, any view', 'one image per available view')}
        for seed in SEEDS:
            d = pd.DataFrame(dict(p=h.pid.values, v=h.view.values, s=IMG[(feat, 'pooled', seed)]))
            for _ in range(20):
                d['r'] = rng.random(len(d))
                g = d[d.v == '4C'].sort_values('r').groupby('p').s.first(); res['one 4C image'].append(roc_auc_score(P.loc[g.index].y, g))
                g = d.sort_values('r').groupby('p').s.first(); res['one image, any view'].append(roc_auc_score(P.loc[g.index].y, g))
                g = d.sort_values('r').groupby(['p', 'v']).s.first().groupby(level=0).mean(); res['one image per available view'].append(roc_auc_score(P.loc[g.index].y, g))
        O['count_controlled'][feat] = {k: [float(np.mean(x)), float(np.std(x))] for k, x in res.items()}
    # view-specific models
    O['per_view'] = {}
    for feat in ('R18', 'Desc'):
        for v in V:
            mv = (h.view == v).values; ps = []; im = []
            for seed in SEEDS:
                sp = IMG[(feat, 'view', seed)]; im.append(roc_auc_score(h.y[mv], sp[mv])); g = pat(sp, mv); ps.append(roc_auc_score(P.loc[g.index].y, g))
            g = pat(IMG[(feat, 'view', 0)], mv)
            O['per_view'][f'{feat}|{v}'] = dict(patient=float(np.mean(ps)), image=float(np.mean(im)), n=int(len(g)), pos=int(P.loc[g.index].y.sum()), ci=boot_ci(P.loc[g.index].y.values, g.values))
    # combinations and missing views on patients with all four views (exploratory)
    A = P[P.nviews == 4].index.values; yA = P.loc[A].y.values; O['combinations'] = {}; O['missing_view'] = {}
    for feat in ('R18', 'Desc'):
        rows = {}
        for k in range(1, 5):
            for sub in itertools.combinations(V, k):
                au = []
                for seed in SEEDS:
                    d = pd.DataFrame(dict(p=h.pid.values, v=h.view.values, s=IMG[(feat, 'view', seed)])); d = d[d.p.isin(A) & d.v.isin(sub)]
                    au.append(roc_auc_score(yA, d.groupby(['p', 'v']).s.mean().groupby(level=0).mean().loc[A]))
                rows['+'.join(sub)] = float(np.mean(au))
        O['combinations'][feat] = rows; mv = {}
        for v in ['none'] + V:
            au = []
            for seed in SEEDS:
                d = pd.DataFrame(dict(p=h.pid.values, v=h.view.values, s=IMG[(feat, 'pooled', seed)])); d = d[d.p.isin(A)]
                if v != 'none': d = d[d.v != v]
                au.append(roc_auc_score(yA, d.groupby('p').s.mean().loc[A]))
            mv[v] = [float(np.mean(au)), float(np.std(au))]
        O['missing_view'][feat] = mv
    json.dump(O, open(out, 'w'), indent=1)

if __name__ == '__main__':
    main(*sys.argv[1:3])
