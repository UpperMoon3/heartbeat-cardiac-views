"""Official four folds, official test partition (with and without two test CHD patients that share a near-identical
image with the training partition), and hyperparameter sensitivity. Usage: python 04_official_and_sensitivity.py features.pkl /path/to/Heartbeat/dataset out.json"""
import sys, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import load, profile, profile_X, classifier, boot_ci

DUPLICATED_TEST_CHD = ['P0567', 'P0658']   # near-identical image with a cross-validation CHD patient (SIFT + RANSAC check)

def official_folds(root):
    out = {}
    for k in range(1, 5):
        df = pd.read_csv(Path(root) / '2T' / 'cross_val' / f'fold{k}.csv'); out[k] = sorted(set(df[df.split == 'val'].patient_code.astype(str)))
    return out

def main(fpath, root, out):
    hc, Rc, Dc = load(fpath, 'cross_val'); ht, Rt, Dt = load(fpath, 'test'); Pc = profile(hc); Pt = profile(ht)
    Xc = {'R18': Rc, 'Desc': Dc}; Xt = {'R18': Rt, 'Desc': Dt}; agg = lambda h, s, how: pd.Series(s).groupby(h.pid.values).agg(how); O = {}
    FO = official_folds(root); res = {}
    for feat in ('R18', 'Desc'):
        sc = np.full(len(hc), np.nan)
        for k, val in FO.items():
            m = hc.pid.isin(val).values; sc[m] = classifier(feat).fit(Xc[feat][~m], hc.y.values[~m]).predict_proba(Xc[feat][m])[:, 1]
        for how in ('mean', 'max'):
            g = agg(hc, sc, how); res[f'{feat}|{how}'] = [roc_auc_score(Pc.loc[g.index].y, g)] + boot_ci(Pc.loc[g.index].y.values, g.values)
    oof = pd.Series(np.nan, index=Pc.index)
    for k, val in FO.items():
        m = Pc.index.isin(val); oof[m] = LogisticRegression(max_iter=2000, class_weight='balanced').fit(profile_X(Pc[~m]), Pc.y[~m]).predict_proba(profile_X(Pc[m]))[:, 1]
    res['profile'] = [roc_auc_score(Pc.y, oof)] + boot_ci(Pc.y.values, oof.values); O['official_folds'] = res
    keep = ~Pt.index.isin(DUPLICATED_TEST_CHD); T = {}
    for feat in ('R18', 'Desc'):
        s = classifier(feat).fit(Xc[feat], hc.y.values).predict_proba(Xt[feat])[:, 1]
        for how in ('mean', 'max'):
            g = agg(ht, s, how).loc[Pt.index]
            T[f'{feat}|{how}'] = dict(all=[roc_auc_score(Pt.y, g)] + boot_ci(Pt.y.values, g.values), without_duplicated=[roc_auc_score(Pt.y[keep], g[keep])] + boot_ci(Pt.y[keep].values, g[keep].values))
    pr = LogisticRegression(max_iter=2000, class_weight='balanced').fit(profile_X(Pc), Pc.y).predict_proba(profile_X(Pt))[:, 1]
    T['profile'] = dict(all=[roc_auc_score(Pt.y, pr)] + boot_ci(Pt.y.values, pr), without_duplicated=[roc_auc_score(Pt.y[keep], pr[keep])] + boot_ci(Pt.y[keep].values, pr[keep]))
    T['n_images_only'] = [roc_auc_score(Pt.y, Pt.n)] + boot_ci(Pt.y.values, Pt.n.values); O['official_test'] = T
    pids = Pc.index.values; fo = np.empty(len(pids), int)
    for k, (_, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100).split(pids, Pc.y.values)): fo[b] = k
    fi = pd.Series(fo, index=pids).loc[hc.pid].values; hs = []
    for feat, grid in (('R18', [(C, k) for C in (0.01, 0.1, 1.0) for k in (32, 64, 128)]), ('Desc', [(C, 64) for C in (0.1, 1.0, 10.0)])):
        for C, k in grid:
            sc = np.empty(len(hc))
            for f in range(5):
                tr = fi != f; sc[fi == f] = classifier(feat, C, k).fit(Xc[feat][tr], hc.y.values[tr]).predict_proba(Xc[feat][fi == f])[:, 1]
            hs.append(dict(feat=feat, C=C, k=k if feat == 'R18' else None, **{how: roc_auc_score(Pc.loc[agg(hc, sc, how).index].y, agg(hc, sc, how)) for how in ('mean', 'max', 'sum')}))
    O['hyperparameters'] = hs; json.dump(O, open(out, 'w'), indent=1, default=float)

if __name__ == '__main__':
    main(*sys.argv[1:4])
