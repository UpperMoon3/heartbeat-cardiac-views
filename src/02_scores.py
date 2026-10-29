"""Out-of-fold image scores of pooled and view-specific classifiers (5-fold, grouped by patient, 5 seeds).
Usage: python 02_scores.py features.pkl scores.pkl"""
import sys, pickle, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import load, profile, classifier, V

def main(fpath, out):
    h, R, D = load(fpath, 'cross_val'); P = profile(h); pids = P.index.values; yP = P.y.values; X = {'R18': R, 'Desc': D}; img = {}
    for seed in range(5):
        fold = np.empty(len(pids), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(pids, yP)): fold[te] = k
        fimg = pd.Series(fold, index=pids).loc[h.pid].values
        for feat in ('R18', 'Desc'):
            pooled = np.full(len(h), np.nan); spec = np.full(len(h), np.nan)
            for k in range(5):
                tr = fimg != k; te = fimg == k
                pooled[te] = classifier(feat).fit(X[feat][tr], h.y.values[tr]).predict_proba(X[feat][te])[:, 1]
                for v in V:
                    trv = tr & (h.view.values == v); tev = te & (h.view.values == v)
                    if tev.any(): spec[tev] = classifier(feat).fit(X[feat][trv], h.y.values[trv]).predict_proba(X[feat][tev])[:, 1]
            img[(feat, 'pooled', seed)] = pooled; img[(feat, 'view', seed)] = spec
    pickle.dump(dict(h=h, P=P, img=img), open(out, 'wb'))

if __name__ == '__main__':
    main(*sys.argv[1:3])
