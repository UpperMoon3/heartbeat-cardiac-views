import pickle, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
V = ['4C', '3VT', 'LVOT', 'RVOT']

def load(path, split):
    F = pickle.load(open(path, 'rb')); m = F['meta']; idx = m[m.split == split].index.values
    return m.loc[idx].reset_index(drop=True), F['r18'][idx], F['desc'][idx]

def profile(h):
    P = h.groupby('pid').agg(y=('y', 'first'), n=('y', 'size'), **{v: ('view', lambda s, v=v: (s == v).sum()) for v in V})
    P['nviews'] = (P[V] > 0).sum(1); return P

def profile_X(P): return np.c_[np.log1p(P[V].values), P.nviews.values]

def classifier(feat, C=None, k=64):
    """Hyperparameters fixed a priori: C = 0.1 and 64 components (ResNet-18), C = 1 (descriptors)."""
    if feat == 'R18':
        return make_pipeline(StandardScaler(), PCA(k, random_state=0), LogisticRegression(C=C or 0.1, max_iter=3000, class_weight='balanced'))
    return make_pipeline(StandardScaler(), LogisticRegression(C=C or 1.0, max_iter=3000, class_weight='balanced'))

def boot_ci(y, p, B=2000, seed=0):
    y = np.asarray(y); p = np.asarray(p); rng = np.random.default_rng(seed); pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0); a = []
    for _ in range(B):
        i = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]; a.append(roc_auc_score(y[i], p[i]))
    return [float(np.quantile(a, .025)), float(np.quantile(a, .975))]
