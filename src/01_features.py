"""Extract ImageNet ResNet-18 embeddings and 12 handcrafted descriptors for the Heartbeat
second-trimester cross-validation and test partitions.
Usage: python 01_features.py /path/to/Heartbeat/dataset /path/to/resnet18-imagenet.pth features.pkl"""
import sys, pickle, numpy as np, pandas as pd, cv2, torch, torchvision
from pathlib import Path

def descriptor(path, size=128):
    x = cv2.resize(cv2.imread(str(path), 0), (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mu, sd = float(x.mean()), float(x.std())
    skew = float(((x - mu) ** 3).mean() / (sd ** 3 + 1e-8)); bright = float((x > mu + sd).mean())
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3); mag = np.sqrt(gx * gx + gy * gy)
    g_mu, g_sd = float(mag.mean()), float(mag.std()); edge = float((mag > g_mu + g_sd).mean())
    ratio = float(np.log((np.abs(gx).mean() + 1e-6) / (np.abs(gy).mean() + 1e-6)))
    lap = float(cv2.Laplacian(x, cv2.CV_32F).var()); t = size // 3
    return np.array([mu, sd, skew, bright, g_mu, g_sd, edge, ratio, lap, x[t:2*t].mean(), x[:t].mean(), x[2*t:].mean()], float)

def main(root, weights, out):
    rows = []
    for split in ('cross_val', 'test'):
        base = Path(root) / '2T' / split; df = pd.read_csv(base / 'metadata_raw.csv')
        for img, pid, cls, view in zip(df['image'], df['patient_code'], df['class'], df['view']):
            rows.append(dict(split=split, path=str(base / img), pid=str(pid), y=int(str(cls).upper() == 'CHD'), view=str(view).upper()))
    meta = pd.DataFrame(rows)
    dev = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    m = torchvision.models.resnet18(); m.load_state_dict(torch.load(weights, map_location='cpu')); m.fc = torch.nn.Identity(); m.eval().to(dev)
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32); E = []
    for i in range(0, len(meta), 64):
        X = []
        for p in meta.path.values[i:i + 64]:
            x = cv2.resize(cv2.imread(p, 0), (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            X.append(((np.repeat(x[:, :, None], 3, 2) - mean) / std).transpose(2, 0, 1))
        with torch.no_grad(): E.append(m(torch.tensor(np.stack(X)).to(dev)).cpu().numpy())
    D = np.vstack([descriptor(p) for p in meta.path])
    pickle.dump(dict(meta=meta, r18=np.vstack(E).astype(np.float32), desc=D), open(out, 'wb'))

if __name__ == '__main__':
    main(*sys.argv[1:4])
