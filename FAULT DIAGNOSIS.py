import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import entropy
from collections import Counter

file_labels = {
    "0_0": 0,
    "7_1": 1, "14_1": 1, "21_1": 1,  # 内圈故障
    "7_2": 2, "14_2": 2, "21_2": 2,  # 滚珠故障
    "7_3": 3, "14_3": 3, "21_3": 3   # 外圈故障
}

signal_keys = {
    "0_0": "X097_DE_time", "7_1": "X105_DE_time", "14_1": "X169_DE_time", "21_1": "X209_DE_time",
    "7_2": "X118_DE_time", "14_2": "X185_DE_time", "21_2": "X222_DE_time",
    "7_3": "X130_DE_time", "14_3": "X197_DE_time", "21_3": "X234_DE_time"
}

window_size = 512
overlap = 0.5
step = int(window_size * (1 - overlap))
data_dir = "D:\\Dr.Gao的研究生生活\\研1\\第二学期结课作业\\非线性信息处理技术\\大作业\\code"

def fractal_dimension(Z):
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where(S > 0)[0])
    Z = np.asarray(Z, dtype=bool)
    p = min(Z.shape)
    n = 2**np.floor(np.log2(p)).astype(int)
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    if len(sizes) == 0: return 1.0
    counts = [boxcount(Z, size) for size in sizes]
    if np.any(np.array(counts) <= 0): return 1.0
    try:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    except Exception:
        return 1.0

def sample_entropy(signal, m=2, r=0.2):
    N = len(signal)
    r *= np.std(signal)
    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m)])
        C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x])
        return C / ((N - m) * (N - m - 1) + 1e-10)
    try:
        return -np.log(_phi(m+1) / _phi(m) + 1e-10)
    except:
        return 0.0

def approximate_entropy(signal, m=2, r=0.2):
    N = len(signal)
    r *= np.std(signal)
    def _phi(m):
        x = np.array([signal[i:i + m] for i in range(N - m)])
        C = np.sum([np.sum(np.max(np.abs(x - xi), axis=1) <= r) for xi in x])
        return C / ((N - m) * (N - m) + 1e-10)
    try:
        return np.log(_phi(m)) - np.log(_phi(m + 1))
    except:
        return 0.0

def permutation_entropy(signal, order=3, delay=1):
    N = len(signal)
    if N < order * delay: return 0.0
    perms = np.array([tuple(np.argsort(signal[i:i + order * delay:delay])) for i in range(N - order * delay + 1)])
    c = Counter(map(tuple, perms))
    p = np.array(list(c.values()), dtype=float)
    p /= np.sum(p)
    return entropy(p)

def extract_all_features(signal):
    N = len(signal)
    tau = 10
    m = 3
    embedded = np.array([signal[i:N-(m-1)*tau+i:tau] for i in range(m)]).T
    lyap = np.mean(np.abs(np.diff(np.log(np.abs(signal) + 1e-8))))
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    dynamic_range = np.max(signal) - np.min(signal)
    reshaped = signal[:256*2].reshape((256, 2))
    normed = (reshaped - np.min(reshaped)) / (np.max(reshaped) - np.min(reshaped) + 1e-8)
    fractal_dim = fractal_dimension(normed > 0.5)
    samp_en = sample_entropy(signal)
    perm_en = permutation_entropy(signal)
    approx_en = approximate_entropy(signal)
    return [lyap, fractal_dim, mean_val, std_val, dynamic_range, samp_en, perm_en, approx_en]

def simple_classifier(features, prototypes):
    dists = [np.linalg.norm(features - p) for p in prototypes]
    return np.argmin(dists)

X = []
y = []

for fname, label in file_labels.items():
    fpath = os.path.join(data_dir, fname + ".mat")
    mat = scipy.io.loadmat(fpath)
    sig = mat[signal_keys[fname]].squeeze()
    for start in range(0, len(sig) - window_size, step):
        window = sig[start:start + window_size]
        features = extract_all_features(window)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

titles = ['Lyapunov', 'Fractal Dim.', 'Mean', 'Std', 'Dynamic Range', 'Sample Entropy', 'Perm Entropy', 'Approx Entropy']
fig, axs = plt.subplots(3, 3, figsize=(18, 12))
for i, ax in enumerate(axs.ravel()[:8]):
    for label in np.unique(y):
        ax.hist(X[y == label, i], bins=50, alpha=0.6, label=f"Class {label}")
    ax.set_title(titles[i])
    ax.legend()
plt.tight_layout()
plt.show()

prototypes = np.array([np.mean(X[y == label], axis=0) for label in range(4)])
y_pred = np.array([simple_classifier(f, prototypes) for f in X])
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Inner", "Ball", "Outer"])
disp.plot(cmap="viridis")
plt.title("Confusion Matrix: Chaos + Fractal + 3 Entropy Features")
plt.show()
