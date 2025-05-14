import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

file_labels = {
    "0_0": 0,
    "7_1": 1, "14_1": 1, "21_1": 1,  # 内圈故障
    "7_2": 2, "14_2": 2, "21_2": 2,  # 滚珠故障
    "7_3": 3, "14_3": 3, "21_3": 3   # 外圈故障
}

signal_keys = {
    "0_0": "X097_DE_time",
    "7_1": "X105_DE_time",
    "14_1": "X169_DE_time",
    "21_1": "X209_DE_time",
    "7_2": "X118_DE_time",
    "14_2": "X185_DE_time",
    "21_2": "X222_DE_time",
    "7_3": "X130_DE_time",
    "14_3": "X197_DE_time",
    "21_3": "X234_DE_time"
}

window_size = 512
overlap = 0.5
step = int(window_size * (1 - overlap))

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

    if len(sizes) == 0:
        return 1.0

    counts = [boxcount(Z, size) for size in sizes]
    if np.any(np.array(counts) <= 0):
        return 1.0

    try:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    except Exception:
        return 1.0


def extract_chaos_features(signal):
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

    return [lyap, fractal_dim, mean_val, std_val, dynamic_range]


def simple_classifier(features, prototypes):
    dists = [np.linalg.norm(features - p) for p in prototypes]
    return np.argmin(dists)


data_dir = "D:\\Dr.Gao的研究生生活\\研1\\第二学期结课作业\\非线性信息处理技术\\大作业\\code"
X = []
y = []

for fname, label in file_labels.items():
    fpath = os.path.join(data_dir, fname + ".mat")
    mat = scipy.io.loadmat(fpath)
    sig = mat[signal_keys[fname]].squeeze()
    for start in range(0, len(sig) - window_size, step):
        window = sig[start:start + window_size]
        features = extract_chaos_features(window)
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)


titles = ['Lyapunov Approx.', 'Fractal Dim.', 'Mean', 'Std', 'Dynamic Range']
fig, axs = plt.subplots(2, 3, figsize=(15, 9))
for i, ax in enumerate(axs.ravel()[:5]):
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
plt.title("Confusion Matrix: Chaos + Fractal Features")
plt.show()
