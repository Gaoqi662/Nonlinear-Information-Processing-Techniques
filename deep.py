import os
import numpy as np
import scipy.io
from scipy.signal import welch
from scipy.stats import entropy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

label_map = {
    "0_0": 0, "7_1": 1, "14_1": 2, "21_1": 3,
    "7_2": 4, "14_2": 5, "21_2": 6,
    "7_3": 7, "14_3": 8, "21_3": 9
}

window_size = 512
step = window_size // 2

def load_and_slice(path, label):
    mat = scipy.io.loadmat(path)
    key = next(k for k in mat if "DE_time" in k)
    signal = mat[key].squeeze()
    segments, labels = [], []
    for i in range(0, len(signal) - window_size, step):
        seg = signal[i:i+window_size]
        segments.append(seg)
        labels.append(label)
    return np.array(segments), np.array(labels)

folder = "D:\\Dr.Gao的研究生生活\\研1\\第二学期结课作业\\非线性信息处理技术\\大作业\\code"  
files = {f"{k}": os.path.join(folder, f"{k}.mat") for k in label_map}

X, y = [], []
for name, path in files.items():
    data, labels = load_and_slice(path, label_map[name])
    X.append(data)
    y.append(labels)
X = np.concatenate(X)
y = np.concatenate(y)

def spectral_entropy(signal):
    f, Pxx = welch(signal, nperseg=256)
    Pxx = Pxx / np.sum(Pxx)
    return entropy(Pxx)

def sample_entropy(signal, m=2, r=0.2):
    N = len(signal)
    def _phi(m):
        x = np.array([signal[i:i+m] for i in range(N - m)])
        C = np.sum([np.sum(np.linalg.norm(x - xi, axis=1) < r) - 1 for xi in x])
        return C / (N - m)
    return -np.log(_phi(m + 1) / _phi(m) + 1e-12)

def approximate_entropy(signal, m=2, r=0.2):
    def _phi(m):
        N = len(signal)
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.mean([np.sum(np.max(np.abs(x - xi), axis=1) <= r) / (N - m + 1) for xi in x])
        return C
    return np.abs(np.log(_phi(m + 1) / _phi(m) + 1e-12))

def energy_entropy(signal):
    energy = np.square(signal)
    P = energy / np.sum(energy)
    return entropy(P)


class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]
        ent_feats = np.array([
            spectral_entropy(signal),
            sample_entropy(signal),
            approximate_entropy(signal),
            energy_entropy(signal)
        ], dtype=np.float32)
        return torch.tensor(signal).unsqueeze(0), torch.tensor(ent_feats), torch.tensor(self.y[idx])

dataset = BearingDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# === 模型 ===
class CNNWithEntropyAttention(nn.Module):
    def __init__(self):
        super(CNNWithEntropyAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.attn = ChannelAttention(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32 + 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x, ent):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.attn(x)
        x = self.pool(x).squeeze(-1)
        x = torch.cat([x, ent], dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


model = CNNWithEntropyAttention().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    model.train()
    total_loss, correct = 0.0, 0
    for signals, entropies, labels in train_loader:
        signals, entropies, labels = signals.to(device), entropies.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals, entropies)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    acc = correct / train_size
    print(f"Epoch {epoch+1}/20 - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for signals, entropies, labels in test_loader:
        signals, entropies = signals.to(device), entropies.to(device)
        outputs = model(signals, entropies)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix of 10-class Fault Diagnosis")
plt.show()
