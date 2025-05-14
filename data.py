import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from scipy.stats import entropy
from numpy.linalg import norm
from glob import glob

mat_files = sorted(glob('D:\\Dr.Gao的研究生生活\\研1\\第二学期结课作业\\非线性信息处理技术\\大作业\\code\\*.mat'))

data_dict = {}
for file in mat_files:
    mat_data = scipy.io.loadmat(file)
    for key in mat_data:
        if "DE_time" in key:
            data_dict[os.path.basename(file)] = mat_data[key].flatten()
            break

plt.figure(figsize=(15, 10))
for i, (name, signal) in enumerate(data_dict.items()):
    plt.subplot(5, 2, i+1)
    plt.plot(signal[:2000])
    plt.title(name)
    plt.tight_layout()
plt.suptitle("DE_time Signals from Bearing Dataset", fontsize=16, y=1.02)
plt.show()
