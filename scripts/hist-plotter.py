import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import skew
from Util import get_train_dataset, get_test_dataset, Dataset, get_dataset_labels
import numpy as np

dataset = Dataset.ACCEL
y = get_dataset_labels(dataset)

labels, counts = np.unique(y.to_numpy(), return_counts=True)

print(f"labels={labels}, counts={counts}")

if labels.dtype == object:
    pass
else:
    print(f"Skewness: {skew(y.to_numpy())}")
    print(f"y.shape: {y.shape}")
    y.hist(bins=2)

plt.yscale('log')
plt.title(dataset.name.lower())
plt.ylabel("Frequency")
# plt.xlabel("bins")
plt.show()