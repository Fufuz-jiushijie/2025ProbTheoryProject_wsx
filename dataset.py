import numpy as np
import torch
from sklearn.datasets import make_circles

def get_dataset(n_samples=10000):
    data, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    data = data.astype(np.float32)
    return torch.from_numpy(data)

def get_batch(data, batch_size):
    idx = torch.randint(0, data.shape[0], (batch_size,))
    return data[idx]
