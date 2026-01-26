import numpy as np
import torch

def load_and_preprocess_file(filename, max_dim = 2**20, pad_value = 0):
    with open(filename, "rb") as f:
        f_bytez = f.read()
    x = np.frombuffer(f_bytez, dtype=np.uint8)[:max_dim]
    if len(x) < max_dim:
        n_pad = max_dim - len(x)
        x = np.pad(x, (0, n_pad), 'constant', constant_values=pad_value)
    
    return torch.tensor(x, dtype=torch.int32).unsqueeze(0), f_bytez


def preprocess_bytes(bytez, max_dim = 2**20, pad_value = 0):
    x = np.frombuffer(bytez, dtype=np.uint8)[:max_dim]
    if len(x) < max_dim:
        n_pad = max_dim - len(x)
        x = np.pad(x, (0, n_pad), 'constant', constant_values=pad_value)
    
    return torch.tensor(x, dtype=torch.int32).unsqueeze(0)