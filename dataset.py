import os, math
import numpy as np
import pandas as pd
from tqdm import tqdm
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
from torch.utils.data import Dataset

def load_data(train_sequences_path, train_labels_path, min_len_filter, max_len_filter):
  train_sequences = pd.read_csv(train_sequences_path)
  train_labels = pd.read_csv(train_labels_path)

  train_labels["pdb_id"] = train_labels["ID"].apply(lambda x: x.split("_")[0] + "_" + x.split("_")[1])

  all_xyz = []
  for pdb_id in tqdm(train_sequences["target_id"], desc="Collecting XYZ data"):
    df = train_labels[train_labels["pdb_id"] == pdb_id]
    xyz = df[["x_1", "y_1", "z_1"]].to_numpy().astype("float32")
    xyz[xyz < -1e17] = float("nan")
    all_xyz.append(xyz)
  
  valid_indices = []
  max_len_seen = 0

  for i, xyz in enumerate(all_xyz):
    if len(xyz) > max_len_seen: max_len_seen = len(xyz)

    nan_ratio = np.isnan(xyz).mean()
    seq_len = len(xyz)
    if (nan_ratio <= 0.5) and (min_len_filter < seq_len < max_len_filter): valid_indices.append(i)

  print(f"Longest sequence in train: {max_len_seen}")

  train_sequences = train_sequences.loc[valid_indices].reset_index(drop=True)
  all_xyz = [all_xyz[i] for i in valid_indices]

  return {
      "sequence": train_sequences["sequence"].tolist(),
      "temporal_cutoff": train_sequences["temporal_cutoff"].tolist(),
      "description": train_sequences["description"].tolist(),
      "all_sequences": train_sequences["all_sequences"].tolist(),
      "xyz": all_xyz,
  }

def train_val_split(data, cutoff_date, test_cutoff_date):
  train_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if pd.Timestamp(date_str) <= cutoff_date]
  test_indices = [i for i, date_str in enumerate(data["temporal_cutoff"]) if cutoff_date < pd.Timestamp(date_str) <= test_cutoff_date]
  return train_indices, test_indices

PAD_IDX = 0

def rand_rot3d(dtype=torch.float32):
  q = torch.randn(4, dtype=dtype)     # i.i.d. N(0,1)
  q = q / q.norm()                    # unit quaternion  → Haar-uniform
  w, x, y, z = q
  return torch.tensor([
    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
  ], dtype=dtype)

class RNA3D_Dataset(Dataset):
  def __init__(self, indices, data_dict, max_len=384, rotate=False):
    self.indices = indices
    self.data = data_dict
    self.max_len = max_len
    self.nt_to_idx = {nt: i for i, nt in enumerate("ACGU")}
    self.rotate = rotate

  def __len__(self): return len(self.indices)
  
  def __getitem__(self, idx):
    data_idx = self.indices[idx]
    seq = [self.nt_to_idx[nt] for nt in self.data["sequence"][data_idx]]
    seq = torch.tensor(seq, dtype=torch.long)        # [seq_len]
    xyz = torch.as_tensor(self.data["xyz"][data_idx], dtype=torch.float32)  # [seq_len, 3]
    xyz[torch.isnan(xyz)] = 0.0
    if self.rotate: xyz = xyz @ rand_rot3d(dtype=torch.float32).to(xyz.device).T

    seq_len = len(seq)
    if seq_len > self.max_len:
      start = np.random.randint(0, seq_len - self.max_len + 1) 
      end = start + self.max_len
      seq = seq[start:end]
      xyz = xyz[start:end]
      seq_len = self.max_len
    elif seq_len < self.max_len:
      pad_len = self.max_len - seq_len
      seq_pad = torch.full((pad_len,), PAD_IDX, dtype=torch.long)
      seq = torch.cat([seq, seq_pad], 0) 
      xyz_pad = torch.zeros((pad_len, 3), dtype=xyz.dtype)
      xyz = torch.cat([xyz, xyz_pad], 0)

    xyz -= xyz[:seq_len].mean(0, keepdim=True) # center
    sigma = xyz[:seq_len].std() + 1e-6
    xyz_norm = xyz / sigma
    sigma_tok = torch.full((1,3), sigma, dtype=xyz.dtype)
    xyz_input = torch.cat([sigma_tok, xyz_norm], dim=0)

    mask = torch.zeros(self.max_len + 1, dtype=torch.bool)
    mask[1:seq_len + 1] = True

    return {"sequence": seq, "xyz": xyz_input, "mask": mask, "sigma": sigma}
