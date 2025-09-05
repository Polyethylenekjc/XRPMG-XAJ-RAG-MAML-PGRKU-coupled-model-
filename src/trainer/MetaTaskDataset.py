from torch.utils.data import Dataset
import os
import torch
import pandas as pd

class MetaTaskDataset(Dataset):
    def __init__(self, meta_dir, physics_dir, target_col, window_size, transform=None):
        self.meta_dir = meta_dir
        self.physics_dir = physics_dir
        self.target_col = target_col
        self.window_size = window_size
        self.transform = transform

        self.meta_files = sorted([f for f in os.listdir(meta_dir) if f.endswith('.csv')])
        self.physics_files = sorted([f for f in os.listdir(physics_dir) if f.endswith('.csv')])
        assert len(self.meta_files) == len(self.physics_files), "Meta and Physics datas must have the same number of files"

    def __len__(self):
        return len(self.meta_files)

    def __getitem__(self, idx):
        meta_path = os.path.join(self.meta_dir, self.meta_files[idx])
        physics_path = os.path.join(self.physics_dir, self.physics_files[idx])

        meta_df = pd.read_csv(meta_path)
        physics_df = pd.read_csv(physics_path)

        meta_df = meta_df.fillna(0)
        physics_df = physics_df.fillna(0)

        meta_target = meta_df[[self.target_col]]
        physics_target = physics_df[[self.target_col]]

        combined_target = pd.concat([meta_target, physics_target], axis=1)

        meta_tensor = torch.tensor(meta_df.values, dtype=torch.float32)
        combined_tensor = torch.tensor(combined_target.values, dtype=torch.float32)

        def apply_window(data, window):
            return torch.stack([data[i:i+window] for i in range(len(data)-window+1)])

        X = apply_window(meta_tensor, self.window_size)
        y = apply_window(combined_tensor, self.window_size)

        return X, y