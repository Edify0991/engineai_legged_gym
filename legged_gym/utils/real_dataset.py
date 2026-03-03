from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class RealRobotLogSchema:
    """Schema for real-robot offline logs saved in `.npz`.

    Fields:
        obs: Current policy observations with shape (N, 705).
        action: Demonstration / executed actions with shape (N, 12).
        next_obs: Optional next-step observations with shape (N, 705).
        done: Episode terminal flags with shape (N,).
    """

    obs: str = 'obs'
    action: str = 'action'
    next_obs: str = 'next_obs'
    done: str = 'done'


class RealRobotDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        schema = RealRobotLogSchema()
        if schema.obs not in data or schema.action not in data or schema.done not in data:
            raise KeyError('Dataset npz must contain keys: obs, action, done')

        self.obs = torch.from_numpy(data[schema.obs]).float()
        self.action = torch.from_numpy(data[schema.action]).float()
        self.done = torch.from_numpy(data[schema.done]).float()

        self.next_obs = None
        if schema.next_obs in data:
            self.next_obs = torch.from_numpy(data[schema.next_obs]).float()

        n = self.obs.shape[0]
        if self.action.shape[0] != n or self.done.shape[0] != n:
            raise ValueError('obs/action/done must have the same first dimension')
        if self.next_obs is not None and self.next_obs.shape[0] != n:
            raise ValueError('next_obs must have the same first dimension as obs')

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        sample = {
            'obs': self.obs[idx],
            'action': self.action[idx],
            'done': self.done[idx],
        }
        if self.next_obs is not None:
            sample['next_obs'] = self.next_obs[idx]
        return sample


def make_dataloader(path, batch_size, shuffle=True, num_workers=0, drop_last=False):
    dataset = RealRobotDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )
