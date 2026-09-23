"""Episode-safe perception dataset. Store frames once, reconstruct causal windows lazily."""

from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .perception import motion_features


class ClearanceWriter:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "x")  # never overwrite a collected dataset silently
        self.file.attrs.update(
            schema="residualguard-clearance-v1",
            max_range_m=4.0,
            imu_order="angular_velocity_times_0.25,projected_gravity",
            quat_order="wxyz",
            time_reference="simulator_label_time",
        )

    def append(self, env):
        window = env._clearance_window
        if window is None:
            raise ValueError("Environment requires collect_clearance=True")
        if "seed" not in self.file.attrs:
            self.file.attrs["seed"] = env.cfg.seed
        frame = {
            "grid": window.grids[:, -1].half(),
            "imu": window.imu[:, -1],
            "position": window.positions[:, -1],
            "quaternion": window.quaternions[:, -1],
            "time": window.times[:, -1],
            "extrapolation": window.extrapolation[:, -1],
            "valid": window.valid[:, -1],
            "target_m": env._pre_geometry[3],
            "episode": env.episode_id,
            "step": env.episode_length_buf,
        }
        for name, tensor in frame.items():
            data = tensor.detach().cpu().numpy()
            if name not in self.file:
                self.file.create_dataset(
                    name,
                    shape=(0, *data.shape[1:]),
                    maxshape=(None, *data.shape[1:]),
                    dtype=data.dtype,
                    chunks=True,
                    compression="lzf",
                )
            dataset = self.file[name]
            start = dataset.shape[0]
            dataset.resize(start + data.shape[0], axis=0)
            dataset[start:] = data

    def close(self):
        self.file.close()


class ClearanceDataset(Dataset):
    def __init__(self, path, split="train", seed=42, validation_fraction=0.2):
        self.path = str(path)
        with h5py.File(path, "r") as file:
            if file.attrs.get("schema") != "residualguard-clearance-v1":
                raise ValueError("Unrecognized clearance dataset")
            self.episodes = file["episode"][:]
            self.steps = file["step"][:]
        unique = np.unique(self.episodes)
        if split not in ("train", "val", "all"):
            raise ValueError(f"Invalid split: {split}")
        if len(unique) < 2 and split != "all":
            raise ValueError(
                "Need at least two episodes for a disjoint train/validation split"
            )
        np.random.default_rng(seed).shuffle(unique)
        n_val = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
        selected = (
            unique
            if split == "all"
            else (unique[:n_val] if split == "val" else unique[n_val:])
        )
        self.indices = np.flatnonzero(np.isin(self.episodes, selected))
        self.lookup = {
            (int(ep), int(step)): i
            for i, (ep, step) in enumerate(zip(self.episodes, self.steps))
        }
        if len(self.lookup) != len(self.steps):
            raise ValueError(
                "Duplicate (episode,step) data; refusing ambiguous temporal pairs"
            )
        self.selected_episodes = set(map(int, selected))
        self._file = None

    def __len__(self):
        return len(self.indices)

    def window(self, index):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        episode, step = int(self.episodes[index]), int(self.steps[index])
        indices = [self.lookup.get((episode, s), -1) for s in range(step - 7, step + 1)]
        valid = torch.tensor([i >= 0 for i in indices], dtype=torch.float32)
        safe_indices = [max(i, 0) for i in indices]
        valid *= torch.tensor([float(self._file["valid"][i]) for i in safe_indices])
        keys = ("grid", "imu", "position", "quaternion", "time", "extrapolation")
        # Individual reads support repeated padding indices (h5py fancy indices must be strictly increasing).
        arrays = {
            k: torch.from_numpy(
                np.stack([self._file[k][i] for i in safe_indices])
            ).float()
            for k in keys
        }
        arrays["grid"][valid == 0] = 1
        arrays["imu"][valid == 0] = 0
        motion = motion_features(
            arrays["position"][None],
            arrays["quaternion"][None],
            arrays["time"][None],
            arrays["extrapolation"][None],
            valid[None],
        )[0]
        return {
            "grids": arrays["grid"],
            "imu": arrays["imu"],
            "motion": motion,
            "target": torch.from_numpy(self._file["target_m"][index]).float(),
            "episode": torch.tensor(episode),
        }

    def __getitem__(self, index):
        index = int(self.indices[index])
        next_index = self.lookup.get(
            (int(self.episodes[index]), int(self.steps[index]) + 1), index
        )
        return (
            self.window(index),
            self.window(next_index),
            torch.tensor(float(next_index != index)),
        )

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
