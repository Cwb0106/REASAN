#!/usr/bin/env python3
"""Episode-equal critical MAE/DOR on a separately collected held-out HDF5 file."""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader
from residualguard.data import ClearanceDataset
from residualguard.perception import ClearancePredictor, critical_region


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        required=True,
        help="Separate held-out episodes, not the training data",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    original = saved.get("config", {}).get("data")
    if original and Path(original).resolve() == Path(args.data).resolve():
        raise ValueError(
            "Use a separate held-out data file; training/validation metrics are already logged"
        )
    if original and Path(original).is_file():
        with (
            h5py.File(original, "r") as train_file,
            h5py.File(args.data, "r") as test_file,
        ):
            if "seed" in train_file.attrs and train_file.attrs.get(
                "seed"
            ) == test_file.attrs.get("seed"):
                raise ValueError(
                    "Held-out collection must use a different seed to avoid replayed training episodes"
                )
    model = (
        ClearancePredictor(
            not saved.get("config", {}).get("no_motion_conditioning", False)
        )
        .to(args.device)
        .eval()
    )
    model.load_state_dict(saved["model"])
    data = ClearanceDataset(args.data, split="all")
    episode_totals = {}
    with torch.no_grad():
        for current, _, _ in DataLoader(data, batch_size=args.batch_size):
            prediction = (
                model(
                    current["grids"].to(args.device),
                    current["imu"].to(args.device),
                    current["motion"].to(args.device),
                ).cpu()
                * 4
            )
            error = prediction - current["target"]
            critical = critical_region(current["target"])
            for i, episode in enumerate(current["episode"].tolist()):
                values = episode_totals.setdefault(episode, np.zeros(3))
                values += np.array(
                    [
                        torch.where(critical[i], error[i].abs(), 0.0).sum().item(),
                        ((error[i] > 0.10) * critical[i]).sum().item(),
                        critical[i].sum().item(),
                    ]
                )
    counts = [x for x in episode_totals.values() if x[2] > 0]
    results = {
        "episodes": len(episode_totals),
        "critical_episodes": len(counts),
        "critical_mae_m": None,
        "dor_percent": None,
    }
    if counts:
        per_episode = np.stack([x[:2] / x[2] for x in counts])
        results["critical_mae_m"] = {
            "mean": per_episode[:, 0].mean().item(),
            "sd": per_episode[:, 0].std().item(),
        }
        results["dor_percent"] = {
            "mean": (100 * per_episode[:, 1]).mean().item(),
            "sd": (100 * per_episode[:, 1]).std().item(),
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    data.close()


if __name__ == "__main__":
    main()
