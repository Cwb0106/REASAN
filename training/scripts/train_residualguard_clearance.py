#!/usr/bin/env python3
"""Train/evaluate the independent 8-frame CNN/FiLM/ConvGRU clearance predictor."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from residualguard.data import ClearanceDataset
from residualguard.perception import ClearancePredictor, clearance_loss
from residualguard.runner import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-batches", type=int, default=None, help="Limit only for smoke validation"
    )
    parser.add_argument("--resume", type=str)
    parser.add_argument("--no-motion-conditioning", action="store_true")
    args = parser.parse_args()
    args.data = str(Path(args.data).resolve())
    seed_everything(args.seed)
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_config.json").write_text(json.dumps(vars(args), indent=2))
    train = ClearanceDataset(args.data, "train", args.seed)
    val = ClearanceDataset(args.data, "val", args.seed)
    assert train.selected_episodes.isdisjoint(val.selected_episodes)
    model = ClearancePredictor(
        use_motion_conditioning=not args.no_motion_conditioning
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if (
            saved["config"].get("no_motion_conditioning", False)
            != args.no_motion_conditioning
        ):
            raise ValueError("Resume must preserve the motion-conditioning ablation")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch = saved["epoch"]
    best = float("inf")
    try:
        for epoch in range(start_epoch, start_epoch + args.epochs):
            metrics = {
                "epoch": epoch + 1,
                "train_episodes": len(train.selected_episodes),
                "val_episodes": len(val.selected_episodes),
            }
            for split, dataset in (("train", train), ("val", val)):
                model.train(split == "train")
                totals, count = {}, 0
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=split == "train",
                    num_workers=0,
                )
                for batch_index, (current, following, valid) in enumerate(loader):
                    if args.max_batches is not None and batch_index >= args.max_batches:
                        break
                    current = {k: v.to(args.device) for k, v in current.items()}
                    following = {k: v.to(args.device) for k, v in following.items()}
                    valid = valid.to(args.device)
                    with torch.set_grad_enabled(split == "train"):
                        prediction = model(
                            current["grids"], current["imu"], current["motion"]
                        )
                        next_prediction = model(
                            following["grids"], following["imu"], following["motion"]
                        )
                        loss, terms = clearance_loss(
                            prediction,
                            current["target"],
                            next_prediction,
                            following["target"],
                            valid,
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite predictor loss")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0, error_if_nonfinite=True
                            )
                            optimizer.step()
                    size = len(valid)
                    values = {
                        **terms,
                        "total": loss,
                        "mae_m": (prediction * 4 - current["target"]).abs().mean(),
                    }
                    for key, value in values.items():
                        totals[key] = totals.get(key, 0) + value.detach().item() * size
                    count += size
                if not count:
                    raise ValueError(f"Empty {split} partition")
                metrics.update({f"{split}/{k}": v / count for k, v in totals.items()})
            print(json.dumps(metrics), flush=True)
            with (output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(metrics) + "\n")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "config": vars(args),
                "max_range": 4.0,
                "validation": metrics,
            }
            torch.save(checkpoint, output / "last.pt")
            if metrics["val/total"] < best:
                best = metrics["val/total"]
                torch.save(checkpoint, output / "best.pt")
        model.eval()
        torch.jit.script(model.cpu()).save(str(output / "predictor.ts"))
        print(
            f"PASS: perception training, disjoint-episode validation, checkpoint and export: {output}"
        )
    finally:
        train.close()
        val.close()


if __name__ == "__main__":
    main()
