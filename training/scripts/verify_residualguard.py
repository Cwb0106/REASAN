#!/usr/bin/env python3
"""Bounded verification of real Go2 physics, PPO, resume, perception, and exported inference.

Run with the env_reasan Python interpreter on a GPU-capable host. Artifacts and
per-stage logs are retained; successful updates do not imply a trained policy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from residualguard.config import Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(TRAINING / "logs/residualguard/verification")
    )
    parser.add_argument(
        "--heldout-only",
        action="store_true",
        help="Add held-out perception checks to a completed run",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = (
        json.loads((output / "verification.json").read_text())
        if args.heldout_only
        else {}
    )
    baseline = [
        TRAINING / "logs/rsl_rl/go2_lidar" / p / "exported/policy.pt"
        for p in ("loco_1", "filter_1")
    ]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline}
    cfg = Config()
    cfg.ppo.num_steps = 16
    cfg.ppo.epochs = 2
    cfg.ppo.mini_batches = 2
    cfg.ppo.encoder_epochs = 1
    cfg.save(output / "smoke_config.json")

    def run(name, command):
        start = time.monotonic()
        print(f"START {name}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                [sys.executable, "-u", *command],
                cwd=TRAINING,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        results[name] = {
            "returncode": result.returncode,
            "seconds": time.monotonic() - start,
            "log": str(output / f"{name}.log"),
            "command": [sys.executable, "-u", *command],
        }
        (output / "verification.json").write_text(json.dumps(results, indent=2))
        if result.returncode:
            print((output / f"{name}.log").read_text()[-6000:], flush=True)
            raise RuntimeError(f"{name} failed; see retained log")
        print(f"PASS {name} ({results[name]['seconds']:.1f}s)", flush=True)

    if args.heldout_only:
        run(
            "collect_heldout",
            [
                "scripts/collect_residualguard_clearance.py",
                "--num-envs",
                "4",
                "--steps",
                "16",
                "--terrain-size",
                "4",
                "--seed",
                "43",
                "--output",
                str(output / "heldout.h5"),
            ],
        )
        run(
            "evaluate_heldout",
            [
                "scripts/evaluate_residualguard_clearance.py",
                "--data",
                str(output / "heldout.h5"),
                "--checkpoint",
                str(output / "perception/best.pt"),
                "--device",
                "cpu",
                "--batch-size",
                "4",
                "--output",
                str(output / "heldout_metrics.json"),
            ],
        )
        return

    run(
        "unit",
        [
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_residualguard.py",
            "-v",
        ],
    )
    run(
        "collect",
        [
            "scripts/collect_residualguard_clearance.py",
            "--num-envs",
            "4",
            "--steps",
            "32",
            "--terrain-size",
            "4",
            "--episode-seconds",
            "0.6",
            "--output",
            str(output / "clearance.h5"),
        ],
    )
    run(
        "perception",
        [
            "scripts/train_residualguard_clearance.py",
            "--data",
            str(output / "clearance.h5"),
            "--output",
            str(output / "perception"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--max-batches",
            "2",
        ],
    )
    base = [
        "scripts/train_residualguard.py",
        "--config",
        str(output / "smoke_config.json"),
        "--num-envs",
        "4",
        "--terrain-rows",
        "4",
        "--terrain-cols",
        "4",
        "--episode-seconds",
        "0.6",
    ]
    run(
        "rl_ground_truth",
        base + ["--iterations", "3", "--log-dir", str(output / "rl_ground_truth")],
    )
    run(
        "resume",
        base
        + [
            "--iterations",
            "1",
            "--log-dir",
            str(output / "resumed"),
            "--resume",
            str(output / "rl_ground_truth/model_3.pt"),
        ],
    )
    run(
        "rl_predicted",
        base
        + [
            "--iterations",
            "2",
            "--log-dir",
            str(output / "rl_predicted"),
            "--clearance-checkpoint",
            str(output / "perception/best.pt"),
        ],
    )
    run(
        "inference",
        [
            "scripts/play_residualguard.py",
            "--policy",
            str(output / "rl_predicted/exported/policy.pt"),
            "--config",
            str(output / "rl_predicted/config.json"),
            "--output",
            str(output / "inference"),
            "--clearance-checkpoint",
            str(output / "perception/best.pt"),
            "--steps",
            "40",
            "--num-envs",
            "4",
            "--terrain-size",
            "4",
            "--episode-seconds",
            "0.6",
        ],
    )
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline}
    if after != before:
        raise AssertionError("A baseline checkpoint was modified")
    results["baseline_sha256_unchanged"] = after
    results["status"] = (
        "PASS (training-chain verification, not convergence or paper performance)"
    )
    (output / "verification.json").write_text(json.dumps(results, indent=2))
    print(results["status"], flush=True)


if __name__ == "__main__":
    main()
