#!/usr/bin/env python3
"""Collect Gazebo Harmonic expert/DAgger rollouts and train/evaluate MultimodalPolicy (BC + DAgger).

Reuses the exact PyTorch training script (`tools/train_local_policy.py`) and safety-filtered
`LocalPolicyExperiment` from `controllers/uav_map_rover/local_policy.py`, replacing only the
rollout simulator with Gazebo Harmonic + ROS 2 Jazzy (`gazebo/run_gazebo_mission.py`).
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MISSION_RUNNER = ROOT / "gazebo" / "run_gazebo_mission.py"
TRAIN_SCRIPT = ROOT / "tools" / "train_local_policy.py"


def _run_rollout(python_exe, map_from, out_dir, profile, goal, obstacle, planner,
                 policy, checkpoint, expert_mix, seed, headless, gz_partition):
    cmd = [
        python_exe, str(MISSION_RUNNER),
        "--mode", "navigation",
        "--profile", profile,
        "--map-from", str(map_from),
        "--planner", planner,
        "--obstacle", obstacle,
        "--local-policy", policy,
        "--local-record",
        "--local-expert-mix", str(expert_mix),
        "--local-seed", str(seed),
        "--out-dir", str(out_dir),
    ]
    if goal:
        cmd.extend(["--goal", goal])
    if checkpoint:
        cmd.extend(["--local-checkpoint", str(checkpoint)])
    if headless:
        cmd.append("--headless")
    if gz_partition:
        cmd.extend(["--gz-partition", gz_partition])
    subprocess.run(cmd, check=True)
    return out_dir / "rover" / "imitation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-from", required=True, type=Path, help="Gazebo uav_map.npz survey directory.")
    parser.add_argument("--profile", default="full", choices=("local", "full"))
    parser.add_argument("--planner", default="igha_vctp", choices=("astar", "igha_vctp"))
    parser.add_argument("--goals", nargs="+", default=["valley"], help="Training goal IDs.")
    parser.add_argument("--obstacles", nargs="+", default=["none", "detour"], choices=("none", "detour", "block"))
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gz-partition", default="gz_local_train")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = (args.out_dir or (ROOT / "runs" / f"gz_local_policy_{stamp}")).resolve()
    work.mkdir(parents=True, exist_ok=True)

    train_runs = []
    val_runs = []
    # 1. Collect Rule Expert BC rollouts in Gazebo Harmonic (last episode held out for validation)
    bc_idx = 0
    for goal in args.goals:
        for obs in args.obstacles:
            bc_idx += 1
            ep_dir = work / f"bc_ep_{bc_idx:02d}_{goal}_{obs}"
            ds = _run_rollout(
                args.python, args.map_from, ep_dir, args.profile, goal, obs, args.planner,
                policy="rule", checkpoint="", expert_mix=1.0, seed=100 + bc_idx,
                headless=args.headless, gz_partition=args.gz_partition,
            )
            train_runs.append(ds)

    val_dir = work / f"val_ep_{args.goals[0]}_{args.obstacles[-1]}"
    val_runs.append(_run_rollout(
        args.python, args.map_from, val_dir, args.profile, args.goals[0], args.obstacles[-1],
        args.planner, policy="rule", checkpoint="", expert_mix=1.0, seed=999,
        headless=args.headless, gz_partition=args.gz_partition,
    ))

    # 2. Train BC checkpoint
    bc_ckpt = work / "bc_policy.pt"
    train_cmd = [
        args.python, str(TRAIN_SCRIPT),
        "--train", *[str(d) for d in train_runs],
        "--val", *[str(d) for d in val_runs],
        "--epochs", str(args.epochs),
        "--output", str(bc_ckpt),
    ]
    subprocess.run(train_cmd, check=True)
    current_ckpt = bc_ckpt

    # 3. DAgger rounds in Gazebo Harmonic
    for r in range(1, args.dagger_rounds + 1):
        beta = max(0.2, 0.7 ** r)
        for idx, goal in enumerate(args.goals, start=1):
            obs = args.obstacles[(r + idx) % len(args.obstacles)]
            ep_dir = work / f"dagger_r{r}_ep{idx:02d}_{goal}_{obs}"
            ds = _run_rollout(
                args.python, args.map_from, ep_dir, args.profile, goal, obs, args.planner,
                policy="neural", checkpoint=current_ckpt, expert_mix=beta, seed=200 + r * 10 + idx,
                headless=args.headless, gz_partition=args.gz_partition,
            )
            train_runs.append(ds)
        dagger_ckpt = work / f"dagger_r{r}_policy.pt"
        train_cmd = [
            args.python, str(TRAIN_SCRIPT),
            "--train", *[str(d) for d in train_runs],
            "--val", *[str(d) for d in val_runs],
            "--init", str(current_ckpt),
            "--epochs", str(max(15, args.epochs // 2)),
            "--output", str(dagger_ckpt),
        ]
        subprocess.run(train_cmd, check=True)
        current_ckpt = dagger_ckpt

    manifest = dict(
        simulator="gazebo_harmonic_ros2",
        bc_checkpoint=str(bc_ckpt),
        final_checkpoint=str(current_ckpt),
        train_runs=[str(d) for d in train_runs],
        val_runs=[str(d) for d in val_runs],
    )
    (work / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Gazebo Local Policy] Complete! Final checkpoint: {current_ckpt}", flush=True)


if __name__ == "__main__":
    main()
