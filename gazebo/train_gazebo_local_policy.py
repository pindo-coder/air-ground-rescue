#!/usr/bin/env python3
"""Collect Gazebo Harmonic expert/DAgger rollouts and train/evaluate MultimodalPolicy (BC + DAgger).

Reuses the exact PyTorch training script (`tools/train_local_policy.py`) and safety-filtered
`LocalPolicyExperiment` from `controllers/uav_map_rover/local_policy.py`, replacing only the
rollout simulator with Gazebo Harmonic + ROS 2 Jazzy (`gazebo/run_gazebo_mission.py`).
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from run_gazebo_mission import BUNDLE_DIR, PROFILES

ROOT = Path(__file__).resolve().parents[1]
MISSION_RUNNER = ROOT / "gazebo" / "run_gazebo_mission.py"
TRAIN_SCRIPT = ROOT / "tools" / "train_local_policy.py"


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_episode_summary(out_dir, goal, obstacle, world_sha256, checkpoint_sha256):
    status = json.loads((out_dir / "rover/nav_status.json").read_text(encoding="utf-8"))
    if status["goal_id"] != goal:
        raise ValueError(f"Rollout selected {status['goal_id']!r} instead of {goal!r}: {out_dir}")
    manifest = json.loads((out_dir / "rover/imitation/manifest.json").read_text(encoding="utf-8"))
    if not manifest["samples"]:
        raise ValueError(f"No imitation samples in {out_dir}: {status['status']} {status['reason']}")

    summary = dict(
        run_id=uuid4().hex,
        condition=dict(world_sha256=world_sha256, goal_id=goal, obstacle_scenario=obstacle,
                       map_sha256=_sha256(out_dir / "uav_map.npz")),
        local_policy=dict(checkpoint_sha256=checkpoint_sha256),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return manifest["decisions"].get("learner_nonzero_applied", 0)


def _run_rollout(python_exe, map_from, out_dir, profile, goal, obstacle, planner,
                 policy, checkpoint, expert_mix, seed, headless, gz_partition,
                 bundle_dir, timeout_s):
    out_dir.mkdir()
    world_sha256 = _sha256(bundle_dir / "search_d.sdf")
    checkpoint_sha256 = _sha256(checkpoint) if checkpoint else None
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
        "--bundle-dir", str(bundle_dir),
        "--timeout-s", str(timeout_s),
        "--out-dir", str(out_dir),
    ]
    if goal:
        cmd.extend(["--goal", goal])
    if checkpoint:
        cmd.extend(["--local-checkpoint", str(checkpoint)])
    if headless:
        cmd.append("--headless")
    if gz_partition:
        cmd.extend(["--gz-partition", f"{gz_partition}_{out_dir.name}"])
    subprocess.run(cmd, check=True)
    learner_actions = _write_episode_summary(out_dir, goal, obstacle, world_sha256, checkpoint_sha256)
    return out_dir, learner_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-from", required=True, type=Path, help="Gazebo uav_map.npz survey directory.")
    parser.add_argument("--val-map-from", type=Path, help="Optional separate survey directory for validation.")
    parser.add_argument("--bundle-dir", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--profile", default="full", choices=("local", "full"))
    parser.add_argument("--planner", default="igha_vctp", choices=("astar", "igha_vctp"))
    parser.add_argument("--goals", nargs="+", help="Training goal IDs.")
    parser.add_argument("--val-goals", nargs="+", help="Held-out goal IDs.")
    parser.add_argument("--obstacles", nargs="+", default=["none", "detour"], choices=("none", "detour", "block"))
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gz-partition", default="gz_local_train")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    config = json.loads(PROFILES[args.profile].read_text(encoding="utf-8"))
    available_goals = [goal["id"] for goal in config.get("route_goals", [{"id": "local"}])]
    args.goals = args.goals or available_goals[:1]
    val_goals = args.val_goals or [next((goal for goal in available_goals if goal not in args.goals), args.goals[0])]
    if not set(args.goals + val_goals).issubset(available_goals):
        parser.error(f"Valid goals for profile {args.profile}: {', '.join(available_goals)}")
    val_map = args.val_map_from or args.map_from
    for path in (args.map_from / "uav_map.npz", val_map / "uav_map.npz", args.bundle_dir / "search_d.sdf"):
        if not path.is_file():
            parser.error(f"Missing {path}")
    if set(args.goals) & set(val_goals) and _sha256(args.map_from / "uav_map.npz") == _sha256(val_map / "uav_map.npz"):
        parser.error("Training and validation require different goals or different survey maps.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = (args.out_dir or (ROOT / "runs" / f"gz_local_policy_{stamp}")).resolve()
    work.mkdir(parents=True, exist_ok=True)

    train_runs = []
    val_runs = []
    rollout_options = dict(headless=args.headless, gz_partition=args.gz_partition,
                           bundle_dir=args.bundle_dir, timeout_s=args.timeout_s)
    # 1. Collect rule expert rollouts for training and validation.
    bc_idx = 0
    for goal in args.goals:
        for obs in args.obstacles:
            bc_idx += 1
            ep_dir = work / f"bc_ep_{bc_idx:02d}_{goal}_{obs}"
            run, _ = _run_rollout(
                args.python, args.map_from, ep_dir, args.profile, goal, obs, args.planner,
                policy="rule", checkpoint="", expert_mix=1.0, seed=100 + bc_idx,
                **rollout_options,
            )
            train_runs.append(run)

    for index, goal in enumerate(val_goals, start=1):
        val_dir = work / f"val_ep_{index:02d}_{goal}"
        run, _ = _run_rollout(
            args.python, val_map, val_dir, args.profile, goal, "none", args.planner,
            policy="rule", checkpoint="", expert_mix=1.0, seed=900 + index,
            **rollout_options,
        )
        val_runs.append(run)

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
        round_learner_actions = 0
        for idx, goal in enumerate(args.goals, start=1):
            obs = args.obstacles[(r + idx) % len(args.obstacles)]
            ep_dir = work / f"dagger_r{r}_ep{idx:02d}_{goal}_{obs}"
            run, learner_actions = _run_rollout(
                args.python, args.map_from, ep_dir, args.profile, goal, obs, args.planner,
                policy="neural", checkpoint=current_ckpt, expert_mix=beta, seed=200 + r * 10 + idx,
                **rollout_options,
            )
            train_runs.append(run)
            round_learner_actions += learner_actions
        if not round_learner_actions:
            raise RuntimeError(f"DAgger round {r} applied no nonzero learner actions from {current_ckpt}")
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
