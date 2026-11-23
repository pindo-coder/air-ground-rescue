"""Compare rule, BC and DAgger local planners in three matched Webots rollouts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from run_uav_navigation import PROFILES, ROOT, run
from safety_metrics import compute_run_safety_smoothness

# Load checkpoints only to audit model lineage and the training/validation split.
sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))
from local_policy import load_policy
from train_local_policy import scenario_key


def metric_row(method, folder, nav, policy):
    runtime = policy.get("runtime") or {}
    decisions = runtime.get("decisions", {})
    inference = runtime.get("model_inference_ms") or {}
    learner = decisions.get("learner", 0)
    fallback = decisions.get("safety_fallback", 0)
    proposals = learner + fallback
    sm = compute_run_safety_smoothness(folder)
    return dict(
        method=method, status=nav["status"], arrived=nav["arrived"], collisions=nav["collisions"],
        unsafe_state=nav["unsafe_state"], arrival_time_s=nav["travel_duration_s"],
        elapsed_s=round(nav["end_time_s"] - nav["drive_start_time_s"], 3),
        path_m=nav["actual_path_length_m"], final_goal_error_m=nav["final_goal_error_m"],
        replan_count=nav["replan_count"], proposals=proposals,
        learner_applied=decisions.get("learner_applied", 0),
        learner_nonzero_applied=decisions.get("learner_nonzero_applied", 0),
        safety_fallbacks=fallback, fallback_fraction=fallback / proposals if proposals else None,
        model_inference_p95_ms=inference.get("p95"),
        min_safety_margin_m=sm.get("min_safety_margin_m"),
        j_avoid_integral=sm.get("j_avoid_integral"),
        rms_jerk_mps3=sm.get("rms_jerk_mps3"),
        cause=nav["failure_reason"], run_dir=str(folder),
    )


def compare(map_from, bc_checkpoint, dagger_checkpoint, profile="local", goal=None,
            planner="astar", obstacle="none", obstacle_xy=None, obstacle_size=None,
            timeout=90.0, port=18575, output=None, visible=False):
    bc_path, dagger_path = Path(bc_checkpoint).resolve(), Path(dagger_checkpoint).resolve()
    _, bc = load_policy(bc_path)
    _, dagger = load_policy(dagger_path)
    bc_sha = hashlib.sha256(bc_path.read_bytes()).hexdigest()
    if bc["stage"] != "bc" or dagger["stage"] != "dagger" or dagger["bc_sha256"] != bc_sha:
        raise ValueError("Provide a BC checkpoint and its DAgger descendant.")
    if output is None:
        output = ROOT / "runs" / f"local_policy_comparison_{time.strftime('%Y%m%d_%H%M%S')}"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    expected = None
    rows = []
    for offset, (method, checkpoint) in enumerate((("rule", None), ("bc", bc_path), ("dagger", dagger_path))):
        folder, nav = run(profile=profile, goal=goal, planner=planner, obstacle=obstacle,
                          obstacle_xy=obstacle_xy, obstacle_size=obstacle_size,
                          map_from=map_from, timeout=timeout, port=port + offset,
                          local_policy="neural" if checkpoint else "rule",
                          checkpoint=checkpoint, visible=visible)
        info = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        condition = info["condition"]
        if expected is not None and condition != expected:
            mismatches = [key for key in condition if condition[key] != expected[key]]
            raise RuntimeError(f"{method} conditions differ on {mismatches}. See {folder / 'summary.json'}")
        expected = condition
        rows.append(metric_row(method, folder, nav, info["local_policy"]))

    known_runs = {
        episode["run"]
        for model in (bc, dagger)
        for split in ("training_episodes", "validation_episodes")
        for episode in model[split]
        if scenario_key(episode["condition"]) == scenario_key(expected)
    }
    summary = dict(conditions=expected, held_out=not known_runs, matching_data_runs=sorted(known_runs),
                   model_checkpoints=dict(bc=str(bc_path), dagger=str(dagger_path)), results=rows)
    (output / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison: {output / 'comparison.json'}", flush=True)
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-from", type=Path, required=True, help="Prior run with frozen uav_map.npz.")
    parser.add_argument("--bc-checkpoint", type=Path, required=True)
    parser.add_argument("--dagger-checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="local")
    parser.add_argument("--goal")
    parser.add_argument("--planner", choices=("astar", "igha_vctp"), default="astar")
    parser.add_argument("--obstacle", choices=("none", "detour", "block"), default="none")
    parser.add_argument("--obstacle-xy", nargs=2, type=float)
    parser.add_argument("--obstacle-size", nargs=3, type=float)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--port", type=int, default=18575)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visible", action="store_true")
    compare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
