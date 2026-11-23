"""Compute quantitative safety margin (S_margin) and avoidance smoothness (J_avoid) metrics."""
from pathlib import Path
import csv
import json
import math
import numpy as np


def compute_run_safety_smoothness(run_dir: Path, rover_radius_m: float = 0.30) -> dict:
    """Compute S_margin (minimum obstacle clearance) and J_avoid (jerk integral) from a completed run."""
    run_dir = Path(run_dir)
    traj_path = run_dir / "evaluation/trajectory.csv"
    map_path = run_dir / "uav_map.npz"
    summary_path = run_dir / "evaluation/summary.json"
    if not traj_path.is_file():
        return {}

    with traj_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 5:
        return {}

    t = np.array([float(r["time_s"]) for r in rows], dtype=float)
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=float)
    speed = np.array([float(r["speed_mps"]) for r in rows], dtype=float)
    yaw = np.array([float(r["yaw_rad"]) for r in rows], dtype=float)
    roll = np.array([abs(float(r["roll_deg"])) for r in rows], dtype=float)
    pitch = np.array([abs(float(r["pitch_deg"])) for r in rows], dtype=float)

    # Smoothness J_avoid = integral ||d a / dt||^2 dt (planar velocity vector + heading rate)
    dt = np.diff(t)
    valid_dt = dt > 1e-4
    vx = speed * np.cos(yaw)
    vy = speed * np.sin(yaw)
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    ax[1:] = np.where(valid_dt, np.diff(vx) / np.where(valid_dt, dt, 1.0), 0.0)
    ay[1:] = np.where(valid_dt, np.diff(vy) / np.where(valid_dt, dt, 1.0), 0.0)
    jx = np.zeros_like(vx)
    jy = np.zeros_like(vy)
    jx[2:] = np.where(valid_dt[1:], np.diff(ax[1:]) / np.where(valid_dt[1:], dt[1:], 1.0), 0.0)
    jy[2:] = np.where(valid_dt[1:], np.diff(ay[1:]) / np.where(valid_dt[1:], dt[1:], 1.0), 0.0)
    # Clip extreme single-frame contact/reset spikes above 50 m/s^3 only if unphysical.
    jerk_sq = jx[2:] ** 2 + jy[2:] ** 2
    dt_j = np.where(valid_dt[1:], dt[1:], 0.0)
    j_avoid = float(np.sum(jerk_sq * dt_j))
    total_dur = max(1e-3, float(np.sum(dt_j)))
    rms_jerk = math.sqrt(j_avoid / total_dur)

    # Safety margin S_margin(t) against UAV BLOCKED cells + injected obstacles.
    obs_centers = []
    obs_radii = []
    if map_path.is_file():
        with np.load(map_path, allow_pickle=False) as data:
            states = data["states"]
            origin = data["origin_xy_m"].astype(float)
            res = float(data["resolution_m"])
        br, bc = np.nonzero(states == 1)
        if len(br):
            bx = origin[0] + (bc + 0.5) * res
            by = origin[1] + (br + 0.5) * res
            obs_centers.append(np.column_stack([bx, by]))
            obs_radii.append(np.full(len(br), 0.5 * res, dtype=float))

    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for ob in summary.get("injected_obstacles", []):
            if ob.get("xy_end_enu_m") is not None:
                continue  # Moving obstacle handled via time-synchronized dynamic_obstacles.csv below.
            oxy = ob.get("xy_enu_m") or ob.get("xy_m")
            osz = ob.get("size_m") or [0.5, 0.5, 0.5]
            if oxy is not None:
                obs_centers.append(np.array([[float(oxy[0]), float(oxy[1])]], dtype=float))
                obs_radii.append(np.array([0.5 * max(float(osz[0]), float(osz[1]))], dtype=float))

    min_margin = None
    p05_margin = None
    mean_margin = None
    if obs_centers:
        centers = np.vstack(obs_centers)
        radii = np.concatenate(obs_radii)
        # Evaluate vectorized pairwise distances in chunks to stay memory-light.
        margins = np.empty(len(xy), dtype=float)
        chunk = 256
        for i in range(0, len(xy), chunk):
            sub = xy[i:i + chunk]
            d = np.linalg.norm(sub[:, None, :] - centers[None, :, :], axis=2) - radii[None, :] - rover_radius_m
            margins[i:i + chunk] = np.min(d, axis=1)
        min_margin = round(float(np.min(margins)), 4)
        p05_margin = round(float(np.quantile(margins, 0.05)), 4)
        mean_margin = round(float(np.mean(margins)), 4)

    dyn_csv = run_dir / "evaluation/dynamic_obstacles.csv"
    min_dyn_margin = None
    if dyn_csv.is_file():
        with dyn_csv.open("r", encoding="utf-8") as df:
            drows = [float(r["clearance_m"]) for r in csv.DictReader(df)]
        if drows:
            min_dyn_margin = round(float(min(drows)), 4)
            if min_margin is None or min_dyn_margin < min_margin:
                min_margin = min_dyn_margin

    return dict(
        min_safety_margin_m=min_margin,
        min_dynamic_margin_m=min_dyn_margin,
        p05_safety_margin_m=p05_margin,
        mean_safety_margin_m=mean_margin,
        j_avoid_integral=round(j_avoid, 3),
        rms_jerk_mps3=round(rms_jerk, 4),
        mean_abs_roll_deg=round(float(np.mean(roll)), 2),
        mean_abs_pitch_deg=round(float(np.mean(pitch)), 2),
        mean_speed_mps=round(float(np.mean(speed)), 4),
    )
