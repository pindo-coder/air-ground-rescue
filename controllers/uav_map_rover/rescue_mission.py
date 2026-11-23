"""Safe approach-point selection and onboard RGB-D confirmation for detected victims.

Consumes ONLY sensor-derived aerial victim candidates (`aerial_victims.json`),
the UAV traversability map (`uav_map.npz`), and onboard rover sensors.
Zero access to simulator ground truth.
"""
from pathlib import Path
import math
import sys

import numpy as np

from grid_route import center_mask, plan, plan_dual
from igha_star import plan_igha_star
from survey_map import BLOCKED, FREE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "controllers/search_agent") not in sys.path:
    sys.path.insert(0, str(ROOT / "controllers/search_agent"))
from perception import covariance_intersection


def _mask_victims_on_states(grid, states, exclude_victims_xy, rover_radius_m,
                            exclusion_radius_m=0.60, start_xy=None):
    """Return a copy of coarse `states` with detected victim footprints marked BLOCKED.

    Preserves usability of `start_xy` if the rover is already parked near a victim so
    departure legs route smoothly around the victim's body without driving over them.
    """
    planning_states = np.asarray(states, dtype=np.int8).copy()
    if not exclude_victims_xy:
        return planning_states

    orig_usable = center_mask(grid, states, rover_radius_m)
    victim_centers = [np.asarray(v[:2], dtype=float) for v in exclude_victims_xy]
    victim_cells = {grid.cell(vxy) for vxy in victim_centers}
    pad = max(1, math.ceil(exclusion_radius_m / grid.resolution))

    for vxy in victim_centers:
        r0, c0 = grid.cell(vxy)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                cell = (r0 + dr, c0 + dc)
                if not grid.contains(cell):
                    continue
                if float(np.linalg.norm(grid.center(cell) - vxy)) <= exclusion_radius_m:
                    planning_states[cell] = BLOCKED

    if start_xy is not None:
        sxy = np.asarray(start_xy[:2], dtype=float)
        scell = grid.cell(sxy)
        if grid.contains(scell) and orig_usable[scell]:
            usable_now = center_mask(grid, planning_states, rover_radius_m)
            if not usable_now[scell]:
                # Carve back only immediate neighbors of `start_xy` that are NOT a victim center cell.
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb = (scell[0] + dr, scell[1] + dc)
                        if not grid.contains(nb) or nb in victim_cells:
                            continue
                        if states[nb] == FREE and planning_states[nb] == BLOCKED:
                            planning_states[nb] = FREE
    return planning_states


def select_victim_approach_point(grid, states, planning_cost, start_xy, victim_xy,
                                 rover_radius_m, vctp=None, planner_mode="igha_vctp",
                                 start_yaw=0.0, min_standoff_m=0.85, max_standoff_m=1.58,
                                 ideal_standoff_m=1.32, exclude_victims_xy=None,
                                 victim_exclusion_radius_m=0.60):
    """Select a safe confirmed approach pose beside a detected victim and plan a global route."""
    start_xy = np.asarray(start_xy, dtype=float)
    victim_xy = np.asarray(victim_xy, dtype=float)
    all_victims = list(exclude_victims_xy) if exclude_victims_xy else [victim_xy]
    if not any(float(np.linalg.norm(np.asarray(v[:2], dtype=float) - victim_xy)) < 0.05 for v in all_victims):
        all_victims.append(victim_xy)

    planning_states = _mask_victims_on_states(
        grid, states, all_victims, rover_radius_m,
        exclusion_radius_m=victim_exclusion_radius_m, start_xy=start_xy,
    )
    usable = center_mask(grid, planning_states, rover_radius_m)

    # Search confirmed usable cells in a ring around the detected victim.
    r0, c0 = grid.cell(victim_xy)
    pad = max(2, math.ceil((max_standoff_m + 0.5) / grid.resolution))
    d_start_to_victim = float(np.linalg.norm(victim_xy - start_xy))
    candidates = []
    for dr in range(-pad, pad + 1):
        for dc in range(-pad, pad + 1):
            cell = (r0 + dr, c0 + dc)
            if not grid.contains(cell) or not usable[cell]:
                continue
            xy = grid.center(cell)
            standoff = float(np.linalg.norm(xy - victim_xy))
            if not (min_standoff_m <= standoff <= max_standoff_m):
                continue
            approach_yaw = math.atan2(victim_xy[1] - xy[1], victim_xy[0] - xy[0])
            vctp_ok = True
            vctp_cost = 1.0
            if vctp is not None:
                vr = vctp.check_pose(xy[0], xy[1], approach_yaw)
                vctp_ok = vr.status == "PASS"
                vctp_cost = vr.cost if vctp_ok else 99.0
            t_cost = (float(planning_cost[cell])
                      if planning_cost is not None and np.isfinite(planning_cost[cell]) else 1.0)
            # Prefer near-side approach points outside the victim's body footprint (>= 1.08 m)
            # so the rover never drives past or grazes a supine victim to reach their far side.
            d_start_to_cand = float(np.linalg.norm(xy - start_xy))
            far_side_pen = 0.75 * max(0.0, d_start_to_cand - d_start_to_victim + 0.25)
            close_pen = 10.0 * max(0.0, 1.08 - standoff)
            score = (
                (0.0 if vctp_ok else 50.0)
                + 1.4 * abs(standoff - ideal_standoff_m)
                + close_pen
                + far_side_pen
                + 0.35 * t_cost
                + 0.25 * min(vctp_cost, 10.0)
            )
            candidates.append((score, standoff, approach_yaw, xy, vctp_ok))

    candidates.sort(key=lambda item: item[0])
    if not candidates:
        dual = plan_dual(grid, states, start_xy, victim_xy, rover_radius_m, planning_cost)
        return dict(
            status="NO_APPROACH",
            reason="NO_CONFIRMED_APPROACH_CELL_NEAR_VICTIM",
            victim_xy_m=victim_xy.tolist(),
            dual_route=dual,
        )

    # First try top ranked approach points with the requested planner on victim-masked states.
    last_reason = "NO_CONFIRMED_ROUTE"
    for _, standoff, approach_yaw, axy, vctp_ok in candidates[:10]:
        # Quick 2D A* check to confirm connectivity and get baseline length.
        try:
            astar_path = plan(grid, planning_states, start_xy, axy, rover_radius_m, planning_cost)
            astar_len = float(np.linalg.norm(np.diff(astar_path, axis=0), axis=1).sum())
        except ValueError as err:
            last_reason = str(err)
            continue

        if planner_mode == "igha_vctp" and vctp is not None:
            igha = plan_igha_star(
                grid, planning_states, vctp, start_xy, axy, rover_radius_m,
                start_yaw=start_yaw, goal_yaw=None, goal_tol_m=0.45,
            )
            if igha["status"] == "PLANNED":
                return dict(
                    status="PLANNED",
                    planner="igha_vctp",
                    victim_xy_m=victim_xy.tolist(),
                    approach_xy_m=axy.tolist(),
                    approach_yaw_rad=round(approach_yaw, 4),
                    standoff_m=round(standoff, 3),
                    route_xy_m=igha["route_xy_m"],
                    route_xyyaw_m=igha["route_xyyaw_m"],
                    route_length_m=float(igha["route_length_m"]),
                    astar_length_m=astar_len,
                    vctp_audit=igha["vctp"],
                    igha_stats=igha["stats"],
                )
            last_reason = igha.get("reason", "IGHA_NO_ROUTE")
            # Fall back to confirmed 2D A* only if no VCTP+IGHA* candidate succeeds across ring.
            continue

        vctp_audit = vctp.audit_route(astar_path) if vctp is not None else None
        return dict(
            status="PLANNED",
            planner="astar",
            victim_xy_m=victim_xy.tolist(),
            approach_xy_m=axy.tolist(),
            approach_yaw_rad=round(approach_yaw, 4),
            standoff_m=round(standoff, 3),
            route_xy_m=astar_path.tolist(),
            route_length_m=astar_len,
            vctp_audit=vctp_audit,
        )

    # If igha_vctp failed on fine bumps right near the victim, check if confirmed A* reaches safely.
    if planner_mode == "igha_vctp":
        for _, standoff, approach_yaw, axy, _ in candidates[:6]:
            try:
                astar_path = plan(grid, planning_states, start_xy, axy, rover_radius_m, planning_cost)
                astar_len = float(np.linalg.norm(np.diff(astar_path, axis=0), axis=1).sum())
                vctp_audit = vctp.audit_route(astar_path) if vctp is not None else None
                return dict(
                    status="PLANNED",
                    planner="astar_fallback",
                    fallback_reason=last_reason,
                    victim_xy_m=victim_xy.tolist(),
                    approach_xy_m=axy.tolist(),
                    approach_yaw_rad=round(approach_yaw, 4),
                    standoff_m=round(standoff, 3),
                    route_xy_m=astar_path.tolist(),
                    route_length_m=astar_len,
                    vctp_audit=vctp_audit,
                )
            except ValueError:
                continue

    dual = plan_dual(grid, states, start_xy, candidates[0][3], rover_radius_m, planning_cost)
    return dict(
        status="NO_ROUTE",
        reason=last_reason,
        victim_xy_m=victim_xy.tolist(),
        approach_xy_m=candidates[0][3].tolist(),
        dual_route=dual,
    )


def plan_next_victim(grid, states, planning_cost, rover_xy, rover_yaw,
                     remaining_victims, rover_radius_m, vctp=None, planner_mode="igha_vctp",
                     exclude_victims_xy=None):
    """Select the shortest-route confirmed reachable detected victim from current rover pose."""
    all_exclude = (list(exclude_victims_xy) if exclude_victims_xy is not None
                   else [v["mean_enu_m"] for v in remaining_victims])
    plans = []
    for victim in remaining_victims:
        res = select_victim_approach_point(
            grid, states, planning_cost, rover_xy, victim["mean_enu_m"],
            rover_radius_m, vctp=vctp, planner_mode=planner_mode, start_yaw=rover_yaw,
            exclude_victims_xy=all_exclude,
        )
        res["victim_candidate"] = victim
        plans.append(res)
    reachable = [p for p in plans if p["status"] == "PLANNED"]
    if not reachable:
        return None, plans
    reachable.sort(key=lambda item: item["route_length_m"])
    return reachable[0], plans


def confirm_victim_onboard(detector, bgr, depth, pose_xy, yaw, roll, pitch,
                           score_min, victim_mean_enu, victim_cov_enu, max_assoc_m=1.8):
    """Run onboard rover RGB-D person detection and fuse with aerial ENU estimate if matched."""
    pose_2d = np.array([float(pose_xy[0]), float(pose_xy[1]), float(yaw)], dtype=float)
    pose_cov = np.diag([0.06 ** 2, 0.06 ** 2, 0.02 ** 2])
    detections = [
        d for d in detector.localize(bgr, depth, pose_2d, pose_cov, (float(roll), float(pitch)))
        if float(d["score"]) >= float(score_min)
    ]
    if not detections:
        return dict(confirmed=False, fused_xy_m=list(map(float, victim_mean_enu)),
                    fused_cov_enu=np.asarray(victim_cov_enu, dtype=float).tolist(),
                    rover_detections=0)

    target = np.asarray(victim_mean_enu, dtype=float)
    best = min(detections, key=lambda d: float(np.linalg.norm(d["mean"] - target)))
    dist = float(np.linalg.norm(best["mean"] - target))
    if dist > max_assoc_m:
        return dict(confirmed=False, fused_xy_m=target.tolist(),
                    fused_cov_enu=np.asarray(victim_cov_enu, dtype=float).tolist(),
                    rover_detections=len(detections), nearest_det_dist_m=round(dist, 3))

    fused_mean, fused_cov = covariance_intersection(
        target, np.asarray(victim_cov_enu, dtype=float), best["mean"], best["cov"]
    )
    return dict(
        confirmed=True,
        fused_xy_m=[round(float(fused_mean[0]), 4), round(float(fused_mean[1]), 4)],
        fused_cov_enu=[[round(float(v), 5) for v in row] for row in fused_cov],
        rover_xy_m=[round(float(best["mean"][0]), 4), round(float(best["mean"][1]), 4)],
        rover_score=round(float(best["score"]), 4),
        rover_depth_m=round(float(best["depth"]), 3),
        association_dist_m=round(dist, 3),
        rover_detections=len(detections),
    )
