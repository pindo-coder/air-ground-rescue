"""IGHA*-inspired heading-aware kinodynamic search for a four-wheel skid-steer rover.

Searches continuous-heading pose states (x, y, yaw) inside the UAV-confirmed
corridor using skid-steer motion primitives (forward arcs, in-place turns, and
reverse maneuvers), verifying every intermediate pose with VCTP on the fine
terrain surface and guiding search via an obstacle-aware 2D Dijkstra heuristic.
"""
import heapq
import math

import numpy as np

from grid_route import center_mask
from survey_map import FREE, NEIGHBORS, Grid
from vctp import VctpChecker


N_HEADINGS = 16
HEADING_STEP_RAD = 2.0 * math.pi / N_HEADINGS


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_to_bin(yaw):
    return int(round((_wrap(yaw) % (2.0 * math.pi)) / HEADING_STEP_RAD)) % N_HEADINGS


def _bin_to_yaw(hbin):
    return _wrap((hbin % N_HEADINGS) * HEADING_STEP_RAD)


def _build_holonomic_heuristic(coarse_grid, coarse_usable, goal_xy, xy_step=0.25):
    """Backward 8-connected Dijkstra on a 0.25 m grid for informed hybrid search."""
    subdiv = max(1, int(round(coarse_grid.resolution / xy_step)))
    h_grid = Grid(
        coarse_grid.origin,
        coarse_grid.resolution / subdiv,
        (coarse_grid.shape[0] * subdiv, coarse_grid.shape[1] * subdiv),
    )
    usable = np.repeat(np.repeat(coarse_usable, subdiv, axis=0), subdiv, axis=1)
    goal_cell = h_grid.cell(goal_xy)
    dist_map = np.full(h_grid.shape, math.inf, dtype=float)
    if not h_grid.contains(goal_cell) or not usable[goal_cell]:
        return h_grid, usable, dist_map

    moves = (*NEIGHBORS, (-1, -1), (-1, 1), (1, -1), (1, 1))
    dist_map[goal_cell] = 0.0
    queue = [(0.0, goal_cell)]
    while queue:
        d, cell = heapq.heappop(queue)
        if d > dist_map[cell]:
            continue
        for dr, dc in moves:
            nxt = (cell[0] + dr, cell[1] + dc)
            if not h_grid.contains(nxt) or not usable[nxt]:
                continue
            nd = d + h_grid.resolution * math.hypot(dr, dc)
            if nd < dist_map[nxt]:
                dist_map[nxt] = nd
                heapq.heappush(queue, (nd, nxt))
    return h_grid, usable, dist_map


# Skid-steer motion primitives: (name, arc_length_m, delta_heading_bins, cost_multiplier)
PRIMITIVES = (
    ("forward", 0.55, 0, 1.00),
    ("forward", 0.55, 1, 1.03),
    ("forward", 0.55, -1, 1.03),
    ("forward", 0.55, 2, 1.10),
    ("forward", 0.55, -2, 1.10),
    ("turn", 0.00, 1, 0.22),
    ("turn", 0.00, -1, 0.22),
    ("turn", 0.00, 2, 0.36),
    ("turn", 0.00, -2, 0.36),
    ("reverse", -0.50, 0, 1.65),
    ("reverse", -0.50, 1, 1.75),
    ("reverse", -0.50, -1, 1.75),
)


def plan_igha_star(coarse_grid, states, vctp: VctpChecker, start_xy, goal_xy,
                   rover_radius_m, start_yaw=0.0, goal_yaw=None,
                   goal_tol_m=0.45, max_expansions=15000, heuristic_weight=1.55):
    """Plan a VCTP-verified heading-aware route with skid-steer motion primitives."""
    start_xy = np.asarray(start_xy, dtype=float)
    goal_xy = np.asarray(goal_xy, dtype=float)

    # Preserve the full rover_radius_m corridor clearance from UAV BLOCKED/UNKNOWN edges
    # so the rover never hugs overhanging brush/tree borders while VCTP checks fine terrain inside.
    corridor_usable = center_mask(coarse_grid, states, rover_radius_m)

    # Hybrid A* discretization: 0.50 m x 0.50 m x 22.5 deg bins storing continuous (x, y, yaw).
    h_grid, h_usable, h_dist = _build_holonomic_heuristic(coarse_grid, corridor_usable, goal_xy, xy_step=0.50)

    vctp_cache = {}
    stats = dict(
        expanded_nodes=0,
        vctp_queries=0,
        vctp_rejected_fail=0,
        vctp_rejected_unknown=0,
    )

    def check_cached(x, y, yaw):
        key = (int(round(x * 8.0)), int(round(y * 8.0)), _yaw_to_bin(yaw))
        res = vctp_cache.get(key)
        if res is None:
            stats["vctp_queries"] += 1
            res = vctp.check_pose(x, y, yaw)
            vctp_cache[key] = res
            if res.status == "FAIL":
                stats["vctp_rejected_fail"] += 1
            elif res.status == "UNKNOWN":
                stats["vctp_rejected_unknown"] += 1
        return res

    start_check = check_cached(start_xy[0], start_xy[1], start_yaw)
    if start_check.status != "PASS":
        return dict(
            status="NO_ROUTE",
            reason=f"START_VCTP_{start_check.status}: {start_check.reason}",
            stats=stats,
        )

    start_cell = h_grid.cell(start_xy)
    if not h_grid.contains(start_cell) or not math.isfinite(h_dist[start_cell]):
        return dict(
            status="NO_ROUTE",
            reason="NO_CONFIRMED_CORRIDOR_TO_GOAL",
            stats=stats,
        )

    def heuristic(x, y, yaw):
        cell = h_grid.cell((x, y))
        h = h_dist[cell] if h_grid.contains(cell) and math.isfinite(h_dist[cell]) else math.dist((x, y), goal_xy)
        if goal_yaw is not None and math.dist((x, y), goal_xy) < 1.2:
            h += 0.15 * abs(_wrap(goal_yaw - yaw))
        return heuristic_weight * h

    start_bin = _yaw_to_bin(start_yaw)
    start_key = (start_cell[0], start_cell[1], start_bin)
    start_pose = (float(start_xy[0]), float(start_xy[1]), _bin_to_yaw(start_bin))

    # Node storage: key -> (g_cost, pose, parent_key, motion_name)
    g_scores = {start_key: 0.0}
    poses = {start_key: start_pose}
    parents = {start_key: None}
    motions = {start_key: "start"}
    queue = [(heuristic(*start_pose), 0.0, start_key)]

    goal_key = None
    while queue and stats["expanded_nodes"] < max_expansions:
        _, g_cur, key = heapq.heappop(queue)
        if g_cur > g_scores.get(key, math.inf) + 1e-9:
            continue
        stats["expanded_nodes"] += 1
        x, y, yaw = poses[key]
        prev_motion = motions[key]

        dist_goal = math.hypot(goal_xy[0] - x, goal_xy[1] - y)
        yaw_ok = goal_yaw is None or abs(_wrap(goal_yaw - yaw)) <= 0.35
        if dist_goal <= goal_tol_m and yaw_ok:
            goal_key = key
            break

        hbin = key[2]
        for kind, ds, dbin, mult in PRIMITIVES:
            dyaw = dbin * HEADING_STEP_RAD
            nbin = (hbin + dbin) % N_HEADINGS
            nyaw = _bin_to_yaw(nbin)

            if kind == "turn":
                # Avoid oscillating turn-left then immediately turn-right.
                check = check_cached(x, y, nyaw)
                if check.status != "PASS":
                    continue
                nx, ny = x, y
                step_cost = mult * check.cost
            else:
                # Simulate circular arc / straight motion in 3 sub-steps and verify every sub-pose with VCTP.
                sub_steps = 3 if abs(ds) >= 0.4 else 2
                valid = True
                cost_accum = 0.0
                sx, sy, sth = x, y, yaw
                for k in range(1, sub_steps + 1):
                    frac = 1.0 / sub_steps
                    mid_th = _wrap(sth + 0.5 * dyaw * frac)
                    sx += (ds * frac) * math.cos(mid_th)
                    sy += (ds * frac) * math.sin(mid_th)
                    sth = _wrap(sth + dyaw * frac)
                    scell = h_grid.cell((sx, sy))
                    if not h_grid.contains(scell) or not h_usable[scell]:
                        valid = False
                        break
                    res = check_cached(sx, sy, sth)
                    if res.status != "PASS":
                        valid = False
                        break
                    cost_accum += res.cost
                if not valid:
                    continue
                nx, ny = sx, sy
                mean_vctp_cost = cost_accum / sub_steps
                switch_pen = (0.45 if (prev_motion == "forward" and kind == "reverse")
                              or (prev_motion == "reverse" and kind == "forward") else 0.0)
                step_cost = abs(ds) * mult * mean_vctp_cost + switch_pen

            ncell = h_grid.cell((nx, ny))
            nkey = (ncell[0], ncell[1], nbin)
            ng = g_cur + step_cost
            if ng + 1e-6 >= g_scores.get(nkey, math.inf):
                continue
            g_scores[nkey] = ng
            poses[nkey] = (nx, ny, nyaw)
            parents[nkey] = key
            motions[nkey] = kind
            heapq.heappush(queue, (ng + heuristic(nx, ny, nyaw), ng, nkey))

    if goal_key is None:
        reason = ("VCTP_BLOCKED_OR_UNKNOWN_CORRIDOR: IGHA* found no VCTP-verified "
                  f"route (fail={stats['vctp_rejected_fail']}, unknown={stats['vctp_rejected_unknown']})")
        return dict(status="NO_ROUTE", reason=reason, stats=stats)

    # Reconstruct pose-annotated path.
    chain = []
    cur = goal_key
    while cur is not None:
        chain.append((poses[cur], motions[cur]))
        cur = parents[cur]
    chain.reverse()

    route_xyyaw = [[round(p[0], 4), round(p[1], 4), round(p[2], 4)] for p, _ in chain]
    motion_list = [m for _, m in chain]

    # Spatial polyline (deduplicating pure in-place turns for length & 2D preview compatibility).
    spatial = [route_xyyaw[0][:2]]
    for pose in route_xyyaw[1:]:
        if math.dist(pose[:2], spatial[-1]) > 1e-4:
            spatial.append(pose[:2])
    length_m = (float(np.linalg.norm(np.diff(np.asarray(spatial), axis=0), axis=1).sum())
                if len(spatial) > 1 else 0.0)
    heading_change = sum(
        abs(_wrap(route_xyyaw[i + 1][2] - route_xyyaw[i][2]))
        for i in range(len(route_xyyaw) - 1)
    )
    vctp_audit = vctp.audit_route(
        [p[:2] for p in route_xyyaw],
        [p[2] for p in route_xyyaw],
    )
    stats.update(
        length_m=round(length_m, 4),
        pose_waypoints=len(route_xyyaw),
        spatial_waypoints=len(spatial),
        turn_in_place_count=sum(m == "turn" for m in motion_list),
        reverse_count=sum(m == "reverse" for m in motion_list),
        total_heading_change_deg=round(math.degrees(heading_change), 2),
    )
    return dict(
        status="PLANNED",
        route_xy_m=spatial,
        route_xyyaw_m=route_xyyaw,
        motions=motion_list,
        route_length_m=length_m,
        vctp=vctp_audit,
        stats=stats,
    )
