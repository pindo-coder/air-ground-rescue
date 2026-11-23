"""Select oblique views from a measured prior and update only supported terrain."""
from pathlib import Path
import math
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from survey_map import FREE, UNKNOWN, ground_slope

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "controllers/uav_map_rover") not in sys.path:
    sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))
from grid_route import plan_dual


def select_patches_density(grid, layers, settings):
    """Baseline unknown-density + straight-line-proximity heuristic."""
    refinement = settings["refinement"]
    states = layers["states"]
    rows, cols = np.indices(grid.shape)
    positions = np.stack((cols + 0.5, rows + 0.5), axis=-1) * grid.resolution + grid.origin
    start = np.asarray(settings["deployment_xy_enu_m"])
    distance = np.full(grid.shape, np.inf)
    for case in settings["route_goals"]:
        direction = np.asarray(case["xy_enu_m"]) - start
        length_squared = np.dot(direction, direction)
        if length_squared == 0:
            continue
        along = np.clip(np.sum((positions - start) * direction, axis=-1) / length_squared, 0, 1)
        distance = np.minimum(distance, np.linalg.norm(positions - start - along[..., None] * direction, axis=-1))
    near_floor = distance_transform_edt(states != FREE) * grid.resolution <= 1.75
    eligible = (states == UNKNOWN) & near_floor & (layers["observed_views"] >= 2)
    radius = refinement["patch_radius_m"]
    window = 2 * math.ceil(radius / grid.resolution) + 1
    density = uniform_filter(eligible.astype(float), size=window, mode="constant") * window**2
    score = np.where(eligible, density * np.exp(-distance / 7), 0.0)
    patches = []
    for _ in range(refinement["patch_count"]):
        cell = np.unravel_index(np.argmax(score), score.shape)
        if score[cell] <= 0:
            break
        center = grid.center(cell)
        offset = np.linalg.norm(positions - center, axis=-1)
        depths = layers["ground_z_m"][(offset <= radius) & (states == FREE)]
        if depths.size:
            patches.append(dict(xy_m=center.tolist(), elevation_m=float(np.median(depths))))
        score[offset < 2.6 * radius] = 0.0
    return patches, []


def select_patches_route_value(grid, layers, settings):
    """Traversing Mars dual-route + DUCPP route-value-guided patch selection."""
    refinement = settings["refinement"]
    states = layers["states"]
    rows, cols = np.indices(grid.shape)
    positions = np.stack((cols + 0.5, rows + 0.5), axis=-1) * grid.resolution + grid.origin
    start = np.asarray(settings["deployment_xy_enu_m"], dtype=float)
    radius_rover = float(settings["rover_radius_m"])
    planning_cost = layers.get("planning_cost") if "max_slope_deg" in settings else None
    unknown_cost = float(refinement.get("unknown_cost", 1.25))
    reachability_bonus = float(refinement.get("reachability_bonus_m", 60.0))

    near_floor = distance_transform_edt(states != FREE) * grid.resolution <= 1.75
    observed_ok = layers["observed_views"] >= 2

    cell_value = np.zeros(grid.shape, dtype=float)
    cell_goals = {}
    dual_summaries = []
    for case in settings["route_goals"]:
        dual = plan_dual(
            grid, states, start, case["xy_enu_m"], radius_rover,
            planning_cost=planning_cost, unknown_cost=unknown_cost,
        )
        summary = dict(
            id=case["id"],
            label=case.get("label", case["id"]),
            conservative_status=dual["conservative_status"],
            conservative_length_m=dual["conservative_length_m"],
            optimistic_status=dual["optimistic_status"],
            optimistic_length_m=dual["optimistic_length_m"],
            changes_reachability=dual["changes_reachability"],
            potential_saving_m=dual["potential_saving_m"],
            unconfirmed_clearance_waypoints=dual["unconfirmed_clearance_waypoints"],
            corridor_unknown_cell_count=dual["corridor_unknown_cell_count"],
            segments=len(dual["unconfirmed_segments"]),
        )
        dual_summaries.append(summary)
        if dual["optimistic_status"] != "PLANNED" or not dual["corridor_unknown_cells"]:
            continue
        goal_value = (reachability_bonus if dual["changes_reachability"]
                      else float(dual["potential_saving_m"] or 0.0))
        if goal_value < 2.0 and not dual["changes_reachability"]:
            continue
        corridor_cells = [
            (r, c) for r, c in dual["corridor_unknown_cells"]
            if states[r, c] == UNKNOWN and observed_ok[r, c]
        ]
        if not corridor_cells:
            continue
        # Distribute goal value across its unknown corridor cells, favoring boundary cells
        # adjacent to verified floor (immediately verifiable via local ground plane fitting).
        weights = np.array([1.0 if near_floor[r, c] else 0.35 for r, c in corridor_cells], dtype=float)
        norm = max(float(weights.sum()), 1.0)
        for (r, c), w in zip(corridor_cells, weights):
            cell_value[r, c] += goal_value * (w / norm)
            cell_goals.setdefault((r, c), set()).add(case["id"])

    patch_radius = float(refinement["patch_radius_m"])
    patch_count = int(refinement["patch_count"])
    min_sep = float(refinement.get("min_patch_separation_m", 1.8 * patch_radius))

    # Build circular kernel mask for exact disk integration over candidate centers.
    k_rad = math.ceil(patch_radius / grid.resolution)
    dy, dx = np.mgrid[-k_rad:k_rad + 1, -k_rad:k_rad + 1] * grid.resolution
    disk_offsets = np.argwhere(np.hypot(dx, dy) <= patch_radius) - k_rad

    remaining = cell_value.copy()
    verifiable_unknown = (states == UNKNOWN) & near_floor & observed_ok
    candidates = np.argwhere(remaining > 0)
    suppressed = np.zeros(grid.shape, dtype=bool)
    patches = []

    for _ in range(patch_count):
        best = None
        for r0, c0 in candidates:
            if suppressed[r0, c0] or remaining[r0, c0] <= 0:
                continue
            rr = r0 + disk_offsets[:, 0]
            cc = c0 + disk_offsets[:, 1]
            inside = (rr >= 0) & (rr < grid.shape[0]) & (cc >= 0) & (cc < grid.shape[1])
            rr, cc = rr[inside], cc[inside]
            free_in_disk = states[rr, cc] == FREE
            if not np.any(free_in_disk):
                continue
            gain = float(remaining[rr, cc].sum())
            # Small tie-breaker toward verifiable boundary unknown cells inside the patch.
            verif_bonus = 0.02 * float(np.count_nonzero(verifiable_unknown[rr, cc]))
            score = gain + verif_bonus
            if best is None or score > best[0]:
                best = (score, gain, int(r0), int(c0), rr, cc, free_in_disk)
        if best is None or best[1] <= 0:
            break
        _, gain, r0, c0, rr, cc, free_in_disk = best
        center = grid.center((r0, c0))
        depths = layers["ground_z_m"][rr[free_in_disk], cc[free_in_disk]]
        covered_goals = sorted({
            gid for r, c in zip(rr, cc) if remaining[r, c] > 0
            for gid in cell_goals.get((int(r), int(c)), ())
        })
        patches.append(dict(
            xy_m=center.tolist(),
            elevation_m=float(np.median(depths)),
            route_value_gain=round(gain, 3),
            target_goals=covered_goals,
        ))
        remaining[rr, cc] = 0.0
        offset = np.linalg.norm(positions - center, axis=-1)
        suppressed[offset < min_sep] = True

    if not patches:
        return select_patches_density(grid, layers, settings)
    return patches, dual_summaries


def select_patches(grid, layers, settings):
    strategy = settings.get("refinement", {}).get("strategy", "density")
    if strategy == "route_value":
        patches, _ = select_patches_route_value(grid, layers, settings)
        return patches
    patches, _ = select_patches_density(grid, layers, settings)
    return patches


def schedule_oblique(grid, baseline, settings, now):
    refinement = settings["refinement"]
    strategy = refinement.get("strategy", "density")
    if strategy == "route_value":
        patches, dual_routes = select_patches_route_value(grid, baseline, settings)
    else:
        patches, dual_routes = select_patches_density(grid, baseline, settings)
    altitude = refinement["altitude_enu_m"]
    stations = []
    for index, patch in enumerate(patches, start=1):
        target = np.asarray(patch["xy_m"])
        height = altitude - patch["elevation_m"]
        if height <= 0:
            continue
        distance = height * math.tan(math.radians(refinement["angle_from_vertical_deg"]))
        for azimuth in (0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            view = np.array([math.cos(azimuth), math.sin(azimuth)])
            lateral = np.array([-view[1], view[0]])
            for spacing in (-2.6, 0.0, 2.6):
                position = target - distance * view + spacing * lateral
                if not grid.contains(grid.cell(position)):
                    continue
                delta = target - position
                stations.append(dict(xyz_m=[*position.tolist(), altitude],
                                     yaw_rad=math.atan2(delta[1], delta[0]),
                                     pitch_rad=-math.atan2(np.linalg.norm(delta), height), patch=index))

    finish = [*settings["flight"]["waypoints_xy_enu_m"][-1], settings["flight"]["altitude_enu_m"]]
    time_s = math.ceil(now) + 1
    position = finish
    ordered = []
    while stations:
        station = min(stations, key=lambda item: math.dist(position, item["xyz_m"]))
        stations.remove(station)
        trip = math.dist(position, station["xyz_m"])
        arrival = time_s + max(1, math.ceil(trip / refinement["transfer_speed_mps"]))
        station.update(arrival_s=arrival, capture_s=arrival + 1)
        ordered.append(station)
        time_s, position = station["capture_s"], station["xyz_m"]
    return dict(start_s=math.ceil(now) + 1, start_xyz_m=finish, end_s=time_s,
                stations=ordered, patches=patches, radius_m=refinement["patch_radius_m"],
                strategy=strategy, dual_routes=dual_routes)


def oblique_pose(schedule, now):
    position, last_time = np.asarray(schedule["start_xyz_m"]), schedule["start_s"]
    for station in schedule["stations"]:
        target = np.asarray(station["xyz_m"])
        if now <= station["capture_s"]:
            progress = np.clip((now - last_time) / (station["arrival_s"] - last_time), 0, 1)
            return position + progress * (target - position), station["yaw_rad"], station["pitch_rad"]
        position, last_time = target, station["capture_s"]
    station = schedule["stations"][-1]
    return position, station["yaw_rad"], station["pitch_rad"]


def local_ground_planes(grid, baseline, survey):
    states = baseline["states"]
    known_ground = states == FREE
    near_floor = distance_transform_edt(~known_ground) * grid.resolution <= 1.25
    rows, cols = np.indices(grid.shape)
    xy = np.stack((cols + 0.5, rows + 0.5), axis=-1) * grid.resolution + grid.origin
    inside = np.zeros(grid.shape, dtype=bool)
    for patch in survey["patches"]:
        inside |= np.linalg.norm(xy - patch["xy_m"], axis=-1) <= survey["radius_m"]
    planes = np.full((*grid.shape, 3), np.nan)
    window = math.ceil(1.6 / grid.resolution)
    for cell in map(tuple, np.argwhere((states == UNKNOWN) & near_floor & inside)):
        row, col = cell
        patch = (slice(max(0, row - window), min(grid.shape[0], row + window + 1)),
                 slice(max(0, col - window), min(grid.shape[1], col + window + 1)))
        offsets = xy[patch] - xy[cell]
        neighbors = known_ground[patch] & (np.linalg.norm(offsets, axis=-1) <= 1.6)
        if neighbors.sum() < 5:
            continue
        offsets = offsets[neighbors]
        matrix = np.column_stack((offsets, np.ones(len(offsets))))
        z = baseline["ground_z_m"][patch][neighbors]
        coef, _, rank, _ = np.linalg.lstsq(matrix, z, rcond=None)
        if rank == 3 and np.median(np.abs(z - matrix @ coef)) <= 0.10:
            planes[cell] = coef
    return planes, xy


def check_oblique_frame(grid, planes, centers, points):
    """Split each 0.5 m cell into four quarters and distinguish floor/body-height hits."""
    indices = np.floor((points[:, :2] - grid.origin) / grid.resolution).astype(int)
    cols, rows = indices.T
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    rows, cols, points = rows[inside], cols[inside], points[inside]
    ids = rows * grid.shape[1] + cols
    size = math.prod(grid.shape)
    observed = np.bincount(ids, minlength=size).reshape(grid.shape) >= 8
    measured = np.isfinite(planes[rows, cols, 2])
    rows, cols, points, ids = rows[measured], cols[measured], points[measured], ids[measured]
    delta = points[:, :2] - centers[rows, cols]
    parameters = planes[rows, cols]
    centered_z = points[:, 2] - np.sum(delta * parameters[:, :2], axis=-1)
    relative_z = centered_z - parameters[:, 2]
    floor = np.abs(relative_z) <= 0.16
    quarter = (delta[:, 0] >= 0).astype(int) + 2 * (delta[:, 1] >= 0)
    quarter_votes = np.bincount(ids[floor] * 4 + quarter[floor], minlength=4 * size)
    floor_hits = np.bincount(ids[floor], minlength=size)
    z_sum = np.bincount(ids[floor], weights=centered_z[floor], minlength=size)
    z = np.divide(z_sum, floor_hits, where=floor_hits > 0, out=np.full(size, np.nan))
    # Persistently raised debris or returns below the local plane may be a step.
    interference = ((relative_z >= 0.22) & (relative_z <= 0.65)) | (relative_z < -0.26)
    body_hits = np.bincount(ids[interference], minlength=size).reshape(grid.shape) >= 3
    return (quarter_votes.reshape(*grid.shape, 4) >= 2, z.reshape(grid.shape), body_hits, observed)


def refine_ground(grid, baseline, point_views, camera_origins, slope_limit, survey):
    """Recover ground only with repeat evidence in all four quarters and clear body space."""
    result = {name: value.copy() for name, value in baseline.items()}
    promoted = np.zeros(grid.shape, dtype=bool)
    quarter_count = np.zeros(grid.shape, dtype=np.uint8)
    result.update(refined_ground=promoted, oblique_ground_quarters=quarter_count)
    if len(point_views) < 3:
        return result
    planes, centers = local_ground_planes(grid, baseline, survey)
    frames = [check_oblique_frame(grid, planes, centers, points) for points in point_views]
    quarter, elevation, obstruction, observed = (np.stack(values) for values in zip(*frames))
    result["observed_views"] += observed.sum(axis=0).astype(np.uint16)
    cameras = np.asarray(camera_origins)[:, :2]
    camera_separation = np.linalg.norm(cameras - cameras[:, None], axis=-1) >= 2.0
    candidates = np.argwhere(np.any(quarter.sum(axis=0) >= 2, axis=-1))
    confirmed_z = np.full(grid.shape, np.nan)
    support = np.zeros(grid.shape, dtype=np.uint16)
    spread = np.zeros(grid.shape, dtype=np.float32)
    for row, col in candidates:
        votes = quarter[:, row, col]
        qualified = 0
        for subcell in range(4):
            indices = np.flatnonzero(votes[:, subcell])
            if camera_separation[np.ix_(indices, indices)].any():
                qualified += 1
        quarter_count[row, col] = qualified
        used = np.any(votes, axis=-1)
        if qualified < 4 or used.sum() < 3 or obstruction[:, row, col].sum() >= 2:
            continue
        z = elevation[used, row, col]
        if np.ptp(z) <= 0.23:
            confirmed_z[row, col] = np.median(z)
            support[row, col] = len(z)
            spread[row, col] = np.ptp(z)
    new_z = np.where(np.isfinite(confirmed_z), confirmed_z, baseline["ground_z_m"])
    slope = ground_slope(new_z, grid.resolution)
    promoted[:] = (support >= 3) & np.isfinite(slope) & (slope <= slope_limit)
    result["states"][promoted] = FREE
    result["ground_z_m"][promoted] = new_z[promoted]
    result["slope_deg"][promoted] = slope[promoted]
    result["supporting_views"][promoted] = support[promoted]
    result["planning_cost"][promoted] = (1.2 + 1.1 * (slope[promoted] / slope_limit)**2
                                         + 0.35 * spread[promoted] / 0.23
                                         + 0.45 * np.maximum(0, 1 - support[promoted] / 6.0))
    return result


DEFAULT_DYNAMIC_REFINEMENT = dict(
    patch_radius_m=3.5,
    angle_from_vertical_deg=37.0,
    altitude_enu_m=16.5,
    transfer_speed_mps=8.0,
)


def select_dynamic_patches(
    grid, states, ground_z_m, start_xy, goal_xy, rover_radius_m,
    planning_cost=None, patch_radius_m=3.5, max_patches=2, unknown_cost=1.25,
    exclude_centers=(),
):
    """Select targeted unknown-corridor patches for mid-mission rover-triggered UAV refinement."""
    dual = plan_dual(
        grid, states, start_xy, goal_xy, rover_radius_m,
        planning_cost=planning_cost, unknown_cost=unknown_cost,
    )
    if dual["optimistic_status"] != "PLANNED":
        # If the rover stopped right next to a detected barrier, snap query start to nearest
        # non-BLOCKED-padded cell within 1.5 m so Traversing Mars can still find unknown bypasses.
        from grid_route import center_mask
        opt_states = np.where(states == 1, 1, FREE).astype(np.int8)
        opt_usable = center_mask(grid, opt_states, rover_radius_m)
        r0, c0 = grid.cell(start_xy)
        cands = []
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                cell = (r0 + dr, c0 + dc)
                if grid.contains(cell) and opt_usable[cell] and states[cell] == FREE:
                    d = math.dist(grid.center(cell), start_xy)
                    if d <= 1.6:
                        cands.append((d, grid.center(cell)))
        for _, anchor in sorted(cands, key=lambda item: item[0]):
            trial = plan_dual(
                grid, states, anchor, goal_xy, rover_radius_m,
                planning_cost=planning_cost, unknown_cost=unknown_cost,
            )
            if trial["optimistic_status"] == "PLANNED":
                dual = trial
                break
    summary = dict(
        conservative_status=dual["conservative_status"],
        conservative_length_m=dual["conservative_length_m"],
        optimistic_status=dual["optimistic_status"],
        optimistic_length_m=dual["optimistic_length_m"],
        changes_reachability=dual["changes_reachability"],
        potential_saving_m=dual["potential_saving_m"],
        corridor_unknown_cell_count=dual["corridor_unknown_cell_count"],
    )
    corridor = [tuple(rc) for rc in dual.get("corridor_unknown_cells", []) if states[tuple(rc)] == UNKNOWN]
    if dual["optimistic_status"] != "PLANNED" or not corridor:
        return [], summary

    rows, cols = np.indices(grid.shape)
    positions = np.stack((cols + 0.5, rows + 0.5), axis=-1) * grid.resolution + grid.origin
    free_mask = states == FREE
    near_floor = distance_transform_edt(~free_mask) * grid.resolution <= 2.0
    free_z_valid = free_mask & np.isfinite(ground_z_m)
    fallback_z = float(np.median(ground_z_m[free_z_valid])) if np.any(free_z_valid) else 0.0

    remaining = set(corridor)
    patches = []
    min_sep = 1.25 * float(patch_radius_m)

    for _ in range(int(max_patches)):
        if not remaining:
            break
        best = None
        for cell in sorted(remaining):
            center = grid.center(cell)
            if any(math.dist(center, p["xy_m"]) < min_sep for p in patches):
                continue
            if any(math.dist(center, ex) < 0.75 * patch_radius_m for ex in exclude_centers):
                continue
            covered = [c for c in remaining if math.dist(grid.center(c), center) <= patch_radius_m]
            if not covered:
                continue
            near_cnt = sum(1 for c in covered if near_floor[c])
            # Prefer patches with the most corridor unknown cells & strong contact with existing FREE floor.
            dist_to_rover = math.dist(center, start_xy)
            score = len(covered) + 0.35 * near_cnt - 0.02 * dist_to_rover
            if best is None or score > best[0]:
                best = (score, center, covered)
        if best is None:
            break
        _, center, covered = best
        offset = np.linalg.norm(positions - center, axis=-1)
        local_free = free_z_valid & (offset <= max(6.0, 2.0 * patch_radius_m))
        elev = float(np.median(ground_z_m[local_free])) if np.any(local_free) else fallback_z
        patches.append(dict(
            xy_m=[round(float(center[0]), 3), round(float(center[1]), 3)],
            elevation_m=round(elev, 3),
            corridor_cells=len(covered),
        ))
        for c in covered:
            remaining.discard(c)
    return patches, summary


def schedule_dynamic_oblique(
    grid, patches, refinement_cfg, current_drone_xyz, now,
    world_bounds=((-39.0, -27.5), (39.0, 27.5)),
):
    """Build a fast targeted oblique refinement flight from the drone's current position."""
    cfg = dict(DEFAULT_DYNAMIC_REFINEMENT)
    if refinement_cfg:
        cfg.update(refinement_cfg)
    altitude = float(cfg["altitude_enu_m"])
    angle_rad = math.radians(float(cfg["angle_from_vertical_deg"]))
    transfer_speed = max(2.0, float(cfg["transfer_speed_mps"]))
    radius_m = float(cfg["patch_radius_m"])
    (xmin, ymin), (xmax, ymax) = world_bounds

    stations = []
    for index, patch in enumerate(patches, start=1):
        target = np.asarray(patch["xy_m"], dtype=float)
        height = altitude - float(patch["elevation_m"])
        if height <= 2.0:
            continue
        distance = height * math.tan(angle_rad)
        for azimuth in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            view = np.array([math.cos(azimuth), math.sin(azimuth)])
            lateral = np.array([-view[1], view[0]])
            for spacing in (-2.4, 0.0, 2.4):
                position = target - distance * view + spacing * lateral
                if not (xmin <= position[0] <= xmax and ymin <= position[1] <= ymax):
                    continue
                delta = target - position
                stations.append(dict(
                    xyz_m=[float(position[0]), float(position[1]), altitude],
                    yaw_rad=float(math.atan2(delta[1], delta[0])),
                    pitch_rad=float(-math.atan2(np.linalg.norm(delta), height)),
                    patch=index,
                ))

    start_xyz = [float(current_drone_xyz[0]), float(current_drone_xyz[1]), float(current_drone_xyz[2])]
    start_s = math.ceil(now) + 1
    time_s = start_s
    position = start_xyz
    ordered = []
    while stations:
        station = min(stations, key=lambda item: math.dist(position, item["xyz_m"]))
        stations.remove(station)
        trip = math.dist(position, station["xyz_m"])
        arrival = time_s + max(1, math.ceil(trip / transfer_speed))
        station.update(arrival_s=arrival, capture_s=arrival + 1)
        ordered.append(station)
        time_s, position = station["capture_s"], station["xyz_m"]
    return dict(
        start_s=start_s,
        start_xyz_m=start_xyz,
        end_s=time_s,
        stations=ordered,
        patches=patches,
        radius_m=radius_m,
        strategy="dynamic_route_value",
    )


def refine_ground_iterative(grid, baseline, point_views, camera_origins, slope_limit, survey, max_passes=4):
    """Iteratively propagate verified oblique ground planes across multi-cell canopy gaps.

    Every promoted cell satisfies the exact same strict 4-quarter multi-view oblique,
    body-clearance, spread, and slope constraints as `refine_ground`, using only the
    already-captured oblique sensor frames.
    """
    current = {name: value.copy() for name, value in baseline.items()}
    total_promoted = np.zeros(grid.shape, dtype=bool)
    base_obs_views = current["observed_views"].copy()
    added_obs_views = np.zeros(grid.shape, dtype=np.uint16)
    for pass_idx in range(max(1, int(max_passes))):
        updated = refine_ground(grid, current, point_views, camera_origins, slope_limit, survey)
        if pass_idx == 0:
            added_obs_views = updated["observed_views"] - base_obs_views
        else:
            # Avoid double-counting observed_views across boundary propagation passes.
            updated["observed_views"] = base_obs_views + added_obs_views
        promoted_now = updated["refined_ground"]
        total_promoted |= promoted_now
        current = updated
        if not np.any(promoted_now):
            break
    current["refined_ground"] = total_promoted
    return current, total_promoted
