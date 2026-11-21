"""A* on an aerial prior, with unknown terrain excluded from route planning."""
import heapq
import math

import numpy as np
from scipy.ndimage import distance_transform_edt

from survey_map import BLOCKED, FREE, NEIGHBORS, UNKNOWN


def center_mask(grid, states, radius):
    # Grid cells represent squares. Subtract their half diagonal from the
    # distance to their centers to bound footprint clearance conservatively.
    open_cells = states == FREE
    distance = distance_transform_edt(np.pad(open_cells, 1))[1:-1, 1:-1]
    clearance = grid.resolution * (distance - math.sqrt(0.5))
    return open_cells & (clearance >= radius)


def plan(grid, states, start_xy, goal_xy, radius, planning_cost=None):
    usable = center_mask(grid, states, radius)
    if planning_cost is not None:
        if planning_cost.shape != states.shape:
            raise ValueError("Planning cost shape does not match the terrain grid.")
        valid_cost = planning_cost[usable]
        if not np.isfinite(valid_cost).all() or np.any(valid_cost < 1.0):
            raise ValueError("Traversable cells require finite planning costs of at least one.")
    start, goal = grid.cell(start_xy), grid.cell(goal_xy)
    for label, cell in (("Start", start), ("Goal", goal)):
        if not grid.contains(cell) or not usable[cell]:
            raise ValueError(f"{label} has no confirmed space for the rover footprint.")
    moves = (*NEIGHBORS, (-1, -1), (-1, 1), (1, -1), (1, 1))
    queue = [(math.dist(start, goal), 0.0, start)]
    costs = {start: 0.0}
    parents = {start: None}
    while queue:
        _, cost, cell = heapq.heappop(queue)
        if cost > costs[cell]:
            continue
        if cell == goal:
            path = []
            while cell is not None:
                path.append(grid.center(cell))
                cell = parents[cell]
            return np.asarray(path[::-1])
        for dr, dc in moves:
            other = (cell[0] + dr, cell[1] + dc)
            if not grid.contains(other) or not usable[other]:
                continue
            if dr and dc and not (usable[other[0], cell[1]] and usable[cell[0], other[1]]):
                continue
            weight = 1.0 if planning_cost is None else (planning_cost[cell] + planning_cost[other]) / 2
            candidate = cost + weight * math.hypot(dr, dc)
            if candidate >= costs.get(other, math.inf):
                continue
            costs[other] = candidate
            parents[other] = cell
            heapq.heappush(queue, (candidate + math.dist(other, goal), candidate, other))
    raise ValueError("No route on currently confirmed terrain.")


def direct_line_cells(grid, states, start, goal):
    """Describe why a straight-line route is unsuitable, using only the prior."""
    samples = max(2, math.ceil(math.dist(start, goal) * 10 / grid.resolution))
    cells = {grid.cell(xy) for xy in np.linspace(start, goal, samples)}
    return dict(blocked=int(sum(grid.contains(c) and states[c] == BLOCKED for c in cells)),
                unknown=int(sum(not grid.contains(c) or states[c] == UNKNOWN for c in cells)))


def _path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def _corridor_unknown_cells(grid, states, waypoints, radius):
    """Return sorted UNKNOWN cell indices (row, col) within rover footprint of waypoints."""
    pad = max(1, math.ceil(radius / grid.resolution))
    found = set()
    for xy in waypoints:
        r0, c0 = grid.cell(xy)
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                cell = (r0 + dr, c0 + dc)
                if not grid.contains(cell) or states[cell] != UNKNOWN:
                    continue
                if math.dist(grid.center(cell), xy) <= radius + 0.5 * math.sqrt(2.0) * grid.resolution:
                    found.add(cell)
    return sorted(found)


def plan_dual(grid, states, start_xy, goal_xy, radius, planning_cost=None, unknown_cost=1.25):
    """Traversing Mars dual-route search: conservative (confirmed) vs optimistic (unknown-allowed).

    Conservative route strictly excludes UNKNOWN and BLOCKED cells.
    Optimistic route excludes only BLOCKED cells and penalizes UNKNOWN traversal by `unknown_cost`,
    then extracts contiguous unconfirmed segments to guide DUCPP aerial refinement.
    """
    usable_cons = center_mask(grid, states, radius)
    cons_path, cons_error = None, None
    try:
        cons_path = plan(grid, states, start_xy, goal_xy, radius, planning_cost)
    except ValueError as err:
        cons_error = str(err)

    opt_states = np.where(states == BLOCKED, BLOCKED, FREE).astype(np.int8)
    if planning_cost is not None:
        opt_cost = np.where(
            states == FREE,
            np.where(np.isfinite(planning_cost), np.maximum(planning_cost, 1.0), 1.0),
            float(unknown_cost),
        ).astype(np.float32)
    else:
        opt_cost = np.where(states == FREE, 1.0, float(unknown_cost)).astype(np.float32)

    opt_path, opt_error = None, None
    try:
        opt_path = plan(grid, opt_states, start_xy, goal_xy, radius, opt_cost)
    except ValueError as err:
        opt_error = str(err)

    cons_len = _path_length(cons_path) if cons_path is not None else None
    opt_len = _path_length(opt_path) if opt_path is not None else None

    segments = []
    total_unconfirmed_wp = 0
    total_direct_unknown_wp = 0
    all_corridor_unknown = set()

    if opt_path is not None:
        current = []
        for point in opt_path:
            cell = grid.cell(point)
            if not (grid.contains(cell) and usable_cons[cell]):
                current.append(point)
                total_unconfirmed_wp += 1
                if grid.contains(cell) and states[cell] == UNKNOWN:
                    total_direct_unknown_wp += 1
            elif current:
                arr = np.asarray(current, dtype=float)
                unk_cells = _corridor_unknown_cells(grid, states, arr, radius)
                all_corridor_unknown.update(unk_cells)
                segments.append(dict(
                    start_xy_m=arr[0].tolist(),
                    end_xy_m=arr[-1].tolist(),
                    mid_xy_m=arr.mean(axis=0).tolist(),
                    waypoint_count=int(len(arr)),
                    unknown_cell_count=int(len(unk_cells)),
                    unknown_cells=[[r, c] for r, c in unk_cells],
                ))
                current = []
        if current:
            arr = np.asarray(current, dtype=float)
            unk_cells = _corridor_unknown_cells(grid, states, arr, radius)
            all_corridor_unknown.update(unk_cells)
            segments.append(dict(
                start_xy_m=arr[0].tolist(),
                end_xy_m=arr[-1].tolist(),
                mid_xy_m=arr.mean(axis=0).tolist(),
                waypoint_count=int(len(arr)),
                unknown_cell_count=int(len(unk_cells)),
                unknown_cells=[[r, c] for r, c in unk_cells],
            ))

    changes_reachability = cons_path is None and opt_path is not None
    potential_saving_m = (max(0.0, cons_len - opt_len)
                          if cons_len is not None and opt_len is not None else None)
    return dict(
        conservative_status="PLANNED" if cons_path is not None else "NO_ROUTE",
        conservative_length_m=cons_len,
        conservative_reason=cons_error,
        optimistic_status="PLANNED" if opt_path is not None else "NO_ROUTE",
        optimistic_length_m=opt_len,
        optimistic_reason=opt_error,
        optimistic_route_xy_m=opt_path.tolist() if opt_path is not None else None,
        optimistic_waypoint_count=int(len(opt_path)) if opt_path is not None else 0,
        unconfirmed_clearance_waypoints=int(total_unconfirmed_wp),
        direct_unknown_waypoints=int(total_direct_unknown_wp),
        corridor_unknown_cell_count=int(len(all_corridor_unknown)),
        corridor_unknown_cells=[[r, c] for r, c in sorted(all_corridor_unknown)],
        changes_reachability=changes_reachability,
        potential_saving_m=potential_saving_m,
        unconfirmed_segments=segments,
    )
