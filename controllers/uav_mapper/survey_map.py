"""Build a conservative, sensor-derived terrain prior in the Webots ENU frame."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import distance_transform_edt


UNKNOWN, FREE, BLOCKED = -1, 0, 1
NEIGHBORS = ((0, 1), (1, 0), (0, -1), (-1, 0))


@dataclass(frozen=True)
class Grid:
    origin: tuple
    resolution: float
    shape: tuple

    @classmethod
    def from_bounds(cls, bounds, resolution):
        origin, upper = np.asarray(bounds, dtype=float)
        size = (upper - origin) / resolution
        if np.any(size <= 0) or not np.allclose(size, np.round(size)):
            raise ValueError("Map bounds must be a positive integer number of cells.")
        return cls(tuple(origin), resolution, tuple(np.round(size[::-1]).astype(int)))

    def cell(self, xy):
        return tuple(np.floor((np.asarray(xy) - self.origin) / self.resolution)
                     .astype(int)[::-1])

    def center(self, cell):
        return np.asarray(self.origin) + (np.asarray(cell)[::-1] + 0.5) * self.resolution

    def contains(self, cell):
        row, col = cell
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]


def sampled_pixels(depth, stride, min_range, max_range):
    """Return selected original pixel coordinates and their optical-axis depths."""
    valid = np.isfinite(depth) & (depth > min_range) & (depth < max_range)
    rows, cols = np.nonzero(valid)
    return np.column_stack((cols * stride, rows * stride, depth[valid])), valid


def binned_surfaces(grid, points, min_points=8):
    """One lower and one upper elevation estimate per cell and per observation."""
    lower = np.full(grid.shape, np.nan, dtype=np.float32)
    upper = lower.copy()
    points = points[np.isfinite(points).all(axis=1)]
    cols, rows = np.floor((points[:, :2] - grid.origin) / grid.resolution).astype(int).T
    inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
    cells = rows[inside] * grid.shape[1] + cols[inside]
    z = points[inside, 2]
    order = np.lexsort((z, cells))
    ids, first, counts = np.unique(cells[order], return_index=True, return_counts=True)
    use = counts >= min_points
    ids, first, counts = ids[use], first[use], counts[use]
    z = z[order]
    lower.flat[ids] = z[first + (0.2 * (counts - 1)).astype(int)]
    upper.flat[ids] = z[first + (0.8 * (counts - 1)).astype(int)]
    return lower, upper


def binned_fine_surface(grid, points, subdiv=4, min_points=2):
    """Fine-resolution (default 0.125 m) per-frame surface elevation for VCTP."""
    fine_shape = (grid.shape[0] * subdiv, grid.shape[1] * subdiv)
    fine_res = grid.resolution / subdiv
    surface = np.full(fine_shape, np.nan, dtype=np.float32)
    if len(points) == 0:
        return surface
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return surface
    cols, rows = np.floor((points[:, :2] - grid.origin) / fine_res).astype(int).T
    inside = (rows >= 0) & (rows < fine_shape[0]) & (cols >= 0) & (cols < fine_shape[1])
    if not np.any(inside):
        return surface
    cells = rows[inside] * fine_shape[1] + cols[inside]
    z = points[inside, 2]
    order = np.lexsort((z, cells))
    ids, first, counts = np.unique(cells[order], return_index=True, return_counts=True)
    use = counts >= min_points
    ids, first, counts = ids[use], first[use], counts[use]
    z = z[order]
    # 75th percentile retains micro-obstacles (rocks/steps) for VCTP belly-clearance checks.
    surface.flat[ids] = z[first + (0.75 * (counts - 1)).astype(int)]
    return surface


def compose_fine_ground(grid, fine_views, states, coarse_ground_z, subdiv=4, max_offset_m=0.45):
    """Fuse multi-view fine surfaces on confirmed FREE cells; keep UNKNOWN/BLOCKED as NaN."""
    fine_shape = (grid.shape[0] * subdiv, grid.shape[1] * subdiv)
    if not fine_views:
        return np.full(fine_shape, np.nan, dtype=np.float32)
    coarse_free = np.repeat(np.repeat(states == FREE, subdiv, axis=0), subdiv, axis=1)
    coarse_z_up = np.repeat(np.repeat(coarse_ground_z, subdiv, axis=0), subdiv, axis=1)
    stack = np.stack(fine_views)
    # Reject high overhanging canopy returns far above confirmed coarse ground.
    in_band = np.isfinite(stack) & coarse_free[None, :, :] & np.isfinite(coarse_z_up[None, :, :])
    in_band &= np.abs(stack - coarse_z_up[None, :, :]) <= max_offset_m
    masked = np.ma.array(stack, mask=~in_band)
    fine_z = np.ma.median(masked, axis=0).filled(np.nan).astype(np.float32)
    fine_z[~coarse_free] = np.nan
    return fine_z


def connected_ground(grid, surface, stable, seed_xy, max_slope_deg=17.0):
    """Grow ground support from visible soil references through continuous observations."""
    ground = np.zeros(grid.shape, dtype=bool)
    pending = []
    for point in np.atleast_2d(seed_xy):
        seed = grid.cell(point)
        if grid.contains(seed) and stable[seed] and not ground[seed]:
            ground[seed] = True
            pending.append(seed)
    if not pending:
        raise ValueError("Ground references lack stable measurements. Choose a visible soil cell.")
    max_rise = grid.resolution * math.tan(math.radians(max_slope_deg)) + 0.045
    while pending:
        cell = pending.pop()
        for dr, dc in NEIGHBORS:
            other = (cell[0] + dr, cell[1] + dc)
            if not grid.contains(other) or ground[other] or not stable[other]:
                continue
            if abs(surface[other] - surface[cell]) <= max_rise:
                ground[other] = True
                pending.append(other)
    return ground


def ground_slope(elevation, resolution):
    """Slope from measured neighbours, with missing derivatives kept unknown."""
    padded = np.pad(elevation, 1, constant_values=np.nan)
    east, west = padded[1:-1, 2:], padded[1:-1, :-2]
    north, south = padded[2:, 1:-1], padded[:-2, 1:-1]

    def derivative(positive, negative):
        return np.where(np.isfinite(positive) & np.isfinite(negative),
                        (positive - negative) * 0.5,
                        np.where(np.isfinite(positive), positive - elevation, elevation - negative))

    rise = np.hypot(derivative(east, west), derivative(north, south))
    return np.where(np.isfinite(elevation), np.degrees(np.arctan(rise / resolution)), np.nan)


def classify(grid, lower_views, upper_views, seed_xy, max_slope_deg=None):
    """Retain areas with insufficient or inconsistent ground evidence as unknown."""
    lower = np.stack(lower_views)
    upper = np.stack(upper_views)
    surface = np.ma.median(np.ma.masked_invalid(lower), axis=0).filled(np.nan)
    view_count = np.sum(np.abs(lower - surface) <= 0.17, axis=0)
    relief = np.ma.median(np.ma.masked_invalid(upper - lower), axis=0).filled(np.nan)
    stable = (view_count >= 2) & (relief <= 0.25)
    ground = connected_ground(grid, surface, stable, seed_xy, 17.0 if max_slope_deg is None else 31.0)

    # Report short, well observed steps/rough surfaces beside established ground.
    # A tall return alone (e.g. canopy or roof above occluded ground) stays unknown.
    distance, nearest = distance_transform_edt(~ground, return_indices=True)
    base = surface[tuple(nearest)]
    offset = surface - base
    step = (offset < -0.26) | ((offset > 0.26) & (offset <= 1.10))
    rough = (np.abs(offset) <= 0.22) & (relief > 0.25) & (relief <= 0.85)
    near_ground = distance <= math.sqrt(2)
    hazards = ~ground & near_ground & (view_count >= 2) & (step | rough)
    ground_z = np.where(ground, surface, np.nan)
    slope_deg = ground_slope(ground_z, grid.resolution)
    steep = np.zeros(grid.shape, dtype=bool) if max_slope_deg is None else slope_deg > max_slope_deg
    states = np.full(grid.shape, UNKNOWN, dtype=np.int8)
    states[ground & ~steep] = FREE
    states[hazards | steep] = BLOCKED
    observed_views = np.count_nonzero(np.isfinite(lower), axis=0)
    hazard_type = np.zeros(grid.shape, dtype=np.uint8)
    hazard_type[steep] = 1
    hazard_type[hazards & step] = 2
    hazard_type[hazards & rough] = 3
    limits = 17.0 if max_slope_deg is None else max_slope_deg
    grade = np.clip(np.nan_to_num(slope_deg, nan=limits) / limits, 0.0, 1.0)
    texture_risk = np.clip(relief / 0.25, 0.0, 1.0)
    uncertainty = np.clip(1 - view_count / 6.0, 0.0, 1.0)
    cost = 1.0 + 1.1 * grade ** 2 + 0.35 * texture_risk + 0.45 * uncertainty
    return dict(states=states, ground_z_m=ground_z, surface_z_m=surface, slope_deg=slope_deg,
                surface_relief_m=relief, observed_views=observed_views.astype(np.uint16),
                supporting_views=view_count.astype(np.uint16), hazard_type=hazard_type,
                planning_cost=np.where(states == FREE, cost, np.nan).astype(np.float32))


def save_prior(path, grid, layers, timestamp):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, version=1, frame="ENU", origin_xy_m=grid.origin,
                            resolution_m=grid.resolution, timestamp_s=timestamp, **layers)
    temporary.replace(path)


def load_prior(path):
    with np.load(path, allow_pickle=False) as data:
        if data["version"].item() != 1 or data["frame"].item() != "ENU":
            raise ValueError("Unsupported terrain map version or frame.")
        states = data["states"]
        origin = data["origin_xy_m"]
        resolution = float(data["resolution_m"])
        if (states.ndim != 2 or not states.size or not np.isin(states, (UNKNOWN, FREE, BLOCKED)).all()
                or origin.shape != (2,) or not np.isfinite(origin).all()
                or not math.isfinite(resolution) or resolution <= 0):
            raise ValueError("Invalid terrain map grid.")
        return Grid(tuple(origin), resolution, states.shape), states
