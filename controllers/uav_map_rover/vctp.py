"""VCTP-inspired four-wheel fine-terrain traversability check for ResearchRover.

Evaluates candidate vehicle poses (x, y, yaw) on a sensor-derived fine ground
surface (0.125 m resolution when available):
- four-wheel contact elevations (FL, FR, RL, RR),
- fitted support-plane pitch and roll,
- four-wheel twist / single-wheel step residual,
- chassis belly clearance against micro-obstacles (rocks, crests, steps).

Where fine ground evidence is missing (UNKNOWN / NaN), poses are kept UNKNOWN
rather than assumed passable, signalling the need for aerial refinement.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
from PIL import Image

from survey_map import FREE, Grid


# ResearchRover.proto four-wheel geometry (metres in body frame: +X forward, +Y left).
HALF_WHEELBASE_M = 0.150
HALF_TRACK_M = 0.205
WHEEL_RADIUS_M = 0.090
BELLY_HEIGHT_M = 0.090

WHEEL_OFFSETS_BODY = np.array([
    [+HALF_WHEELBASE_M, +HALF_TRACK_M],  # FL
    [+HALF_WHEELBASE_M, -HALF_TRACK_M],  # FR
    [-HALF_WHEELBASE_M, +HALF_TRACK_M],  # RL
    [-HALF_WHEELBASE_M, -HALF_TRACK_M],  # RR
], dtype=float)

BELLY_OFFSETS_BODY = np.array([
    [bx, by]
    for bx in (-0.21, 0.0, 0.21)
    for by in (-0.14, 0.0, 0.14)
], dtype=float)


@dataclass(frozen=True)
class VctpResult:
    status: str  # "PASS" | "FAIL" | "UNKNOWN"
    reason: str
    pitch_deg: float
    roll_deg: float
    wheel_twist_m: float
    max_wheel_step_m: float
    max_belly_protrusion_m: float
    cost: float


class FineTerrainSurface:
    """Sensor-derived fine elevation layer gated strictly by confirmed FREE cells."""

    def __init__(self, fine_grid: Grid, fine_z_m: np.ndarray, coarse_grid: Grid,
                 coarse_states: np.ndarray, source: str = "uav_map_fine"):
        self.fine_grid = fine_grid
        self.fine_z_m = np.asarray(fine_z_m, dtype=float)
        self.coarse_grid = coarse_grid
        self.coarse_states = np.asarray(coarse_states, dtype=np.int8)
        self.source = source
        # Pre-mask fine elevation strictly to confirmed FREE coarse cells for fast O(1) lookup.
        sub_r = max(1, self.fine_grid.shape[0] // self.coarse_grid.shape[0])
        sub_c = max(1, self.fine_grid.shape[1] // self.coarse_grid.shape[1])
        if (self.coarse_grid.shape[0] * sub_r == self.fine_grid.shape[0]
                and self.coarse_grid.shape[1] * sub_c == self.fine_grid.shape[1]):
            free_mask = np.repeat(np.repeat(self.coarse_states == FREE, sub_r, axis=0), sub_c, axis=1)
            self._free_z = np.where(free_mask, self.fine_z_m, np.nan)
        else:
            self._free_z = self.fine_z_m.copy()
        self._ox = float(self.fine_grid.origin[0])
        self._oy = float(self.fine_grid.origin[1])
        self._inv_res = 1.0 / float(self.fine_grid.resolution)
        self._rows, self._cols = self.fine_grid.shape

    @classmethod
    def from_uav_map(cls, map_path, dsm_fallback_path=None):
        map_path = Path(map_path)
        with np.load(map_path, allow_pickle=False) as data:
            states = data["states"]
            origin = tuple(map(float, data["origin_xy_m"]))
            res = float(data["resolution_m"])
            coarse_grid = Grid(origin, res, states.shape)
            coarse_z = data["ground_z_m"].astype(float)

            if "fine_ground_z_m" in data:
                fine_z = data["fine_ground_z_m"].astype(float)
                fine_res = float(data["fine_resolution_m"]) if "fine_resolution_m" in data else res / (fine_z.shape[0] / states.shape[0])
                fine_grid = Grid(origin, fine_res, fine_z.shape)
                return cls(fine_grid, fine_z, coarse_grid, states, source="uav_map_fine")

        # Fallback: load 0.125 m sensor-derived DSM from aerial_products if present,
        # strictly gated by confirmed FREE cells and within 0.45 m of coarse ground.
        dsm_path = Path(dsm_fallback_path) if dsm_fallback_path else map_path.parent / "aerial_products/surface_dsm.tif"
        if dsm_path.is_file():
            with Image.open(dsm_path) as img:
                dsm = np.flipud(np.asarray(img, dtype=float))
            dsm[dsm < -9000.0] = np.nan
            subdiv = dsm.shape[0] // states.shape[0]
            if subdiv >= 1 and dsm.shape == (states.shape[0] * subdiv, states.shape[1] * subdiv):
                free_up = np.repeat(np.repeat(states == FREE, subdiv, axis=0), subdiv, axis=1)
                cz_up = np.repeat(np.repeat(coarse_z, subdiv, axis=0), subdiv, axis=1)
                near_ground = free_up & np.isfinite(dsm) & np.isfinite(cz_up) & (np.abs(dsm - cz_up) <= 0.45)
                # Under overhanging high canopy (> 0.90 m above confirmed FREE ground),
                # DSM records canopy top rather than the already-confirmed ground; use confirmed coarse z there.
                overhanging = free_up & np.isfinite(dsm) & np.isfinite(cz_up) & (dsm > cz_up + 0.90)
                fine_z = np.where(near_ground, dsm, np.where(overhanging, cz_up, np.nan))
                fine_grid = Grid(origin, res / subdiv, dsm.shape)
                return cls(fine_grid, fine_z, coarse_grid, states, source="aerial_products_dsm")

        return cls(coarse_grid, np.where(states == FREE, coarse_z, np.nan),
                   coarse_grid, states, source="coarse_ground_z")

    def sample_z(self, xy):
        """Sample fine ground height at ENU (N, 2) points; non-FREE or missing -> NaN."""
        pts = np.atleast_2d(np.asarray(xy, dtype=float))
        n = len(pts)
        if n == 0:
            return np.empty(0, dtype=float)

        c_f = (pts[:, 0] - self._ox) * self._inv_res - 0.5
        r_f = (pts[:, 1] - self._oy) * self._inv_res - 0.5
        r0 = np.floor(r_f).astype(int)
        c0 = np.floor(c_f).astype(int)
        dr = r_f - r0
        dc = c_f - c0

        out = np.full(n, np.nan, dtype=float)
        in_b = (r0 >= 0) & (r0 + 1 < self._rows) & (c0 >= 0) & (c0 + 1 < self._cols)
        if np.any(in_b):
            rr, cc2 = r0[in_b], c0[in_b]
            z00 = self._free_z[rr, cc2]
            z01 = self._free_z[rr, cc2 + 1]
            z10 = self._free_z[rr + 1, cc2]
            z11 = self._free_z[rr + 1, cc2 + 1]
            ok = np.isfinite(z00) & np.isfinite(z01) & np.isfinite(z10) & np.isfinite(z11)
            idx = np.flatnonzero(in_b)[ok]
            if idx.size:
                a = dc[idx]
                b = dr[idx]
                out[idx] = ((1.0 - b) * ((1.0 - a) * z00[ok] + a * z01[ok])
                            + b * ((1.0 - a) * z10[ok] + a * z11[ok]))

        missing = ~np.isfinite(out)
        if np.any(missing):
            rn = np.floor((pts[missing, 1] - self._oy) * self._inv_res).astype(int)
            cn = np.floor((pts[missing, 0] - self._ox) * self._inv_res).astype(int)
            ins = (rn >= 0) & (rn < self._rows) & (cn >= 0) & (cn < self._cols)
            idx = np.flatnonzero(missing)[ins]
            if idx.size:
                out[idx] = self._free_z[rn[ins], cn[ins]]
        return out


class VctpChecker:
    """Four-wheel kinematic/geometric contact & belly-clearance checker."""

    def __init__(self, surface: FineTerrainSurface, max_pitch_deg=18.0, max_roll_deg=16.0,
                 max_wheel_twist_m=0.085, max_wheel_step_m=0.16, max_belly_protrusion_m=0.075):
        self.surface = surface
        self.max_pitch_deg = float(max_pitch_deg)
        self.max_roll_deg = float(max_roll_deg)
        self.max_wheel_twist_m = float(max_wheel_twist_m)
        self.max_wheel_step_m = float(max_wheel_step_m)
        self.max_belly_protrusion_m = float(max_belly_protrusion_m)

    def check_pose(self, x, y, yaw) -> VctpResult:
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=float)
        center = np.array([x, y], dtype=float)

        wheel_enu = center + WHEEL_OFFSETS_BODY @ rot.T
        belly_enu = center + BELLY_OFFSETS_BODY @ rot.T
        all_z = self.surface.sample_z(np.vstack((wheel_enu, belly_enu)))
        zw = all_z[:4]
        zb = all_z[4:]

        if not np.isfinite(zw).all() or int(np.count_nonzero(~np.isfinite(zb))) > 1:
            return VctpResult("UNKNOWN", "INSUFFICIENT_GROUND_EVIDENCE",
                              math.nan, math.nan, math.nan, math.nan, math.nan, math.inf)

        z_fl, z_fr, z_rl, z_rr = zw
        plane_c = 0.25 * (z_fl + z_fr + z_rl + z_rr)
        dz_dx = ((z_fl + z_fr) - (z_rl + z_rr)) / (4.0 * HALF_WHEELBASE_M)
        dz_dy = ((z_fl + z_rl) - (z_fr + z_rr)) / (4.0 * HALF_TRACK_M)

        pitch_deg = math.degrees(-math.atan(dz_dx))
        roll_deg = math.degrees(math.atan(dz_dy))
        wheel_twist_m = 0.25 * abs((z_fl + z_rr) - (z_fr + z_rl))
        max_wheel_step_m = max(
            abs(z_fl - z_rl), abs(z_fr - z_rr), abs(z_fl - z_fr), abs(z_rl - z_rr)
        )

        plane_belly = plane_c + BELLY_OFFSETS_BODY[:, 0] * dz_dx + BELLY_OFFSETS_BODY[:, 1] * dz_dy
        valid_b = np.isfinite(zb)
        protrusions = zb[valid_b] - plane_belly[valid_b]
        max_belly_protrusion_m = float(np.max(protrusions)) if protrusions.size else 0.0

        reasons = []
        if abs(pitch_deg) > self.max_pitch_deg:
            reasons.append(f"EXCESSIVE_PITCH({pitch_deg:.1f}deg)")
        if abs(roll_deg) > self.max_roll_deg:
            reasons.append(f"EXCESSIVE_ROLL({roll_deg:.1f}deg)")
        if wheel_twist_m > self.max_wheel_twist_m:
            reasons.append(f"WHEEL_TWIST_STEP({wheel_twist_m:.3f}m)")
        if max_wheel_step_m > self.max_wheel_step_m:
            reasons.append(f"WHEEL_DROP_STEP({max_wheel_step_m:.3f}m)")
        if max_belly_protrusion_m > self.max_belly_protrusion_m:
            reasons.append(f"BELLY_CLEARANCE({max_belly_protrusion_m:.3f}m)")

        if reasons:
            return VctpResult("FAIL", "; ".join(reasons),
                              pitch_deg, roll_deg, wheel_twist_m, max_wheel_step_m,
                              max_belly_protrusion_m, math.inf)

        cost = (
            1.0
            + 0.85 * (abs(pitch_deg) / self.max_pitch_deg) ** 2
            + 1.05 * (abs(roll_deg) / self.max_roll_deg) ** 2
            + 0.55 * (wheel_twist_m / self.max_wheel_twist_m) ** 2
            + 0.75 * (max(0.0, max_belly_protrusion_m) / self.max_belly_protrusion_m) ** 2
        )
        return VctpResult("PASS", "OK", pitch_deg, roll_deg, wheel_twist_m,
                          max_wheel_step_m, max_belly_protrusion_m, float(cost))

    def audit_route(self, route_xy, yaws=None):
        """Evaluate every pose along a route with VCTP and compute wheel-cell diversity."""
        pts = np.asarray(route_xy, dtype=float)
        if len(pts) == 0:
            return dict(checked_poses=0, pass_count=0, fail_count=0, unknown_count=0, pass_rate=0.0)
        if yaws is None:
            yaws = []
            for i in range(len(pts)):
                if i + 1 < len(pts):
                    d = pts[i + 1] - pts[i]
                elif i > 0:
                    d = pts[i] - pts[i - 1]
                else:
                    d = np.array([1.0, 0.0])
                yaws.append(math.atan2(d[1], d[0]))

        passes = fails = unknowns = 0
        max_pitch = max_roll = max_twist = max_step = max_protrusion = 0.0
        same_coarse = 0
        distinct_fine = 0
        first_failure = None

        for idx, ((x, y), yaw) in enumerate(zip(pts, yaws)):
            res = self.check_pose(x, y, yaw)
            c, s = math.cos(yaw), math.sin(yaw)
            rot = np.array([[c, -s], [s, c]], dtype=float)
            wheel_enu = np.array([x, y]) + WHEEL_OFFSETS_BODY @ rot.T
            coarse_cells = {self.surface.coarse_grid.cell(w) for w in wheel_enu}
            fine_cells = {self.surface.fine_grid.cell(w) for w in wheel_enu}
            if len(coarse_cells) == 1:
                same_coarse += 1
            if len(fine_cells) == 4:
                distinct_fine += 1

            if res.status == "PASS":
                passes += 1
            elif res.status == "FAIL":
                fails += 1
                if first_failure is None:
                    first_failure = dict(index=idx, xy_m=[round(float(x), 3), round(float(y), 3)],
                                         yaw_rad=round(float(yaw), 3), reason=res.reason)
            else:
                unknowns += 1
                if first_failure is None:
                    first_failure = dict(index=idx, xy_m=[round(float(x), 3), round(float(y), 3)],
                                         yaw_rad=round(float(yaw), 3), reason=res.reason)
            if math.isfinite(res.pitch_deg):
                max_pitch = max(max_pitch, abs(res.pitch_deg))
                max_roll = max(max_roll, abs(res.roll_deg))
                max_twist = max(max_twist, res.wheel_twist_m)
                max_step = max(max_step, res.max_wheel_step_m)
                max_protrusion = max(max_protrusion, res.max_belly_protrusion_m)

        n = len(pts)
        return dict(
            surface_source=self.surface.source,
            fine_resolution_m=self.surface.fine_grid.resolution,
            checked_poses=n,
            pass_count=passes,
            fail_count=fails,
            unknown_count=unknowns,
            pass_rate=round(passes / n, 4),
            all_passed=bool(passes == n),
            max_abs_pitch_deg=round(max_pitch, 2),
            max_abs_roll_deg=round(max_roll, 2),
            max_wheel_twist_m=round(max_twist, 4),
            max_wheel_step_m=round(max_step, 4),
            max_belly_protrusion_m=round(max_protrusion, 4),
            all_four_wheels_same_coarse_cell_count=same_coarse,
            all_four_wheels_distinct_fine_cells_count=distinct_fine,
            first_failure=first_failure,
        )
