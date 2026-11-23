"""Convert a real UAV photogrammetry DSM GeoTIFF (e.g. WebODM / DJI Mini 4 Pro) into uav_map.npz.

Allows real-world aerial surveys to feed directly into the same VCTP + IGHA* planning stack
validated in Webots, without requiring any external GIS dependencies beyond Pillow + NumPy + SciPy.
"""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, median_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "controllers/uav_mapper"))
sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))

from igha_star import plan_igha_star
from survey_map import BLOCKED, FREE, UNKNOWN, Grid, connected_ground, ground_slope, save_prior
from vctp import FineTerrainSurface, VctpChecker


def _read_geotiff_dsm(path: Path, pixel_size_m: float | None = None, nodata: float = -9999.0):
    with Image.open(path) as img:
        tags = getattr(img, "tag_v2", {})
        scale_tag = tags.get(33550)  # ModelPixelScaleTag: (sx, sy, sz)
        if pixel_size_m is None:
            if scale_tag and len(scale_tag) >= 2 and float(scale_tag[0]) > 0:
                pixel_size_m = float(scale_tag[0])
            else:
                raise ValueError(
                    "GeoTIFF lacks ModelPixelScaleTag (33550); please pass --pixel-size-m explicitly."
                )
        arr = np.array(img, dtype=np.float32, copy=True)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = np.flipud(arr)  # Image top-down -> ENU south-to-north row indexing.
    arr[~np.isfinite(arr) | (arr <= nodata + 1.0) | (arr < -1000.0)] = np.nan
    return arr, float(pixel_size_m)


def _resample_nan(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    h, w = arr.shape
    oh, ow = out_shape
    ry = np.linspace(0, h - 1, oh)
    rx = np.linspace(0, w - 1, ow)
    r0 = np.clip(np.round(ry).astype(int), 0, h - 1)
    c0 = np.clip(np.round(rx).astype(int), 0, w - 1)
    return arr[np.ix_(r0, c0)].astype(np.float32)


def dsm_to_uav_map(
    dsm_path: Path,
    output_dir: Path,
    pixel_size_m: float | None = None,
    coarse_res_m: float = 0.50,
    subdiv: int = 4,
    max_slope_deg: float = 17.0,
    seed_xy: tuple[float, float] | None = None,
    origin_xy: tuple[float, float] = (0.0, 0.0),
):
    """Build a VCTP-compatible uav_map.npz from a real-world photogrammetry DSM GeoTIFF."""
    raw_dsm, src_res = _read_geotiff_dsm(dsm_path, pixel_size_m=pixel_size_m)
    extent_y = raw_dsm.shape[0] * src_res
    extent_x = raw_dsm.shape[1] * src_res
    coarse_rows = max(4, int(math.floor(extent_y / coarse_res_m)))
    coarse_cols = max(4, int(math.floor(extent_x / coarse_res_m)))
    fine_shape = (coarse_rows * subdiv, coarse_cols * subdiv)
    fine_res = coarse_res_m / subdiv

    fine_raw = _resample_nan(raw_dsm, fine_shape)
    # Aggregate fine cells into coarse blocks for robust lower/upper elevation & relief.
    blocks = fine_raw.reshape(coarse_rows, subdiv, coarse_cols, subdiv).transpose(0, 2, 1, 3)
    flat_blocks = blocks.reshape(coarse_rows, coarse_cols, subdiv * subdiv)
    valid_cnt = np.sum(np.isfinite(flat_blocks), axis=2)
    masked = np.ma.masked_invalid(flat_blocks)
    lower = np.ma.array(np.nanpercentile(np.where(valid_cnt[:, :, None] > 0, flat_blocks, np.nan), 20, axis=2)).filled(np.nan).astype(np.float32)
    upper = np.ma.array(np.nanpercentile(np.where(valid_cnt[:, :, None] > 0, flat_blocks, np.nan), 80, axis=2)).filled(np.nan).astype(np.float32)
    lower[valid_cnt < 2] = np.nan
    upper[valid_cnt < 2] = np.nan
    relief = np.where(np.isfinite(lower) & np.isfinite(upper), upper - lower, np.nan).astype(np.float32)

    grid = Grid(tuple(map(float, origin_xy)), float(coarse_res_m), (coarse_rows, coarse_cols))
    stable = np.isfinite(lower) & (relief <= 0.25)

    if seed_xy is None:
        # Automatically pick the flattest stable cell near the center of observed terrain.
        slope_est = ground_slope(np.where(stable, lower, np.nan), coarse_res_m)
        cand = stable & np.isfinite(slope_est) & (slope_est <= 8.0)
        if not np.any(cand):
            cand = stable
        if not np.any(cand):
            raise ValueError("DSM contains no valid stable ground cells.")
        rr, cc = np.nonzero(cand)
        cy, cx = coarse_rows / 2.0, coarse_cols / 2.0
        best = int(np.argmin((rr - cy) ** 2 + (cc - cx) ** 2))
        seed_xy = tuple(grid.center((int(rr[best]), int(cc[best]))).tolist())

    ground = connected_ground(grid, lower, stable, seed_xy, max_slope_deg=max(25.0, max_slope_deg + 8.0))
    distance, nearest = distance_transform_edt(~ground, return_indices=True)
    base = lower[tuple(nearest)]
    offset = lower - base
    step = (offset < -0.26) | ((offset > 0.26) & (offset <= 1.50))
    rough = (np.abs(offset) <= 0.22) & (relief > 0.25) & (relief <= 1.00)
    hazards = ~ground & (distance <= math.sqrt(2)) & np.isfinite(lower) & (step | rough)

    ground_z = np.where(ground, lower, np.nan).astype(np.float32)
    slope_deg = ground_slope(ground_z, coarse_res_m).astype(np.float32)
    steep = np.isfinite(slope_deg) & (slope_deg > max_slope_deg)

    states = np.full(grid.shape, UNKNOWN, dtype=np.int8)
    states[ground & ~steep] = FREE
    states[hazards | steep] = BLOCKED

    coarse_free = np.repeat(np.repeat(states == FREE, subdiv, axis=0), subdiv, axis=1)
    coarse_z_up = np.repeat(np.repeat(ground_z, subdiv, axis=0), subdiv, axis=1)
    fine_z = np.where(coarse_free & np.isfinite(fine_raw) & (np.abs(fine_raw - coarse_z_up) <= 0.45),
                      fine_raw, np.nan).astype(np.float32)
    # Mild 3x3 despiking for single-pixel photogrammetry outliers while preserving micro-obstacles.
    finite_mask = np.isfinite(fine_z)
    if np.any(finite_mask):
        filled = np.where(finite_mask, fine_z, np.nanmedian(fine_z))
        med = median_filter(filled, size=3)
        fine_z = np.where(finite_mask & (np.abs(fine_z - med) > 0.20), med, fine_z).astype(np.float32)

    grade = np.clip(np.nan_to_num(slope_deg, nan=max_slope_deg) / max_slope_deg, 0.0, 1.0)
    texture_risk = np.clip(np.nan_to_num(relief, nan=0.25) / 0.25, 0.0, 1.0)
    cost = (1.0 + 1.1 * grade ** 2 + 0.35 * texture_risk).astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / "uav_map.npz"
    layers = dict(
        states=states,
        ground_z_m=ground_z,
        surface_z_m=lower,
        slope_deg=slope_deg,
        surface_relief_m=relief,
        observed_views=np.where(np.isfinite(lower), 4, 0).astype(np.uint16),
        supporting_views=np.where(stable, 4, 0).astype(np.uint16),
        hazard_type=np.where(steep, 1, np.where(hazards & step, 2, np.where(hazards & rough, 3, 0))).astype(np.uint8),
        planning_cost=np.where(states == FREE, cost, np.nan).astype(np.float32),
        fine_ground_z_m=fine_z,
        fine_resolution_m=np.float32(fine_res),
    )
    save_prior(map_path, grid, layers, timestamp=0.0)
    meta = dict(
        source_dsm=str(dsm_path),
        source_pixel_size_m=src_res,
        coarse_resolution_m=coarse_res_m,
        fine_resolution_m=fine_res,
        grid_shape=list(grid.shape),
        origin_xy_m=list(grid.origin),
        seed_xy_m=[round(float(seed_xy[0]), 3), round(float(seed_xy[1]), 3)],
        free_cells=int(np.sum(states == FREE)),
        blocked_cells=int(np.sum(states == BLOCKED)),
        unknown_cells=int(np.sum(states == UNKNOWN)),
    )
    (output_dir / "real_dsm_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return map_path, meta


def plan_and_preview_real_map(map_path: Path, start_xy, goal_xy, out_png: Path, rover_radius_m: float = 0.40):
    surface = FineTerrainSurface.from_uav_map(map_path)
    vctp = VctpChecker(surface)
    res = plan_igha_star(
        surface.coarse_grid, surface.coarse_states, vctp,
        start_xy=start_xy, goal_xy=goal_xy, rover_radius_m=rover_radius_m,
    )
    states = surface.coarse_states
    palette = {UNKNOWN: (84, 94, 107), FREE: (139, 178, 144), BLOCKED: (217, 104, 74)}
    rgb = np.zeros((*states.shape, 3), dtype=np.uint8)
    for st, col in palette.items():
        rgb[states == st] = col
    img = Image.fromarray(np.flipud(rgb)).resize((states.shape[1] * 6, states.shape[0] * 6), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (max(760, img.width + 48), img.height + 120), "#111923")
    canvas.paste(img, (24, 78))
    dr = ImageDraw.Draw(canvas)
    try:
        f_title = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 20)
        f_sub = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 14)
    except Exception:
        f_title = ImageFont.load_default()
        f_sub = ImageFont.load_default()

    def to_px(xy):
        e, n = (np.asarray(xy, dtype=float) - surface.coarse_grid.origin) / (np.asarray(states.shape[::-1]) * surface.coarse_grid.resolution)
        return 24 + int(round(img.width * e)), 78 + int(round(img.height * (1.0 - n)))

    if res.get("status") == "PLANNED":
        pts = [to_px(p) for p in res["route_xy_m"]]
        if len(pts) >= 2:
            dr.line(pts, fill="#50ddfa", width=4)
    sx, sy = to_px(start_xy)
    gx, gy = to_px(goal_xy)
    dr.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), fill="#2ec4b6", outline="#ffffff", width=2)
    dr.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), fill="#ff9f1c", outline="#ffffff", width=2)

    dr.text((24, 14), "真实无人机摄影测量 DSM → VCTP + IGHA* 可通行路径规划预览", font=f_title, fill="#edf4f8")
    vctp_rate = res.get("vctp", {}).get("pass_rate", 0.0) * 100.0
    status_txt = f"状态：{res.get('status')} | 路径长：{res.get('length_m', 0.0):.2f} m | VCTP 通过率：{vctp_rate:.1f}%"
    dr.text((24, 46), status_txt, font=f_sub, fill="#a3f7bf" if res.get("status") == "PLANNED" else "#ffd166")
    canvas.save(out_png)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsm", type=Path, required=True, help="Path to WebODM / DJI Terra GeoTIFF DSM.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for uav_map.npz and preview.")
    parser.add_argument("--pixel-size-m", type=float, default=None, help="DSM pixel size in metres (optional if GeoTIFF tagged).")
    parser.add_argument("--coarse-res", type=float, default=0.50)
    parser.add_argument("--subdiv", type=int, default=4)
    parser.add_argument("--max-slope-deg", type=float, default=17.0)
    parser.add_argument("--seed-xy", nargs=2, type=float, default=None)
    parser.add_argument("--start-xy", nargs=2, type=float, default=None)
    parser.add_argument("--goal-xy", nargs=2, type=float, default=None)
    args = parser.parse_args()

    map_path, meta = dsm_to_uav_map(
        args.dsm, args.output, pixel_size_m=args.pixel_size_m,
        coarse_res_m=args.coarse_res, subdiv=args.subdiv,
        max_slope_deg=args.max_slope_deg, seed_xy=tuple(args.seed_xy) if args.seed_xy else None,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    if args.start_xy and args.goal_xy:
        res = plan_and_preview_real_map(map_path, args.start_xy, args.goal_xy, args.output / "real_dsm_igha_plan_cn.png")
        print(json.dumps({k: v for k, v in res.items() if k != "route_xy_m"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
