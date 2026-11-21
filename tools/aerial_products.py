"""Sensor-derived aerial survey products in the rover's local ENU frame.

The DSM-backed image projection follows Hinzmann et al., Mapping on the Fly.
The Webots implementation uses measured depth both to form the DSM and to
reject occluded or inconsistent source-image projections.
"""
from pathlib import Path
import argparse
import json
import math

import numpy as np
from PIL import Image, TiffImagePlugin, TiffTags

from uav_map_preview import load_prior, project
from multiview import aerial_camera_rotation
from survey_map import Grid, sampled_pixels


METHOD = "https://github.com/ethz-asl/aerial_mapper"
NO_ELEVATION = -9999.0
NO_TRAVERSABILITY = 255
DEPTH_TOLERANCE_M = 0.28


def subsurface_grid(prior, factor=4):
    return Grid(prior.origin, prior.resolution / factor, tuple(n * factor for n in prior.shape))


def build_surface(grid, observations):
    """Rasterize the upper visible surface and retain the measured colored points."""
    stride = int(observations["stride_px"])
    min_range, max_range = observations["range_m"]
    points, colors, cell_ids, boxes = [], [], [], []
    for frame, depth in enumerate(observations["depth_m"]):
        pixels, valid = sampled_pixels(depth, stride, min_range, max_range)
        xyz = project(observations, frame, pixels)
        cols, rows = np.floor((xyz[:, :2] - grid.origin) / grid.resolution).astype(int).T
        inside = ((rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1]))
        points.append(xyz[inside].astype(np.float32))
        colors.append(observations["rgb"][frame][valid][inside])
        rr, cc = rows[inside], cols[inside]
        cell_ids.append(rr * grid.shape[1] + cc)
        boxes.append((int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1)
                     if rr.size else None)
    xyz = np.concatenate(points)
    rgb = np.concatenate(colors)
    if not len(xyz):
        raise ValueError("No camera depth returns inside the survey footprint.")
    ids = np.concatenate(cell_ids)
    order = np.lexsort((xyz[:, 2], ids))
    occupied, starts, counts = np.unique(ids[order], return_index=True, return_counts=True)
    good = counts >= 2
    # Prefer the upper measured surface, while suppressing isolated high returns.
    selected = starts[good] + np.floor(0.85 * (counts[good] - 1)).astype(int)
    dsm = np.full(grid.shape, np.nan, dtype=np.float32)
    dsm.flat[occupied[good]] = xyz[order[selected], 2]
    return dsm, xyz, rgb, boxes


def world_to_image(points, camera, rotation, width, height, fov):
    """Project ENU points into the Webots FLU camera and return optical-axis depth."""
    relative = (points - camera) @ rotation
    depth = relative[:, 0]
    focal = width / (2 * math.tan(fov / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (width - 1) / 2 - focal * relative[:, 1] / depth
        v = (height - 1) / 2 - focal * relative[:, 2] / depth
    return u, v, depth


def bilinear_color(image, u, v):
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    x1, y1 = np.minimum(x + 1, image.shape[1] - 1), np.minimum(y + 1, image.shape[0] - 1)
    sx, sy = (u - x)[:, None], (v - y)[:, None]
    upper = (1 - sx) * image[y, x] + sx * image[y, x1]
    lower = (1 - sx) * image[y1, x] + sx * image[y1, x1]
    return np.rint((1 - sy) * upper + sy * lower).astype(np.uint8)


def orthorectify(grid, dsm, observations, boxes, frames, depth_tolerance=DEPTH_TOLERANCE_M):
    width, height = observations["size_wh"]
    fov = float(observations["fov_rad"])
    stride = int(observations["stride_px"])
    min_range, max_range = observations["range_m"]
    rotations = observations.get("body_rotation_enu")
    scores = np.full(grid.shape, -np.inf, dtype=np.float32)
    frame_ids = np.full(grid.shape, -1, dtype=np.int32)
    color = np.zeros((*grid.shape, 4), dtype=np.uint8)
    matched_error = np.full(grid.shape, np.nan, dtype=np.float32)
    depth_rejections = 0
    for index, box in enumerate(boxes):
        if box is None:
            continue
        y0, y1, x0, x1 = box
        rows, cols = np.nonzero(np.isfinite(dsm[y0:y1, x0:x1]))
        rows, cols = rows + y0, cols + x0
        xyz = np.column_stack((
            grid.origin[0] + (cols + 0.5) * grid.resolution,
            grid.origin[1] + (rows + 0.5) * grid.resolution,
            dsm[rows, cols],
        ))
        gps = observations["gps_enu_m"][index]
        attitude = observations["attitude_rpy_rad"][index]
        orientation = aerial_camera_rotation(attitude, None if rotations is None else rotations[index])
        u, v, depth = world_to_image(xyz, gps, orientation, width, height, fov)
        in_frame = ((depth > min_range) & (depth < max_range) & (u >= 0) & (u < width - 1)
                    & (v >= 0) & (v < height - 1))
        ids = np.flatnonzero(in_frame)
        if not ids.size:
            continue
        depth_frame = observations["depth_m"][index]
        sample_x = np.clip(np.rint(u[ids] / stride).astype(int), 0, depth_frame.shape[1] - 1)
        sample_y = np.clip(np.rint(v[ids] / stride).astype(int), 0, depth_frame.shape[0] - 1)
        observed = depth_frame[sample_y, sample_x]
        error = np.abs(depth[ids] - observed)
        visible = (observed > min_range) & (observed < max_range) & (error <= depth_tolerance)
        depth_rejections += int((~visible).sum())
        ids, error = ids[visible], error[visible]
        if not ids.size:
            continue
        ray = gps - xyz[ids]
        # Closest to nadir. The value is the sine of the view elevation angle.
        angle_score = ray[:, 2] / np.linalg.norm(ray, axis=1)
        better = (angle_score > 0) & (angle_score > scores[rows[ids], cols[ids]])
        ids, error, angle_score = ids[better], error[better], angle_score[better]
        if not ids.size:
            continue
        path = frames / f"{index:03d}.jpg"
        if path.is_file():
            with Image.open(path) as photo:
                rgb = np.asarray(photo.convert("RGB"))
            sampled = bilinear_color(rgb, u[ids], v[ids])
        else:
            rgb = observations["rgb"][index]
            sampled = bilinear_color(rgb, np.minimum(u[ids] / stride, rgb.shape[1] - 1),
                                     np.minimum(v[ids] / stride, rgb.shape[0] - 1))
        cell = (rows[ids], cols[ids])
        scores[cell] = angle_score
        color[cell] = np.column_stack((sampled, np.full(len(ids), 255, dtype=np.uint8)))
        frame_ids[cell] = index
        matched_error[cell] = error
    return color, frame_ids, matched_error, depth_rejections


def write_raster(path, grid, layer, nodata=None):
    """Write a north-up GeoTIFF with local metric ENU coordinates and a matching world file."""
    west, south = grid.origin
    north = south + grid.shape[0] * grid.resolution
    pixel = grid.resolution
    directory = TiffImagePlugin.ImageFileDirectory_v2()
    directory.tagtype[33550] = TiffTags.DOUBLE
    directory[33550] = (pixel, pixel, 0.0)
    directory.tagtype[33922] = TiffTags.DOUBLE
    directory[33922] = (0.0, 0.0, 0.0, west, north, 0.0)
    citation = "Webots local ENU metres|"
    directory.tagtype[34735] = TiffTags.SHORT
    directory[34735] = (1, 1, 1, 3, 1024, 0, 1, 32767, 1025, 0, 1, 1,
                        1026, 34737, len(citation), 0)
    directory.tagtype[34737] = TiffTags.ASCII
    directory[34737] = citation
    if nodata is not None:
        directory.tagtype[42113] = TiffTags.ASCII
        directory[42113] = str(nodata)
    Image.fromarray(np.flipud(layer)).save(path, tiffinfo=directory, compression="tiff_deflate")
    world_file = (pixel, 0.0, 0.0, -pixel, west + pixel / 2, north - pixel / 2)
    path.with_suffix(".tfw").write_text("\n".join(str(value) for value in world_file) + "\n",
                                        encoding="ascii")


def write_cloud(path, points, colors, voxel=0.15):
    voxels = np.floor((points - points.min(axis=0)) / voxel).astype(np.int64)
    dimensions = voxels.max(axis=0) + 1
    _, selected = np.unique(np.ravel_multi_index(voxels.T, tuple(dimensions)), return_index=True)
    records = np.empty(len(selected), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                             ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    for column, name in enumerate(("x", "y", "z")):
        records[name] = points[selected, column]
    for column, name in enumerate(("red", "green", "blue")):
        records[name] = colors[selected, column]
    header = (
        "ply\nformat binary_little_endian 1.0\ncomment Webots local ENU metres\n"
        f"element vertex {len(records)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as output:
        output.write(header.encode("ascii"))
        records.tofile(output)
    return len(records)


def export_products(run, observations):
    folder = run / "aerial_products"
    folder.mkdir(exist_ok=True)
    prior, states = load_prior(run / "uav_map.npz")
    with np.load(run / "uav_map.npz", allow_pickle=False) as mapping:
        ground = mapping["ground_z_m"]
    raster = subsurface_grid(prior)
    dsm, points, colors, boxes = build_surface(raster, observations)
    ortho, sources, error, rejected = orthorectify(raster, dsm, observations, boxes, run / "frames")
    write_raster(folder / "orthomosaic.tif", raster, ortho)
    write_raster(folder / "surface_dsm.tif", raster, np.nan_to_num(dsm, nan=NO_ELEVATION), NO_ELEVATION)
    write_raster(folder / "ground_dem.tif", prior, np.nan_to_num(ground, nan=NO_ELEVATION), NO_ELEVATION)
    write_raster(folder / "traversability.tif", prior, np.where(states < 0, 255, states).astype("uint8"),
                 NO_TRAVERSABILITY)
    count = write_cloud(folder / "measured_cloud.ply", points, colors)
    np.savez_compressed(folder / "orthomosaic_sources.npz", source_frame_index=sources,
                        depth_difference_m=error)
    filled = np.isfinite(error)
    report = dict(method_reference=METHOD, frame="Webots local ENU metres", origin_xy_m=prior.origin,
                  orthorectification="DSM backward projection; depth-gated, closest to nadir",
                  depth_tolerance_m=DEPTH_TOLERANCE_M,
                  extent_xy_m=[n * prior.resolution for n in prior.shape[::-1]],
                  surface_resolution_m=raster.resolution, ground_resolution_m=prior.resolution,
                  surface_cells=int(dsm.size), dsm_cells=int(np.isfinite(dsm).sum()),
                  orthomosaic_cells=int(filled.sum()), measured_points_before_voxel_filter=len(points),
                  exported_cloud_points=count, cloud_voxel_m=0.15, depth_rejections=rejected,
                  selected_depth_difference_m=(dict(median=float(np.median(error[filled])),
                                                     p90=float(np.quantile(error[filled], 0.9)))
                                                if filled.any() else None),
                  ground_elevation_cells=int(np.isfinite(ground).sum()),
                  traversability_codes={"unknown": 255, "candidate_free": 0, "blocked": 1})
    (folder / "products.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Completed Webots survey directory.")
    args = parser.parse_args()
    with np.load(args.run / "observations.npz", allow_pickle=False) as captures:
        observations = dict(captures)
    report = export_products(args.run, observations)
    from uav_full_preview import draw_aerial_products
    draw_aerial_products(args.run, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
