"""Presentation figure for the measured D-scene survey."""
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uav_map_preview import aerial_mosaic, load_prior
from survey_map import BLOCKED, FREE, UNKNOWN


FONT_PATHS = (
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts/msyh.ttc",
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
BACKGROUND = "#101923"
PANEL = "#172530"
TEXT = "#edf4f6"
MUTED = "#b8c8d2"
UNKNOWN_COLOR = (82, 94, 105)
FREE_COLOR = (138, 171, 137)
REFINED_COLOR = (45, 205, 187)
PARTIAL_COLOR = (81, 105, 137)
HAZARD_COLOR = ((238, 170, 87), (209, 91, 74), (226, 132, 83))
ROUTE_COLORS = ("#46e0fc", "#eec2ff", "#ffd487")


def risk_colors(states, hazards, added=None):
    risk = np.full((*states.shape, 3), UNKNOWN_COLOR, dtype=np.uint8)
    risk[states == FREE] = FREE_COLOR
    for code, color in enumerate(HAZARD_COLOR, start=1):
        risk[hazards == code] = color
    if added is not None:
        risk[added] = REFINED_COLOR
    return risk


def draw_full_preview(folder, report, observations, config, schedule=None):
    grid, states = load_prior(folder / "uav_map.npz")
    with np.load(folder / "uav_map.npz", allow_pickle=False) as data:
        elevation, seen, hazards = (data[key] for key in ("ground_z_m", "observed_views", "hazard_type"))
        added = data["refined_ground"] if "refined_ground" in data else None
    counts = {label: int(np.count_nonzero(states == state)) for label, state in
              (("candidate_free", FREE), ("blocked", BLOCKED), ("unknown", UNKNOWN))}
    ground = np.isfinite(elevation)
    low, high = float(elevation[ground].min()), float(elevation[ground].max())
    extent = np.asarray(grid.shape[::-1]) * grid.resolution

    cn_font = next((path for path in FONT_PATHS if path.is_file()), None)

    def font(size):
        return ImageFont.truetype(str(cn_font), size) if cn_font else ImageFont.load_default(size=size)

    def lang(zh, en):
        return zh if cn_font else en

    margin, space, w = 38, 28, 825
    h = round(w * extent[1] / extent[0])
    x_left, x_right = margin, margin + w + space
    y_first, y_second = 159, 159 + h + 99
    poster = Image.new("RGB", (2 * w + 2 * margin + space, y_second + h + 139), BACKGROUND)
    draw = ImageDraw.Draw(poster)
    draw.text((margin, 26), lang("援来有路  ·  山洪灾区航测与导航", "Flood response | aerial survey and navigation"),
              font=font(43), fill=TEXT)
    metadata = f"Webots D   ·   {extent[0]:.0f} × {extent[1]:.0f} m   ·   {grid.resolution:.2f} m   ·   "
    phase = observations.get("survey_phase", np.zeros(len(observations["timestamp_s"]), dtype=int))
    nadir_count = int(np.count_nonzero(phase == 0))
    oblique_count = int(np.count_nonzero(phase == 1))
    caption = lang(f"{nadir_count} 帧俯拍 + {oblique_count} 帧斜视补拍",
                   f"{nadir_count} nadir and {oblique_count} oblique observations")
    if not oblique_count:
        caption = lang(f"{nadir_count} 帧机载观测", f"{nadir_count} aerial observations")
    draw.text((margin, 87), metadata + caption, font=font(22), fill=MUTED)

    def put_map(values, left, top, resampling=Image.Resampling.NEAREST):
        image = Image.fromarray(np.flipud(values)).resize((w, h), resampling)
        poster.paste(image, (left, top))
        draw.rectangle((left, top, left + w, top + h), outline="#445564", width=2)

    def position(xy, left, top):
        east, north = (np.asarray(xy) - grid.origin) / extent
        return float(left + w * east), float(top + h * (1 - north))

    ortho_file = folder / "aerial_products/orthomosaic.tif"
    headings = (
        (x_left, y_first, lang("正射图与实测航迹", "Orthomosaic and sampled survey track")
         if ortho_file.is_file()
         else lang("航拍合成与实测航迹", "Aerial mosaic and sampled survey track")),
        (x_right, y_first, lang("地面高程 · 米", "Observed ground elevation | m")),
        (x_left, y_second, lang("通行风险与未知区域", "Traversability and unknown terrain")),
        (x_right, y_second, lang("小车全局规划", "Rover global route options")),
    )
    for left, top, label in headings:
        draw.text((left, top - 36), label, font=font(24), fill=TEXT)

    if ortho_file.is_file():
        with Image.open(ortho_file) as source:
            optical = source.convert("RGBA")
        composite = Image.new("RGBA", optical.size, BACKGROUND)
        composite.alpha_composite(optical)
        put_map(np.flipud(np.asarray(composite.convert("RGB"))), x_left, y_first,
                Image.Resampling.BILINEAR)
    else:
        put_map(aerial_mosaic(grid, observations), x_left, y_first, Image.Resampling.BILINEAR)
    track = [position(xy, x_left, y_first) for xy in observations["gps_enu_m"][phase == 0, :2]]
    draw.line(track, fill="#183642", width=5)
    draw.line(track, fill="#71e4f4", width=2)
    for px, py in track[::8]:
        draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill="#f0fafb")
    if schedule is not None:
        for index, patch in enumerate(schedule["patches"], start=1):
            px, py = position(patch["xy_m"], x_left, y_first)
            radius = w * schedule["radius_m"] / extent[0]
            draw.ellipse((px - radius, py - radius, px + radius, py + radius),
                         outline="#ffda92", width=3)
            draw.text((px, py), str(index), font=font(21), fill="#ffda92", anchor="mm",
                      stroke_width=2, stroke_fill=BACKGROUND)
    scale = w * 10 / extent[0]
    sx, sy = x_left + 18, y_first + h - 21
    draw.line((sx, sy, sx + scale, sy), fill="white", width=4)
    draw.text((sx, sy - 29), "10 m", font=font(19), fill="white", stroke_width=1, stroke_fill=BACKGROUND)
    draw.text((x_left + w - 49, y_first + 13), "N ↑", font=font(19), fill="white",
              stroke_width=1, stroke_fill=BACKGROUND)

    span = max(high - low, 1e-6)
    scalar = np.uint8(255 * np.clip((np.nan_to_num(elevation, nan=low) - low) / span, 0, 1))
    relief = cv2.applyColorMap(scalar, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    relief[~ground] = UNKNOWN_COLOR
    put_map(relief, x_right, y_first)
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 16, axis=0)
    colorbar = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    poster.paste(Image.fromarray(colorbar).resize((268, 16)), (x_right + 67, y_first + h + 18))
    draw.text((x_right, y_first + h + 13), f"{low:.1f}", font=font(20), fill=MUTED)
    draw.text((x_right + 345, y_first + h + 13), f"{high:.1f} m", font=font(20), fill=MUTED)
    draw.text((x_right + 485, y_first + h + 13), lang("灰色  地面高程未确认", "Grey  |  ground unverified"),
              font=font(19), fill=MUTED)
    seen_pct = 100 * float(np.mean(seen > 0))
    strips = sum(a[1] == b[1] for a, b in zip(config["flight"]["waypoints_xy_enu_m"],
                                              config["flight"]["waypoints_xy_enu_m"][1:]))
    draw.text((x_left, y_first + h + 13),
              lang(f"{strips} 条测线    深度回波覆盖 {seen_pct:.1f}% 的格子",
                   f"{strips} survey strips    Depth returns in {seen_pct:.1f}% of cells"),
              font=font(20), fill=MUTED)

    risk = risk_colors(states, hazards, added)
    put_map(risk, x_left, y_second)
    subdued = np.uint8(risk * 0.62 + np.array([9, 15, 20]))
    put_map(subdued, x_right, y_second)
    start = position(report["start_xy_m"], x_right, y_second)

    def marker(point, label, color):
        x, y = point
        draw.ellipse((x - 13, y - 13, x + 13, y + 13), fill=PANEL, outline=color, width=3)
        draw.text((x, y - 1), label, font=font(16), fill=color, anchor="mm")

    paths = [[position(xy, x_right, y_second) for xy in case.get("route_xy_m", [])]
             for case in report["routes"]]
    for path in paths:
        if len(path) > 1:
            draw.line(path, fill="#101923", width=11, joint="curve")
    for i, path in enumerate(paths):
        if len(path) > 1:
            draw.line(path, fill=ROUTE_COLORS[i % len(ROUTE_COLORS)],
                      width=8 if i == 0 else 4, joint="curve")

    for i, case in enumerate(report["routes"]):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        tag = chr(ord("A") + i)
        target = position(case["goal_xy_m"], x_right, y_second)
        if case["status"] == "PLANNED":
            detail = f"{case['route_length_m']:.1f} m"
        else:
            detail = lang("待地面核实", "ground check required")
        marker(target, tag, color)
        y = y_second + h + 14 + 30 * i
        draw.text((x_right, y), f"{tag}  {case['label'] if cn_font else case['id']}    {detail}",
                  font=font(20), fill=color)
    marker(start, "S", "white")
    draw.text((x_right, y_second + h + 105),
              lang("S  小车 GPS 起点   ·   青紫叠线为共用路段",
                   "S  Rover GPS origin   |   Two colors show a shared segment"),
              font=font(17), fill=MUTED)

    legend = (
        (FREE_COLOR, lang(f"候选通行 {100*counts['candidate_free']/states.size:.1f}%",
                          f"Candidate {100*counts['candidate_free']/states.size:.1f}%")),
        (HAZARD_COLOR[0], lang("陡坡", "Steep")),
        (HAZARD_COLOR[1], lang("台阶 / 落差", "Step / drop")),
        (HAZARD_COLOR[2], lang("近地不平整", "Rough")),
        (UNKNOWN_COLOR, lang(f"未知 {100*counts['unknown']/states.size:.1f}%",
                             f"Unknown {100*counts['unknown']/states.size:.1f}%")),
    )
    x = x_left
    y = y_second + h + 17
    for color, title in legend:
        draw.rectangle((x, y, x + 15, y + 15), fill=color)
        draw.text((x + 23, y - 5), title, font=font(18), fill=TEXT)
        x += 49 + round(draw.textlength(title, font=font(18)))
    draw.text((x_left, y + 34), lang("灰区含林下及屋顶遮挡，靠近后由小车复核。",
                                     "Grey includes occluded canopy and roof areas."),
              font=font(20), fill=MUTED)
    draw.text((x_left, y + 67), lang("路线按实测高程、坡度和车身通行余量计算。",
                                     "Routes use measured terrain, grade and rover clearance."),
              font=font(20), fill=MUTED)
    if added is not None:
        draw.rectangle((x_left, y + 105, x_left + 15, y + 120), fill=REFINED_COLOR)
        detail = lang(f"青绿色 斜视补拍新增地面证据 {int(added.sum())} 格",
                      f"Turquoise  |  {int(added.sum())} cells supported by oblique views")
        draw.text((x_left + 23, y + 100), detail, font=font(18), fill=TEXT)
    poster.save(folder / "preview.png")
    return counts


def draw_refinement_preview(folder, schedule, metrics):
    grid, before = load_prior(folder / "uav_map_nadir.npz")
    _, after = load_prior(folder / "uav_map.npz")
    with np.load(folder / "uav_map_nadir.npz", allow_pickle=False) as base:
        original = risk_colors(before, base["hazard_type"])
    with np.load(folder / "uav_map.npz", allow_pickle=False) as final:
        added = final["refined_ground"]
        revised = risk_colors(after, final["hazard_type"], added)
        revised[(after == UNKNOWN) & (final["oblique_ground_quarters"] > 0)] = PARTIAL_COLOR

    path = next((item for item in FONT_PATHS if item.is_file()), None)

    def font(size):
        return ImageFont.truetype(str(path), size) if path else ImageFont.load_default(size=size)

    def word(zh, en):
        return zh if path else en

    margin, gutter, side = 32, 25, 342
    patches = schedule["patches"]
    canvas = Image.new("RGB", (2 * margin + side * len(patches) + gutter * (len(patches) - 1), 960), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 22), word("未知区斜视补拍前后对照", "Oblique survey | before and after"),
              font=font(32), fill=TEXT)
    detail = (f"{metrics['newly_supported_cells']} 格获得新增地面证据（{metrics['newly_supported_area_m2']:.2f} m²）"
              if path else f"{metrics['newly_supported_cells']} newly supported cells")
    draw.text((margin, 70), detail, font=font(19), fill=MUTED)
    for index, patch in enumerate(patches):
        row, col = grid.cell(patch["xy_m"])
        radius = max(2, round((schedule["radius_m"] + 1.0) / grid.resolution))
        size = min(2 * radius + 1, *grid.shape)
        row0 = min(max(0, row - radius), grid.shape[0] - size)
        col0 = min(max(0, col - radius), grid.shape[1] - size)
        region = (slice(row0, row0 + size), slice(col0, col0 + size))
        left = margin + index * (side + gutter)
        label = word(f"补拍区 {index+1}", f"Focus area {index+1}")
        draw.text((left, 112), label, font=font(21), fill="#ffda92")
        for top, source, title in ((168, original, word("初始俯拍", "Nadir baseline")),
                                   (564, revised, word("加入斜视补拍", "With oblique views"))):
            draw.text((left, top - 26), title, font=font(17), fill=TEXT)
            tile = Image.fromarray(np.flipud(source[region])).resize((side, side), Image.Resampling.NEAREST)
            canvas.paste(tile, (left, top))
    draw.rectangle((margin, 922, margin + 15, 937), fill=REFINED_COLOR)
    footer = word("新增候选通行", "New candidate ground")
    draw.text((margin + 25, 918), footer, font=font(18), fill=TEXT)
    second_key = margin + 206
    draw.rectangle((second_key, 922, second_key + 15, 937), fill=PARTIAL_COLOR)
    draw.text((second_key + 25, 918),
              word("有部分地面证据，仍待核实", "Some floor evidence, still unknown"), font=font(18), fill=TEXT)
    canvas.save(folder / "refinement.png")


def draw_aerial_products(folder, report):
    products = folder / "aerial_products"
    cn_font = next((path for path in FONT_PATHS if path.is_file()), None)

    def font(size):
        return ImageFont.truetype(str(cn_font), size) if cn_font else ImageFont.load_default(size=size)

    def lang(chinese, english):
        return chinese if cn_font else english

    with Image.open(products / "orthomosaic.tif") as source:
        optical = source.convert("RGBA")
    with Image.open(products / "surface_dsm.tif") as source:
        dsm = np.asarray(source).copy()
    with Image.open(products / "ground_dem.tif") as source:
        dem = np.asarray(source).copy()
    background = Image.new("RGBA", optical.size, BACKGROUND)
    background.alpha_composite(optical)
    known_surface = dsm > -9998
    zmin, zmax = float(dsm[known_surface].min()), float(dsm[known_surface].max())

    def elevation_image(z):
        known = z > -9998
        values = np.uint8(255 * np.clip((z - zmin) / max(zmax - zmin, 1e-6), 0, 1))
        color = cv2.applyColorMap(values, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
        color[~known] = UNKNOWN_COLOR
        return Image.fromarray(color)

    margin, gap, width = 29, 18, 470
    height = round(width * optical.height / optical.width)
    top = 137
    panel = Image.new("RGB", (3 * width + 2 * gap + 2 * margin, top + height + 102), BACKGROUND)
    draw = ImageDraw.Draw(panel)
    draw.text((margin, 21), lang("援来有路  ·  无人机航测成果", "Aerial survey products"),
              font=font(35), fill=TEXT)
    east, north = report["extent_xy_m"]
    draw.text((margin, 68), f"Webots D   ·   {east:g} × {north:g} m   ·   " +
              lang("本地 ENU 米制坐标", "Local ENU, metres"), font=font(19), fill=MUTED)
    ortho_fraction = 100 * report["orthomosaic_cells"] / report["surface_cells"]
    surface_resolution, ground_resolution = report["surface_resolution_m"], report["ground_resolution_m"]
    surfaces = (
        (lang("反向投影正射图", "Orthorectified mosaic"), background.convert("RGB"),
         lang(f"{surface_resolution:g} m 栅格 · 有效像素 {ortho_fraction:.1f}%",
              f"{surface_resolution:g} m pixels · {ortho_fraction:.1f}% textured")),
        (lang("可见表面高程 DSM", "Visible surface elevation (DSM)"), elevation_image(dsm),
         lang("含树冠、屋顶和裸露地形", "Canopy, roofs and exposed terrain")),
        (lang("地面高程 DEM", "Ground elevation (DEM)"), elevation_image(dem),
         lang(f"{ground_resolution:g} m 栅格 · 灰色为地面未知",
              f"{ground_resolution:g} m cells · grey: ground unknown")),
    )
    for i, (title, artwork, caption) in enumerate(surfaces):
        x = margin + i * (width + gap)
        draw.text((x, top - 33), title, font=font(23), fill=TEXT)
        sampling = Image.Resampling.BILINEAR if i == 0 else Image.Resampling.NEAREST
        panel.paste(artwork.resize((width, height), sampling), (x, top))
        draw.rectangle((x, top, x + width, top + height), outline="#445564", width=2)
        draw.text((x, top + height + 13), caption, font=font(18), fill=MUTED)
    ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :], 12, axis=0)
    colors = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS)[..., ::-1].copy()
    bar_x, bar_y = margin + width + gap + 62, top + height + 60
    panel.paste(Image.fromarray(colors).resize((228, 12)), (bar_x, bar_y))
    draw.text((bar_x - 58, bar_y - 4), f"{zmin:.1f}", font=font(17), fill=MUTED)
    draw.text((bar_x + 239, bar_y - 4), f"{zmax:.1f} m", font=font(17), fill=MUTED)
    panel.save(products / "terrain_layers.png")
