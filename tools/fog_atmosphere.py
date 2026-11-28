"""Realistic overcast mountain-valley fog atmosphere for Webots R2025a scenes.

Replaces harsh LINEAR cutoff fog and sunny-blue skyboxes with:
  1. Seamless 3D procedural overcast stratus skybox + HDR irradiance cubemap
     horizon-locked to the exact tone-mapped fog airlight color.
  2. Beer-Lambert EXPONENTIAL fog curve calibrated for natural close-to-far
     atmospheric perspective (no blank grey cutoff wall at 38 m).
  3. Shadowless overcast diffused daylight + ambient skylight + damp terrain.
  4. Stratified perimeter & high-flank volumetric valley-mist ribbon banks
     placed strictly outside the sensor-active rescue corridor so RGB-D / LiDAR
     depth measurements remain uncorrupted.
"""
from pathlib import Path
import json
import math
import re

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SKY_DIR = ROOT / "worlds/textures/sky"
TEX_DIR = ROOT / "worlds/textures"

SKY_PREFIX = "D_misty_valley"
MIST_TEX_NAME = "D_mist_layer_soft.png"

# Linear fog color in Webots PBR / Fog shader (bright milky mountain mist)
FOG_COLOR_LINEAR = (0.70, 0.74, 0.77)
FOG_VISIBILITY_RANGE_M = 42.0
FOG_TYPE = "EXPONENTIAL"
BG_LUMINOSITY = 0.88


def _fbm3(x, y, z, octaves=4, seed=71):
    """Smooth vectorized 3D trigonometric-warped FBM on unit directions (zero seams)."""
    val = np.zeros_like(x, dtype=np.float32)
    amp = 0.52
    freq = 2.6
    for oct_idx in range(octaves):
        p0 = seed * 0.13 + oct_idx * 1.91
        n1 = np.sin(freq * x + 0.7 * np.cos(freq * 1.3 * y + p0) + p0)
        n2 = np.cos(freq * 1.17 * z - 0.65 * np.sin(freq * 0.9 * x - p0) + 1.3 * p0)
        n3 = np.sin(freq * 0.83 * y + 0.6 * np.sin(freq * 1.4 * z + p0) - 0.8 * p0)
        val += amp * (0.38 * n1 + 0.34 * n2 + 0.28 * n3)
        freq *= 2.07
        amp *= 0.50
    return 0.5 + 0.5 * np.clip(val, -1.0, 1.0)


def _cube_face_dirs(face: str, size: int):
    """Return unit 3D direction vectors (dx, dy_up, dz) matching Webots VRML Background."""
    lin = np.linspace(-1.0 + 1.0 / size, 1.0 - 1.0 / size, size, dtype=np.float32)
    u, v = np.meshgrid(lin, lin)  # u: left->right, v: top->bottom
    if face == "front":
        dx, dy, dz = u, -v, np.ones_like(u)
    elif face == "back":
        dx, dy, dz = -u, -v, -np.ones_like(u)
    elif face == "left":
        dx, dy, dz = -np.ones_like(u), -v, u
    elif face == "right":
        dx, dy, dz = np.ones_like(u), -v, -u
    elif face == "top":
        dx, dy, dz = u, np.ones_like(u), v
    elif face == "bottom":
        dx, dy, dz = u, -np.ones_like(u), -v
    else:
        raise ValueError(f"Unknown cubemap face: {face}")
    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    return dx / norm, dy / norm, dz / norm


def ensure_misty_skybox_assets(force: bool = False):
    """Generate seamless overcast stratus cubemap (.jpg + .hdr) and soft mist sprite."""
    SKY_DIR.mkdir(parents=True, exist_ok=True)
    TEX_DIR.mkdir(parents=True, exist_ok=True)

    faces = ("back", "bottom", "front", "left", "right", "top")
    expected = [SKY_DIR / f"{SKY_PREFIX}_{f}.jpg" for f in faces] + [
        SKY_DIR / f"{SKY_PREFIX}_{f}.hdr" for f in faces
    ] + [TEX_DIR / MIST_TEX_NAME]
    if not force and all(p.is_file() for p in expected):
        return

    # Calibrated overcast airlight RGB matching Fog color (0.72, 0.75, 0.77)
    horizon_lin = np.array(FOG_COLOR_LINEAR, dtype=np.float32)
    zenith_lin = np.array([0.56, 0.61, 0.66], dtype=np.float32)
    cloud_bright_lin = np.array([0.84, 0.87, 0.89], dtype=np.float32)
    sun_dir = np.array([0.52, 0.64, -0.56], dtype=np.float32)
    sun_dir /= np.linalg.norm(sun_dir)

    for face in faces:
        # 512x512 sRGB JPG visual skybox
        dx, dy, dz = _cube_face_dirs(face, 512)
        # Lock -0.05..+0.08 horizon band closely to horizon_lin so distant foggy ridges vanish smoothly,
        # while fading below-ground nadir directions (dy < -0.05) to dark soil tone so terrain stitching seams never glow white.
        sky_mix = np.power(np.clip((dy - 0.06) / 0.85, 0.0, 1.0), 0.78)
        # Keep elevated overview sky (dy >= -0.50) bright misty overcast; fade only steep downward rays (dy < -0.52)
        # to dark soil so 1 px nadir terrain-boundary cracks never glow bright white.
        ground_fade = np.clip((-dy - 0.52) / 0.22, 0.0, 1.0)
        ground_dark = np.array([0.13, 0.12, 0.10], dtype=np.float32)
        clouds = _fbm3(dx * 1.4, dy * 2.8, dz * 1.4, octaves=5, seed=71)
        cloud_band = np.clip((dy - 0.08) / 0.36, 0.0, 1.0)
        sun_cos = np.clip(dx * sun_dir[0] + dy * sun_dir[1] + dz * sun_dir[2], 0.0, 1.0)
        sun_glow = 0.12 * np.power(sun_cos, 4.0) * cloud_band

        base_lin = horizon_lin[None, None, :] * (1.0 - sky_mix[..., None]) + zenith_lin[None, None, :] * sky_mix[..., None]
        # Layer soft overcast stratus modulation only above the misty horizon
        c_weight = cloud_band * (0.48 * (clouds - 0.34))
        c_weight = np.clip(c_weight, 0.0, 0.68)
        sky_lin = base_lin * (1.0 - c_weight[..., None]) + cloud_bright_lin[None, None, :] * c_weight[..., None]
        sky_lin = np.clip(sky_lin + sun_glow[..., None] * np.array([1.0, 0.98, 0.95], dtype=np.float32), 0.0, 1.0)
        sky_lin = sky_lin * (1.0 - ground_fade[..., None]) + ground_dark[None, None, :] * ground_fade[..., None]

        # Webots skybox.frag applies pow(tex, 2.0) before hdr_resolve tone mapping;
        # encode inverse sqrt so rendered skybox linear energy matches horizon_lin seamlessly.
        srgb = np.clip(np.sqrt(sky_lin) * 255.0, 0.0, 255.0).astype(np.uint8)
        Image.fromarray(srgb, mode="RGB").save(SKY_DIR / f"{SKY_PREFIX}_{face}.jpg", quality=95)

        # 256x256 diffuse Radiance HDR irradiance cubemap (soft bright overcast ambient IBL)
        hx, hy, hz = _cube_face_dirs(face, 256)
        up_w = 0.5 + 0.5 * hy
        irr = np.empty((256, 256, 3), dtype=np.float32)
        irr_low = np.array([0.22, 0.24, 0.25], dtype=np.float32)
        irr_high = np.array([0.52, 0.55, 0.58], dtype=np.float32)
        for c in range(3):
            irr[:, :, c] = irr_low[c] * (1.0 - up_w) + irr_high[c] * up_w
        cv2.imwrite(str(SKY_DIR / f"{SKY_PREFIX}_{face}.hdr"), irr[:, :, ::-1])

    # 512x512 RGBA radial soft mist cloud texture (edges decay smoothly to alpha=0)
    lin = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    gx, gy = np.meshgrid(lin, lin)
    r = np.sqrt(gx * gx + gy * gy)
    radial = np.exp(-3.2 * (r ** 2)) * np.clip(1.0 - np.clip((r - 0.48) / 0.44, 0.0, 1.0), 0.0, 1.0) ** 2
    wisp = _fbm3(gx * 1.9, gy * 1.9, 0.35 * (gx - gy), octaves=4, seed=109)
    alpha = np.clip(radial * (0.45 + 0.55 * wisp), 0.0, 1.0)
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[:, :, 0] = 230
    rgba[:, :, 1] = 235
    rgba[:, :, 2] = 238
    rgba[:, :, 3] = np.clip(np.round(alpha * 88.0), 0, 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(TEX_DIR / MIST_TEX_NAME)


def _distant_hill_z(x: float, y: float, base_z: float) -> float:
    """Match tools/forest_terrain.py outer_mountains() elevation envelope outside [-42..42, -30..30]."""
    dx, dy = max(abs(x) - 42.0, 0.0), max(abs(y) - 30.0, 0.0)
    fade = min(1.0, math.hypot(dx, dy) / 20.0)
    distant = (
        17.0 * math.exp(-((x + 63.0) / 38.0) ** 2 - ((y - 58.0) / 32.0) ** 2)
        + 21.0 * math.exp(-((x - 64.0) / 34.0) ** 2 - ((y - 72.0) / 38.0) ** 2)
        + 12.0 * math.exp(-((x - 78.0) / 40.0) ** 2 - ((y + 53.0) / 32.0) ** 2)
    )
    return float(base_z + fade * distant)


def build_valley_mist_vrml(include_outer_banks: bool = True, include_gully_wisps: bool = False, seed: int = 71) -> str:
    """Build non-clipping perimeter valley-mist drifts (+ optional inner gully mist for standalone scenes)."""
    if not include_outer_banks and not include_gully_wisps:
        return ""
    terrain_file = ROOT / "evaluation/search/D.json"
    if not terrain_file.is_file():
        terrain_file = ROOT / "evaluation/scenarios/B.json"
    terrain = json.loads(terrain_file.read_text(encoding="utf-8"))
    import sys
    eval_dir = str(ROOT / "evaluation")
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    from metrics import height_at

    rng = np.random.default_rng(seed)
    nodes = []

    def _safe_max_z(cx, cy, sx, sy):
        zs = []
        for dx_s in (-0.45 * sx, 0.0, 0.45 * sx):
            for dy_s in (-0.45 * sy, 0.0, 0.45 * sy):
                px, py = cx + dx_s, cy + dy_s
                bz = float(height_at(terrain, max(-41.5, min(41.5, px)), max(-29.5, min(29.5, py))))
                zs.append(_distant_hill_z(px, py, bz))
        return max(zs)

    if include_outer_banks:
        rings = [
            (48.0, 62.0, 14, (16.0, 13.0), (1.0, 3.2)),
            (64.0, 82.0, 14, (22.0, 16.0), (2.2, 5.5)),
        ]
        for r_min, r_max, count, (sx, sy), (dz_lo, dz_hi) in rings:
            for k in range(count):
                ang = 2.0 * math.pi * (k + rng.uniform(-0.20, 0.20)) / count
                rad = rng.uniform(r_min, r_max)
                x = rad * math.cos(ang) * 1.12
                y = rad * math.sin(ang) * 0.88
                z = _safe_max_z(x, y, sx, sy) + rng.uniform(dz_lo, dz_hi)
                yaw = ang + rng.uniform(-0.5, 0.5)
                nodes.append(
                    f'    Transform {{ translation {x:.2f} {y:.2f} {z:.2f} rotation 0 0 1 {yaw:.3f}\n'
                    f'      children [\n'
                    f'        Shape {{\n'
                    f'          appearance PBRAppearance {{ baseColor 0.85 0.89 0.91 baseColorMap ImageTexture {{ url [ "textures/{MIST_TEX_NAME}" ] }} roughness 1 metalness 0 IBLStrength 0.45 emissiveColor 0.26 0.29 0.31 }}\n'
                    f'          geometry Plane {{ size {sx:.1f} {sy:.1f} }}\n'
                    f'        }}\n'
                    f'      ]\n'
                    f'    }}'
                )

    if include_gully_wisps:
        # Low-lying inhomogeneous mist ribbons hovering along the flood gully & eastern/northern hollows
        # (placed away from static audit rover spawn cone so occlusion audit stays clean)
        gully_spots = [
            (-1.5, -13.0, 10.5, 7.5), (1.5, -6.5, 11.5, 8.0), (-0.5, 1.0, 12.0, 8.5),
            (2.0, 8.0, 12.0, 8.5), (0.0, 14.5, 13.0, 9.0),
            (11.0, -14.0, 12.0, 8.5), (16.0, -8.5, 12.5, 9.0), (22.0, 2.0, 13.5, 9.5),
            (-16.0, 8.0, 12.5, 9.0), (-11.0, 16.0, 13.0, 9.0), (14.0, 16.5, 13.0, 9.0),
            (28.0, -15.0, 13.0, 9.0), (-26.0, 14.0, 13.5, 9.5), (30.0, 14.0, 14.0, 9.5),
            (7.0, -21.0, 12.0, 8.5), (-7.5, 22.0, 13.5, 9.5),
        ]
        for gx, gy, sx, sy in gully_spots:
            z = _safe_max_z(gx, gy, sx, sy) + float(rng.uniform(0.60, 1.45))
            yaw = float(rng.uniform(-0.8, 0.8))
            nodes.append(
                f'    Transform {{ translation {gx:.2f} {gy:.2f} {z:.2f} rotation 0 0 1 {yaw:.3f}\n'
                f'      children [\n'
                f'        Shape {{\n'
                f'          appearance PBRAppearance {{ baseColor 0.86 0.90 0.92 baseColorMap ImageTexture {{ url [ "textures/{MIST_TEX_NAME}" ] }} roughness 1 metalness 0 IBLStrength 0.45 emissiveColor 0.22 0.25 0.27 }}\n'
                f'          geometry Plane {{ size {sx:.1f} {sy:.1f} }}\n'
                f'        }}\n'
                f'      ]\n'
                f'    }}'
            )

    return "DEF VALLEY_MIST_BANKS Group {\n  children [\n" + "\n".join(nodes) + "\n  ]\n}\n"


def apply_realistic_fog_to_text(scene: str, include_mist_banks: bool = True, include_gully_wisps: bool = False) -> str:
    """Transform a Webots scene string into a realistic overcast misty mountain valley."""
    ensure_misty_skybox_assets(force=False)

    # 1. Replace clear mountains skybox with overcast horizon-matched stratus cubemap
    faces = ("back", "bottom", "front", "left", "right", "top")
    bg_fields = []
    for f in faces:
        bg_fields.append(f'  {f}Url [ "textures/sky/{SKY_PREFIX}_{f}.jpg" ]')
        bg_fields.append(f'  {f}IrradianceUrl [ "textures/sky/{SKY_PREFIX}_{f}.hdr" ]')
    new_bg = "Background {\n" + "\n".join(bg_fields) + f"\n  luminosity {BG_LUMINOSITY}\n}}"
    scene = re.sub(r"Background\s*\{[^}]*\}", new_bg, scene, count=1)

    # 2. Replace LINEAR cutoff fog with smooth Beer-Lambert EXPONENTIAL fog
    new_fog = (
        f"Fog {{\n"
        f"  color {FOG_COLOR_LINEAR[0]} {FOG_COLOR_LINEAR[1]} {FOG_COLOR_LINEAR[2]}\n"
        f'  fogType "{FOG_TYPE}"\n'
        f"  visibilityRange {FOG_VISIBILITY_RANGE_M}\n"
        f"}}"
    )
    if re.search(r"Fog\s*\{[^}]*\}", scene):
        scene = re.sub(r"Fog\s*\{[^}]*\}", new_fog, scene, count=1)
    else:
        scene = scene.replace(new_bg, new_bg + "\n" + new_fog, 1)

    # 3. Replace harsh sunny directional shadows with diffused overcast sky lighting + soft AO
    lights = list(re.finditer(r"DirectionalLight\s*\{[^}]*\}", scene))
    if len(lights) >= 1:
        b0 = (
            "DirectionalLight {\n"
            "  ambientIntensity 0.62\n"
            "  color 0.92 0.95 0.98\n"
            "  direction -0.55 -0.35 -0.82\n"
            "  intensity 0.36\n"
            "  castShadows FALSE\n"
            "}"
        )
        scene = scene[:lights[0].start()] + b0 + scene[lights[0].end():]
    lights = list(re.finditer(r"DirectionalLight\s*\{[^}]*\}", scene))
    if len(lights) >= 2:
        b1 = (
            "DirectionalLight {\n"
            "  ambientIntensity 0.22\n"
            "  color 0.80 0.86 0.92\n"
            "  direction 0.45 0.35 -0.90\n"
            "  intensity 0.44\n"
            "}"
        )
        scene = scene[:lights[1].start()] + b1 + scene[lights[1].end():]

    # 4. Boost AmbientOcclusionRadius softly for shadowless contact grounding and tune exposure
    scene = re.sub(r"\bambientOcclusionRadius\s+[0-9.]+", "ambientOcclusionRadius 1.35", scene)
    scene = re.sub(r"\bexposure\s+1\.[23]\d*", "exposure 1.12", scene)

    # 5. Subtle damp soil sheen on terrain without breaking perimeter seamlessness
    scene = re.sub(r"(baseColorMap\s+ImageTexture\s*\{\s*url\s*\[\s*\"textures/D_ground\.jpg\"\s*\]\s*\}\s*roughness\s+)0\.84",
                   r"\g<1>0.74", scene, count=1)

    # 6. Inject perimeter (+ optional inner gully) stratified valley mist ribbons
    if (include_mist_banks or include_gully_wisps) and "DEF VALLEY_MIST_BANKS" not in scene:
        scene += "\n" + build_valley_mist_vrml(include_outer_banks=include_mist_banks, include_gully_wisps=include_gully_wisps)
    return scene


def apply_realistic_fog_to_world(world_path: Path, include_mist_banks: bool = True, include_gully_wisps: bool = False):
    """Apply overcast valley fog transformation to a Webots .wbt world file on disk."""
    world_path = Path(world_path)
    txt = world_path.read_text(encoding="utf-8")
    world_path.write_text(
        apply_realistic_fog_to_text(txt, include_mist_banks=include_mist_banks, include_gully_wisps=include_gully_wisps),
        encoding="utf-8",
    )


def build_fog_world_d() -> Path:
    """Build standalone worlds/search_D_fog.wbt (+ .wbproj layout) for direct Webots opening."""
    import shutil
    import sys
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from build_search_worlds import build

    src = build("D")
    dst = ROOT / "worlds/search_D_fog.wbt"
    shutil.copy2(src, dst)
    apply_realistic_fog_to_world(dst, include_mist_banks=True, include_gully_wisps=True)
    src_proj = ROOT / "worlds/.search_D.wbproj"
    dst_proj = ROOT / "worlds/.search_D_fog.wbproj"
    if src_proj.is_file():
        shutil.copy2(src_proj, dst_proj)
    return dst


if __name__ == "__main__":
    out = build_fog_world_d()
    print(f"[FogAtmosphere] Built standalone realistic fog world: {out}", flush=True)
