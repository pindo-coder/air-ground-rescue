"""Materials and scenery for the flash-flood rescue environment."""
from functools import lru_cache
from pathlib import Path
import math
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def tile(name):
    return np.asarray(Image.open(ROOT / "worlds/textures/materials" / name).convert("RGB"))


def appearance(style, color="1 1 1", scale=(1, 1)):
    files = {
        "rock": ("rock_face_03_Diffuse.jpg", "rock_face_03_nor_gl.jpg", "rock_face_03_Rough.jpg"),
        "plaster": ("worn_cracked_plaster_Diffuse.jpg", "worn_cracked_plaster_nor_gl.jpg", "worn_cracked_plaster_Rough.jpg"),
        "brick": ("broken_brick_wall_Diffuse.jpg", "broken_brick_wall_nor_gl.jpg", "broken_brick_wall_Rough.jpg"),
        "wood": ("pine_trunk.jpg", None, None),
        "roof": ("corrugated_iron_Diffuse.jpg", "corrugated_iron_nor_gl.jpg", "corrugated_iron_Rough.jpg"),
        "mud": ("brown_mud_02_Diffuse.jpg", "brown_mud_02_nor_gl.jpg", "brown_mud_02_Rough.jpg"),
    }
    diffuse, normal, rough = files[style]
    maps = f'baseColorMap ImageTexture {{ url [ "textures/materials/{diffuse}" ] }}'
    if normal:
        maps += f' normalMap ImageTexture {{ url [ "textures/materials/{normal}" ] }} normalMapFactor 0.65'
    if rough:
        maps += f' roughnessMap ImageTexture {{ url [ "textures/materials/{rough}" ] }}'
    return (f'PBRAppearance {{ baseColor {color} {maps} roughness 0.88 metalness 0 '
            f'IBLStrength 0.85 textureTransform TextureTransform {{ scale {scale[0]} {scale[1]} }} }}')


def flood_ground(p, size):
    x, y = np.meshgrid(np.linspace(-21, 21, size, dtype=np.float32),
                       np.linspace(15, -15, size, dtype=np.float32))
    phase = (p["seed"] % 37) * .07
    macro = .50 + .25 * np.sin(.34 * x + .23 * y + phase) * np.cos(.31 * y - .17 * x)
    macro += .14 * np.sin(.85 * x - .51 * y)
    mix = np.clip(macro, .20, .80)
    # World-space tiling retains ground-scale detail without drawing a route.
    u = np.floor((x / 2.4 + .14 * np.sin(y * .27)) * 1024).astype(np.int32) % 1024
    v = np.floor((-y / 2.4 + .12 * np.sin(x * .21)) * 1024).astype(np.int32) % 1024
    soil = tile("brown_mud_02_Diffuse.jpg")[v, u].astype(np.float32)
    litter = tile("forest_floor_Diffuse.jpg")[(v + 137) % 1024, (u + 349) % 1024].astype(np.float32)
    litter *= np.array([.64, .72, .61], dtype=np.float32)
    rgb = soil * mix[..., None] + litter * (1 - mix[..., None])
    light = .88 + .17 * np.sin(x * .23 - y * .38 + phase) * np.cos(y * .40)
    rgb *= light[..., None]
    # Silt exposed by floodwater and darker, wet pockets across the whole ground.
    if p["cut"]:
        cx, cy = p["cut"]
        along = .88 * (x - cx) + .475 * (y - cy)
        across = -.475 * (x - cx) + .88 * (y - cy)
        wash = np.exp(-(along / 2.1) ** 2) * np.exp(-(across / 5.2) ** 4)
        rgb = rgb * (1 - .27 * wash[..., None]) + np.array([91, 74, 56]) * (.16 * wash[..., None])
    normal = tile("brown_mud_02_nor_gl.jpg")[v, u]
    rough = tile("brown_mud_02_Rough.jpg")[v, u].astype(np.float32)
    rough = np.clip(rough * .60 + 55, 80, 240).astype(np.uint8)
    return (Image.fromarray(np.clip(rgb * 1.27, 0, 255).astype(np.uint8)),
            Image.fromarray(normal), Image.fromarray(rough))


def skyline(parameters, elevation):
    # This surrounding terrain frames the 42x30 m test area with distant mountains.
    xs = [-115, -90, -65, -45, -31, *np.arange(-21, 21.001, .125).tolist(), 31, 45, 65, 90, 115]
    ys = [-105, -80, -55, -35, -23, *np.arange(-15, 15.001, .125).tolist(), 23, 35, 55, 80, 105]
    points, faces, lookup = [], [], {}

    def vertex(i, j):
        key = i, j
        if key in lookup:
            return lookup[key]
        x, y = xs[i], ys[j]
        dx, dy = max(abs(x) - 21, 0), max(abs(y) - 15, 0)
        fade = min(1, math.hypot(dx, dy) / 18)
        h = float(elevation(x, y))
        distant = (15 * math.exp(-((x + 56) / 35) ** 2 - ((y - 49) / 40) ** 2)
                   + 21 * math.exp(-((x - 49) / 32) ** 2 - ((y - 65) / 26) ** 2)
                   + 13 * math.exp(-((x - 63) / 36) ** 2 - ((y + 39) / 36) ** 2)
                   + 8 * math.exp(-((x + 64) / 31) ** 2 - ((y + 58) / 30) ** 2))
        index = len(points)
        points.append((x, y, h + fade * distant))
        lookup[key] = index
        return index

    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            if xs[i] >= -21 and xs[i+1] <= 21 and ys[j] >= -15 and ys[j+1] <= 15:
                continue
            a, b, c, d = vertex(i, j), vertex(i+1, j), vertex(i+1, j+1), vertex(i, j+1)
            faces.append(f"{a} {b} {c} {d} -1")
    vertices = ", ".join(" ".join(f"{v:.4f}" for v in xyz) for xyz in points)
    uv = ", ".join(f"{(x+21)/42:.6f} {(y+15)/30:.6f}" for x, y, _ in points)
    key = parameters["key"]
    return f'''DEF DISTANT_MOUNTAINS Shape {{
      appearance PBRAppearance {{
        baseColorMap ImageTexture {{ url [ "textures/{key}_ground.png" ] }}
        normalMap ImageTexture {{ url [ "textures/{key}_ground_normal.jpg" ] }} normalMapFactor .7
        roughnessMap ImageTexture {{ url [ "textures/{key}_ground_rough.jpg" ] }} metalness 0
      }}
      geometry IndexedFaceSet {{ coord Coordinate {{ point [ {vertices} ] }}
        texCoord TextureCoordinate {{ point [ {uv} ] }}
        coordIndex [ {' '.join(faces)} ] texCoordIndex [ {' '.join(faces)} ] creaseAngle 0.75 }}
    }}'''


def sky():
    urls = []
    for direction in ("back", "bottom", "front", "left", "right", "top"):
        urls.append(f'{direction}Url [ "textures/sky/mountains_{direction}.jpg" ]')
        urls.append(f'{direction}IrradianceUrl [ "textures/sky/mountains_{direction}.hdr" ]')
    return f'Background {{ {" ".join(urls)} luminosity 0.8 }}\nFog {{ color .63 .68 .69 visibilityRange 350 fogType "LINEAR" }}'

def residual_water(parameters, elevation):
    if not parameters["cut"]:
        return ""
    cx, cy = parameters["cut"]
    points = []
    for i in range(41):
        across = -2.6 + 5.2 * i / 40
        along = -.12 * math.sin(across)
        for side in (-1, 1):
            local = along + side * .14
            x, y = cx + .88 * local - .475 * across, cy + .475 * local + .88 * across
            points.append((x, y, float(elevation(x, y)) + .045))
    faces = [f"{2*i} {2*i+1} {2*i+3} {2*i+2} -1" for i in range(40)]
    vertices = ", ".join(" ".join(f"{v:.5f}" for v in q) for q in points)
    return f"""DEF RESIDUAL_STREAM Shape {{
      appearance PBRAppearance {{ baseColor .25 .19 .12 roughness .24 metalness 0
        transparency .12 IBLStrength 1.1 }}
      geometry IndexedFaceSet {{ coord Coordinate {{ point [ {vertices} ] }}
        coordIndex [ {' '.join(faces)} ] creaseAngle .8 }}
    }}"""

def foliage_cluster(radius):
    vertices, uv, faces = [], [], []
    for i in range(9):
        angle = i * 2.399963
        center = (.24 * radius * math.cos(angle), .24 * radius * math.sin(angle), .18 * radius * math.sin(i))
        tangent = (.78 * radius * math.cos(angle), .78 * radius * math.sin(angle), .25 * radius * math.cos(i))
        up = (-.17 * radius * math.sin(angle), .17 * radius * math.cos(angle), .72 * radius)
        start = len(vertices)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append(tuple(center[j] + a * tangent[j] + b * up[j] for j in range(3)))
            uv.append(((a + 1) / 2, (b + 1) / 2))
        faces.extend((f"{start} {start+1} {start+2} {start+3} -1",
                      f"{start+3} {start+2} {start+1} {start} -1"))
    coords = ", ".join(" ".join(f"{v:.6f}" for v in row) for row in vertices)
    texcoords = ", ".join(f"{u} {v}" for u, v in uv)
    return f"""IndexedFaceSet {{
        coord Coordinate {{ point [ {coords} ] }}
        texCoord TextureCoordinate {{ point [ {texcoords} ] }}
        coordIndex [ {' '.join(faces)} ] texCoordIndex [ {' '.join(faces)} ]
    }}"""
