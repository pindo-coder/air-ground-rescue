"""Local textured casualty rendering, independent of collision and policy code."""
from functools import lru_cache
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The body already contains a short neck. Seat the scan inside that collar
# instead of exposing a second, long cylindrical neck between the two meshes.
HEAD_SCAN_CENTER = (-.60, 0, .143)
HEAD_PROXY_POSITIONS = {
    "HEAD": (-.60, 0, .133),
    "HAIR": (-.63, 0, .128),
    "NOSE": (-.58, 0, .238),
    "EYE_L": (-.62, .043, .236),
    "EYE_R": (-.62, -.043, .236),
    "NECK": (-.475, 0, .125),
}
NECK_PROXY_DIMENSIONS = (.055, .07)
VISUAL_MODEL = "textured_mesh_v2"


@lru_cache(maxsize=1)
def human_catalog():
    return json.loads((ROOT / "assets/render/catalog.json").read_text(encoding="utf-8"))["human"]


def casualty_render(key):
    shapes = []
    for item in human_catalog()[key]["parts"]:
        color = item.get("color", "1 1 1")
        maps = []
        for field, source in (("baseColorMap", "albedo"), ("normalMap", "normal")):
            if item.get(source):
                url = json.dumps("../assets/" + item[source])
                maps.append(f"{field} ImageTexture {{ url [ {url} ] filtering 5 }}")
        if item["mesh"].endswith(("_shirt.obj", "_pants.obj")):
            maps.append('roughnessMap ImageTexture { url [ "../assets/upstream/webots/projects/appearances/protos/textures/fabric/fabric_roughness.jpg" ] filtering 5 }')
        app = f"PBRAppearance {{ baseColor {color} {' '.join(maps)} normalMapFactor .55 roughness .95 metalness 0 }}"
        mesh = json.dumps("../assets/render/" + item["mesh"])
        shapes.append(f"Shape {{ appearance {app} geometry Mesh {{ url [ {mesh} ] }} }}")
    return "DEF CASUALTY_RENDER Group { children [\n" + "\n".join(shapes) + "\n] }"
