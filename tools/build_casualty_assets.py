"""Rebuild bundled human meshes offline from attributed source files."""
import json
from pathlib import Path
import struct

import numpy as np
from human_visual import human_assets
from casualty_visual import HEAD_SCAN_CENTER

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
UP = ASSETS / "upstream"
RENDER = ASSETS / "render"


def write_obj(path, vertices, uv, faces, normals=None):
    used = np.unique(np.asarray(faces))
    mapping = np.zeros(len(vertices), dtype=int)
    mapping[used] = np.arange(len(used))
    faces = mapping[np.asarray(faces)]
    vertices, uv, normals = vertices[used], uv[used], normals[used]
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.write("# Converted render mesh. Source and license: assets/CREDITS.md\n")
        for prefix, rows in (("v", vertices), ("vt", uv), ("vn", normals)):
            for row in rows:
                out.write(prefix + " " + " ".join(f"{v:.6f}" for v in row) + "\n")
        for face in faces:
            out.write("f " + " ".join(f"{int(v)+1}/{int(v)+1}/{int(v)+1}" for v in face) + "\n")


def head_asset():
    raw = (UP / "head_scan/LeePerrySmith.glb").read_bytes()
    length = struct.unpack_from("<I", raw, 12)[0]
    model = json.loads(raw[20:20 + length])
    binary = raw[28 + length:]

    def accessor(index):
        a = model["accessors"][index]
        view = model["bufferViews"][a["bufferView"]]
        dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[a["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        return np.frombuffer(binary, dtype=dtype, count=a["count"] * width,
                             offset=view.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(-1, width).copy()

    primitive = model["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    vertex = accessor(attributes["POSITION"])
    normal = accessor(attributes["NORMAL"])
    uv = accessor(attributes["TEXCOORD_0"])
    uv[:, 1] = 1 - uv[:, 1]
    indices = accessor(primitive["indices"]).reshape(-1, 3)
    indices = indices[np.all(vertex[indices, 1] > -1.25, axis=1)]
    source = vertex[np.unique(indices)][:, [1, 0, 2]] * [-1, 1, 1]
    lo, hi = source.min(0), source.max(0)
    scale = .26 / (hi[0] - lo[0])
    vertex = (vertex[:, [1, 0, 2]] * [-1, 1, 1] - (lo + hi) / 2) * scale + HEAD_SCAN_CENTER
    normal = normal[:, [1, 0, 2]] * [-1, 1, 1]
    # Extend only the cropped neck end into the shirt, leaving the face intact.
    # Its raw cut edge is then hidden below the collar rather than exposed.
    t = np.clip((vertex[:, 0] + .535) / .065, 0, 1)
    vertex[:, 0] += .035 * t * t * (3 - 2 * t)
    normal[:, 0] /= 1 + .035 * 6 * t * (1 - t) / .065
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    write_obj(RENDER / "head_scan.obj", vertex, uv, indices, normals=normal)


def main():
    RENDER.mkdir(parents=True, exist_ok=True)
    head_asset()
    data = {"revision": 2, "head_center_local_m": list(HEAD_SCAN_CENTER), "human": human_assets(UP, RENDER, write_obj)}
    (RENDER / "catalog.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("Rebuilt 12 body-part meshes and one scanned-head mesh from local sources.")


if __name__ == "__main__":
    main()
