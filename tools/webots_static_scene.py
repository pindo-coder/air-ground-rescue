"""Read and export the static nodes used by the repository's Webots B world."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET


TOKEN = re.compile(r'#[^\n]*|"(?:\\.|[^"\\])*"|[{}\[\]]|[^\s,{}\[\]"#]+')
NUMBER = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')
IDENTITY = (0, 0, 0, 1)
GEOMETRIES = {"Box", "Cylinder", "Sphere", "IndexedFaceSet", "Mesh"}


@dataclass(eq=False)
class Node:
    kind: str
    fields: dict
    source: Path
    name: str = ""


@dataclass(frozen=True)
class Parameter:
    name: str


class Parser:
    def __init__(self, text, source):
        self.tokens = [token for token in TOKEN.findall(text) if not token.startswith("#")]
        self.source = source
        self.index = 0
        self.definitions = {}

    def peek(self, offset=0):
        index = self.index + offset
        return self.tokens[index] if index < len(self.tokens) else ""

    def take(self, expected=None):
        token = self.peek()
        if not token or expected is not None and token != expected:
            raise ValueError(f"{self.source.name}: expected {expected or 'a token'}, found {token!r}")
        self.index += 1
        return token

    def atom(self):
        token = self.take()
        if token.startswith('"'):
            return json.loads(token)
        if token == "TRUE":
            return True
        if token == "FALSE":
            return False
        if token == "NULL":
            return None
        if NUMBER.fullmatch(token):
            return float(token)
        if token in {"[", "]", "{", "}"}:
            raise ValueError(f"{self.source.name}: unexpected token {token!r}")
        return token

    def node(self):
        if self.peek() == "USE":
            self.take()
            return self.definitions[self.take()]
        name = ""
        if self.peek() == "DEF":
            self.take()
            name = self.take()
        kind = self.take()
        self.take("{")
        result = Node(kind, {}, self.source, name)
        if name:
            self.definitions[name] = result
        while self.peek() != "}":
            key = self.take()
            if key == "hidden":
                key = self.take()
            result.fields[key] = self.value()
        self.take("}")
        return result

    def node_ahead(self):
        return self.peek() in {"DEF", "USE"} or self.peek(1) == "{"

    def value(self):
        if self.peek() == "[":
            self.take()
            items = []
            while self.peek() != "]":
                items.append(self.node() if self.node_ahead() else self.atom())
            self.take("]")
            return items
        if self.node_ahead():
            return self.node()
        if self.peek() == "IS":
            self.take()
            return Parameter(self.take())
        first = self.atom()
        if isinstance(first, float) and NUMBER.fullmatch(self.peek()):
            numbers = [first]
            while NUMBER.fullmatch(self.peek()):
                numbers.append(self.atom())
            return tuple(numbers)
        return first

    def world(self):
        nodes = []
        while self.peek():
            if self.peek() == "EXTERNPROTO":
                self.take()
                self.atom()
                continue
            nodes.append(self.node())
        return nodes


def read_pine(source):
    text = source.read_text(encoding="utf-8")
    # Pine's template only makes its own boundingObject conditional. B supplies
    # a separate trunk collider in each outer Solid.
    interface_end = text.index("]", text.index("PROTO Pine"))
    content = re.sub(r"%<.*?>%", "", text[interface_end + 1:], flags=re.S)
    parser = Parser(content, source)
    parser.take("{")
    result = parser.node()
    parser.take("}")
    return result


def number_list(items):
    return " ".join(f"{value:.12g}" for value in items)


def element(parent, tag, value=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    child.text = value
    return child


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    x, y, z = a
    ax, ay, az = b
    return y * az - z * ay, z * ax - x * az, x * ay - y * ax


def unit(vector):
    length = math.hypot(*vector)
    if not length:
        raise ValueError("Cannot normalize a zero vector")
    return tuple(value / length for value in vector)


def multiply(a, b):
    va, sa = a[:3], a[3]
    vb, sb = b[:3], b[3]
    return (*[sa * y + sb * x + c for x, y, c in zip(va, vb, cross(va, vb))],
            sa * sb - dot(va, vb))


def rotated(q, v):
    first = cross(q[:3], v)
    second = cross(q[:3], first)
    return tuple(x + 2 * (q[3] * y + z) for x, y, z in zip(v, first, second))


@dataclass(frozen=True)
class Placement:
    xyz: tuple = (0, 0, 0)
    rotation: tuple = IDENTITY
    scale: float = 1

    def child(self, node, parameters=None):
        def get(key, default):
            value = node.fields.get(key, default)
            if isinstance(value, Parameter):
                return parameters[value.name]
            return value

        xyz = get("translation", (0, 0, 0))
        x, y, z, angle = get("rotation", (0, 0, 1, 0))
        axis = unit((x, y, z)) if angle else (0, 0, 1)
        quat = (*[a * math.sin(angle / 2) for a in axis], math.cos(angle / 2))
        scales = get("scale", (1, 1, 1))
        if not (scales[0] == scales[1] == scales[2] and scales[0] > 0):
            raise ValueError(f"Nonuniform or nonpositive scale on {node.kind} in {node.source.name}")
        offset = rotated(self.rotation, tuple(value * self.scale for value in xyz))
        return Placement(tuple(a + b for a, b in zip(self.xyz, offset)),
                         multiply(self.rotation, quat), self.scale * scales[0])

    def write(self, node):
        element(node, "pose", number_list((*self.xyz, *self.rotation)), rotation_format="quat_xyzw")


def vectors(items, width):
    if len(items) % width:
        raise ValueError(f"Invalid {width}-component vector list")
    return [tuple(items[start:start + width]) for start in range(0, len(items), width)]


def polygon_indices(items):
    result, polygon = [], []
    for value in items:
        if value == -1:
            if polygon:
                result.append(polygon)
                polygon = []
        else:
            if value < 0 or int(value) != value:
                raise ValueError(f"Invalid mesh index {value}")
            polygon.append(int(value))
    if polygon:
        result.append(polygon)
    return result


def polygon_normal(points):
    edges = [cross(a, b) for a, b in zip(points, points[1:] + points[:1])]
    return unit(tuple(map(sum, zip(*edges))))


def signed_area(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def triangles(points, normal):
    if len(points) == 3:
        yield (0, 1, 2)
        return
    axis = max(range(3), key=lambda index: abs(normal[index]))
    plane = [tuple(v for i, v in enumerate(p) if i != axis) for p in points]
    orientation = 1 if normal[axis] * (-1 if axis == 1 else 1) > 0 else -1
    remaining = list(range(len(points)))
    # Ear clipping retains concave wall profiles and carries polygon-corner UVs along.
    while len(remaining) > 3:
        for index, b in enumerate(remaining):
            a, c = remaining[index - 1], remaining[(index + 1) % len(remaining)]
            if signed_area(plane[a], plane[b], plane[c]) * orientation <= 1e-12:
                continue
            if any(all(signed_area(plane[i], plane[j], plane[k]) * orientation >= -1e-12
                       for i, j in ((a, b), (b, c), (c, a))) for k in remaining if k not in (a, b, c)):
                continue
            yield a, b, c
            del remaining[index]
            break
        else:
            raise ValueError("Could not triangulate a Webots polygon")
    yield tuple(remaining)


def texture_coordinates(node, points, faces, parameters):
    texture = node.fields.get("texCoord")
    if texture:
        uv = vectors(texture.fields["point"], 2)
    else:
        mins = [min(axis) for axis in zip(*points)]
        sizes = [max(axis) - low for axis, low in zip(zip(*points), mins)]
        first, second, _ = sorted(range(3), key=lambda i: -sizes[i])
        uv = [((point[first] - mins[first]) / sizes[first],
               (point[second] - mins[second]) / sizes[first]) for point in points]
    if node.fields.get("texCoordIndex"):
        tex_faces = polygon_indices(node.fields["texCoordIndex"])
    else:
        tex_faces = faces
    if [len(face) for face in tex_faces] != [len(face) for face in faces]:
        raise ValueError("Mesh texture indices do not match its face indices")
    center, offset, scale, rotation = parameters
    cos, sin = math.cos(rotation), math.sin(rotation)
    transformed = []
    for u, v in uv:
        x, y = (u - center[0]) * scale[0], (v - center[1]) * scale[1]
        transformed.append((center[0] + offset[0] + cos * x - sin * y,
                            center[1] + offset[1] + sin * x + cos * y))
    return transformed, tex_faces


def write_indexed_mesh(path, node, tex_transform):
    points = vectors(node.fields["coord"].fields["point"], 3)
    faces = polygon_indices(node.fields["coordIndex"])
    if node.fields.get("ccw", True) is False:
        faces = [face[::-1] for face in faces]
    uv, tex_faces = texture_coordinates(node, points, faces, tex_transform)
    if node.fields.get("ccw", True) is False and node.fields.get("texCoordIndex"):
        tex_faces = [face[::-1] for face in tex_faces]
    face_points = [[points[index] for index in face] for face in faces]
    face_normals = [polygon_normal(polygon) for polygon in face_points]
    crease = node.fields.get("creaseAngle", 0)
    neighbors = defaultdict(list)
    if crease > 0:
        for face_index, face in enumerate(faces):
            for vertex in face:
                neighbors[vertex].append(face_index)

    with path.open("w", encoding="ascii", newline="\n") as mesh:
        for prefix, items in (("v", points), ("vt", uv)):
            for row in items:
                mesh.write(f"{prefix} {number_list(row)}\n")

        explicit = node.fields.get("normal")
        if explicit:
            normals = vectors(explicit.fields["vector"], 3)
            indices = node.fields.get("normalIndex") or []
            if node.fields.get("normalPerVertex", True):
                normal_faces = polygon_indices(indices) if indices else faces
                if indices and node.fields.get("ccw", True) is False:
                    normal_faces = [face[::-1] for face in normal_faces]
            else:
                normal_faces = [[int(indices[i]) if indices else i] * len(face)
                                for i, face in enumerate(faces)]
        else:
            normals, normal_faces = [], []
            threshold = math.cos(crease)
            for face_index, face in enumerate(faces):
                direction = face_normals[face_index]
                normal_face = []
                for vertex in face:
                    incident = [face_normals[i] for i in neighbors[vertex]
                                if dot(direction, face_normals[i]) >= threshold]
                    smooth = tuple(map(sum, zip(*incident))) if incident else direction
                    normals.append(unit(smooth) if math.hypot(*smooth) > 1e-12 else direction)
                    normal_face.append(len(normals) - 1)
                normal_faces.append(normal_face)

        for normal in normals:
            mesh.write(f"vn {number_list(normal)}\n")
        for i, face in enumerate(faces):
            for indices in triangles(face_points[i], face_normals[i]):
                corners = (f"{face[k] + 1}/{tex_faces[i][k] + 1}/{normal_faces[i][k] + 1}"
                           for k in indices)
                mesh.write("f " + " ".join(corners) + "\n")


class StaticScene:
    def __init__(self, project, output):
        self.project = project
        self.output = output
        self.assets = {}
        self.meshes = {}
        self.pine = None

    def asset(self, node, url):
        path = (node.source.parent / url).resolve()
        relative = Path("resources") / path.relative_to(self.project)
        if not path.is_file():
            raise FileNotFoundError(f"Missing referenced asset: {path}")
        self.assets[path] = relative
        return relative.as_posix()

    def copy_assets(self):
        for source, relative in sorted(self.assets.items()):
            output = self.output / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, output)

    @staticmethod
    def tex_transform(appearance):
        transform = appearance.fields.get("textureTransform") if appearance else None
        fields = transform.fields if transform else {}
        return (fields.get("center", (0, 0)), fields.get("translation", (0, 0)),
                fields.get("scale", (1, 1)), fields.get("rotation", 0))

    def material(self, visual, appearance):
        if appearance is None:
            return
        if appearance.kind != "PBRAppearance":
            raise ValueError(f"Unsupported appearance: {appearance.kind}")
        material = element(visual, "material")
        fields = appearance.fields
        rgb = fields.get("baseColor", (1, 1, 1))
        element(material, "diffuse", number_list((*rgb, 1)))
        metal = element(element(material, "pbr"), "metal")
        element(metal, "roughness", number_list((fields.get("roughness", 0),)))
        element(metal, "metalness", number_list((fields.get("metalness", 1),)))
        for source, target in (("baseColorMap", "albedo_map"), ("normalMap", "normal_map"),
                               ("roughnessMap", "roughness_map")):
            texture = fields.get(source)
            if texture is not None:
                element(metal, target, self.asset(texture, texture.fields["url"][0]))
        if fields.get("transparency"):
            element(visual, "transparency", number_list((fields["transparency"],)))

    def geometry(self, owner, node, scale, appearance=None):
        geometry = element(owner, "geometry")
        if node.kind in {"Box", "Cylinder", "Sphere"}:
            solid = element(geometry, node.kind.lower())
            if node.kind == "Box":
                size = tuple(v * scale for v in node.fields.get("size", (2, 2, 2)))
                element(solid, "size", number_list(size))
            else:
                element(solid, "radius", number_list((node.fields.get("radius", 1) * scale,)))
                if node.kind == "Cylinder":
                    element(solid, "length", number_list((node.fields.get("height", 2) * scale,)))
            return

        if node.kind == "Mesh":
            uri = self.asset(node, node.fields["url"][0])
        elif node.kind == "IndexedFaceSet":
            transform = self.tex_transform(appearance)
            key = node, transform
            if key not in self.meshes:
                # The same original geometry object is reused for matching collision
                # shapes and all Pine instances when the UV transform agrees.
                stem = (node.name or node.kind).lower()
                path = Path("meshes") / f"{len(self.meshes):04d}_{stem}.obj"
                (self.output / path).parent.mkdir(parents=True, exist_ok=True)
                write_indexed_mesh(self.output / path, node, transform)
                self.meshes[key] = path.as_posix()
            uri = self.meshes[key]
        else:
            raise ValueError(f"Unsupported geometry {node.kind} from {node.source.name}")
        mesh = element(geometry, "mesh")
        element(mesh, "uri", uri)
        if scale != 1:
            element(mesh, "scale", number_list((scale, scale, scale)))

    def shapes(self, node, placement, collision=False, parameters=None):
        if node is None:
            return
        if node.kind == "Shape":
            appearance = node.fields.get("appearance")
            if collision or not appearance or appearance.fields.get("transparency", 0) < 1:
                yield node.fields["geometry"], appearance, placement
        elif node.kind in GEOMETRIES:
            if not collision:
                raise ValueError(f"Expected Shape wrapper for {node.kind}")
            yield node, None, placement
        elif node.kind == "Pine":
            if collision or node.fields.get("enableBoundingObject", True):
                raise ValueError("B expects Pine instances with an explicit outer trunk collider")
            if self.pine is None:
                self.pine = read_pine(self.project / "protos/Pine.proto")
            fields = {"translation": (0, 0, 0), "rotation": (0, 0, 1, 0),
                      "scale": (1, 1, 1), **node.fields}
            yield from self.shapes(self.pine, placement, parameters=fields)
        elif node.kind in {"Group", "Pose", "Transform", "Solid"}:
            relative = placement.child(node, parameters)
            for child in node.fields.get("children", []):
                yield from self.shapes(child, relative, collision, parameters)
        else:
            raise ValueError(f"Unsupported B scene node {node.kind} in {node.source.name}")

    def static_model(self, world, node):
        if not node.name:
            raise ValueError(f"Expected a DEF name for root {node.kind}")
        model = element(world, "model", name=node.name.lower())
        Placement().child(node).write(model)
        element(model, "static", "true")
        link = element(model, "link", name="body")
        visual_roots = node.fields.get("children", []) if node.kind == "Solid" else [node]
        visuals = chain.from_iterable(self.shapes(child, Placement()) for child in visual_roots)
        collisions = self.shapes(node.fields.get("boundingObject"), Placement(), collision=True)
        for tag, shapes in (("visual", visuals), ("collision", collisions)):
            for index, (geometry, appearance, placement) in enumerate(shapes, 1):
                name = f"{tag}_{index}_{(geometry.name or geometry.kind).lower()}"
                item = element(link, tag, name=name)
                placement.write(item)
                self.geometry(item, geometry, placement.scale, appearance if tag == "visual" else None)
                if tag == "visual":
                    self.material(item, appearance)
                    if node.kind == "Shape":
                        element(item, "cast_shadows", "false")
