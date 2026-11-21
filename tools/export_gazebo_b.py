"""Export the static environment from the committed Webots B world."""

import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from webots_static_scene import Parser, Placement, StaticScene, element, number_list


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "worlds" / "B_washout.wbt"
DEFAULT_OUTPUT = ROOT / "gazebo" / "generated" / "B-full"


def terrain_grid(terrain):
    shape, = terrain.fields["children"]
    grid = shape.fields["geometry"]
    if grid.kind != "ElevationGrid":
        raise ValueError("B terrain must contain an ElevationGrid")
    fields = grid.fields
    nx, ny = (int(fields[key]) for key in ("xDimension", "yDimension"))
    dx, dy = (fields[key] for key in ("xSpacing", "ySpacing"))
    heights = fields["height"]
    if nx < 2 or ny < 2 or dx <= 0 or dy <= 0 or len(heights) != nx * ny:
        raise ValueError("ElevationGrid dimensions, spacing and height count do not agree")
    if not all(map(math.isfinite, heights)):
        raise ValueError("ElevationGrid contains a non-finite height")
    return nx, ny, dx, dy, heights


def write_terrain_obj(path, grid):
    nx, ny, dx, dy, heights = grid
    with path.open("w", encoding="ascii", newline="\n") as mesh:
        mesh.write("# Webots B ElevationGrid, in its original local coordinates\n")
        for j in range(ny):
            for i in range(nx):
                mesh.write(f"v {number_list((i * dx, j * dy, heights[j * nx + i]))}\n")
        for j in range(ny):
            for i in range(nx):
                # OBJ UV starts at the bottom left; the image top faces positive Y.
                mesh.write(f"vt {number_list((i / (nx - 1), j / (ny - 1)))}\n")
        for j in range(ny):
            south, north = max(0, j - 1), min(ny - 1, j + 1)
            for i in range(nx):
                west, east = max(0, i - 1), min(nx - 1, i + 1)
                dz_dx = (heights[j * nx + east] - heights[j * nx + west]) / ((east - west) * dx)
                dz_dy = (heights[north * nx + i] - heights[south * nx + i]) / ((north - south) * dy)
                length = math.hypot(dz_dx, dz_dy, 1)
                mesh.write(f"vn {number_list((-dz_dx / length, -dz_dy / length, 1 / length))}\n")
        for j in range(ny - 1):
            for i in range(nx - 1):
                sw = j * nx + i + 1  # OBJ indices start at one.
                se, nw, ne = sw + 1, sw + nx, sw + nx + 1
                # Match the SW-to-NE diagonal used by the Webots grid.
                for face in ((sw, se, ne), (sw, ne, nw)):
                    mesh.write("f " + " ".join(f"{k}/{k}/{k}" for k in face) + "\n")


def add_terrain(world, terrain, exporter):
    model = element(world, "model", name="terrain")
    Placement().child(terrain).write(model)
    element(model, "static", "true")
    link = element(model, "link", name="body")
    for tag in ("visual", "collision"):
        name = "elevation_grid_visual" if tag == "visual" else "elevation_grid"
        item = element(link, tag, name=name)
        element(element(element(item, "geometry"), "mesh"), "uri", "terrain.obj")
        if tag == "visual":
            shape, = terrain.fields["children"]
            exporter.material(item, shape.fields["appearance"])


def add_overview_gui(world):
    gui = element(world, "gui")
    view = element(gui, "plugin", filename="MinimalScene", name="3D View")
    layout = element(view, "gz-gui")
    element(layout, "title", "3D View")
    element(layout, "property", "false", key="showTitleBar", type="bool")
    element(layout, "property", "docked", key="state", type="string")
    element(view, "engine", "ogre2")
    element(view, "scene", "scene")
    element(view, "camera_pose", number_list((0, 0, 53, 0, math.pi / 2, math.pi / 2)))

    # Harmonic uses the SDF's GUI plugins in place of its default layout.
    for filename, name in (("GzSceneManager", "Scene Manager"),
                           ("InteractiveViewControl", "Interactive view control")):
        layout = element(element(gui, "plugin", filename=filename, name=name), "gz-gui")
        element(layout, "property", "floating", key="state", type="string")
        element(layout, "property", "false", key="showTitleBar", type="bool")
        for dimension in ("width", "height"):
            element(layout, "property", "5", key=dimension, type="double")

    control = element(gui, "plugin", filename="WorldControl", name="World control")
    layout = element(control, "gz-gui")
    element(layout, "property", "floating", key="state", type="string")
    element(layout, "property", "false", key="showTitleBar", type="bool")
    element(layout, "property", "72", key="height", type="double")
    element(layout, "property", "1", key="z", type="double")
    anchors = element(layout, "anchors", target="3D View")
    for edge in ("left", "bottom"):
        element(anchors, "line", own=edge, target=edge)
    for option in ("play_pause", "step", "use_event"):
        element(control, option, "true")

    tree = element(gui, "plugin", filename="EntityTree", name="Entity tree")
    layout = element(tree, "gz-gui")
    element(layout, "property", "docked", key="state", type="string")


def export(output):
    nodes = Parser(SOURCE.read_text(encoding="utf-8"), SOURCE).world()
    terrain, = (node for node in nodes if node.name == "TERRAIN")
    grid = terrain_grid(terrain)

    sdf = ET.Element("sdf", version="1.10")
    world = element(sdf, "world", name="b_washout")
    physics = element(world, "physics", name="default", type="ignored")
    element(physics, "max_step_size", "0.001")
    element(physics, "real_time_factor", "1.0")
    scene = element(world, "scene")
    element(scene, "ambient", "0.6 0.6 0.6 1")
    element(scene, "background", "0.62 0.73 0.81 1")
    add_overview_gui(world)
    lights = (node for node in nodes if node.kind == "DirectionalLight")
    for index, source in enumerate(lights):
        fields = source.fields
        light = element(world, "light", name="sun" if index == 0 else f"fill_{index}",
                        type="directional")
        element(light, "cast_shadows", str(fields.get("castShadows", False)).lower())
        element(light, "diffuse", number_list((*fields.get("color", (1, 1, 1)), 1)))
        element(light, "direction", number_list(fields["direction"]))
        element(light, "intensity", number_list((fields.get("intensity", 1),)))

    output.mkdir(parents=True, exist_ok=True)
    exporter = StaticScene(ROOT, output)
    add_terrain(world, terrain, exporter)
    for node in nodes:
        if node is not terrain and node.kind in {"Solid", "Shape"}:
            exporter.static_model(world, node)
    write_terrain_obj(output / "terrain.obj", grid)
    exporter.copy_assets()
    path = output / "b_washout.sdf"
    ET.indent(sdf, space="  ")
    ET.ElementTree(sdf).write(path, encoding="utf-8", xml_declaration=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="directory for the SDF, terrain mesh, and textures")
    args = parser.parse_args()
    print(export(args.output.resolve()))


if __name__ == "__main__":
    main()
