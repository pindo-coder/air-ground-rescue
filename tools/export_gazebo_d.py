"""Export the regenerated Webots D scenario to a portable Gazebo Harmonic bundle."""

import argparse
import json
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from export_gazebo_b import add_overview_gui, terrain_grid, terrain_normal, write_terrain_obj
from gazebo_d_robots import add_drone, add_rover
from webots_static_scene import Parser, Placement, StaticScene, element, number_list


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "worlds" / "search_D.wbt"
DEFAULT_OUTPUT = ROOT / "gazebo" / "generated" / "D"
TILE_CELLS = 48


def terrain_collision_tiles(output, grid):
    nx, ny, dx, dy, heights = grid
    directory = output / "collision"
    directory.mkdir(exist_ok=True)
    (directory / "ground.mtl").write_text("newmtl ground\nKd 1 1 1\n", encoding="ascii")
    for start_y in range(0, ny - 1, TILE_CELLS):
        for start_x in range(0, nx - 1, TILE_CELLS):
            end_x = min(start_x + TILE_CELLS, nx - 1)
            end_y = min(start_y + TILE_CELLS, ny - 1)
            width = end_x - start_x + 1
            path = directory / f"tile_{start_y}_{start_x}.obj"
            with path.open("w", encoding="ascii", newline="\n") as mesh:
                mesh.write("mtllib ground.mtl\nusemtl ground\n")
                for j in range(start_y, end_y + 1):
                    for i in range(start_x, end_x + 1):
                        mesh.write(f"v {number_list((i * dx, j * dy, heights[j * nx + i]))}\n")
                # Harmonic's DART mesh importer requires a normal for each vertex.
                for j in range(start_y, end_y + 1):
                    for i in range(start_x, end_x + 1):
                        mesh.write(f"vn {number_list(terrain_normal(grid, i, j))}\n")
                for j in range(end_y - start_y):
                    for i in range(width - 1):
                        sw = j * width + i + 1
                        se, nw, ne = sw + 1, sw + width, sw + width + 1
                        mesh.write(f"f {sw}//{sw} {se}//{se} {ne}//{ne}\n")
                        mesh.write(f"f {sw}//{sw} {ne}//{ne} {nw}//{nw}\n")
            yield path


def add_terrain(world, source, exporter, grid):
    model = element(world, "model", name="terrain")
    Placement().child(source).write(model)
    element(model, "static", "true")
    link = element(model, "link", name="body")
    visual = element(link, "visual", name="surface")
    element(element(element(visual, "geometry"), "mesh"), "uri", "terrain.obj")
    shape, = source.fields["children"]
    exporter.material(visual, shape.fields["appearance"])
    for path in terrain_collision_tiles(exporter.output, grid):
        collision = element(link, "collision", name=path.stem)
        uri = path.relative_to(exporter.output).as_posix()
        element(element(element(collision, "geometry"), "mesh"), "uri", uri)
    write_terrain_obj(exporter.output / "terrain.obj", grid)


def add_lights(world, nodes):
    for index, source in enumerate(node for node in nodes if node.kind == "DirectionalLight"):
        fields = source.fields
        light = element(world, "light", name="sun" if index == 0 else f"fill_{index}",
                        type="directional")
        element(light, "cast_shadows", str(fields.get("castShadows", False)).lower())
        element(light, "diffuse", number_list((*fields.get("color", (1, 1, 1)), 1)))
        element(light, "direction", number_list(fields["direction"]))
        element(light, "intensity", str(fields["intensity"]))


def add_gui(world, viewpoint):
    eye = viewpoint.fields["position"]
    forward = (-eye[0], -eye[1], 2 - eye[2])
    pitch = math.atan2(-forward[2], math.hypot(*forward[:2]))
    yaw = math.atan2(forward[1], forward[0])
    gui = add_overview_gui(world, (*eye, 0, pitch, yaw))
    for title, namespace in (("Rover RGB", "rover"), ("Drone RGB", "drone")):
        image = element(gui, "plugin", filename="ImageDisplay", name=title)
        layout = element(image, "gz-gui")
        element(layout, "title", title)
        element(layout, "property", "docked", key="state", type="string")
        element(layout, "property", "320", key="width", type="double")
        element(layout, "property", "240", key="height", type="double")
        element(image, "topic", f"/{namespace}/rgb")
        element(image, "topic_picker", "true")


def write_environment(output, config, terrain, grid):
    nx, ny, dx, dy, flat = grid
    aerial = config["aerial"]
    routes = [
        {"model": "drone", "path": aerial["path"], "speed_mps": aerial["speed_mps"],
         "phase_s": 0.0, "altitude_enu_m": aerial["altitude_enu_m"]}
    ]
    for obstacle in config["moving_obstacles"]:
        routes.append({
            "model": obstacle["def"].lower(),
            "path": obstacle["path"],
            "speed_mps": obstacle["speed_mps"],
            "phase_s": obstacle["phase_s"],
            "height_m": obstacle["size"][2],
        })
    runtime = {
        "origin": terrain.fields["translation"][:2],
        "spacing": [dx, dy],
        "heights": [flat[j * nx:(j + 1) * nx] for j in range(ny)],
        "routes": routes,
    }
    (output / "environment.json").write_text(json.dumps(runtime), encoding="utf-8")
    shutil.copyfile(ROOT / "gazebo" / "rescue_d_environment.py", output / "rescue_d_environment.py")
    shutil.copyfile(ROOT / "controllers" / "disaster_environment" / "motion.py", output / "motion.py")
    shutil.copyfile(ROOT / "gazebo" / "run_d.sh", output / "run.sh")
    (output / "run.sh").chmod(0o755)


def export(output):
    source = SOURCE.read_text(encoding="utf-8")
    nodes = Parser(source, SOURCE).world()
    named = {node.name: node for node in nodes if node.name}
    world_info = next(node for node in nodes if node.kind == "WorldInfo")
    friction = {entry.fields["material1"]: entry.fields["coulombFriction"][0]
                for entry in world_info.fields["contactProperties"]}
    config = json.loads((ROOT / "config" / "scenes" / "D.json").read_text(encoding="utf-8"))
    terrain = named["TERRAIN"]
    grid = terrain_grid(terrain)
    sdf = ET.Element("sdf", version="1.10")
    world = element(sdf, "world", name="search_d")
    physics = element(world, "physics", name="default", type="ignored")
    element(physics, "max_step_size", "0.002")
    element(physics, "real_time_factor", "1.0")
    scene = element(world, "scene")
    element(scene, "ambient", "0.5 0.5 0.5 1")
    element(scene, "background", "0.62 0.73 0.81 1")

    # A world with explicit plugins needs Gazebo's base world systems as well.
    for filename, name in (
        ("physics", "Physics"), ("user-commands", "UserCommands"),
        ("scene-broadcaster", "SceneBroadcaster"), ("imu", "Imu")
    ):
        element(world, "plugin", filename=f"gz-sim-{filename}-system",
                name=f"gz::sim::systems::{name}")
    sensors = element(world, "plugin", filename="gz-sim-sensors-system", name="gz::sim::systems::Sensors")
    element(sensors, "render_engine", "ogre2")
    environment = element(world, "plugin", filename="gz-sim-python-system-loader-system",
                          name="gz::sim::systems::PythonSystemLoader")
    element(environment, "module_name", "rescue_d_environment")
    add_lights(world, nodes)
    add_gui(world, named["SCENE_VIEW"])

    output.mkdir(parents=True, exist_ok=True)
    exporter = StaticScene(ROOT, output)
    add_terrain(world, terrain, exporter, grid)
    for node in nodes:
        if node is not terrain and node.kind in {"Solid", "Shape"}:
            exporter.static_model(world, node)
    for node in named["D_FOREST"].fields["children"]:
        exporter.static_model(world, node)
    add_rover(world, named["ROVER"], exporter, friction)
    add_drone(world, named["OBSERVER_DRONE"], exporter, config["aerial"]["image_period_s"])
    exporter.copy_assets()
    write_environment(output, config, terrain, grid)
    path = output / "search_d.sdf"
    ET.indent(sdf, space="  ")
    ET.ElementTree(sdf).write(path, encoding="utf-8", xml_declaration=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not SOURCE.is_file():
        parser.error("run python3 tools/build_search_worlds.py --case D first")
    print(export(args.output.resolve()))


if __name__ == "__main__":
    main()
