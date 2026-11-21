"""Run the aerial-map handoff in a derived D world and retain all run evidence."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from build_search_worlds import build, replace_node
from metrics import height_at, look_at_rotation
from run_search import ROOT, executable_path, referenced_assets, simulator_has_errors
from uav_map_preview import audit_alignment, draw_preview, load_prior
from uav_full_preview import draw_aerial_products, draw_full_preview, draw_refinement_preview
from aerial_products import export_products


PROFILES = {"full": "uav_mapping_full_D.json", "local": "uav_mapping_D.json"}


def survey_metrics(map_file):
    """Virtual terrain reference is read here after publishing/planning, for evaluation."""
    grid, states = load_prior(map_file)
    with np.load(map_file, allow_pickle=False) as data:
        elevation = data["ground_z_m"]
        observations = data["observed_views"]
        hazards = data["hazard_type"]
    valid = np.isfinite(elevation)
    truth = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8"))
    error = [abs(elevation[cell] - height_at(truth, *grid.center(cell)))
             for cell in zip(*np.nonzero(valid))]
    return dict(
        extent_m=(np.asarray(grid.shape[::-1]) * grid.resolution).tolist(),
        total_cells=int(states.size), observed_cells=int(np.count_nonzero(observations)),
        cells_observed_twice_or_more=int(np.count_nonzero(observations >= 2)),
        cells_with_ground_elevation=int(valid.sum()),
        ground_elevation_range_m=[float(elevation[valid].min()), float(elevation[valid].max())],
        terrain_hazards={name: int(np.count_nonzero(hazards == code))
                         for code, name in ((1, "steep"), (2, "step_or_drop"), (3, "rough"))},
        observed_ground_cell_error_m=dict(median=float(np.median(error)),
                                          p90=float(np.quantile(error, 0.9))),
    )


def refinement_metrics(folder, observations, schedule, config=None):
    grid, nadir = load_prior(folder / "uav_map_nadir.npz")
    final_grid, final = load_prior(folder / "uav_map.npz")
    if final_grid != grid:
        raise ValueError("Oblique correction and nadir survey must use the same grid.")
    added = (nadir == -1) & (final == 0)
    unchanged_known = bool(np.array_equal(nadir[nadir != -1], final[nadir != -1]))
    with np.load(folder / "uav_map.npz", allow_pickle=False) as data:
        elevation = data["ground_z_m"]
        quarters = data["oblique_ground_quarters"]
    truth = json.loads((ROOT / "evaluation/search/D.json").read_text(encoding="utf-8"))
    errors = [abs(elevation[cell] - height_at(truth, *grid.center(cell)))
              for cell in zip(*np.nonzero(added))]
    corridor_metrics = {}
    if config is not None and config.get("route_goals"):
        sys.path.insert(0, str(ROOT / "controllers/uav_map_rover"))
        from grid_route import plan_dual
        with np.load(folder / "uav_map_nadir.npz", allow_pickle=False) as nadir_data:
            nadir_cost = nadir_data["planning_cost"]
        start = config["deployment_xy_enu_m"]
        radius = config["rover_radius_m"]
        corridor_cells = set()
        for case in config["route_goals"]:
            dual = plan_dual(grid, nadir, start, case["xy_enu_m"], radius, nadir_cost)
            if dual["changes_reachability"] or (dual["potential_saving_m"] or 0.0) >= 2.0:
                corridor_cells.update(map(tuple, dual["corridor_unknown_cells"]))
        if corridor_cells:
            corridor_mask = np.zeros(grid.shape, dtype=bool)
            for cell in corridor_cells:
                corridor_mask[cell] = True
            corridor_metrics = dict(
                high_value_corridor_unknown_before=int(np.count_nonzero(corridor_mask & (nadir == -1))),
                high_value_corridor_unknown_after=int(np.count_nonzero(corridor_mask & (final == -1))),
                high_value_corridor_newly_supported=int(np.count_nonzero(corridor_mask & added)),
                high_value_corridor_partial_evidence=int(np.count_nonzero(corridor_mask & (final == -1) & (quarters > 0))),
            )
    return dict(
        strategy=schedule.get("strategy", "density"),
        nadir_frames=int(np.count_nonzero(observations["survey_phase"] == 0)),
        oblique_frames=int(np.count_nonzero(observations["survey_phase"] == 1)),
        oblique_finish_s=schedule["end_s"],
        patches=len(schedule["patches"]),
        patch_centers_xy_m=[patch["xy_m"] for patch in schedule["patches"]],
        unknown_before=int(np.count_nonzero(nadir == -1)),
        unknown_after=int(np.count_nonzero(final == -1)),
        newly_supported_cells=int(added.sum()),
        newly_supported_area_m2=float(added.sum() * grid.resolution**2),
        unknown_with_some_floor_evidence=int(np.count_nonzero((final == -1) & (quarters > 0))),
        previous_known_states_preserved=unchanged_known,
        added_ground_height_reference=dict(median_m=float(np.median(errors)),
                                           p90_m=float(np.quantile(errors, 0.9)),
                                           above_0_20_m=int(np.count_nonzero(np.asarray(errors) > 0.20)))
        if errors else None,
        **corridor_metrics,
    )


def demo_world(path, config):
    source = ROOT / "worlds/search_D.wbt"
    terrain_file = ROOT / "evaluation/search/D.json"
    if not source.is_file() or not terrain_file.is_file():
        source = build("D")
    scene = source.read_text(encoding="utf-8")
    terrain = json.loads(terrain_file.read_text(encoding="utf-8"))
    flight = config["flight"]
    x, y = flight["waypoints_xy_enu_m"][0]
    z = flight["altitude_enu_m"]
    scene = replace_node(scene, r"\bDEF\s+OBSERVER_DRONE\s+ObserverDrone",
                         f'DEF OBSERVER_DRONE ObserverDrone {{ translation {x} {y} {z} controller "uav_mapper" }}')
    # These coordinates only place the vehicle in the simulation. The rover
    # receives its starting position from its GPS after deployment.
    start = config["deployment_xy_enu_m"]
    position = [*start, height_at(terrain, *start) + 0.035]
    fixture = json.dumps(json.dumps({"deployment_xyz_m": position}))
    scene = replace_node(scene, r"\bDEF\s+DISASTER_ENVIRONMENT\s+Robot",
                         'DEF SURVEY Robot { name "Aerial grid survey" supervisor TRUE '
                         f'controller "uav_survey" customData {fixture} window "<none>" }}')
    scene = replace_node(scene, r"\bDEF\s+SEARCH_REFEREE\s+Robot", "")
    west, south = config["bounds_enu_m"][0]
    staging = [west - 12, south - 10]
    stage_z = height_at(terrain, *staging) + 0.035
    scene = replace_node(scene, r"\bDEF\s+ROVER\s+ResearchRover",
                         f'DEF ROVER ResearchRover {{ name "Map reading rover" controller "uav_map_rover" '
                         f'translation {staging[0]} {staging[1]} {stage_z} }}')
    (xmin, ymin), (xmax, ymax) = terrain["bounds_xy"]
    if not (xmin <= staging[0] <= xmax and ymin <= staging[1] <= ymax):
        # Stationary holding platform outside the survey footprint.
        scene += (f'\nSolid {{ translation {staging[0]} {staging[1]} {stage_z - 0.085} '
                  'boundingObject Box { size 2 2 .1 } }\n')
    center = [(low + high) / 2 for low, high in zip(*config["bounds_enu_m"])]
    width = config["bounds_enu_m"][1][0] - config["bounds_enu_m"][0][0]
    eye = [center[0] - 0.6 * width, center[1] - max(20, 0.82 * width), max(25, 0.60 * width)]
    angles = " ".join(map(str, look_at_rotation(eye, [*center, 2.0])))
    scene = replace_node(scene, r"\bDEF\s+SCENE_VIEW\s+Viewpoint",
                         f'DEF SCENE_VIEW Viewpoint {{ position {" ".join(map(str, eye))} '
                         f'orientation {angles} fieldOfView 1.0 exposure 1.25 }}')
    path.write_text(scene, encoding="utf-8")
    referenced_assets(path)


def run(visible=False, port=18554, profile="full", nadir_only=False, refinement_strategy=None):
    config = json.loads((ROOT / "config/scenes" / PROFILES[profile]).read_text(encoding="utf-8"))
    if nadir_only:
        config.pop("refinement", None)
    elif refinement_strategy and "refinement" in config:
        config["refinement"]["strategy"] = refinement_strategy
    strategy_tag = (f"_{refinement_strategy}" if refinement_strategy and not nadir_only else "")
    run_id = f"uav_mapping_D_{profile}{strategy_tag}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    folder = ROOT / "runs" / run_id
    folder.mkdir(parents=True)
    (folder / "map_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    world = ROOT / "worlds" / f"_validation_{run_id}.wbt"
    demo_world(world, config)
    executable = executable_path()
    if not executable.is_file():
        raise FileNotFoundError(f"Webots executable not found: {executable}")
    runtime = f"[python]\nCOMMAND = {Path(sys.executable).as_posix()}\n"
    for controller in ("uav_survey", "uav_mapper", "uav_map_rover"):
        (ROOT / f"controllers/{controller}/runtime.ini").write_text(runtime, encoding="utf-8")
    flight = config["flight"]
    points = flight["waypoints_xy_enu_m"]
    duration = sum(math.dist(a, b) for a, b in zip(points, points[1:])) / flight["speed_mps"]
    env = dict(os.environ, UAV_MAPPING_RUN=run_id, UAV_MAPPING_FLIGHT_END=str(duration),
               PYTHONDONTWRITEBYTECODE="1")
    command = [str(executable), "--batch", "--mode=fast", f"--port={port}", "--stdout", "--stderr"]
    if not visible:
        command.append("--no-rendering")
    command.append(str(world))
    log = folder / "webots.log"
    print(f"Run: {folder}", flush=True)
    budget = duration + (160 if "refinement" in config else 0)
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=max(240, budget * 4), text=True,
                            encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    output = result.stdout
    log.write_text(output, encoding="utf-8")
    if result.returncode or simulator_has_errors(output):
        raise RuntimeError(f"Webots exited with code {result.returncode}. See {log}\n{output[-2500:]}")
    plan = json.loads((folder / "route.json").read_text(encoding="utf-8"))
    with np.load(folder / "observations.npz", allow_pickle=False) as measurements:
        observations = dict(measurements)
    products = export_products(folder, observations)
    draw_aerial_products(folder, products)
    supplement = folder / "oblique_survey.json"
    schedule = json.loads(supplement.read_text(encoding="utf-8")) if supplement.is_file() else None
    cells = (draw_full_preview(folder, plan, observations, config, schedule) if profile == "full"
             else draw_preview(folder, plan, observations))
    summary = dict(status=plan["status"], profile=profile,
                   captured_views=len(observations["timestamp_s"]),
                   survey_end_s=duration, alignment=audit_alignment(folder, observations),
                   cells=cells, mapping=survey_metrics(folder / "uav_map.npz"),
                   aerial_products=products)
    summary["routes"] = [{key: case[key] for key in
                          ("id", "status", "goal_state", "terrain", "distance_only", "reason", "nadir", "dual_route")
                          if key in case} for case in plan["routes"]]
    if schedule is not None:
        summary["refinement"] = refinement_metrics(folder, observations, schedule, config)
        if schedule["patches"]:
            draw_refinement_preview(folder, schedule, summary["refinement"])
    for key in ("route_length_m", "straight_line", "reason"):
        if key in plan:
            summary[key] = plan[key]
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return folder, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--nadir-only", action="store_true", help="Run the initial sweep without follow-up views.")
    parser.add_argument("--refinement-strategy", choices=("density", "route_value"), default=None,
                        help="Oblique patch selector: 'density' baseline or 'route_value' (Traversing Mars + DUCPP).")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--open-preview", action="store_true", help="Open the resulting PNG on Windows.")
    parser.add_argument("--port", type=int, default=18554)
    args = parser.parse_args()
    folder, plan = run(args.visible, args.port, args.profile, args.nadir_only, args.refinement_strategy)
    if args.open_preview and os.name == "nt":
        os.startfile(folder / "preview.png")
    return 0 if plan["status"] == "PLANNED" else 1


if __name__ == "__main__":
    sys.exit(main())
