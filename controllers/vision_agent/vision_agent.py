"""Camera-only adapter. No Supervisor, terrain reader, oracle pose or navigation."""
from pathlib import Path
import json
import math
import os
import time

from controller import Robot
from policy import Policy

robot = Robot()
task = json.loads(robot.getCustomData())
scenario_id = task["scenario_id"]
period = task["camera"]["period_ms"]
root = Path(__file__).resolve().parents[2]
run_id = os.environ.get("VISION_RUN_ID", time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
public_out = root / "runs" / run_id / scenario_id / "observations"
public_out.mkdir(parents=True, exist_ok=True)
(public_out / "initial_task.json").write_text(json.dumps(task, indent=2), encoding="utf-8")
cameras = [robot.getDevice(n) for n in ("camera_left", "camera_right")]
motors = [robot.getDevice(n) for n in ("left_motor::front", "right_motor::front")]
for camera in cameras:
    camera.enable(period)
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)
policy = Policy()
policy.reset(task)
saved = set()
ended = False
record_every = max(0,int(os.environ.get("VISION_RECORD_EVERY_N", "0")))
record_limit = max(0,int(os.environ.get("VISION_RECORD_MAX_PAIRS", "300")))
frame_number = 0
recorded_pairs = 0
record_file = (public_out / "frames.jsonl").open("w",encoding="utf-8") if record_every else None
print(f"[Vision Agent] {scenario_id}: initial direction {task['goal_hint']['center_deg_left_positive']:+.0f} +/- {task['goal_hint']['half_width_deg']} deg; no live goal bearing.", flush=True)
print("[Vision Agent] Low-speed stereo exploration enabled." if policy.enabled else "[Vision Agent] Preview policy: stationary.", flush=True)
last_mode = None
while robot.step(period) != -1:
    now = robot.getTime()
    try:
        message = json.loads(robot.getCustomData() or "{}")
    except json.JSONDecodeError:
        message = {}
    ended = ended or bool(message.get("episode_end"))
    left, right = cameras[0].getImage(), cameras[1].getImage()
    if left is None or right is None:
        continue
    observation = {"left_bgra": bytes(left), "right_bgra": bytes(right), "width": cameras[0].getWidth(), "height": cameras[0].getHeight(), "timestamp_s": now}
    action = policy.act(observation)
    if policy.mode != last_mode:
        print(f"[Vision Agent] {policy.mode} at {now:.1f}s", flush=True)
        last_mode = policy.mode
    v, omega = float(action.linear_m_s), float(action.angular_rad_s)
    if not math.isfinite(v) or not math.isfinite(omega):
        raise ValueError("Policy returned a non-finite action")
    lim = task["action_limits"]
    v = max(-lim["linear_m_s"], min(lim["linear_m_s"], v))
    omega = max(-lim["angular_rad_s"], min(lim["angular_rad_s"], omega))
    if ended:
        v = omega = 0.0
    motors[0].setVelocity((v - .41 * omega / 2) / .09)
    motors[1].setVelocity((v + .41 * omega / 2) / .09)
    telemetry = {"scenario_id": scenario_id, "run_id": run_id, "timestamp_s": now, "active": bool(action.active), "claim_arrival": bool(action.claim_arrival), "v_m_s": v, "omega_rad_s": omega}
    if ended:
        telemetry["episode_end"] = True
    robot.setCustomData(json.dumps(telemetry))
    frame_number += 1
    if record_file and frame_number%record_every==0 and recorded_pairs<record_limit:
        names=[]
        for camera,side in zip(cameras,("left","right")):
            name=f"record_{recorded_pairs:05d}_{side}.png"
            camera.saveImage(str(public_out/name),100)
            names.append(name)
        record_file.write(json.dumps({"timestamp_s":now,"left":names[0],"right":names[1],"command_v":v,"command_omega":omega})+'\n')
        record_file.flush()
        recorded_pairs += 1
    for capture in (2, 8):
        if now >= capture and capture not in saved:
            for camera, side in zip(cameras, ("left", "right")):
                camera.saveImage(str(public_out / f"{side}_{capture:02d}s.png"), 100)
            saved.add(capture)
            print(f"[Vision Agent] Saved stereo pair at {now:.1f}s -> {public_out}", flush=True)
