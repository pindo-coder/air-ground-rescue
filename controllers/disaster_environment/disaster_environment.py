"""Environment-only motion; no detections or target coordinates are transmitted."""
from pathlib import Path
import json
import os
import sys
from controller import Supervisor
from motion import path_pose

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'evaluation'))
from metrics import height_at

robot = Supervisor()
config = json.loads((ROOT/'config/scenes/D.json').read_text())
terrain = json.loads((ROOT/'evaluation/search/D.json').read_text())
run_id = os.environ.get('RESCUE_RUN_ID', 'manual_D')
out = ROOT/'runs'/run_id/'environment'; out.mkdir(parents=True, exist_ok=True)
nodes = [(item, robot.getFromDef(item['def'])) for item in config['moving_obstacles']]
drone = robot.getFromDef('OBSERVER_DRONE')
duration = float(os.environ.get('RESCUE_DURATION',600))
tick = 0
with (out/'motion.jsonl').open('w', encoding='utf-8') as log:
    while robot.step(100) != -1:
        now = robot.getTime(); tick += 1
        seconds = min(now, duration)
        actual = {}
        if os.environ.get('RESCUE_DYNAMIC','on')=='off':seconds=0
        for item, node in nodes:
            xy, yaw = path_pose(item['path'],item['speed_mps'],seconds,item['phase_s'])
            z = height_at(terrain,*xy)+item['size'][2]/2
            node.getField('translation').setSFVec3f([*map(float,xy),float(z)])
            node.getField('rotation').setSFRotation([0,0,1,float(yaw)])
            actual[item['def']] = node.getPosition()
        aerial = config['aerial']
        xy, yaw = path_pose(aerial['path'],aerial['speed_mps'],min(now,duration))
        drone.getField('translation').setSFVec3f([*map(float,xy),aerial['altitude_enu_m']])
        drone.getField('rotation').setSFRotation([0,0,1,float(yaw)])
        if tick % 10 == 0:
            log.write(json.dumps(dict(t=now,obstacles=actual,drone=drone.getPosition()))+'\n');log.flush()
