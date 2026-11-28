"""Render D in Webots and audit tree occlusion with the actual rover RGB-D."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation
from build_search_worlds import build,replace_node
from run_search import ROOT,executable_path,referenced_assets,simulator_has_errors
from upgrade_casualties import node_span
sys.path.insert(0,str(ROOT/'evaluation'))
from metrics import height_at,look_at_rotation


def preview_world(weather='clear'):
    source=build('D');text=source.read_text(encoding='utf-8')
    private=json.loads((ROOT/'evaluation/search/D.json').read_text())
    x,y=private['spawn'];z=height_at(private,x,y)+.035;yaw=private['spawn_yaw']
    proto = (ROOT / 'protos/ResearchRover.proto').read_text(encoding='utf-8')
    imports = re.findall(r'^EXTERNPROTO "([^"]+)"', proto, flags=re.MULTILINE)
    for filename in imports:
        text = text.replace('EXTERNPROTO "../protos/ResearchRover.proto"',
                            f'EXTERNPROTO "../protos/{filename}"\n'
                            'EXTERNPROTO "../protos/ResearchRover.proto"', 1)
    a,b=node_span(proto,r'\bRobot');body=proto[a:b]
    substitutions=dict(translation=f'{x} {y} {z}',rotation=f'0 0 1 {yaw}',
        name='"Forest render rover"',controller='"forest_preview"',customData='"{}"')
    for key,value in substitutions.items():body=body.replace(f'{key} IS {key}',f'{key} {value}')
    for c_field in ('cargo1Scale','cargo2Scale','cargo3Scale'):
        body=body.replace(f'scale IS {c_field}','scale 1 1 1')
    body=body.replace('Robot {','Robot { supervisor TRUE',1)
    # Audit-only static copy of the exact rover body/camera: no motion or sensor noise.
    while re.search(r'\bphysics\s+Physics\s*\{',body):
        a,b=node_span(body,r'\bphysics\s+Physics');body=body[:a]+body[b:]
    body=re.sub(r'\bnoise\s+[\d.]+','noise 0',body)
    render_cameras=[]
    rotation=Rotation.from_euler('z',yaw)
    for name,eye,target,fov in (
        ('overview',[-58,-69,48],[0,0,2.0],.97),
        ('rescue_core',[-29,-36,24],[-.5,.6,2.0],.93),
        ('forest_detail',[-21,-16,5.6],[-7,-1,2.3],1.08),
        ('hills',[-42,-25,10],[-21,9,4.5],1.03)):
        local=rotation.inv().apply(np.asarray(eye)-[x,y,z])
        axis_angle=look_at_rotation(eye,target)
        relative=rotation.inv()*Rotation.from_rotvec(np.array(axis_angle[:3])*axis_angle[3])
        vector=relative.as_rotvec();angle=np.linalg.norm(vector);axis=vector/max(angle,1e-9)
        render_cameras.append(f'''Camera {{ name "render_{name}" translation {' '.join(map(str,local))}
 rotation {' '.join(map(str,[*axis,angle]))} width 1920 height 1080 fieldOfView {fov}
 near .05 far 240 exposure 1.22 ambientOcclusionRadius .65 }}''')
    body=body.replace('children [','children [\n'+'\n'.join(render_cameras),1)
    text=replace_node(text,r'\bDEF\s+ROVER\s+ResearchRover','DEF ROVER '+body)
    text=text.replace('controller "disaster_environment"','controller "<none>"')
    text=text.replace('controller "mission_referee"','controller "<none>"')
    text=text.replace('DEF OBSERVER_DRONE ObserverDrone {','DEF OBSERVER_DRONE ObserverDrone { controller "<none>"',1)
    world=ROOT/'worlds/_validation_forest_D.wbt';world.write_text(text,encoding='utf-8')
    if weather == 'fog':
        from fog_atmosphere import apply_realistic_fog_to_world
        apply_realistic_fog_to_world(world, include_mist_banks=True, include_gully_wisps=True)
        text = world.read_text(encoding='utf-8')
        text = text.replace('ambientOcclusionRadius .65', 'ambientOcclusionRadius 1.35')
        world.write_text(text, encoding='utf-8')
    elif weather and weather != 'clear':
        from run_uav_rescue import apply_weather_to_world
        apply_weather_to_world(world, weather)
        text = world.read_text(encoding='utf-8')
        text = text.replace('ambientOcclusionRadius .65', 'ambientOcclusionRadius 1.35')
        world.write_text(text, encoding='utf-8')
    layout=world.parent/f'.{world.stem}.wbproj'
    layout_text=(ROOT/'config/view_layout.wbproj').read_text()
    if layout.exists():
        with layout.open('r+',encoding='utf-8') as handle:handle.write(layout_text);handle.truncate()
    else:layout.write_text(layout_text,encoding='utf-8')
    referenced_assets(world)
    runtime=ROOT/'controllers/forest_preview/runtime.ini'
    runtime.write_text(f'[python]\nCOMMAND = {Path(sys.executable).as_posix()}\n',encoding='utf-8')
    return world


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--visible',action='store_true')
    parser.add_argument('--keep-open',action='store_true')
    parser.add_argument('--weather',choices=('clear','lowlight','fog'),default='clear')
    parser.add_argument('--port',type=int,default=18553)
    args=parser.parse_args()
    world=preview_world(weather=args.weather)
    run_id=time.strftime('forest_render_%Y%m%d_%H%M%S')+f'_{os.getpid()}'
    folder=ROOT/'runs'/run_id;folder.mkdir()
    env=os.environ.copy();env.update(FOREST_RENDER_RUN=run_id,FOREST_KEEP_OPEN='1' if args.keep_open else '0')
    command=[str(executable_path()),'--batch','--mode=fast',f'--port={args.port}','--stdout','--stderr']
    if not args.visible:command.append('--no-rendering')
    command.append(str(world))
    print(folder,flush=True)
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,encoding='utf-8',errors='replace',
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    def capture():
        with (folder/'webots.log').open('w',encoding='utf-8') as log:
            for line in process.stdout:log.write(line);log.flush()
    reader=threading.Thread(target=capture,daemon=True);reader.start()
    deadline=time.monotonic()+180
    while process.poll() is None:
        if args.keep_open and (folder/'render_summary.json').is_file():break
        if time.monotonic()>deadline:
            process.terminate();raise TimeoutError(f'Render timed out: {folder}')
        time.sleep(.25)
    if not args.keep_open:reader.join(timeout=5)
    errors=(folder/'webots.log').read_text(encoding='utf-8',errors='replace')
    if simulator_has_errors(errors) or not (folder/'render_summary.json').is_file():
        raise RuntimeError(f'Forest render failed: {errors[-2500:]}')
    summary=json.loads((folder/'render_summary.json').read_text())
    if not summary['occlusion_pass']:raise RuntimeError(f'Trees did not sufficiently occlude rover: {summary}')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
