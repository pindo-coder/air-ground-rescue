"""Build derived worlds, run a shared-sensor method and retain auditable metrics."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from build_search_worlds import CASES,build

ROOT=Path(__file__).resolve().parents[1]
METHODS=("frontier","weighted","active_only","schedule_only","full","no_prior","no_covariance","no_view_groups")


def simulator_has_errors(output):
    return bool(re.search(r'(?m)^\s*ERROR:',output) or
                re.search(r'(?m)^WARNING:.*Skipped unexpected',output) or
                'Traceback (most recent call last)' in output or
                'controller exited with status:' in output)


def referenced_assets(world):
    found=set();pending=[world]
    while pending:
        source=pending.pop()
        if source in found:continue
        found.add(source)
        if source.suffix.lower() not in (".wbt",".proto"):continue
        for reference in re.findall(r'"([^"\n]+\.(?:proto|png|jpg|jpeg|hdr|obj))"',source.read_text(encoding="utf-8")):
            if reference.startswith(("webots://","http://","https://")):continue
            asset=(source.parent/reference).resolve()
            if not asset.is_file():raise FileNotFoundError(f"Missing local scene asset: {asset}")
            pending.append(asset)
    return found


def configure_runtime():
    content=f"[python]\nCOMMAND = {Path(sys.executable).as_posix()}\n"
    for controller in ("search_agent","mission_referee","aerial_observer","disaster_environment"):
        (ROOT/"controllers"/controller/"runtime.ini").write_text(content,encoding="utf-8")


def executable_path():
    if os.environ.get("SEARCH_WEBOTS_EXE"):return Path(os.environ["SEARCH_WEBOTS_EXE"])
    base=Path(os.environ.get("WEBOTS_HOME",r"C:\Program Files\Webots"))
    if os.name=="nt":return base/"msys64/mingw64/bin/webots.exe"
    found=shutil.which("webots")
    if found:return Path(found)
    return base/"webots"


def run(case="B",method="full",detector="yolo",duration=600,visible=False,
        realtime=False,keep_open=False,port=18540,fusion="on",dynamic="on"):
    if duration<=0 or method not in METHODS or case not in CASES:raise ValueError("Invalid run parameters")
    if detector=="yolo":
        importlib.metadata.version("ultralytics")
        if not (ROOT/"models/yolo11n.pt").is_file():
            raise FileNotFoundError("Run python tools/fetch_search_model.py or use --detector surrogate_rgbd for controlled testing")
    world=build(case)
    configure_runtime()
    run_id=f"search_{case}_{method}_{detector}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    folder=ROOT/"runs"/run_id;folder.mkdir(parents=True)
    env=os.environ.copy()
    env.update(RESCUE_RUN_ID=run_id,RESCUE_METHOD=method,RESCUE_DETECTOR=detector,
               RESCUE_DURATION=str(duration),RESCUE_AUTO_QUIT="0" if keep_open else "1",
               RESCUE_FUSION=fusion,RESCUE_DYNAMIC=dynamic,
               PYTHONDONTWRITEBYTECODE="1")
    files=list((ROOT/"controllers/search_agent").glob("*.py"))+list((ROOT/"controllers/mission_referee").glob("*.py"))
    files+=[ROOT/"protos/ResearchRover.proto",ROOT/"config/search/common.json",
            ROOT/"config/search/public"/f"{case}.json",ROOT/"evaluation/search"/f"{case}.json",
            ROOT/"evaluation/search_metrics.py",world]
    files+=list(referenced_assets(world))+[ROOT/"assets/render/catalog.json"]
    if case=='D':
        files+=[ROOT/'config/scenes/D.json']+[ROOT/'tools'/name for name in (
            'complex_disaster.py','forest_render.py','forest_terrain.py','build_forest_materials.py')]
        for name in ('aerial_observer','disaster_environment'):
            files+=list((ROOT/'controllers'/name).glob('*.py'))
    if detector=="yolo":files.append(ROOT/"models/yolo11n.pt")
    packages=("numpy","scipy","Pillow","opencv-python")+(("ultralytics","torch") if detector=="yolo" else ())
    dependencies={name:importlib.metadata.version(name) for name in packages}
    hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))}
    signature=hashlib.sha256(json.dumps(dict(files=hashes,detector=detector,duration=duration,dependencies=dependencies,
                                            fusion=fusion,dynamic=dynamic),sort_keys=True).encode()).hexdigest()
    manifest=dict(case=case,method=method,detector=detector,duration=duration,config_signature=signature,
                  files_sha256=hashes,experiment_stage="development_preflight",python=sys.version,
                  dependencies=dependencies,fusion=fusion,dynamic=dynamic)
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    executable=executable_path()
    if not executable.is_file():raise FileNotFoundError("Set WEBOTS_HOME or SEARCH_WEBOTS_EXE to the Webots installation")
    command=[str(executable),"--batch",f"--mode={'realtime' if realtime else 'fast'}",f"--port={port}","--stdout","--stderr"]
    if not visible:command.append("--no-rendering")
    command.append(str(world))
    print("Run:",run_id,flush=True)
    log_path=folder/"webots.log"
    process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding="utf-8",errors="replace",
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
    def capture():
        with log_path.open("w",encoding="utf-8") as log:
            for line in process.stdout:
                log.write(line);log.flush()
    reader=threading.Thread(target=capture,daemon=True);reader.start()
    deadline=time.monotonic()+max(180,duration*8)
    try:
        while process.poll() is None:
            summary=folder/"evaluation/summary.json"
            if keep_open and summary.is_file():
                try:finished=json.loads(summary.read_text()).get("status")=="FINISHED"
                except json.JSONDecodeError:finished=False
                if finished:
                    print("Episode finished; Webots remains open.",flush=True)
                    return folder
            if time.monotonic()>deadline:
                process.terminate()
                raise TimeoutError(f"Timed out; partial run retained at {folder}")
            time.sleep(.5)
    except KeyboardInterrupt:
        if not keep_open:process.terminate()
        raise
    reader.join(timeout=5)
    output=log_path.read_text(encoding="utf-8",errors="replace")
    summary=folder/"evaluation/summary.json"
    result=json.loads(summary.read_text()) if summary.is_file() else {}
    if (process.returncode or result.get("status")!="FINISHED" or result.get("method")!=method or
        simulator_has_errors(output)):
        raise RuntimeError(f"Controller/world validation failed; inspect {log_path}\n{output[-3500:]}")
    print({key:result[key] for key in ("effective_reports","false_reports","delivered_people","collisions","mission_success")},flush=True)
    return folder


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=CASES,default="B")
    parser.add_argument("--method",choices=METHODS,default="full")
    parser.add_argument("--detector",choices=("yolo","surrogate_rgbd"),default="yolo")
    parser.add_argument("--duration",type=float,default=600)
    parser.add_argument("--visible",action="store_true")
    parser.add_argument("--realtime",action="store_true")
    parser.add_argument("--keep-open",action="store_true")
    parser.add_argument("--port",type=int,default=18540)
    parser.add_argument('--fusion',choices=('on','off'),default='on')
    parser.add_argument('--dynamic',choices=('on','off'),default='on')
    args=parser.parse_args()
    try:run(**vars(args))
    except (ImportError,importlib.metadata.PackageNotFoundError,FileNotFoundError) as error:
        parser.exit(2,f"{error}\nInstall requirements-search.txt, plus requirements-search-yolo.txt for yolo.\n")
