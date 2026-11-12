"""Run each scene for 12 simulated seconds; no navigation policy is exercised."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import time
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--webots",required=True,help="Full path to the Webots executable on this computer")
    parser.add_argument("--scenario",choices=["A","B","C","all"],default="all")
    args=parser.parse_args()
    run_id=time.strftime("qa_%Y%m%d_%H%M%S")
    result=[]
    for key in "ABC" if args.scenario=="all" else args.scenario:
        meta=json.loads((ROOT/"evaluation"/"scenarios"/f"{key}.json").read_text())
        env=os.environ.copy()
        env.pop("VISION_DRIVE",None)  # Validation always checks the static preview.
        env.update(VISION_VALIDATE="1",VISION_RUN_ID=run_id,PYTHONDONTWRITEBYTECODE="1")
        # Hidden/minimized Qt views sometimes export a 1-pixel-wide viewport on
        # Windows. Validate in a normal rendered window, with overlay-free copies.
        qa_world=ROOT/"worlds"/f"_validation_{key}.wbt"
        qa_layout=qa_world.parent/f".{qa_world.stem}.wbproj"
        qa_world.write_text((ROOT/"worlds"/meta["world_file"]).read_text(encoding="utf-8"),encoding="utf-8")
        layout=(ROOT/"config"/"view_layout.wbproj").read_text(encoding="utf-8").rstrip()+'\nrenderingDevicePerspectives: Vision Rescue Rover:camera_left;0;0.42;0.02;0.73\nrenderingDevicePerspectives: Vision Rescue Rover:camera_right;0;0.42;0.235;0.73\n'
        if qa_layout.exists():
            with qa_layout.open('r+',encoding='utf-8') as handle:
                handle.write(layout);handle.truncate()
        else:
            qa_layout.write_text(layout,encoding='utf-8')
        command=[args.webots,"--port=18455","--batch","--fullscreen","--mode=realtime","--stdout","--stderr",str(qa_world)]
        print(f"Validating {key} ...",flush=True)
        process=subprocess.Popen(command,cwd=str(Path(args.webots).parent),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
        try:
            output,_=process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError(f"Webots {key} did not finish its smoke test")
        folder=ROOT/"runs"/run_id/key
        folder.mkdir(parents=True,exist_ok=True)
        (folder/"webots.log").write_text(output,encoding="utf-8")
        qa_world.unlink(missing_ok=True)
        qa_layout.unlink(missing_ok=True)
        smoke=folder/"evaluation"/"smoke_test.json"
        model_errors=re.findall(r"^ERROR:.*",output,re.MULTILINE)
        if process.returncode!=0 or not smoke.exists() or model_errors:
            print(output[-7000:],flush=True)
            raise RuntimeError(f"Webots {key} failed (exit={process.returncode}); see {folder}")
        check=json.loads(smoke.read_text())
        images=list((folder/"observations").glob("*.png"))
        if len(images)!=4:
            raise RuntimeError(f"Expected 4 stereo images, found {len(images)}")
        for image_path in images:
            with Image.open(image_path) as image:
                assert image.size == (640,480)
        overview=folder/"evaluation"/"scene_overview.png"
        with Image.open(overview) as image:
            if image.width<640 or image.height<360:
                raise RuntimeError(f"Invalid viewport export size: {image.size}")
        for name in ("casualty_detail.png","obstacles_detail.png"):
            with Image.open(folder/"evaluation"/name) as image:
                assert image.width>=640 and image.height>=360
        result.append({"scenario":key,"exit_code":process.returncode,"camera_images":len(images),**check})
        print(f"{key}: upright={check['upright']}, speed={check['stationary_speed_m_s']:.5f}, 4 camera images",flush=True)
    report=ROOT/"runs"/run_id/"validation.json"
    report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Validation report: {report}",flush=True)


if __name__=="__main__":
    main()
