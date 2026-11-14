"""Serial paired-method runs; preserves failed runs and makes no gain claims."""
from pathlib import Path
import argparse
import json
import time
from run_search import run,METHODS

ROOT=Path(__file__).resolve().parents[1]

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",nargs="+",choices=("A","B","C","smoke"),default=["A","B","C"])
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=["frontier","active_only","schedule_only","full","weighted"])
    parser.add_argument("--detector",choices=("yolo","surrogate_rgbd"),default="yolo")
    parser.add_argument("--duration",type=float,default=150)
    args=parser.parse_args();records=[]
    for case in args.cases:
        for method in args.methods:
            try:
                folder=run(case,method,args.detector,args.duration)
                records.append(dict(case=case,method=method,run=folder.name,status="finished"))
            except Exception as error:
                records.append(dict(case=case,method=method,status="error",error=str(error)))
                print("Recorded run failure:",error,flush=True)
    folder=ROOT/"runs";folder.mkdir(exist_ok=True)
    report=folder/f"search_matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding="utf-8")
    print(report)
    raise SystemExit(1 if any(r["status"]=="error" for r in records) else 0)
