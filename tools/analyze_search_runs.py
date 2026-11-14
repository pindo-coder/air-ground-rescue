"""Raw metrics only; null/censored observations and aborted runs remain visible."""
from pathlib import Path
import csv
import json

ROOT=Path(__file__).resolve().parents[1]
FIELDS=("run","scenario","method","detector","simulation_seconds","first_effective_report_s",
        "first_report_censored","effective_reports","false_reports","miss_rate","false_discovery_rate",
        "localization_rmse_m","reported_95pct_ellipse_coverage","delivered_people",
        "on_time_delivered_people","collisions","path_length_m","mission_success","status",
        "config_signature","experiment_stage")

if __name__=="__main__":
    destination=ROOT/"runs/search_metrics.csv"
    with destination.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
        for folder in sorted((ROOT/"runs").glob("search_*/")):
            if not (folder/"manifest.json").is_file():continue
            manifest=json.loads((folder/"manifest.json").read_text())
            row=dict(run=folder.name,status="ERROR_OR_INCOMPLETE",scenario=manifest["case"],
                     method=manifest["method"],detector=manifest["detector"],
                     config_signature=manifest["config_signature"],experiment_stage=manifest["experiment_stage"])
            summary=folder/"evaluation/summary.json"
            if summary.is_file():
                try:row.update(json.loads(summary.read_text()))
                except json.JSONDecodeError:pass
            writer.writerow({field:row.get(field) for field in FIELDS})
    print(destination)
