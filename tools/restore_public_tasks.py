"""Restore initial public tasks after Webots Save serializes runtime telemetry.

Only ROVER.customData is changed; world geometry and viewpoints are preserved.
"""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORLD_NAMES = {"A": "A_trail.wbt", "B": "B_washout.wbt", "C": "C_generalization.wbt"}
CUSTOM_DATA = re.compile(r'\bcustomData\s+("(?:\\.|[^"\\])*")')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report stale tasks without writing files")
    args = parser.parse_args()
    stale = []
    for key, name in WORLD_NAMES.items():
        path = ROOT / "worlds" / name
        with path.open(encoding="utf-8", newline="") as handle:
            text = handle.read()
        rover = re.search(r"\bDEF\s+ROVER\s+VisionRover\s*\{", text)
        if rover is None:
            raise ValueError(f"{name}: ROVER not found")
        evaluator = re.search(r"\bDEF\s+EVALUATOR\b", text[rover.end():])
        end = rover.end() + evaluator.start() if evaluator else len(text)
        match = CUSTOM_DATA.search(text, rover.end(), end)
        if match is None:
            raise ValueError(f"{name}: ROVER.customData not found")
        task = json.loads((ROOT / "config" / "public" / f"{key}_task.json").read_text(encoding="utf-8"))
        current = json.loads(json.loads(match.group(1)))
        if current == task:
            print(f"{key}: initial public task OK")
            continue
        stale.append(key)
        if args.check:
            print(f"{key}: saved runtime data; run python tools/restore_public_tasks.py")
            continue
        value = json.dumps(json.dumps(task, ensure_ascii=True))
        restored = text[:match.start(1)] + value + text[match.end(1):]
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(restored)
        print(f"{key}: restored initial public task")
    return 1 if args.check and stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
