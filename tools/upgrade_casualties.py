"""Update casualty rendering and head proxies, preserving the surrounding world."""
import json
from pathlib import Path
import re

from casualty_visual import casualty_render, HEAD_PROXY_POSITIONS, NECK_PROXY_DIMENSIONS, VISUAL_MODEL

ROOT = Path(__file__).resolve().parents[1]
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|#[^\r\n]*|[{}\[\]]')


def closing(text, opening):
    stack = []
    pairs = {"}": "{", "]": "["}
    for token in TOKENS.finditer(text, opening):
        value = token.group()
        if value in ("{", "["):
            stack.append(value)
        elif value in pairs:
            if not stack or stack.pop() != pairs[value]:
                raise ValueError("Unbalanced Webots node")
            if not stack:
                return token.end()
    raise ValueError("Unclosed Webots node")


def node_span(text, pattern):
    match = re.search(pattern + r"\s*\{", text)
    if match is None:
        raise ValueError("Missing node: " + pattern)
    return match.start(), closing(text, match.end() - 1)


def upgrade_world_text(text, key):
    start, end = node_span(text, r"\bDEF\s+CASUALTY\s+Solid")
    node = text[start:end]
    if re.search(r"\bDEF\s+CASUALTY_RENDER\s+Group\b", node):
        a, b = node_span(node, r"\bDEF\s+CASUALTY_RENDER\s+Group")
        node = node[:a] + node[b:]
    # Keep each original primitive DEF/USE pair as the exact existing collision proxy.
    for match in reversed(list(re.finditer(r"\bPBRAppearance\s*\{", node))):
        end_app = closing(node, match.end() - 1)
        node = node[:match.start()] + "PBRAppearance { transparency 1 roughness 1 metalness 0 }" + node[end_app:]
    # Update both the hidden DEF poses and the collision USE poses. Coordinates
    # are absolute, so applying the tool twice cannot shift the head twice.
    for match in reversed(list(re.finditer(r"\bPose\s*\{", node))):
        pose_end = closing(node, match.end() - 1)
        pose = node[match.start():pose_end]
        reference = re.search(r"\b(?:DEF|USE)\s+GEO_CASUALTY_(HEAD|HAIR|NOSE|EYE_L|EYE_R|NECK)\b", pose)
        if reference is None:
            continue
        name = reference[1]
        position = " ".join(f"{v:.6f}" for v in HEAD_PROXY_POSITIONS[name])
        pose = re.sub(r"\btranslation\s+[-+.\deE]+\s+[-+.\deE]+\s+[-+.\deE]+", "translation " + position, pose, count=1)
        if name == "NECK":
            pose = re.sub(r"\bradius\s+[-+.\deE]+", f"radius {NECK_PROXY_DIMENSIONS[0]:.5f}", pose, count=1)
            pose = re.sub(r"\bheight\s+[-+.\deE]+", f"height {NECK_PROXY_DIMENSIONS[1]:.5f}", pose, count=1)
        node = node[:match.start()] + pose + node[pose_end:]
    children = re.search(r"\bchildren\s*\[", node)
    if children is None:
        raise ValueError("CASUALTY has no children")
    insert = closing(node, children.end() - 1) - 1
    node = node[:insert].rstrip() + "\n      " + casualty_render(key) + "\n    " + node[insert:]
    return text[:start] + node + text[end:]


def main():
    for key, name in (("A", "A_trail.wbt"), ("B", "B_washout.wbt"), ("C", "C_generalization.wbt")):
        path = ROOT / "worlds" / name
        with path.open(encoding="utf-8", newline="") as handle:
            original = handle.read()
        upgraded = upgrade_world_text(original, key)
        with path.open("r+", encoding="utf-8", newline="") as handle:
            handle.write(upgraded)
            handle.truncate()
        metadata_path = ROOT / "evaluation/scenarios" / f"{key}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        casualty = next(e for e in metadata["elements"] if e["kind"] == "casualty")
        casualty["visual_model"] = VISUAL_MODEL
        casualty["body_length_m"] = 1.67
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        print(f"{key}: textured casualty installed; world outside CASUALTY preserved")


if __name__ == "__main__":
    main()
