"""Fetch the optional, version-pinned upstream detector with a SHA256 check."""
from pathlib import Path
import hashlib
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
SHA256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"

if __name__=="__main__":
    destination=ROOT/"models/yolo11n.pt"
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest()==SHA256:
        print("Model already present and verified:",destination)
    else:
        with urllib.request.urlopen(URL,timeout=60) as response:content=response.read()
        if hashlib.sha256(content).hexdigest()!=SHA256:
            raise RuntimeError("Upstream file hash differs; model was not installed")
        destination.write_bytes(content)
        print("Downloaded and verified:",destination)
