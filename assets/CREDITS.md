# Human visual assets: sources and modifications

These human assets were reused from the local `rescue_vision_lab` visual v3
project. All files required to render and rebuild them are bundled here; the
simulation does not read from that other project or download files at runtime.

## Body

Cesium Man, Copyright 2017 Cesium, via Khronos glTF Sample Assets.
Creative Commons Attribution 4.0 International. The original README and Cesium
mark notice are retained in `upstream/cesium_man/README.md`.

- https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/CesiumMan
- https://creativecommons.org/licenses/by/4.0/

The neutral mesh was posed into three supine variants and separated into shirt,
trousers, exposed skin and shoes. The original logo texture is source reference
only and is not displayed. Existing simplified collision primitives are kept
as invisible proxies; the rendered mesh is not a clinical or articulated body
model. Clothing colors remain different in A/B/C.

## Scanned head

**Infinite, 3D Head Scan**, Lee Perry-Smith / Infinite Realities, CC BY 3.0,
based on a work at www.triplegangers.com, via three.js r170.
The original notice is retained in
`upstream/head_scan/LeePerrySmith_License.txt`.

- https://github.com/mrdoob/three.js/tree/r170/examples/models/gltf/LeePerrySmith
- https://creativecommons.org/licenses/by/3.0/

The lower bust was cropped, UVs converted, and the head rotated/scaled to the
existing casualty footprint. Color and normal photographs are retained.
No endorsement by the subject or original authors is implied.

## Fabric

Cyberbotics Webots R2025a Fabric photographic color, normal and roughness maps,
Apache 2.0. The source `Fabric.proto` retains its copyright and license notice;
the original license is bundled in `upstream/webots/LICENSE`.

- https://github.com/cyberbotics/webots/tree/R2025a/projects/appearances/protos/textures/fabric

This integration adds fabric roughness rendering and seats the head scan 6 cm
closer to the shoulders and 1.2 cm lower. The extra cylindrical visual neck is
removed; head/neck collision proxies are aligned with the new position.
The overlapping neck faces of the body proxy are removed, and the cropped
scan neck end is extended into the shirt collar to conceal the raw cut edge.
`tools/build_casualty_assets.py` and `tools/human_visual.py` rebuild the meshes
from the bundled glTF/GLB sources without network access. Existing flood-scene
terrain and other material credits remain in the root `MATERIAL_SOURCES.md`.
