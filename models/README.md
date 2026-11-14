# Optional detector

`python tools/fetch_search_model.py` downloads YOLO11n from the official Ultralytics assets release v8.3.0. Weights are ignored by Git, and are not claimed as a project-trained detector.

SHA256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.

Model origin: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt

Model/library source and license: https://github.com/ultralytics/ultralytics and https://docs.ultralytics.com/models/yolo11/ (AGPL-3.0 / upstream Enterprise options). Preserve the upstream source/license when redistributing the model. Map/human assets retain the separate sources in MATERIAL_SOURCES.md.

The provided integration runs class `person` locally with four image orientations and box NMS. This is a detection interface and test-time transform, not a newly trained rescue-person model. A and C inspection renders produced person detections at some orientations in a development probe; B inspection renders did not. These observations are not detector accuracy results and do not guarantee identification from the moving robot's camera.
