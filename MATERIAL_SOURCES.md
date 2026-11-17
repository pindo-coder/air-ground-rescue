# 素材来源

- 本轮人物网格、扫描头部与织物材质复用自本地 `rescue_vision_lab` 视觉 v3 工程，已复制到本项目的 `assets/`，不依赖原文件夹。Cesium Man 身体、Lee Perry-Smith 扫描头部及 Webots Fabric 的出处、许可和修改说明见 [assets/CREDITS.md](assets/CREDITS.md)。

- 泥土、落叶、岩石、金属屋顶、开裂抹灰墙与破砖墙的照片材质来自 Poly Haven，按 CC0 提供。来源为 [Brown Mud 02](https://polyhaven.com/a/brown_mud_02)、[Forest Floor](https://polyhaven.com/a/forest_floor)、[Rock Face 03](https://polyhaven.com/a/rock_face_03)、[Corrugated Iron](https://polyhaven.com/a/corrugated_iron)、[Worn Cracked Plaster](https://polyhaven.com/a/worn_cracked_plaster)、[Broken Brick Wall](https://polyhaven.com/a/broken_brick_wall)。[Poly Haven 许可](https://polyhaven.com/license)
- Pine.proto 及松树贴图来自 Cyberbotics Webots R2025a，原文件按 CC BY 4.0 提供。本项目增加了 scale 参数以适配原场景树木大小，原版权与许可注释保留在文件中。[Pine 原始文件](https://github.com/cyberbotics/webots/blob/R2025a/projects/objects/trees/protos/Pine.proto)
- D 森林使用 Cyberbotics Webots R2025a 的 [Oak](https://github.com/cyberbotics/webots/blob/R2025a/projects/objects/trees/protos/Oak.proto)、[Sassafras](https://github.com/cyberbotics/webots/blob/R2025a/projects/objects/trees/protos/Sassafras.proto)、[Cypress](https://github.com/cyberbotics/webots/blob/R2025a/projects/objects/trees/protos/Cypress.proto) 及配套树皮/叶片照片贴图，原文件按 CC BY 4.0 提供，原版权与许可注释保留。Oak/Sassafras 增加视觉缩放和多方向树冠复用，碰撞由外层树干代理提供。ForestBush 的多枝/弯曲叶片网格由本项目生成，复用 Sassafras 叶片、法线和树皮纹理，不是球体模型。D 地面照片材质由上述 Poly Haven CC0 泥土与林地材质生成；制作脚本是 `tools/build_forest_materials.py`。
- mountains 天空 JPG/HDR 贴图来自 Cyberbotics Webots R2025a，适用 [Webots Assets License](https://cyberbotics.com/webots_assets_license)。[素材目录](https://github.com/cyberbotics/webots/tree/R2025a/projects/default/worlds/textures/cubic)
- 坍塌房屋几何、建筑碎片、外部山体和本场景材质组合由本项目的 tools/flood_ruins.py 与 tools/render_style.py 生成。
