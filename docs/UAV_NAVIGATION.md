# D 场景无人机先验图到四轮车闭环导航

本文档记录在 Webots R2025a 的 D 复杂山洪场景中，四轮车读取无人机生成的通行图（`uav_map.npz`），沿全局路线实际行驶、利用车载 LiDAR 与 RGB-D 检查并规避附近障碍、在遇无法确认通行区域时安全停车并记录原因，以及到达配置接近点后停稳的闭环实现与验证状态。

原航测与路线生成基线入口 `tools/run_uav_mapping.py`（及 `start_uav_mapping.cmd`）保持不变，仅在启用闭环导航工具 `tools/run_uav_navigation.py`（`UAV_NAV_ENABLE=1`）时衔接车端行驶与独立真值评测。

## 本轮已实现与验证的功能

| 模块 | 实现位置 | 本轮状态与说明 |
| --- | --- | --- |
| 无人机航测与正射/高程产品基线 | `tools/run_uav_mapping.py`、`controllers/uav_mapper/`、`tools/aerial_products.py` | 保持基线不变；已复现 `--profile local`（47 帧观测，输出 `uav_map.npz` 与 9.33 m `route.json`）。 |
| 全局地形 A* 路线规划 | `controllers/uav_map_rover/grid_route.py` | 沿用现有 0.5 m 栅格车体净空掩码与地形代价 A*，严格排除 `UNKNOWN` 与 `BLOCKED` 区域。 |
| 车端 ENU 位姿与停稳估计 | `controllers/uav_map_rover/route_tracker.py`（`RoverPoseEstimator`） | 仅使用车载轮速编码器、IMU 姿态和带噪 GPS 融合本地 ENU 位姿，并结合多帧滑窗抑制 GPS/IMU 噪声对停稳判定的干扰。 |
| 车载 LiDAR + RGB-D 地形感知障碍检查 | `controllers/uav_map_rover/route_tracker.py`（`extract_lidar_obstacles`、`extract_rgbd_obstacles`） | 将 3 层 360° LiDAR 与前视 RGB-D 深度按 IMU 横滚/俯仰与航向投影至 ENU，对照无人机先验地面高程 `ground_z_m` 区分平滑坡面与真实凸起障碍/落差。 |
| 细栅格车端障碍叠加与保守重规划 | `controllers/uav_map_rover/route_tracker.py`（`OnboardObstacleMap`） | 在 0.125 m 细子栅格上叠加车载确认障碍，严格保留无人机 `UNKNOWN`/`BLOCKED` 边界；若确认通行区内存在安全绕行则局部重规划，若走廊被封堵且绕行须进入未知/危险区则拒绝强行通过。 |
| 规则速度采样局部控制器 | `controllers/uav_map_rover/route_tracker.py`（`RuleLocalPlanner`、`LocalPlannerInput`） | 接收前瞻路点、车载 LiDAR/RGB-D 障碍与确认边界，输出 `(v, w)`；具备车体净空前瞻、姿态安全保护、到达减速停稳（`speed < 0.03 m/s` 持续 `≥ 1.5 s`）与无法通行时的安全停车及原因记录。 |
| 独立 Supervisor 真值评测与留证 | `controllers/uav_survey/uav_survey.py`、`tools/run_uav_navigation.py` | 仿真真值（真实位置、速度、高程、物理接触点与测试障碍注入）仅在 Supervisor 侧使用，不传给车端控制器；自动保存轨迹 CSV、评测汇总 JSON、车端遥测、相机证据图与轨迹对比图至 `runs/`。 |

## 四种全局方法与神经网络局部规划的接入状态

| 模块 | 实现位置与接口 | 当前状态与验证情况 |
| --- | --- | --- |
| **Traversing Mars（双路线搜索）** | `controllers/uav_map_rover/grid_route.py`（`plan_dual`） | **已实现并验证**：同步计算已确认保守路线与允许穿越未知格的乐观候选路线，提取关键未知差异段、可达性变化及绕行缩短收益，写入 `route.json` 的 `dual_route`。 |
| **DUCPP（关键路段价值导向补拍）** | `controllers/uav_mapper/survey_refinement.py`（`select_patches_route_value`），通过 `--refinement-strategy route_value` 开启 | **已实现并完成同预算消融**：在 D 全场同为 3 补拍区 / 36 帧斜拍预算下，全部 3 个补拍区均精准落在 `north` 42.83 m 乐观走廊上；当前为航测后单轮补拍，行进中动态多轮交互补拍待接。 |
| **VCTP（四轮通过条件精细验路）** | `controllers/uav_map_rover/vctp.py`（`FineTerrainSurface`、`VctpChecker`），配合 `uav_mapper.py` 在线导出 0.125 m `fine_ground_z_m` | **已实现并验证**：基于 0.125 m 细地面图检查四轮独立落点、支撑面俯仰/横滚、车轮扭转落差与车底净空，缺测处保持 `UNKNOWN`；所有姿态 100% 四轮落入不同 0.125 m 格（原 0.5 m 粗栅格下 61/76～110/151 姿态四轮重合）。 |
| **IGHA\*（带朝向滑移转向搜索）** | `controllers/uav_map_rover/igha_star.py`（`plan_igha_star`），通过 `--planner igha_vctp` 开启 | **已实现并在 Webots 实跑验证**：在 VCTP 验路通道内以 2D 障碍感知 Dijkstra 为启发式，使用前进弧线、原地转向与倒车基元搜索 `(x, y, yaw)` 路点；D 东侧接近点实跑 75/75 姿态 100% 通过 VCTP、0 碰撞到达停稳，且最大横滚/俯仰均优于 2D A\* 基线。 |
| **神经网络车端局部规划（老师要求）** | 复用 `controllers/uav_map_rover/route_tracker.py` 的 `LocalPlannerInput` 标准接口（前方路点 + 车载 LiDAR/RGB-D + 先验边界 → `(v, w)`） | **已接入试跑**。支持 Webots 采集、轻量 PyTorch 网络、BC 预训练、DAgger 聚合微调与三组同条件评测。见[局部规划模仿学习](LOCAL_POLICY.md)。 |

## 启动命令

```powershell
# 1. 原航测与路线生成基线（保持不变）
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile local
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full

# 2. 同预算 DUCPP 路线价值补拍对照（Traversing Mars + DUCPP + 在线 0.125 m fine_ground_z_m）
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full --refinement-strategy route_value

# 3. 闭环行驶：2D A* 基线 vs VCTP + IGHA*（D 全场东侧接近点）
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --planner astar --obstacle none
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --planner igha_vctp --obstacle none

# 4. 闭环行驶：沿路出现可绕行障碍（车载 LiDAR/RGB-D 检测并重规划绕过）
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --obstacle detour

# 5. 闭环行驶：沿路出现封堵窄通道障碍（无法确认绕行时安全停车并记录原因）
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile local --obstacle block
```

附加常用选项：
- `--visible`：显示 Webots 图形界面；`--realtime`：实时倍率；`--keep-open`：结束后保留窗口。
- `--map-from runs/<已有航测目录>`：复用已生成的 `uav_map.npz` 快速调试车端控制（正式端到端验证不带该参数，完整运行无人机采图与小车闭环）。

## 输出产物（`runs/uav_nav_D_*/`）

- `uav_map.npz`、`route.json`、`preview.png`、`aerial_products/`：无人机航测与全局路线产物。
- `rover/telemetry.jsonl`、`rover/events.json`、`rover/nav_status.json`、`rover/*.jpg`：车端传感器估计状态、障碍检测/重规划/停车事件与车载相机证据图。
- `evaluation/trajectory.csv`：10 Hz 仿真真值轨迹、速度、目标距离、碰撞计数、定位误差与姿态角。
- `evaluation/summary.json`：独立真值评测结果（到达与否、到达时间、行驶耗时、停稳结果、碰撞次数、失败/停车原因）。
- `evaluation/nav_preview.png`：先验通行图、初始/重规划路线、注入障碍与真值轨迹叠加总览图。

## 本轮 Webots 实测证据索引

| 测试项 | 运行目录 | 结果 | 路程（实际/规划） | 到达耗时 | 停稳 | 碰撞 | 说明 / 原因 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 航测基线复现（`local`） | `runs/uav_mapping_D_local_20261121_124517_1148` | `PLANNED` | — / 9.33 m | — | — | — | 47 帧观测，852 候选通行 / 125 障碍 / 623 未知格 |
| 可通行路线闭环（`full` / `valley` 东侧接近点） | `runs/uav_nav_D_full_clear_20261121_130543_12840` | `ARRIVED` | 39.34 m / 40.61 m | 127.2 s | 是（2.9 s，0.0 m/s） | 0 | 终点误差 0.32 m，LiDAR 1099 帧 / RGB-D 561 帧，定位中位误差 0.049 m |
| 沿路可绕行障碍（`full` / `valley` + 0.65 m 碎石） | `runs/uav_nav_D_full_detour_20261121_130931_38364` | `ARRIVED` | 39.47 m / 40.61 m | 133.0 s | 是（4.5 s，0.0 m/s） | 0 | LiDAR+RGB-D 检出航测后障碍并重规划 5 次，绕行后终点误差 0.31 m |
| 沿路封堵窄通道障碍（`local` 端到端航测 + 1.8 m 路障） | `runs/uav_nav_D_local_block_20261121_131102_30056` | `STOPPED_SAFE` | 0.05 m / 9.33 m | — | 是（2.5 s，0.0 m/s） | 0 | LiDAR+RGB-D 检出封堵障碍，确认绕行须进入未知区，安全停车并记录 `OBSTACLE_BLOCKED_NO_CONFIRMED_DETOUR` |
| 同预算 DUCPP 补拍对照（`full` / `route_value`） | `runs/uav_mapping_D_full_route_value_20261121_133722_31008` | `PLANNED` | — / 44.90 m | — | — | — | 3 个补拍区全部指向 `north` 42.83 m 乐观走廊；获得 39 个走廊部分地面证据格与 12 个新增确认格（中位误差 1.17 cm），并在线导出 202,099 个 0.125 m `fine_ground_z_m` |
| VCTP + IGHA\* 闭环对照（`full` / `valley` 同图配对） | `runs/uav_nav_D_full_igha_vctp_clear_20261121_133902_38772` | `ARRIVED` | 39.43 m / 39.87 m | 126.2 s | 是（2.5 s，0.0 m/s） | 0 | 75/75 姿态 100% 通过 VCTP（原 A\* 为 96.05%），最大横滚/俯仰降至 10.46°/12.13°（原 A\* 11.34°/13.11°），终点误差 0.32 m |
| 空地协同伤员搜救投送（`full` / `surrogate_rgbd` + `igha_vctp`） | `runs/uav_rescue_D_full_surrogate_rgbd_igha_vctp_clear_20261121_143906_5632` | `MISSION_SUCCESS` | 68.42 m / 11.46+57.99 m | 222.4 s | 是（2.3 s × 2） | 0 | 受控检测器基线：3/3 检出，2/2 有效投送（1.58 m、1.42 m），1/2 跨视角融合，定位 RMSE 0.258 m |
| 空地协同纯 YOLO 搜救投送（`full` / 微调 `yolo` + `igha_vctp`） | `runs/uav_rescue_D_full_yolo_igha_vctp_clear_20261121_163823_37812` | `MISSION_SUCCESS` | 65.93 m / 11.47+57.83 m | 224.0 s | 是（3.5 s、2.3 s） | 0 | **纯 YOLO 闭环**：无人机微调模型 `yolo11n_victim.pt`（仅用 A/B 训练、C 验证、D 严格留出）检出 3/3 伤员（误差 0.029–0.166 m，RMSE 0.134 m，0 误报，提速 5×）；地面端 `yolo11n.pt` 复核 2/2 全部成功（score 0.91、0.86，**2/2 跨视角融合**），2/2 有效投送（1.53 m、1.46 m），0 碰撞 |

## 空地协同伤员检测、接近点规划与物资投送（`tools/run_uav_rescue.py`）

在保留原有固定接近点导航基线（`tools/run_uav_navigation.py`）的前提下，新增端到端“**人即终点（动态检出伤员 → 自动安全接近点 → VCTP + IGHA\* 全局规划 → 闭环抵达与车载确认 → 上报与物理投送）**”救援闭环，**全程严格禁止无人机或小车读取 `evaluation/search/D.json` 仿真真值**：

1. **无人机航拍多帧伤员检测与 ENU 聚类**（`controllers/uav_mapper/aerial_victims.py`、`controllers/uav_mapper/uav_mapper.py`）：对俯拍 RGB-D 帧运行 `PersonDetector` 与反投影定位，通过协方差交集（`covariance_intersection`）跨帧聚类并过滤单帧噪声，输出 `aerial_victims.json`。
2. **动态安全接近点选择与多目标调度**（`controllers/uav_map_rover/rescue_mission.py`）：小车不直接压到伤员中心，而是在检出位置周边 `0.80–1.58 m` 环形确认 `FREE` 区域内，综合理想停靠距离（`1.20 m`）、面向伤员的 VCTP 四轮通过性和地形代价挑选最佳接近点，并用 `VCTP + IGHA*`（或 `A*`）规划无碰撞路径，按最短已确认可达路径贪心调度下一名伤员。
3. **车载 RGB-D 抵近确认、位置上报与物资投送**（`controllers/uav_map_rover/uav_map_rover.py`）：小车到达安全停靠距离后原地转向面向伤员、停稳 `≥ 2.2 s`，调用前视 RGB-D 复核并与航拍先验跨视角融合（保存 `rover/victim_*_confirm.jpg` 证据图），发布 `report` 与 `delivery` 事件。
4. **独立真值评测**（`controllers/uav_survey/uav_survey.py` 的 `_evaluate_rescue`）：仅在 Supervisor 侧对照 `evaluation/search/D.json` 核验上报匹配度、定位 RMSE、停稳时长（`≥ 1.5 s`）、投送距离（`≤ 1.8 m`）、0 碰撞及可视化投送橙色物资箱，并渲染 `evaluation/rescue_preview.png`。

### 救援闭环与 YOLO 微调启动命令

```powershell
# 1. 零泄漏采集 A/B/C 场景数据集并微调航拍 YOLO（D 场景严格留出）
.venv\Scripts\python.exe tools/collect_yolo_dataset.py --out runs/yolo_victim_dataset
.venv\Scripts\python.exe tools/train_yolo_victim.py --data runs/yolo_victim_dataset/dataset.yaml --epochs 35 --output models/yolo11n_victim.pt

# 2. 纯 YOLO 端到端救援闭环（默认 --detector yolo，可加 --visible --realtime --keep-open 或 --record-video）
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --max-victims 2

# 3. 复用已有全场航测帧快速验证车端救援闭环（仍仅使用纯传感器观测，不读真值）
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --max-victims 2 --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
```

> **空地双视角 YOLO 分工与诚实边界**：`tools/train_yolo_victim.py` 仅使用 A/B 训练（496 张）、C 验证（175 张，mAP@50 0.613、Recall 0.695），在严格留出的 D 场景 192 帧真实航拍上由未微调 COCO 的 `0/3`（25.67 s）提升至 **`3/3`、0 误报、定位 RMSE 0.134 m（5.13 s）**。运行时 `PersonDetector("yolo", ...)` 对航拍俯视自动加载 `models/yolo11n_victim.pt`，对小车地面近景加载通用 `models/yolo11n.pt`（在侧面、正头顶及变色/灰度图下均稳定检出 `0.41–0.81`），使 D 场景闭环实跑达到 **`2/2` 全部空地跨视角融合确认**。需诚实说明的是：高空 19 m 约 25 px 极小目标在极端 `+60°` 色相偏移或纯灰度下置信度仍会下降，后续如需高空全色域泛化可在采集时为训练伤员直接随机替换多套服装材质。


