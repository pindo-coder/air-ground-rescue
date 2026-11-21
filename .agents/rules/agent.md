---
trigger: always_on
description: "援来有路 Webots 工程最新交接文档（2026-11-21 晚）。涵盖已完成闭环、四种全局方法、纯 YOLO 空地搜救投送、载具外观升级、神经网络局规试跑现状、迁移清单与下一步接续顺序。"
---

# 援来有路 · Webots 项目交接（2026-11-21 迁移快照）

**参赛项目**：《援来有路——一种面向应急救援场景的无人车自主探索导航系统》。
从仓库根目录打开项目，以下所有代码、命令与证据路径均以仓库根目录为起点。

**核心目标**：在未知山洪灾后场景（Webots R2025a，D 主场景 84 × 60 m）中，由无人机先验建图、补拍并检测疑似伤员，四轮无人车自主规划可行路径至安全接近点，利用车载 LiDAR / RGB-D 避障与复核，完成伤员位置上报与物资投送。**老师硬性要求：车端局部规划使用神经网络**，且各阶段均须保留可复现、零仿真真值泄漏的实跑证据。

---

## 1. 接手与跨机迁移必查清单

- **Git 基线**：本地工作区基于提交 `2a9eab7`（仓库 [pindo-coder/webots-vision-rescue](https://github.com/pindo-coder/webots-vision-rescue)）累积了大量尚未 commit 的完整功能改动。接手后**务必先运行 `git status --short`** 确认本地文件均已完整迁入，切勿覆盖或丢弃未提交工作。
- **跨目录 / 跨机器迁移必须带走的文件**：
  1. 全部已修改与新增源码：`controllers/uav_mapper/`、`controllers/uav_map_rover/`、`controllers/uav_survey/`、`controllers/yolo_dataset_collector/`、`controllers/search_agent/`（含 `perception.py`、`multiview.py`、`search_agent.py`）、`controllers/vision_agent/`；
  2. 载具模型与 PBR 贴图：`protos/ObserverDrone.proto`、`protos/ResearchRover.proto`、`protos/VisionRover.proto`、`protos/RescueRoverBody.proto`、`protos/RescueRoverWheel.proto`、`protos/BeveledBox.proto`、`protos/RoundedHousing.proto`、`protos/textures/*.png`；
  3. 配置、工具、启动脚本、测试与文档：`config/scenes/uav_mapping*.json`、`tools/` 下全部新建脚本、`start_uav_*.cmd`、`tests/search/`、`docs/UAV_MAPPING.md`、`docs/UAV_NAVIGATION.md`、`docs/LOCAL_POLICY.md`、`requirements-local-learning.txt`、`.agents/rules/agent.md`；
  4. **模型权重说明**：微调航拍权重 `models/yolo11n_victim.pt`（5.5 MB）已直接纳入 Git 仓库随代码同步；上游 COCO 基线权重 `models/yolo11n.pt` 仍由 `.gitignore` 忽略，新机克隆后执行 `.venv\Scripts\python.exe tools/fetch_search_model.py` 即可获取；
  5. **需要带走的历史跑测证据与缓存航测图**（`runs/` 也被 `.gitignore` 忽略）：至少建议复制下表中的核心参考目录，特别是 `runs/uav_mapping_D_full_route_value_20261121_133722_31008`（全场 192 帧 + `fine_ground_z_m` + `surface_dsm.tif`，供 `--map-from` 秒级复现）及 `runs/local_learning_pilot_20261121/`（神经网络局规首轮试跑权重与对照）。
- **虚拟环境重建**（目标机 Webots R2025a + Python 3.12–3.14）：
  ```powershell
  python -m venv .venv
  .venv\Scripts\python.exe -m pip install -r requirements-search.txt -r requirements-search-yolo.txt
  .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  .venv\Scripts\python.exe -m pip install -r requirements-local-learning.txt
  .venv\Scripts\python.exe tools/build_search_worlds.py --case D
  ```

---

## 2. 当前总体完成度

| 模块 | 当前状态 | 关键指标与验证事实 |
| --- | --- | --- |
| **场景与基线航测** | **已完成并保留基线** | D 主场景 `84 × 60 m`，含冲沟、倒塌房屋、碎石倒木、252 棵树、140 丛灌木、2 个移动障碍、3 名模拟伤员。`tools/run_uav_mapping.py` 保持不变，全场 192 帧（156 俯拍 + 36 斜拍），输出 `0.5 m` 通行图、`0.125 m` DSM/正射图，并已支持在线导出 202,099 格 `0.125 m` `fine_ground_z_m`。 |
| **载具模型外观升级** | **已完成并验零遮挡** | 重设计 `RescueRoverBody.proto`、`RescueRoverWheel.proto`、`ObserverDrone.proto`（3K 碳纤维 + 拉丝枪灰金属 PBR 贴图、外露双叉臂悬架/防脱轮毂/管状机臂/三轴云台），全部传感器位姿、FOV 与下视/前视净空严格保持零自遮挡。 |
| **定点闭环导航与安全停车** | **已完成并在 Webots 实测** | `tools/run_uav_navigation.py` 闭环实现轮速+IMU+GPS 本地 ENU 估计、3 层 360° LiDAR + RGB-D 地形感知障碍提取、0.125 m 细子栅格局部重规划与安全停车。D 全场无障碍 `ARRIVED`（0 碰撞）、沿路碎石重规划绕行 `ARRIVED`（0 碰撞）、封堵窄通道 `STOPPED_SAFE`（拒绝进入未知区，0 碰撞）。 |
| **四种全局规划方法** | **全部已实现并实跑验证** | ① **Traversing Mars**：`grid_route.py` (`plan_dual`) 输出保守/乐观双路线及关键未知段；② **DUCPP**：`survey_refinement.py` (`--refinement-strategy route_value`) 同预算 3/3 补拍区全部精准命中 `north` 42.83 m 关键走廊（行进中动态多轮补拍待接）；③ **VCTP**：`vctp.py` 基于 0.125 m `fine_ground_z_m` 检四轮独立落点、俯仰/横滚、扭转与底盘净空；④ **IGHA\***：`igha_star.py` (`--planner igha_vctp`) 带朝向滑移转向搜索，D 东侧接近点 75/75 姿态 100% 通过 VCTP、0 碰撞到达，最大横滚/俯仰均优于 2D A\*。 |
| **空地双视角 YOLO 微调** | **已完成（D 场景零泄漏）** | 仅用 A+B 场景采集 496 张训练图（含 90 航拍+36 车载纯背景难负样本）、C 场景 175 张验证图（Recall 0.695、mAP@50 0.613），D 场景严格留出。留出 D 192 帧真实航拍对照：未微调 COCO `0/3`（25.67 s）→ 微调 `models/yolo11n_victim.pt` **`3/3`、0 误报、定位 RMSE `0.134 m`（伤员 2 误差 `2.9 cm`）、耗时 `5.13 s`（提速 5×）**。运行时航拍自动用 `yolo11n_victim.pt`，地面近景自动用通用 `yolo11n.pt`（侧面/正头顶/变色/灰度均稳定检出 `0.41–0.81`）。 |
| **空地协同“人即终点”救援投送** | **已跑通 `MISSION_SUCCESS`** | `tools/run_uav_rescue.py`（默认 `--detector yolo --planner igha_vctp`）串联：无人机纯 YOLO 检出 3/3 伤员 → `rescue_mission.py` 在 `0.80–1.58 m` 确认 `FREE` 环带选安全停靠点 → `VCTP + IGHA*` 避障抵达 → 转向正对伤员停稳 `≥ 2.2 s` 车载 YOLO 复核与跨视角融合 → 上报并物理投送橙色物资箱。D 全场实跑 `2/2` 全部空地跨视角融合确认、`2/2` 有效投送（`1.53 m`、`1.46 m`）、65.93 m / 224.0 s、**0 碰撞**、定位 RMSE `0.406 m`。已支持 `--record-video` 导出带实时小地图 PiP 的 720p H.264 实况视频。 |
| **神经网络局部规划（老师要求）** | **已搭通框架并首轮试跑，仍需迭代提效** | 详见 `docs/LOCAL_POLICY.md`。已完成 18,949 参数轻量多模态 PyTorch 网络（5×24×32 RGB-D + 36 扇区 LiDAR/RGB-D + 导航/净空状态）、Webots 专家采集、BC 预训练、带共用安全护栏的 DAgger 回采微调、三组同条件评测器（`tools/compare_local_policies.py`）。首轮仅在 D 局部图用 478/928 帧试跑，两组未见碎石测试中三组成功率均为 0（BC+DAgger 0 碰撞但行驶 1.9 m 后过早安全停车；规则与 BC 发生轻微接触），**这是迁移后的第一优先攻关项**。 |
| **单元测试健康度** | **全量通过** | `unittest discover -s tests` 全部 **77 / 77 通过**（含 `test_uav_rescue.py` 新增 6 项），`git diff --check` 无异常。 |

---

## 3. 核心入口与常用复现命令

若 Webots 不在默认路径，先设置 `SEARCH_WEBOTS_EXE` 或 `WEBOTS_HOME`。

```powershell
# 1. 全量单元测试回归
.venv\Scripts\python.exe -m unittest discover -s tests

# 2. 航测与 DUCPP 补拍基线（不跑小车行驶）
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile local
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full --refinement-strategy route_value

# 3. 定点闭环导航对照（2D A* vs VCTP+IGHA* / 障碍绕行 / 封堵安全停车）
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --planner igha_vctp --obstacle none --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --obstacle detour --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile local --obstacle block

# 4. 纯 YOLO 端到端空地协同伤员搜救投送（默认 --detector yolo，可加 --visible --realtime --keep-open 或 --record-video）
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --max-victims 2 --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008

# 5. 零泄漏重建 YOLO 数据集与微调（仅采 A/B/C，D 严格留出）
.venv\Scripts\python.exe tools/collect_yolo_dataset.py --out runs/yolo_victim_dataset
.venv\Scripts\python.exe tools/train_yolo_victim.py --data runs/yolo_victim_dataset/dataset.yaml --epochs 35 --output models/yolo11n_victim.pt
# 仅跑留出 D 场景 A/B 评测：
.venv\Scripts\python.exe tools/train_yolo_victim.py --benchmark-only --output models/yolo11n_victim.pt --benchmark-survey runs/uav_mapping_D_full_route_value_20261121_133722_31008

# 6. 神经网络局部规划采集、BC/DAgger 训练与同条件评测（详见 docs/LOCAL_POLICY.md）
.venv\Scripts\python.exe tools/train_local_policy.py --help
.venv\Scripts\python.exe tools/compare_local_policies.py --help
```

Windows 快捷入口：`start_uav_mapping.cmd`、`start_uav_navigation.cmd`、`start_uav_rescue.cmd`。

---

## 4. 代码架构与铁律约定

| 位置 | 职责 |
| --- | --- |
| `controllers/uav_mapper/` | `uav_mapper.py`（俯拍/斜拍采集、在线 `fine_ground_z_m` 与 `aerial_victims.json`）、`survey_map.py`（0.5 m 粗图 + 0.125 m 细地面合成）、`survey_refinement.py`（密度与 DUCPP `route_value` 补拍选区）、`aerial_victims.py`（航拍多帧检测与协方差交集聚类）。 |
| `controllers/uav_map_rover/` | `uav_map_rover.py`（读先验图→全局规划→`drive_route` 定点导航 / `run_rescue_mission` 伤员救援）、`grid_route.py`（A\* 与 Traversing Mars `plan_dual`）、`vctp.py`（四轮精细地形验路）、`igha_star.py`（滑移转向带朝向搜索）、`rescue_mission.py`（动态安全接近点选择、多伤员贪心调度、车载复核融合）、`route_tracker.py`（位姿估计、LiDAR/RGB-D 地形感知避障、`OnboardObstacleMap`、`RuleLocalPlanner` 与神经网络接口 `LocalPlannerInput`）。 |
| `controllers/uav_survey/uav_survey.py` | Supervisor：运动学驱动无人机、投放小车、注入测试障碍，并在 `evaluation/` 侧独立运行定点真值评测（`_evaluate_navigation`）、伤员救援真值评测（`_evaluate_rescue`）及平滑跟车录屏帧采集。 |
| `controllers/search_agent/perception.py` | `PersonDetector`：支持 `yolo`（航拍 `max_depth>=12` 自动用 `models/yolo11n_victim.pt` 单次前向，地面 `max_depth<12` 自动用 `models/yolo11n.pt`；支持 `UAV_YOLO_WEIGHTS` 环境变量覆写）与 `surrogate_rgbd`。 |
| `tools/` | `run_uav_mapping.py`、`run_uav_navigation.py`、`run_uav_rescue.py`（含 H.264 + 小地图 PiP 合成）、`collect_yolo_dataset.py`、`train_yolo_victim.py`、`train_local_policy.py`、`compare_local_policies.py`。 |

**必须严守的工程红线**：
1. **零真值泄漏**：`evaluation/search/D.json` 中的伤员坐标、真实高程、碰撞点只能由 `uav_survey.py` / `mission_referee.py` 读取，**绝不允许**传入 `uav_mapper` 或 `uav_map_rover`；任何检测器/局规训练都不得使用 D 场景真值或测试帧。
2. **坐标系与数组约定**：`uav_map.npz` 为 Webots 本地 ENU 米制，数组行由南向北、列由西向东；`states` 中 `-1=UNKNOWN`、`0=FREE`、`1=BLOCKED`。
3. **保留基线兼容**：`tools/run_uav_mapping.py` 与 `tools/run_uav_navigation.py` 的既有参数与行为不得破坏。
4. **Windows 注意事项**：Webots R2025a 在 Windows `--batch` 下调用 `movieStartRecording` 会触发 FFmpeg `0xC0000005` 崩溃，因此录屏统一采用 `exportImage` + OpenCV `avc1` 合成（已内置于 `run_uav_rescue.py --record-video`）。PowerShell 里内联 Python 如含 `{}` 务必使用单引号 here-string `@' ... '@ | .venv\Scripts\python.exe -`。

---

## 5. 诚实局限与下一阶段优先接续顺序

1. **【最高优先】打磨老师要求的神经网络车端局部规划（BC + DAgger）**：
   - 首轮试跑（`runs/local_learning_pilot_20261121/`）仅在 D 局部图采样 928 帧且停止样本为 0，留出测试到达率为 0/2。
   - 下一步：先排查局部碎石场景中规则专家 5.7–5.8 cm 轻微接触及 DAgger 1.9 m 早停的根因 → 扩充 D 全场多路段 + 多障碍构型的专家轨迹与停止/减速样本 → 多轮 DAgger 聚合微调 → 在 D 局部与 D 全场（`valley` 及救援路段）产出稳定高成功率、低碰撞率的**规则 vs BC vs BC+DAgger**同条件对比表，再接入 `run_uav_rescue.py` 跑通“**纯 YOLO + 四种全局方法 + 神经网络局规**”终极合体闭环。
2. **动态交互补拍与 3 伤员全覆盖**：
   - 当前 DUCPP 补拍为航测后单轮补拍；可进一步接通行进中遇 `UNKNOWN` 阻断时向无人机实时派发补拍并刷新地图。
   - 当前单趟救援按 `cargo_capacity=2` 投送 2 名伤员；如需单次任务救全 3 人，可传 `--max-victims 3` 或增加返航起点 `reload` 逻辑。
3. **高空 YOLO 极端换色泛化**：
   - 当前 `yolo11n_victim.pt` 在未见 D 场景上已达 3/3、0 误报，但 19 m 约 25 px 目标在极端 `+60°` 全局色相偏移或纯灰度图下置信度会下降；若需更强跨服装颜色泛化，可在 `collect_yolo_dataset.py` 中直接随机替换训练伤员的上衣/裤子材质颜色后再微调一轮。
4. **阶段性 Git 提交**：建议在开始下一轮大规模神经局规实验前，先对当前已过测代码做一次干净的阶段性 commit。

---

## 6. 核心实测证据索引（`runs/`）

| 验证项目 | 目录 / 关键文件 | 核心结果 |
| --- | --- | --- |
| 全场 DUCPP `route_value` 航测（推荐 `--map-from` 复用源） | `runs/uav_mapping_D_full_route_value_20261121_133722_31008/` | 192 帧，3/3 补拍区命中 `north` 42.83 m 走廊，导出 202,099 格 0.125 m `fine_ground_z_m` |
| 载具外观重设计对比图 | `runs/vehicle_redesign_final_20261121_141232/vehicle_showcase.jpg` | 四轮车与无人机 PBR 重设计，零光学自遮挡 |
| 定点闭环：2D A\* 无障碍 / 可绕行碎石 / 封堵停车 | `runs/uav_nav_D_full_clear_20261121_130543_12840/`、`runs/uav_nav_D_full_detour_20261121_130931_38364/`、`runs/uav_nav_D_local_block_20261121_131102_30056/` | 分别 `ARRIVED`(0 碰)、`ARRIVED`(5 次重规划, 0 碰)、`STOPPED_SAFE`(0 碰) |
| 定点闭环：`VCTP + IGHA*` 同图对照 | `runs/uav_nav_D_full_igha_vctp_clear_20261121_133902_38772/` | `ARRIVED` 126.2 s，VCTP 75/75 (100%)，最大横滚/俯仰 `10.46°/12.13°`（优于 A\*） |
| 零泄漏 YOLO 数据集与微调训练 | `runs/yolo_victim_dataset/preview_labels.jpg`、`runs/yolo_victim_train_20261121_154709/`、`models/yolo11n_victim.pt` | A+B 496 训练 / C 175 验证；留出 D 航拍 `3/3`、0 误报、RMSE `0.134 m`（COCO 基线 `0/3`） |
| **纯 YOLO 空地协同双伤员救援闭环** | `runs/uav_rescue_D_full_yolo_igha_vctp_clear_20261121_163823_37812/evaluation/summary.json`、`rescue_preview.png` | **`MISSION_SUCCESS`**，3/3 检出、2/2 跨视角融合确认、2/2 投送（1.53/1.46 m）、0 碰撞 |
| 720p 跟车仿真视频（带实时小地图 PiP） | `runs/uav_rescue_D_full_surrogate_rgbd_igha_vctp_clear_20261121_145756_27572/evaluation/rescue_simulation.mp4`、`video_keyframes.jpg` | 55.9 s (4× 速, 1118 帧, H.264, 82 MB)，`MISSION_SUCCESS` |
| 神经网络局部规划首轮试跑 | `runs/local_learning_pilot_20261121/`、`docs/LOCAL_POLICY.md` | BC + DAgger 权重与两组未见碎石对照（待调优提效） |
