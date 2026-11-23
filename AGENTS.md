# 援来有路 · Webots 工程完整技术交接文档

**参赛项目**：《援来有路——一种面向应急救援场景的无人车自主探索导航系统》
**仿真与验证平台**：Webots R2025a · Windows · Python 3.14（`.venv`）
**代码基线与提交状态**：本地 `main` 分支已基于 `2a9eab7` 完成阶段性提交 **`110fbd4`**（60 个文件变更，含微调权重 `models/yolo11n_victim.pt`；本次自动 `git push` 因网络/凭证挂起已手动取消，后续可在本机终端执行 `git push origin main` 推送，或直接拷贝仓库目录迁移）。

**系统总目标**：在未知山洪灾后场景（D 主场景 `84 × 60 m`）中，由无人机先验建图、价值导向补拍与空中伤员检测，四轮无人车自主规划安全可行路径至伤员停靠点，结合车载 LiDAR / RGB-D 局部避障与近距复核，完成位置上报与物理物资投送。**老师硬性要求：车端局部规划必须使用神经网络**，且全链路严格禁止控制器读取任何仿真真值。

---

## 一、迁移清单与目标机环境重建

### 1. 跨机 / 跨目录迁移核对
- **Git 已跟踪（随仓库直接带走）**：全部控制器源码（`controllers/uav_mapper/`、`controllers/uav_map_rover/`、`controllers/uav_survey/`、`controllers/yolo_dataset_collector/`、`controllers/search_agent/`、`controllers/vision_agent/`）、全套重设计载具 PROTO 与 PBR 贴图（`protos/`）、场景配置（`config/scenes/uav_mapping*.json`）、工具脚本（`tools/`）、单元测试（`tests/search/`）、文档（`docs/`、`.agents/rules/agent.md`）、快捷入口（`start_uav_*.cmd`）以及微调航拍权重 **`models/yolo11n_victim.pt`**（5.49 MB，已在 `.gitignore` 中豁免跟踪）。
- **`.gitignore` 忽略、按需拷贝或重建的内容**：
  1. 上游 COCO 权重 `models/yolo11n.pt`：新机运行 `.venv\Scripts\python.exe tools/fetch_search_model.py` 自动下载；
  2. 派生世界文件 `worlds/search_*.wbt` 及 D 地形纹理：新机运行 `.venv\Scripts\python.exe tools/build_search_worlds.py --case D`（及 `A/B/C`）重建；
  3. 核心跑测证据与航测缓存 `runs/`（如需免重飞直接复用全场图或留档视频，请复制以下目录）：
     - `runs/uav_mapping_D_full_route_value_20261121_133722_31008/`（192 帧全场航测 + 0.125 m `fine_ground_z_m` + `surface_dsm.tif`，供 `--map-from` 秒级复现）
     - `runs/uav_rescue_D_full_yolo_igha_vctp_clear_20261121_163823_37812/`（纯 YOLO 闭环 `MISSION_SUCCESS` 留证）
     - `runs/uav_rescue_D_full_surrogate_rgbd_igha_vctp_clear_20261121_145756_27572/evaluation/rescue_simulation.mp4`（720p 跟车 + 实时小地图 PiP 实况视频）
     - `runs/vehicle_redesign_final_20261121_141232/vehicle_showcase.jpg`（无人机/无人车外观重设计对比图）
     - `runs/local_learning_pilot_20261121/`（神经网络局规首轮试跑权重与对照结果）

### 2. 目标机虚拟环境初始化
```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-search.txt -r requirements-search-yolo.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-local-learning.txt
.venv\Scripts\python.exe tools/fetch_search_model.py
.venv\Scripts\python.exe tools/build_search_worlds.py --case D
.venv\Scripts\python.exe -m unittest discover -s tests
```
若 Webots 未安装在默认路径，请设置环境变量 `SEARCH_WEBOTS_EXE` 或 `WEBOTS_HOME`。

---

## 二、无人机航测建图基线

- **入口**：`tools/run_uav_mapping.py`、`start_uav_mapping.cmd`（保持原基线接口完全兼容）。
- **核心模块**：`controllers/uav_mapper/uav_mapper.py`、`survey_map.py`、`survey_refinement.py`、`tools/aerial_products.py`。
- **关键机制**：
  - 无人机在 19 m 高度执行俯拍巡航（全场 156 帧）与 3 个关键补拍区 36 帧斜拍（共 192 帧；局部模式 47 帧）。
  - 深度图按 GPS/IMU 位姿反投影至本地 ENU 米制坐标，融合生成 `0.5 m` 粗通行图 `uav_map.npz`（`states`：`-1=UNKNOWN`、`0=FREE`、`1=BLOCKED`；含 `ground_z_m`、`slope_deg`、`planning_cost`）以及纯俯拍基线 `uav_map_nadir.npz`。
  - **新增在线细地面层导出**：`survey_map.py` 新增 `binned_fine_surface()` 与 `compose_fine_ground()`，`uav_mapper.py` 在线导出 `0.125 m` 细地面高程 `fine_ground_z_m`（全场共 202,099 有效细格），供 VCTP 四轮验路直接使用（对历史旧图亦可回退至 `aerial_products/surface_dsm.tif`，并自动滤除高出已确认地面 `> 0.90 m` 的悬空树冠像素）。
  - 离线产品导出（`tools/aerial_products.py`）：112 万点彩色点云、0.125 m DSM 与反向投影正射图（正射有效像素 98.3%，地面高程有效格 75.8%，未知格 20.4%）。

---

## 三、四种全局规划算法（全部实现并完成 Webots 验证）

| 算法 | 对应文件与核心接口 | 核心机制与关键参数 | 实测验证结论 |
| --- | --- | --- | --- |
| **1. Traversing Mars（双路线搜索）** | `controllers/uav_map_rover/grid_route.py` (`plan_dual`) | 在同一先验图上并行计算两条路线：① **保守路线**（仅允许已确认 `FREE` 且满足 `rover_radius_m=0.40 m` 净空）；② **乐观候选路线**（允许穿越 `UNKNOWN` 格但避开已确认 `BLOCKED`）。自动统计差异未知路点数、走廊未知格集合、是否改变可达性（`changes_reachability`）及潜在绕行缩短收益（`potential_saving_m`），写入 `route.json` 的 `dual_route`。 | D 全场识别出 `north` 乐观走廊 `42.83 m` 与保守路线显著绕行差距，为 DUCPP 补拍提供量化价值信号。 |
| **2. DUCPP（关键路段价值导向补拍）** | `controllers/uav_mapper/survey_refinement.py` (`select_patches_route_value`)，`tools/run_uav_mapping.py --refinement-strategy route_value` | 替代原“纯未知格密度”选点：先用 `plan_dual` 对各任务目标计算双路线差异走廊，按走廊未知格密度、是否解锁原本不可达目标、以及潜在缩短距离对候选补拍区评分，在固定补拍预算（3 个补拍区 / 36 帧斜拍）下优先补拍最具通行价值的关键未知段。 | 同预算消融（`uav_mapping_D_full_route_value_20261121_133722_31008`）：**3/3 补拍区全部精准落在 `north` 42.83 m 关键走廊**，新增 39 个走廊地面证据格与 12 个确认通行格（高程中位误差 1.17 cm）。（注：当前为航测后单轮补拍，行进中动态多轮交互补拍待后续接入。） |
| **3. VCTP（四轮精细地形通过性检验）** | `controllers/uav_map_rover/vctp.py` (`FineTerrainSurface`、`VctpChecker`) | 基于 0.125 m 细地面高程按 `ResearchRover` 真实四轮几何（半轴距 `0.150 m`、半轮距 `0.205 m`、轮半径 `0.090 m`、底盘离地 `0.090 m`）计算 `(x, y, yaw)` 姿态下四轮 FL/FR/RL/RR 独立落点高程，拟合支撑面俯仰/横滚角、四轮扭转残差、单轮阶跃与 9 点底盘腹板净空；细地面缺测处保持 `UNKNOWN` 绝不误判通行。 | D 全场验证：在 0.125 m 细图上 **100% 姿态四轮落入不同细格**（原 0.5 m 粗图下 61/76～110/151 姿态四轮重合）；成功识别并规避细小石块与陡横坡。 |
| **4. IGHA\*（带朝向滑移转向混合搜索）** | `controllers/uav_map_rover/igha_star.py` (`plan_igha_star`)，`--planner igha_vctp` | 在完整 `center_mask`（`0.40 m` 净空）+ VCTP 验路通道内，以 2D 障碍感知 Dijkstra 距离场为启发式，使用适配四轮滑移转向的运动基元（前进直线/左右弧线、原地转向、倒车微调）在 `(x, y, yaw)` 状态空间搜索并输出带朝向路点 `route_xyyaw_m`。 | 与 2D A\* 同图配对实跑 D 东侧接近点（`uav_nav_D_full_igha_vctp_clear_20261121_133902_38772`）：**75/75 姿态 100% 通过 VCTP**（原 A\* 为 96.05%）、0 碰撞 `ARRIVED`（126.2 s，规划 39.87 m），**最大横滚/俯仰降至 `10.46° / 12.13°`**（优于 A\* 的 `11.34° / 13.11°`）。 |

---

## 四、四轮车定点闭环导航与局部避障控制

- **核心模块**：`controllers/uav_map_rover/route_tracker.py`、`uav_map_rover.py`、`controllers/uav_survey/uav_survey.py`（`_evaluate_navigation`）、`tools/run_uav_navigation.py`。
- **关键算法设计**：
  1. **抗噪位姿与停稳估计（`RoverPoseEstimator`）**：仅用轮速编码器 + IMU + 带噪 GPS 融合本地 ENU 位姿；针对 4 cm GPS 噪声和 3 mrad IMU 噪声导致无法累计停稳时长的问题，采用 **12 步 GPS 滑窗均值滑移检测 + 6 步 IMU 航向滑窗**，实现在坡面准确识别真实滑动与静止停稳（`speed < 0.03 m/s` 且 `yaw_rate < 0.08 rad/s` 持续 `≥ 1.5 s`）。
  2. **地形感知车载障碍检测（`extract_lidar_obstacles`、`extract_rgbd_obstacles`）**：将 3 层 360° LiDAR 与前视 320×240 RGB-D 深度按实时 IMU 横滚/俯仰/航向投影至 ENU，对照无人机先验地面高程 `ground_z_m`（带在线高度偏置校准 `z_bias_m`），有效区分 14° 正常缓坡与真实凸起障碍/悬崖落差。
  3. **0.125 m 细子栅格局部障碍叠加与保守重规划（`OnboardObstacleMap`）**：在粗图每格 4×4 细分（0.125 m）上叠加车载新检出障碍，避免 0.5 m 粗格过度膨胀堵死 1.5 m 窄通道，同时严格锁定无人机原始 `UNKNOWN`/`BLOCKED` 边界；遇可绕行障碍时仅在确认通行区内重规划，遇完全封堵且绕行必入未知区时拒绝冒险并触发安全停车（`STOPPED_SAFE`）。
  4. **规则速度采样局部控制器（`RuleLocalPlanner` / `RouteNavigationController`）**：10 Hz 接收 `LocalPlannerInput`（前瞻路点、LiDAR/RGB-D 障碍、先验净空、姿态与速度），前瞻 6 步（1.2 s）动力学采样输出 `(v, w)`，带横滚/俯仰保护、减速进点与停稳确认。
- **Webots D 实测结果**：
  - 无障碍 `valley`（A\*）：`ARRIVED`，127.2 s，39.34 m，0 碰撞，终点误差 0.32 m（`uav_nav_D_full_clear_20261121_130543_12840`）。
  - 沿路可绕行碎石（A\*）：`ARRIVED`，133.0 s，重规划 5 次，0 碰撞，终点误差 0.31 m（`uav_nav_D_full_detour_20261121_130931_38364`）。
  - 沿路封堵障碍（`local` 端到端航测）：`STOPPED_SAFE`，停稳 2.5 s，0 碰撞，记录 `OBSTACLE_BLOCKED_NO_CONFIRMED_DETOUR`（`uav_nav_D_local_block_20261121_131102_30056`）。

---

## 五、神经网络车端局部规划（老师要求 · 已全场达标并接入主救援闭环）

详细技术文档见 `docs/LOCAL_POLICY.md`。
- **核心模块**：`controllers/uav_map_rover/policy_features.py`、`local_policy.py`、`tools/train_local_policy.py`、`tools/compare_local_policies.py`、`tests/search/test_local_policy.py`，已接入 `tools/run_uav_rescue.py --local-policy neural --checkpoint models/local_policy_dagger.pt`。
- **网络架构（18,949 参数，CPU 10 Hz 实时推理，中位耗时 `1.07 ms`）**：
  - **视觉分支**：对齐的 RGB + 深度 + 深度有效掩码（`5 × 24 × 32`）→ 两层轻量卷积；
  - **近场障碍分支**：LiDAR 与 RGB-D 分别按 36 个车体角扇区提取的障碍距离与命中掩码 → 全连接；
  - **导航状态分支**：车体系前瞻路点、目标距离、轮速、上一拍 `(v, w)`、横滚/俯仰、11 个先验+车载地图净空查询 → 全连接；
  - **输出与损失**：三分支融合输出归一化 `(v, w)`（分母 `0.42 m/s`、`0.85 rad/s`，Smooth L1 损失）与 3 类动作模式（停止/原地转向/行驶，类别加权交叉熵 × 0.35）。
- **D 全场大规模 BC + DAgger 训练（`runs/local_learning_D_full/`）**：
  - 采集 2 条 100% 零碰撞 D 全场 40 m 河谷专家轨迹（2,818 帧训练）+ 1 条独立障碍构型验证轨迹（1,340 帧）预训练 **`models/local_policy_bc.pt`**（`val_loss=0.0290`，`mae_v=0.0133 m/s`，`mae_w=0.0294 rad/s`，模式准确率 **`99.78%`**）；
  - 用 BC 策略在 2 个全新障碍构型上实跑 DAgger 回采（`expert_mix=0.15`，两条均 **`ARRIVED`、0 碰撞**，共聚合 **5,361 帧训练样本**），以 `lr=0.00025` 微调导出 **`models/local_policy_dagger.pt`**（验证误差进一步全面超越 BC：`val_loss=0.0276`、`mae_v=0.0126 m/s`、`mae_w=0.0280 rad/s`、模式准确率 **`99.78%`**）。
- **D 全场留出 0.65 m 碎石三组同条件对比（`runs/local_learning_D_full/compare_full_detour_v4/comparison.json`，`held_out=true`，0% 专家混合，200 s 预算，已含最小安全裕度 $S_{\text{margin}}$ 与避障平滑度 $J_{\text{avoid}}$）**：
  - **规则专家**：`ARRIVED`，0 碰撞，`160.0 s`，`42.95 m`，终点误差 `0.323 m`，重规划 `8` 次，$S_{\text{margin}}=0.112\text{ m}$，$J_{\text{avoid}}=4352.1$；
  - **Neural BC（`bc.pt`）**：**`ARRIVED`**，**0 碰撞**，`163.4 s`，`43.00 m`，终点误差 **`0.274 m`**，重规划 **`6` 次**，**`97.4%` 直接神经网络控制**（回退 `2.60%`），P95 推理 `1.27 ms`，$S_{\text{margin}}=0.122\text{ m}$，$J_{\text{avoid}}=2562.8$（平滑度提升 `41.1%`）；
  - **Neural BC + DAgger（`dagger_v2.pt` / `models/local_policy_dagger.pt`）**：**`ARRIVED`**，**0 碰撞**，`193.3 s`，**`39.48 m`（全组最短 -8.1%）**，重规划最低 **`5` 次（-37.5%）**，**`98.5%` 直接神经网络控制**（回退最低 **`1.46%`**），P95 推理 **`1.25 ms`**，**$S_{\text{margin}}=0.191\text{ m}$（+70.6%）**，**$J_{\text{avoid}}=1561.3$（平滑度提升 64.1%）**；
- **2026-11-22 交科赛冲刺优化新增模块（全仓库 81/81 单元测试通过）**：
  - **IGHA\* 起步 GPS 抖动恢复（`controllers/uav_map_rover/igha_star.py`）**：起步位姿因 4 cm GPS 抖动落于细高程单像素噪点时，自动按角度就近搜索同位置可通行原地转向角及 1 细格容差，消除无谓 `astar_fallback`；
  - **车载模块化双仓物资精准投放机构（`protos/ResearchRover.proto` + `controllers/uav_survey/uav_survey.py`）**：后甲板加装低剖面 CNC 双仓释放架（高度 ≤ `0.344 m`，零雷达/前视遮挡），满载 2 个带白十字橙色物资箱，每次物理投送时同步卸载对应舱位箱体；
  - **7 组多随机位置 + 全色域随机换装实跑统计**：涵盖标准 D 全程全栈（`224203_1688`，89.1 s 中文 HUD 全程实录 `MISSION_SUCCESS` · 0 碰撞）、双仓机构 100% IGHA\*（`161638_22012`，`MISSION_SUCCESS` · 0 碰撞）、全非蓝衣随机 `seed=17`（`200923_13096`，`MISSION_SUCCESS` · 2/2 投送 · RMSE `0.239 m` · 0 碰撞）、`seed=42`、`seed=7` 及极端不可达 `seed=101/202`（安全拒入 0 碰撞），空中纯 YOLO 总检出 **`21/21 = 100%`**、**`0` 误报**、平均定位误差 **`0.130 ± 0.043 m`**；
  - **安全平滑指标、真机摄影测量接口与中文指挥调度大屏**：`tools/safety_metrics.py`、`tools/import_real_dsm.py`（WebODM/Mini 4 Pro GeoTIFF DSM → `uav_map.npz` + `VCTP+IGHA*` 规划）、`tools/rescue_dashboard.py`（支持本地交互服务与单文件离线 HTML 导出 `runs/援来有路_空地协同指挥调度大屏_离线版.html`），配套单元测试 `tests/search/test_competition_tools.py`。
- **2026-11-23 极端灾后工况与多姿态抗遮挡短板补齐（全仓库 83/83 单元测试通过）**：
  - **动态障碍物主动避让三组同条件对比（`--obstacle dynamic`，`runs/local_learning_D_full/compare_full_dynamic/comparison.json`，`held_out=true`）**：在 `OnboardObstacleMap` 加入近场 `0.45–2.6 m` 自由空间 1.8 s 动态残影衰减清除；河谷走廊部署 `0.25 m/s` 横向穿越移动障碍后，**规则专家（1 碰撞 `-6.4 cm` 失败）**与**纯 BC（1 碰撞 `-7.8 cm` 失败）**双双撞击停摆，**唯有 `BC + DAgger` 以最小 `+57.5 cm` 动态安全裕度、0 碰撞、P95 推理 `0.995 ms`（<1 ms）、`95.5%` 神经直控安全到达终点（`158.7 s`，误差 `0.285 m`）**；
  - **Webots 真实天气视觉退化测试（`tools/run_uav_rescue.py --weather {clear,lowlight,fog}`）**：在真实 Webots 渲染下注入黄昏低照度（光强 -80%，`20261123_170102_8096`）与山谷 38 m 浓雾（`20261123_171203_25240`），空中检出均保持 **`3/3 = 100%`、`0` 误报**，车载抵近复核置信度达 **`0.853` / `0.897`**（融合 RMSE **`0.220 m` / `0.184 m`**）；
  - **v3 零泄漏六姿态 + 废墟半掩埋 + 全天候抗雾航拍 YOLO11n（`tools/posture_casualty.py`、`tools/collect_yolo_dataset.py --multi-posture`、`models/yolo11n_victim.pt`）**：针对仅仰卧训练的 v2 在留出 D 场景 5 类非仰卧姿态（`俯卧 / 侧卧 / 蜷缩成团 / 抱膝坐姿 / 废墟半掩埋`，`randv53`）空中召回率跌至 `3/5 (60%)`（漏检蜷缩成团与废墟半掩埋）的短板，仅在 **A+B 场景**采集 6 种真实灾后姿态 + 瓦砾半掩埋 + 16 色换装 + 纯灰度 + Koschmieder 大气散射雾化样本（**784 训练图**，C 场景 **175 验证图** `mAP@50 = 0.830` / `Precision = 0.859`，**D 场景 100% 严格留出**），热启动微调并加入 `erasing=0.20` 随机擦除增强。留出 D 场景 **7 项全工况同条件对照（`runs/cn_figures_20261121/v3_posture_benchmark.json`）全部达到 `100%` 召回与 `0` 误报（合计 `23/23`、1,092 航拍帧零误报）**：**5 类非仰卧姿态由 `3/5 (60%)` 跃升至 `5/5 (100%)`（RMSE `0.213 m`，其中原漏检的蜷缩成团 7 视角 `0.69`、废墟半掩埋 7 视角 `0.72`）**，且标准仰卧（`3/3`, `0.121 m`）、`Hue+60°`（`3/3`, `0.116 m`）、纯灰度（`3/3`, `0.110 m`）、黄昏低照度（`3/3`, `0.160 m`）、38 m 浓雾（`3/3`, `0.185 m`）与随机全非蓝衣 `seed=17`（`3/3`, `0.182 m`）**全部零退化**。

---

## 六、无人机与四轮无人车 3D 模型与 PBR 材质重设计

针对原模型“偏玩具感”的问题，按工业级救援平台标准重设计了全部外观与贴图，同时**严格锁定所有物理碰撞体、轴距/轮距/轮径和全部传感器位姿/FOV，经实测零光学自遮挡**：

| 文件 | 重设计内容 | 关键约束校验 |
| --- | --- | --- |
| `protos/RescueRoverBody.proto` | 工业级 UGV 底盘与上装：3K 碳纤维主甲板、CNC 拉丝枪灰铝合金护框、外露双叉臂悬架与橙色螺旋减震器、前部防撞管杠 + 双 LED 探照灯舱、带锁扣防水 Pelican 器材箱、多面体传感器塔（3 层 360° 光学 LiDAR 窗 + 前置 RGB-D 双目舱 + 后部 UHF 天线 + 警示灯与 `UGV-01` 标识）。已修复首版渲染中箱盖浮空缝、标识镜像、天线穿模锁扣与灯舱干涉问题。 | 轴距 `0.30 m`、轮距 `0.41 m`、轮半径 `0.09 m`、离地 `0.09 m`、相机 `[0.26, 0, 0.30]`（320×240, FOV $\pi/2$, 下倾 `0.08 rad`）、LiDAR 高度与视野 100% 保留。 |
| `protos/RescueRoverWheel.proto` | 深齿越野橡胶胎面 + 防脱轮毂（Beadlock）与金属螺栓环。 | 碰撞体圆柱半径 `0.09 m`、摩擦属性与电机绑定不变。 |
| `protos/ObserverDrone.proto` | Matrice 级工业四旋翼：3K 碳纤维上下中心板 + 拉丝枪灰飞控核心舱、碳纤维管状机臂与外转子电机座、高展弦比渐变碳纤桨叶、三轴下视云台横梁与光学双目载荷舱、轻量化碳纤起落架滑橇与 `UAV-01` 顶板标识。**另加装第三人称跟飞相机 `drone_fpv_cam`（480×270）**，配合 `ResearchRover.proto` 新增的**第三人称跟车相机 `rover_chase_cam`（480×270）**与车头前视 `rgb`，在 Webots 实时窗口及 `--record-video` 双阶段录屏中可直接观看无人机飞行与小车行驶全过程。 | 下视 RGB-D（640×480, FOV `1.1 rad`, 载荷中心 `z = -0.20 m`）光学视野 100% 无起落架/机臂自遮挡；新增观察相机无物理碰撞体、无遮挡网格。 |
| `protos/BeveledBox.proto`、`RoundedHousing.proto`、`protos/textures/*.png` | 程序化倒角几何体 + 6 张无缝高分 PBR 贴图：`carbon_weave_base.png` / `carbon_weave_normal.png`、`brushed_metal_base.png` / `brushed_metal_normal.png`、`rescue_rover_id.png`、`rescue_drone_id.png`。 | 对比验收图见 `runs/vehicle_redesign_final_20261121_141232/vehicle_showcase.jpg`。 |

---

## 七、零泄漏航拍 YOLO 微调与空地协同“人即终点”救援投送闭环

### 1. 零泄漏多视角数据集采集与 YOLO11n v2 全色域微调
- **痛点**：原受控检测器 `surrogate_rgbd` 本质为“深蓝上衣颜色 + 尺寸”规则块，换装即 `0/3`；未微调通用 COCO `models/yolo11n.pt` 在 19 m 俯拍平躺伤员（仅约 25 px）上检出率为 `0/3`；首版微调 v1 在极端 `+60°` 色相偏移或纯灰度图上高空置信度会明显下降。
- **v2 零泄漏随机位置 + 16 色换装采集（`controllers/yolo_dataset_collector/yolo_dataset_collector.py`、`tools/collect_yolo_dataset.py`）**：
  - **严格排除 D 场景**：仅在 Webots **A + B 场景**采训练集（**525 张**：340 多色随机位置航拍正样本 + 110 航拍纯背景难负样本 + 75 张纯灰度复制样本），在独立地形 **C 场景**采验证集（**140 张**：105 正 + 35 难负）；每 4 帧动态将全部 6 名伤员随机刷新至新地形点、新坡面朝向，并实时随机切换 **16 种上衣 PBR 颜色（红、橙、黄、绿、橄榄、白灰、炭黑、深蓝、天蓝、酒红、棕、紫等）与 8 种裤子颜色**。
  - **深度遮挡校验自动标注**：将伤员 3D 骨架关键点投影至相机像素，与 RangeFinder 实测深度逐点比对过滤完全树冠遮挡帧，生成紧致 YOLO 标注框（目视抽检见 `runs/yolo_victim_dataset_v2/preview_labels.jpg`）。
- **v2 热启动微调（`tools/train_yolo_victim.py`）**：从 v1 权重热启动 25 轮（`imgsz=640`, `AdamW`, `lr0=0.0018`, `hsv_h=0.35`, `degrees=180.0`, `flipud=0.5`, `mosaic=0.8`），导出新版 `models/yolo11n_victim.pt`（5.49 MB）。C 场景留出验证集跃升至 **`Precision = 0.887`、`Recall = 0.786`、`mAP@50 = 0.862`、`mAP@50-95 = 0.533`**（单类高空部署阈值校准为 `conf = 0.50`）。
- **严格留出 D 场景 192 帧真实航拍三重压力对照**：

| 测试条件（192 帧留出 D 航拍） | 未微调 COCO `yolo11n.pt` | **v2 微调 `yolo11n_victim.pt`（`conf=0.50`）** |
| --- | --- | --- |
| 原图（`orig`） | `0 / 3`（24.80 s） | **`3 / 3` · `0` 误报 · RMSE `0.117 m`**（`5.24 s`，提速 4.7×） |
| 极端 `+60°` 色相偏移（蓝衣变红紫） | — | **`3 / 3` · `0` 误报 · RMSE `0.103 m`**（`5.62 s`） |
| **纯灰度图（完全剥离颜色信息）** | — | **`3 / 3` · `0` 误报 · RMSE `0.092 m`**（`5.14 s`） |

- **空地双视角 YOLO 分工（`controllers/search_agent/perception.py`）**：
  - `PersonDetector("yolo", ROOT, max_depth=...)` 在航拍视角（`max_depth >= 12.0`）自动加载微调专精模型 **`models/yolo11n_victim.pt`**（单次前向，`conf=0.50`）；在地面近景视角（`max_depth < 12.0`）自动加载通用 **`models/yolo11n.pt`**（侧面/正头顶/变色/灰度均稳定检出 `0.41–0.81`）。支持 `UAV_YOLO_WEIGHTS` 环境变量覆写以便 A/B 测试。

### 2. 空地协同“人即终点”救援投送、伤员随机刷新与录屏闭环
- **核心模块**：`controllers/uav_mapper/aerial_victims.py`、`controllers/uav_map_rover/rescue_mission.py`、`uav_map_rover.py`（`run_rescue_mission`）、`controllers/uav_survey/uav_survey.py`（`_evaluate_rescue`）、`tools/run_uav_rescue.py`、`start_uav_rescue.cmd`。
- **支持伤员位置与服装随机刷新**：`tools/run_uav_rescue.py` 新增 `--random-victims` / `--victim-seed <seed>` / `--random-clothing`（及 `--nadir-only` 快速全图重飞）。启动前在 D 场景可通行坡度（`< 14.5°`）、避开树干/灌木/房屋/出生点的位置随机生成 3 名伤员（可选随机非蓝服装），真值仅写入评测专属 `evaluation/truth_override.json`（控制器完全不可见），强制无人机全新起飞实检。
- **完整执行链路（全程零真值输入）**：
  1. **空中检人聚类**：`AerialVictimTracker` 对无人机俯拍 RGB-D 帧执行检测与 ENU 反投影，用 `covariance_intersection` 多帧聚类、过滤单帧噪点（`min_views=2`），输出 `aerial_victims.json`。
  2. **动态安全停靠点与多目标调度**：`select_victim_approach_point` 绝不压向伤员本体，而是在检出位置周边 `0.80–1.58 m` 确认 `FREE` 环带内按理想停靠距（`1.20 m`）、面向伤员的 VCTP 四轮通过性与地形代价选点，用 `VCTP + IGHA*` 规划路线；`plan_next_victim` 按最短已确认可达路径贪心调度下一名伤员。
  3. **安全停靠、抵近复核与物资投送**：小车到达目标接近点（或在伤员本体进入车载 LiDAR 时安全停于 `≤ 1.55 m` 停靠距外），原地转向正对伤员、停稳 `2.2 s` 并调用前视 RGB-D YOLO 复核与航拍先验跨视角融合（保存 `rover/victim_*_confirm.jpg`），发布 `report` 与 `delivery` 事件。
  4. **独立真值评测与可视化**：Supervisor 侧 `_evaluate_rescue` 对照 `evaluation/truth_override.json`（或默认 `evaluation/search/D.json`）校验上报匹配（`≤ 1.0 m`）、停稳时长（`≥ 1.5 s`）、投送距离（`≤ 1.8 m`）与 0 碰撞，在车头旁生成可视化橙色物资箱并输出 `evaluation/rescue_preview.png`。
  5. **跟车实况录屏（`--record-video --video-speed 4`）**：规避 Windows 下 Webots `movieStartRecording` 的 FFmpeg `0xC0000005` 崩溃，采用平滑 3/4 跟车视角 `exportImage` + OpenCV H.264（`avc1`）合成，自动叠加顶部状态栏与右下角**实时无人机通行图 + 小车轨迹小地图 PiP**，输出 `evaluation/rescue_simulation.mp4`。
- **Webots D 场景实跑验收**：
  - **标准 D 场景纯 YOLO 闭环（`uav_rescue_D_full_yolo_igha_vctp_clear_20261121_163823_37812`）**：**`MISSION_SUCCESS`**，无人机检出 `3/3`，车载复核 **`2/2` 全部空地跨视角融合确认**（score `0.909`、`0.864`），**`2/2` 有效投送**（`1.53 m`、`1.46 m`），`65.93 m` / `224.0 s`，**`0` 碰撞、`0` 车体接触**，RMSE `0.406 m`。
  - **D 场景随机位置 + 酒红/绯红换装全新重飞实测（`uav_rescue_D_full_yolo_igha_vctp_clear_randv42_20261121_190007_11920`）**：无人机 156 帧全新航飞实时检出 **全部 3/3 名非蓝衣随机伤员、0 误报、实时航拍误差 `0.080–0.138 m`**；四轮车自主穿谷行驶 **`111.68 m`、`0` 碰撞、`0` 车体接触**，第 1 段 `60.28 m`（VCTP `112/112` 通过）精准抵近酒红上衣伤员完成车载 YOLO 跨视角复核与 `1.42 m` 有效投送（融合误差 `0.204 m`），第 2 段再行 `51.4 m` 抵近东侧边缘伤员附近遇灌木未知区安全停车（`STOPPED_SAFE`）。

---

## 八、全量启动命令速查

```powershell
# 1. 全量单元测试回归（当前 77/77 全通过）
.venv\Scripts\python.exe -m unittest discover -s tests

# 2. 无人机航测与 DUCPP 补拍基线
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile local
.venv\Scripts\python.exe tools/run_uav_mapping.py --profile full --refinement-strategy route_value

# 3. 定点闭环导航（A* vs VCTP+IGHA* / 可绕行障碍 / 封堵停车）
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --planner igha_vctp --obstacle none --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile full --goal valley --obstacle detour --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
.venv\Scripts\python.exe tools/run_uav_navigation.py --profile local --obstacle block

# 4. 纯 YOLO 空地协同双伤员救援投送（默认 --detector yolo，可加 --visible --realtime --keep-open 或 --record-video）
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --max-victims 2 --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008
# 随机刷新伤员位置与随机服装颜色（需全新无人机重飞，可加 --nadir-only 加速）：
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --nadir-only --detector yolo --planner igha_vctp --max-victims 2 --victim-seed 42 --random-clothing
# 终极全栈合体（纯 v2 YOLO + VCTP/IGHA* + 神经网络 DAgger 局部规划）：
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --local-policy neural --checkpoint models/local_policy_dagger.pt --max-victims 2 --map-from runs/uav_mapping_D_full_route_value_20261121_133722_31008

# 5. 零泄漏多色随机位置 YOLO 数据采集、微调与留出 D 场景三重压力评测
.venv\Scripts\python.exe tools/collect_yolo_dataset.py --out runs/yolo_victim_dataset_v2
.venv\Scripts\python.exe tools/train_yolo_victim.py --data runs/yolo_victim_dataset_v2/dataset.yaml --epochs 25 --init models/yolo11n_victim.pt --output models/yolo11n_victim.pt
.venv\Scripts\python.exe tools/train_yolo_victim.py --benchmark-only --output models/yolo11n_victim.pt --benchmark-survey runs/uav_mapping_D_full_route_value_20261121_133722_31008

# 6. 神经网络局部规划训练与三组同条件评测（详见 docs/LOCAL_POLICY.md）
.venv\Scripts\python.exe tools/train_local_policy.py --help
.venv\Scripts\python.exe tools/compare_local_policies.py --help
```

---

## 九、工程红线、诚实局限与下一阶段路线

### 工程红线与已知坑
1. **严格零真值泄漏**：`evaluation/search/D.json` 仅限 Supervisor 评测使用；任何控制器与训练脚本绝不得读取 D 场景真值或测试帧。
2. **坐标约定**：`uav_map.npz` 为 Webots 本地 ENU 米制，数组行由南向北、列由西向东；`states` 为 `-1=UNKNOWN, 0=FREE, 1=BLOCKED`。
3. **保留基线**：`tools/run_uav_mapping.py` 与 `tools/run_uav_navigation.py` 必须保持向后兼容。
4. **Windows / PowerShell 注意**：勿在 `--batch` 下调用 Webots `movieStartRecording`（会触发 `0xC0000005` 崩溃，应使用内置 `exportImage` + OpenCV 方案）；PowerShell 中内联含 `{}` 的 Python 脚本须使用单引号 here-string `@' ... '@ | .venv\Scripts\python.exe -`。

### 诚实边界与下一步优先级
1. **【最高优先】神经网络车端局部规划提效与全场接入（老师硬指标）**：
   - 定位首轮局规试跑（`runs/local_learning_pilot_20261121/`）中规则专家 5.7–5.8 cm 轻微接触及 DAgger 1.9 m 早停的根因；
   - 扩大 D 全场多路段 + 多障碍构型的专家采集（补齐减速/停止样本），进行多轮 DAgger 聚合微调；
   - 在 D 局部与 D 全场产出高成功率的**规则 vs BC vs BC+DAgger** 同条件评测表，并将最终神经网络局规接入 `tools/run_uav_rescue.py`，完成“**纯 YOLO + 四种全局规划 + 神经网络局部规划 + 物理投送**”完整合体。
2. **行进中动态交互补拍与 3 伤员全覆盖**：接通小车遇 `UNKNOWN` 阻断时实时向无人机请求 DUCPP 补拍并刷新通行图（如随机伤员落在东侧林缘窄缝时补拍解锁）；支持 `--max-victims 3` 或返航起点 `reload` 覆盖全部 3 名伤员。
3. **Git 远端同步**：在网络畅通的本机终端执行 `git push origin main` 将本地提交同步到 `https://github.com/pindo-coder/webots-vision-rescue.git`。

---

## 十、关键实测证据总索引（`runs/`）

| 类别 | 目录 / 文件 | 核心结果 |
| --- | --- | --- |
| 全场 DUCPP 航测（推荐 `--map-from` 源） | `runs/uav_mapping_D_full_route_value_20261121_133722_31008/` | 192 帧，3/3 补拍区命中 `north` 42.83 m 走廊，导出 202,099 格 0.125 m `fine_ground_z_m` |
| 载具外观重设计留证 | `runs/vehicle_redesign_final_20261121_141232/vehicle_showcase.jpg` | 四轮车 + 无人机 PBR 重设计，零光学自遮挡 |
| 定点闭环：A\* 无障碍 / 绕行 / 封堵停车 | `runs/uav_nav_D_full_clear_20261121_130543_12840/`、`runs/uav_nav_D_full_detour_20261121_130931_38364/`、`runs/uav_nav_D_local_block_20261121_131102_30056/` | `ARRIVED`(0 碰)、`ARRIVED`(5 重规划, 0 碰)、`STOPPED_SAFE`(0 碰) |
| 定点闭环：`VCTP + IGHA*` 对照 | `runs/uav_nav_D_full_igha_vctp_clear_20261121_133902_38772/` | `ARRIVED` 126.2 s，VCTP 75/75 (100%)，最大横滚/俯仰 `10.46°/12.13°` |
| 零泄漏 v2 全色域 YOLO 数据集与微调 | `runs/yolo_victim_dataset_v2/preview_labels.jpg`、`runs/yolo_victim_train_20261121_182431/`、`models/yolo11n_victim.pt` | A+B 525 / C 140（mAP@50 `0.862`）；留出 D 原图/色偏/纯灰度均 `3/3`、0 误报、RMSE `0.092–0.117 m` |
| **纯 YOLO 救援闭环验收（标准 D）** | `runs/uav_rescue_D_full_yolo_igha_vctp_clear_20261121_163823_37812/evaluation/summary.json`、`rescue_preview.png` | **`MISSION_SUCCESS`**，3/3 检出、2/2 跨视角融合、2/2 投送（1.53/1.46 m）、0 碰撞 |
| **随机位置 + 随机换装全新重飞实测** | `runs/uav_rescue_D_full_yolo_igha_vctp_clear_randv42_20261121_190007_11920/evaluation/rescue_preview.png`、`rover/victim_03_confirm.jpg` | 实时航飞检出 3/3 酒红/绯红随机伤员（0 误报，误差 `0.08–0.14 m`）；车行 `111.68 m` 0 碰撞，完成酒红伤员跨视角确认与 `1.42 m` 投送 |
| 720p 跟车仿真视频（带实时小地图） | `runs/uav_rescue_D_full_surrogate_rgbd_igha_vctp_clear_20261121_145756_27572/evaluation/rescue_simulation.mp4`、`video_keyframes.jpg` | 55.9 s (4× 速, 1118 帧, H.264, 82 MB)，`MISSION_SUCCESS` |
| **终极全栈合体搜救（纯 YOLO + VCTP/IGHA\* + 神经 DAgger）** | `runs/uav_rescue_D_full_yolo_igha_vctp_clear_20261121_205200_21688/evaluation/rescue_preview.png`、`summary.json` | **`MISSION_SUCCESS`**，95.1% 神经网络控制（1.07 ms）、69.26 m 0 碰撞、2/2 跨视角确认、2/2 投送（1.59/1.49 m）、RMSE `0.372 m` |
| D 全场神经网络局规训练与三组留出对比 | `runs/local_learning_D_full/compare_full_detour_v2/comparison.json`、`models/local_policy_bc.pt`、`models/local_policy_dagger.pt` | BC `ARRIVED` 0 碰（0.274 m 误差，97.4% 神经）；DAgger 0 碰最低回退 `1.56%` / 最少重规划 `5` 次 |
