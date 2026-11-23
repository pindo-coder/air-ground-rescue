# Gazebo Harmonic + ROS 2 Jazzy · D 山洪森林场景完整复现手册

本分支已同步 Webots `main` 至 `8fa1611`，在 **Gazebo Sim Harmonic（8.x）+ ROS 2 Jazzy（`ros_gz_bridge`）** 上完整复现以下空地协同救援能力：

- **4WD 工业级载具与静态双仓架**：完整导出 `RescueRoverBody` + 4 个 `RescueRoverWheel`（轴距 `0.300 m`、轮距 `0.410 m`、轮径 `0.090 m`、离地 `0.090 m`，与 VCTP 几何 100% 一致）以及后甲板双仓物资释放架（静态外观，零雷达/相机遮挡、无多余碰撞体）。
- **全量传感器与真值隔离**：
  - 地面车：`/rover/rgb`、`/rover/depth`、`/rover/chase_cam`、`/rover/lidar`（+ `/rover/lidar/points`）、ENU `/rover/imu`、`/rover/gps`（NavSat）、`/rover/odom`、`/rover/joint_states`（4 轮）。
  - 无人机：`/drone/rgb`、`/drone/depth`、`/drone/fpv_cam`、`/drone/gps`（光心同位）、ENU `/drone/imu`。
  - 评测隔离：`/eval/world_poses`、`/eval/drone_pose`、`/eval/rover_contact`、`/eval/rover_wheel_contacts` 仅供 `evaluator_node.py` 消费，建图与车端规划节点零真值泄漏。
- **全链路算法与评测**：0.5 m / 0.125 m 粗细建图 → Traversing Mars 双路线 → DUCPP 价值导向斜拍补拍 → VCTP + IGHA\*（含起步 GPS 抖动恢复）→ 规则 / PyTorch BC+DAgger 神经网络 10 Hz 局规 → 零泄漏 v2 航拍 YOLO11n + 车端复核投送 → 自动输出速度、定位误差、安全裕度 `S_margin`、避障平滑度 `J_avoid` 与离线中文指挥调度大屏。

---

## 一、运行环境与模型权重准备（Ubuntu 24.04 / WSL2）

### 1. 系统前提
- **操作系统**：Ubuntu 24.04 LTS（原生或 Windows 11 WSL2 + WSLg；无图形界面服务器也可在所有任务命令后加 `--headless` 运行）。
- **前置软件源**：需已按官方文档配置 [ROS 2 Jazzy apt 源](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)（Ubuntu 24.04 自带 Gazebo Harmonic 对应依赖包）。

### 2. 安装 Gazebo Harmonic、ROS 2 Jazzy 与桥接包
```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-tf2-ros \
  python3-venv \
  python3-gz-sim8 python3-gz-math7 python3-numpy
```

### 3. 创建 Python 虚拟环境并补齐权重
> **关键**：必须带 `--system-site-packages`，让 `.venv` 直接访问系统 ROS 2 的 `rclpy` 和 Gazebo 的 `gz.sim8`。

```bash
# 在仓库根目录执行
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv

.venv/bin/python -m pip install -r requirements-build.txt -r requirements-search.txt -r requirements-search-yolo.txt
# 有 NVIDIA GPU（CUDA 12.4）用第一条；纯 CPU 环境改用第二条
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
# .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-local-learning.txt

# 下载上游 COCO 通用检测权重 models/yolo11n.pt（车端近景复核使用）
.venv/bin/python tools/fetch_search_model.py
```

权重清单说明：
| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `models/yolo11n_victim.pt` | **随仓库自带** | 无人机 19 m 航拍零泄漏 v2 全色域伤员检测 |
| `models/yolo11n.pt` | `fetch_search_model.py` 下载 | 无人车近距前视伤员复核 |
| `models/local_policy_dagger.pt` / `local_policy_bc.pt` | **随仓库自带** | 神经网络局部规划权重（可直接推理，或见第三节在 Gazebo 中自行微调） |

### 4. 环境一键自检（建议首次必跑）
```bash
source /opt/ros/jazzy/setup.bash
gz sim --version
ros2 --help >/dev/null && echo "ROS 2 OK"
.venv/bin/python -c "import rclpy, gz.sim8, numpy, scipy, cv2, torch, ultralytics; print('Python deps OK')"
ls -lh models/*.pt
```
确认 `gz sim` 输出 `8.x`、打印 `ROS 2 OK` 和 `Python deps OK`、且 `models/` 下 4 个 `.pt` 文件均存在后，再进入下一步。

---

## 二、构建并校验 D 场景 SDF 包

首次拉取代码，或之后改动过 `protos/*.proto` / 场景配置时，必须重新生成 SDF：

```bash
.venv/bin/python tools/build_search_worlds.py --case D
.venv/bin/python tools/export_gazebo_d.py
gz sdf -k gazebo/generated/D/search_d.sdf
```

- **通过标志**：`gz sdf -k` 输出 `Valid.`（返回码 0），且生成 `gazebo/generated/D/search_d.sdf`、`environment.json`、`run.sh` 与网格/贴图资产。
- **零 Node.js 依赖说明**：若本机未安装 Node.js，`tools/webots_static_scene.py` 会自动读取 `gazebo/proto_cache/` 预展开缓存，不影响导出完整工业级外观与 PBR 贴图。
- **仅预览场景（不跑算法）**：
  ```bash
  ./gazebo/generated/D/run.sh
  ```

---

## 三、按顺序完整复现（Step by Step）

> **每次新开终端必须先执行**：`source /opt/ros/jazzy/setup.bash`
> 如无需实时 3D 窗口（如远程终端或批量评测），在下列 `run_gazebo_mission.py` 命令末尾追加 `--headless` 即可。

### Step 1：无人机全场航测 + DUCPP 价值导向斜拍补拍
先跑一次航测生成固定目录 `runs/gz_survey_D_full`，后续导航、救援、神经局规训练全部可通过 `--map-from` 秒级复用该先验图（无需重复飞行）：

```bash
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode mapping --profile full --refinement-strategy route_value \
  --out-dir runs/gz_survey_D_full
```
- **预期产物**：`runs/gz_survey_D_full/` 下生成 `uav_map.npz`（含 0.5 m 粗通行图与 0.125 m `fine_ground_z_m`）、`uav_map_nadir.npz`、`observations.npz`、`oblique_survey.json`。

### Step 2：定点闭环导航验证（VCTP + IGHA\* + 局部避障）
复用 Step 1 的先验图，测试河谷目标 `valley`（支持 `--obstacle none` 无障碍、`detour` 沿路可绕行碎石、`block` 完全封堵安全停车）：

```bash
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode navigation --profile full --goal valley --planner igha_vctp --obstacle detour \
  --map-from runs/gz_survey_D_full \
  --out-dir runs/gz_nav_D_detour
```
- **验收标准**：终端打印的 `evaluation/summary.json` 中 `status` 为 `ARRIVED`（`block` 场景为 `STOPPED_SAFE`）、`collisions == 0`、`chassis_contacts == 0`，且包含 `min_safety_margin_m` 与 `j_avoid_integral`。

### Step 3：纯 YOLO11n 空地协同救援投送全闭环
提供两种复现方式：

- **方式 A（推荐，快速复现）**：复用 Step 1 航测帧离线跑 YOLO 空中检人后直接部署小车执行抵近复核与投送：
  ```bash
  .venv/bin/python gazebo/run_gazebo_mission.py \
    --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
    --map-from runs/gz_survey_D_full --record-cameras \
    --out-dir runs/gz_rescue_D_yolo
  ```
- **方式 B（端到端从头起飞）**：不传 `--map-from`，无人机现场起飞航测、补拍、空中检人并落盘，随后小车自动部署并完成救援（耗时会多出航测飞行时间）：
  ```bash
  .venv/bin/python gazebo/run_gazebo_mission.py \
    --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
    --record-cameras \
    --out-dir runs/gz_rescue_D_e2e
  ```

> **说明**：场景中有 3 名伤员，但无人车默认载货量为 2 箱（`--max-victims 2`），无人机空中会检出全部伤员，小车按最短已确认安全路径完成 2 次抵近复核与物资投送后即判定 `MISSION_SUCCESS` 正常结束；如需尝试投送第 3 人，改为 `--max-victims 3`。

### Step 4：车端神经网络局部规划（直接推理或 Gazebo 重训）
- **方式 A（直接用仓库自带权重推理）**：
  ```bash
  .venv/bin/python gazebo/run_gazebo_mission.py \
    --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
    --local-policy neural --local-checkpoint models/local_policy_dagger.pt \
    --map-from runs/gz_survey_D_full --record-cameras \
    --out-dir runs/gz_rescue_D_neural
  ```
- **方式 B（在 Gazebo 中采集数据并训练专属 BC + DAgger 权重）**：
  ```bash
  .venv/bin/python gazebo/train_gazebo_local_policy.py \
    --map-from runs/gz_survey_D_full --profile full --goals valley \
    --obstacles none detour --dagger-rounds 2 --headless \
    --out-dir runs/gz_local_train

  # 用新训权重跑全场救援闭环
  .venv/bin/python gazebo/run_gazebo_mission.py \
    --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
    --local-policy neural --local-checkpoint runs/gz_local_train/dagger_r2_policy.pt \
    --map-from runs/gz_survey_D_full --record-cameras \
    --out-dir runs/gz_rescue_D_neural_gz
  ```

### Step 5：核对输出产物并导出离线中文指挥调度大屏
每次运行结束后，在对应 `--out-dir`（如 `runs/gz_rescue_D_yolo/`）下会产出：

| 路径 | 内容与验收重点 |
| --- | --- |
| `evaluation/summary.json` | `status`（`MISSION_SUCCESS`/`ARRIVED`）、`collisions`、`chassis_contacts`、`delivered_people`、`localization_rmse_m`、`min_safety_margin_m`（$S_{\text{margin}}$）、`j_avoid_integral`（$J_{\text{avoid}}$）、`rms_jerk_mps3`、`deliveries`、`aerial_candidates_audit` |
| `evaluation/trajectory.csv` | 含 `time_s, x, y, z, yaw_rad, roll_deg, pitch_deg, speed_mps, loc_error_m` |
| `evaluation/collision_events.json` | 碰撞事件明细（理想情况为空列表 `[]`） |
| `aerial_victims.json` | 无人机空中检出并聚类的伤员候选列表 |
| `rover/chase_frames/`、`rover/victim_*_confirm.jpg` | 加 `--record-cameras` 后的跟车视角帧与车载近距复核图 |

导出**单文件离线中文大屏 HTML**（无需起服务，直接用浏览器双击打开，兼容本次新跑目录及历史旧 Gazebo 归档）：
```bash
.venv/bin/python tools/rescue_dashboard.py \
  --run runs/gz_rescue_D_yolo \
  --export-static runs/gz_rescue_D_yolo/dashboard_gazebo.html
```
或启动本地交互式 Web 服务，在下拉框中切换浏览 `runs/` 下全部运行目录：
```bash
.venv/bin/python tools/rescue_dashboard.py --port 8765
```

---

## 四、扩展能力：真机 GeoTIFF DSM 导入与回归测试

### 1. 导入外部航测 DSM（WebODM / DJI Terra）
```bash
.venv/bin/python tools/import_real_dsm.py \
  --dsm <你的_DSM.tif> --output runs/real_dsm_prior
```
生成的 `runs/real_dsm_prior/uav_map.npz` 可直接作为 Gazebo 定点导航的 `--map-from` 输入（前提是输入坐标与运行场景采用同一本地 ENU；救援模式另需对应 `aerial_victims.json`）。

### 2. 自动化单元测试
```bash
# 仅跑 Gazebo 导出、ROS 2 适配与比赛工具回归测试（~2 s）
.venv/bin/python -m unittest \
  tests/test_gazebo_d_export.py \
  tests/test_gazebo_b_export.py \
  tests/test_gazebo_ros2_adapters.py \
  tests/search/test_competition_tools.py -v

# 全仓库完整测试（含 Webots 规划/感知/局规全部单测，105 项）
.venv/bin/python -m unittest discover -s tests -v
```

---

## 五、界面说明与常见问题排坑（FAQ）

1. **为什么右侧 `Rover Depth` 窗口是黑白的（下面发白、远处纯黑）？**
   这是**正常的深度图渲染**，不是摄像头故障：越近越亮（车前地面和近处伤员偏白）、中距离树干偏灰、超出量程（`maxRange = 12 m`）的天空返回 `inf` 显示纯黑；彩色画面在正上方的 `Rover RGB` 窗口，两者视角严格对齐。
2. **为什么场景里有 3 名伤员，只找到/投送 2 人就结束了？**
   无人机空中会把 3 人全部检出写入 `aerial_victims.json`，但无人车默认双仓载货 2 箱（`--max-victims 2`），投送完 2 箱即按 `MISSION_SUCCESS` 结束；需要尝试投送 3 人请传 `--max-victims 3`。
3. **报错 `ModuleNotFoundError: No module named 'rclpy'` 或 `gz.sim8`？**
   请检查两点：① 当前终端是否执行了 `source /opt/ros/jazzy/setup.bash`；② `.venv` 创建时是否带了 `--system-site-packages`（若漏加，删掉 `.venv` 后按第一节重建）。
4. **WSLg 下 Gazebo 窗口黑屏、闪退或卡顿？**
   - 请勿同时手动多开两个 D 场景 GUI（`run.sh` 已内置防多开锁 `.launch.lock`）；
   - 批量跑测或训练请加 `--headless` 跳过 GUI 渲染；
   - 并行隔离多个无头实例时请指定不同的 `--gz-partition <名称>`；
   - 若上次异常中断残留进程，可执行 `pkill -f "gz sim|parameter_bridge"` 清理后再重跑。
5. **话题映射与坐标系细节去哪看？**
   详见 [ROS 2 Jazzy 运行时与话题说明](ros2_rescue/README.md)。
