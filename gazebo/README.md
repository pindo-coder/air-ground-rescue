# Gazebo Harmonic + ROS 2 Jazzy · D 山洪森林场景完整迁移说明

本分支已同步 Webots `main` 至 `8fa1611`，面向 **Gazebo Sim Harmonic（8.15.0）+ ROS 2 Jazzy（`ros_gz_bridge`）** 提供以下能力。

- **四轮滑移转向底盘（4WD）**：完整导出 `RescueRoverBody` + 4 个 `RescueRoverWheel`（轴距 `0.300 m`、轮距 `0.410 m`、轮径 `0.090 m`、底盘离地 `0.090 m`），与 VCTP 四轮精细几何 100% 一致。
- **全量传感器补齐**：
  - 地面车：`/rover/rgb`、`/rover/depth`、`/rover/chase_cam`、`/rover/lidar`（+ `/rover/lidar/points`）、ENU `/rover/imu`、`/rover/gps`（NavSat）、`/rover/odom`、`/rover/joint_states`（4 轮）、`/rover/contact`（评测专用）。
  - 无人机：`/drone/rgb`、`/drone/depth`、`/drone/fpv_cam`、`/drone/gps`（NavSat，光心同位）、ENU `/drone/imu`。
- **全链路算法栈**：无人机 0.5 m / 0.125 m 粗细建图 → Traversing Mars 双路线搜索 → DUCPP 价值导向斜拍补拍（支持 6-DoF 斜拍站位）→ VCTP + IGHA\* 混合全局规划 → 10 Hz 车载 LiDAR/RGB-D 局部避障 → PyTorch 多模态神经网络局部规划（BC + DAgger）→ 零泄漏 v2 航拍 YOLO11n + 车端复核与物资箱投送。

新版本的 IGHA\* 可在起点姿态未通过 VCTP 时尝试邻近朝向与位置。双仓物资架作为静态外观导出到 SDF。独立评测输出速度、定位误差、安全裕度 `S_margin`、避障平滑度 `J_avoid` 和 RMS 加加速度。

---

## 一、WSL2 Ubuntu 24.04 环境准备

1. **安装 Gazebo Harmonic + ROS 2 Jazzy + `ros_gz_bridge`**（Ubuntu 24.04 LTS）：
```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-base \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-tf2-ros \
  python3-gz-sim8 python3-gz-math7 python3-numpy
```

2. **创建与系统 ROS 2 共存的 Python 虚拟环境**（关键：`--system-site-packages` 让 `.venv` 直接读取 `rclpy` 和 `gz.sim8`）：
```bash
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-build.txt -r requirements-search.txt -r requirements-search-yolo.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # RTX 4060 GPU（或 cpu）
.venv/bin/python -m pip install -r requirements-local-learning.txt
.venv/bin/python tools/fetch_search_model.py
```

---

## 二、构建并校验 D 场景 SDF 包

```bash
.venv/bin/python tools/build_search_worlds.py --case D
.venv/bin/python tools/export_gazebo_d.py
gz sdf -k gazebo/generated/D/search_d.sdf
```

> 注：如果目标机器未安装 Node.js，`tools/webots_static_scene.py` 会自动读取 `gazebo/proto_cache/` 中随仓库预生成的模板展开快照，零外部依赖即可导出完整工业级载具与 PBR 贴图。

---

## 三、全链路运行命令速查

运行前请先 `source /opt/ros/jazzy/setup.bash`：

```bash
# 1. 仅预览 Gazebo 场景（默认巡航 + WSLg 防多开锁）
./gazebo/generated/D/run.sh

# 2. 无人机全场航测 + DUCPP 价值导向斜拍补拍（生成 uav_map.npz + 0.125 m 细高程）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode mapping --profile full --refinement-strategy route_value

# 3. 定点闭环导航（VCTP + IGHA* / 可绕行障碍 detour / 封堵停车 block，复用已有航测图秒级启动）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode navigation --profile full --goal valley --planner igha_vctp --obstacle detour \
  --map-from runs/<你的航测目录>

# 4. 纯 YOLO11n 空地协同双伤员救援投送全闭环（支持 --record-cameras 留证）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
  --record-cameras

# 5. Gazebo 车端神经网络局部规划（BC + DAgger 采集与训练）
.venv/bin/python gazebo/train_gazebo_local_policy.py \
  --map-from runs/<你的航测目录> --profile full --goals valley --obstacles none detour --dagger-rounds 2 --headless

# 6. 使用训练好的神经网络局部规划跑全场救援闭环
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode rescue --profile full --detector yolo --planner igha_vctp \
  --local-policy neural --local-checkpoint runs/<训练目录>/dagger_r2_policy.pt \
  --map-from runs/<你的航测目录>
```

---

## 四、话题与权限隔离清单

详细 ROS 2 节点设计、坐标系转换与话题映射见 [ROS 2 Jazzy 运行时文档](ros2_rescue/README.md)。

- 小车与无人机的 PosePublisher 分别桥接至 `/eval/world_poses` 与 `/eval/drone_pose`，保留实体帧名和仿真时间戳。`evaluator_node.py` 使用小车位姿、车体与车轮接触真值独立评测。建图与车端规划节点只使用传感器输入。
- 本地回归测试：
```bash
python3 -m unittest discover -s tests -p 'test_gazebo_*.py' -v
```

## 五、地形导入与离线大屏

使用 `tools/import_real_dsm.py --dsm <DSM.tif> --output <目录>` 可将 GeoTIFF 地形生成 `uav_map.npz`。该先验可用于 Gazebo 定点导航的 `--map-from`，输入坐标需与运行场景采用同一本地 ENU。救援模式还需要对应的航拍伤员候选。

Gazebo 的 `evaluation/summary.json` 包含 `safety_smoothness`。离线大屏读取轨迹、规划路线、投送核验和局规统计，并可从历史轨迹的位置和时间补算速度。

```bash
.venv/bin/python tools/rescue_dashboard.py --run "runs/<Gazebo 运行目录>" \
  --export-static "runs/<Gazebo 运行目录>/dashboard_gazebo.html"
```
