# 援来有路 · Gazebo Harmonic & Webots 空地协同应急救援系统

本分支（`feat/gazebo-models`）已同步 Webots `main` 至 `8fa1611`，完整保留核心算法、零泄漏 v2 全色域航拍 YOLO11n 模型、BC+DAgger 神经局部规划权重、工业级载具外观（含后甲板双仓物资架）与全套回归测试，并提供 **Gazebo Sim Harmonic（8.x）+ ROS 2 Jazzy（`ros_gz_bridge`）** 端到端仿真运行与离线指挥大屏支持。

- **D 山洪森林场景（`84 × 60 m`）**：`673 × 481` 原始高程网格、冲沟、废墟、252 棵树、140 丛灌木、3 名平躺伤员与动态漂木/碎石障碍。
- **全链路核心能力**：无人机 0.5 m / 0.125 m 粗细建图 → Traversing Mars 双路线搜索 → DUCPP 价值导向斜拍补拍 → VCTP 四轮精细验路 + IGHA\*（含起步 GPS 抖动恢复）→ 10 Hz 车载 LiDAR / RGB-D 闭环避障与神经网络局规 → 空地跨视角 YOLO11n 复核与物资投送 → 安全裕度 $S_{\text{margin}}$ / 避障平滑度 $J_{\text{avoid}}$ 量化评测与离线中文指挥大屏。

---

## 快速复现路径（Ubuntu 24.04 / WSL2）

详细环境安装、参数说明与排坑见 [gazebo/README.md](gazebo/README.md)。标准复现按以下 5 步顺序执行：

```bash
# 1. 准备 ROS 2 Jazzy + 虚拟环境（首次）
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-build.txt -r requirements-search.txt -r requirements-search-yolo.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # 无独显改为 .../whl/cpu
.venv/bin/python -m pip install -r requirements-local-learning.txt
.venv/bin/python tools/fetch_search_model.py

# 2. 构建并校验 D 场景 SDF 包（每次改动 PROTO/场景后需重跑）
.venv/bin/python tools/build_search_worlds.py --case D
.venv/bin/python tools/export_gazebo_d.py
gz sdf -k gazebo/generated/D/search_d.sdf

# 3. 无人机全场航测 + DUCPP 补拍（生成可复用的先验地图）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode mapping --profile full --refinement-strategy route_value \
  --out-dir runs/gz_survey_D_full

# 4. 纯 YOLO11n 空地协同双伤员救援闭环（复用上一步先验图秒级开跑；无界面加 --headless）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
  --map-from runs/gz_survey_D_full --record-cameras --out-dir runs/gz_rescue_D_demo

# 5. 导出单文件离线中文指挥调度大屏
.venv/bin/python tools/rescue_dashboard.py \
  --run runs/gz_rescue_D_demo --export-static runs/gz_rescue_D_demo/dashboard_gazebo.html
```

---

## 文档导航

- **[Gazebo 完整复现手册（环境、分步命令、验收标准与 FAQ）](gazebo/README.md)**
- [ROS 2 Jazzy 节点职责、话题映射与坐标系说明](gazebo/ros2_rescue/README.md)
- [Webots 航测 / 导航 / 局规技术文档](docs/UAV_NAVIGATION.md) · [神经网络局部规划说明](docs/LOCAL_POLICY.md) · [工程交接总览](AGENTS.md)
- [素材来源与许可](MATERIAL_SOURCES.md)
