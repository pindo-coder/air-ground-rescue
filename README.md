# 援来有路 · 空地协同应急救援自主规划系统（Gazebo Harmonic & Webots 双平台）

本项目为《援来有路——一种面向应急救援场景的无人车自主探索导航系统》工程代码库。核心算法层完全跨仿真器共享，采用**单主干 `main` 双平台共存**架构，同时支持 **Gazebo Sim Harmonic + ROS 2 Jazzy（团队主开发与训练线）** 与 **Webots R2025a（并行验证与训练线）**。

- **D 山洪森林场景（`84 × 60 m`）**：`673 × 481` 原始高程网格、冲沟、废墟、252 棵树、140 丛灌木、3 名平躺伤员与动态漂木/碎石障碍。
- **共享核心算法栈**：无人机 0.5 m / 0.125 m 粗细建图 → Traversing Mars 双路线搜索 → DUCPP 价值导向斜拍补拍 → VCTP 四轮精细验路 + IGHA\*（含起步 GPS 抖动恢复）→ 10 Hz 车载 LiDAR / RGB-D 闭环避障与 PyTorch BC+DAgger 神经网络局规 → 零泄漏 v2 全色域航拍 YOLO11n 检测与空地融合投送 → 安全裕度 $S_{\text{margin}}$ / 避障平滑度 $J_{\text{avoid}}$ 量化评测与离线中文指挥大屏。

---

## 一、Gazebo Harmonic + ROS 2 Jazzy 快速复现（主开发线）

完整环境安装、权重自检、分步命令与排坑见 **[gazebo/README.md](gazebo/README.md)**。在 Ubuntu 24.04 / WSL2 上按以下 5 步即可跑通闭环：

```bash
# 1. 准备 ROS 2 Jazzy + 虚拟环境（首次）
source /opt/ros/jazzy/setup.bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-build.txt -r requirements-search.txt -r requirements-search-yolo.txt
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124   # 纯 CPU 改用 .../whl/cpu
.venv/bin/python -m pip install -r requirements-local-learning.txt
.venv/bin/python tools/fetch_search_model.py

# 2. 构建并校验 D 场景 SDF 包
.venv/bin/python tools/build_search_worlds.py --case D
.venv/bin/python tools/export_gazebo_d.py
gz sdf -k gazebo/generated/D/search_d.sdf

# 3. 无人机全场航测 + DUCPP 补拍（生成可复用的先验图）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode mapping --profile full --refinement-strategy route_value \
  --out-dir runs/gz_survey_D_full

# 4. 纯 YOLO11n 空地协同双伤员救援闭环（复用先验图秒级启动；无桌面加 --headless）
.venv/bin/python gazebo/run_gazebo_mission.py \
  --mode rescue --profile full --detector yolo --planner igha_vctp --max-victims 2 \
  --map-from runs/gz_survey_D_full --record-cameras --out-dir runs/gz_rescue_D_demo

# 5. 导出单文件离线中文指挥调度大屏
.venv/bin/python tools/rescue_dashboard.py \
  --run runs/gz_rescue_D_demo --export-static runs/gz_rescue_D_demo/dashboard_gazebo.html
```

---

## 二、Webots R2025a 快速复现（并行验证与训练线）

完整技术说明与参数表见 **[AGENTS.md](AGENTS.md)**、[docs/UAV_NAVIGATION.md](docs/UAV_NAVIGATION.md) 与 [docs/LOCAL_POLICY.md](docs/LOCAL_POLICY.md)。Windows 环境初始化与运行：

```powershell
# 1. 创建虚拟环境并补齐依赖与世界文件（首次）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-search.txt -r requirements-search-yolo.txt
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-local-learning.txt
.venv\Scripts\python.exe tools/fetch_search_model.py
.venv\Scripts\python.exe tools/build_search_worlds.py --case D

# 2. 运行航测 / 救援闭环（也可直接双击 start_uav_mapping.cmd / start_uav_rescue.cmd）
.venv\Scripts\python.exe tools/run_uav_rescue.py --profile full --detector yolo --planner igha_vctp --local-policy neural --checkpoint models/local_policy_dagger.pt --record-video
```

---

## 三、团队协作与分支规范（双仿真器共维护）

仓库采用 **单主干 `main` + 目录隔离 + 短命特性分支** 协作，不再长期分叉独立仿真器分支：

1. **目录分工边界**
   - **公共核心层（双方共用）**：`controllers/uav_mapper/`、`controllers/uav_map_rover/`、`controllers/search_agent/`、`protos/`、`config/`、`evaluation/`、`tools/safety_metrics.py`、`tools/rescue_dashboard.py`、`tools/import_real_dsm.py`、`tools/train_local_policy.py`
   - **Gazebo 专属层**：`gazebo/`、`tools/export_gazebo_*.py`、`tools/gazebo_d_robots.py`、`tools/webots_static_scene.py`、`tests/test_gazebo_*.py`
   - **Webots 专属层**：`controllers/uav_survey/`、`tools/run_uav_*.py`、`start_uav_*.cmd`、`docs/`
2. **分支与 PR 规则**
   - 不直接向 `main` 强推；按任务从 `main` 切短命分支（`feat/gz-*`、`feat/webots-*`、`feat/algo-*`、`fix/*`），1~3 天内通过 Pull Request 合并回 `main`；
   - GitHub Actions（[.github/workflows/tests.yml](.github/workflows/tests.yml)）会在每次向 `main` 提 PR 时自动执行**全量双端单元测试**与 **D 场景 Gazebo SDF 导出冒烟**，防止单边改动破坏另一仿真器。
3. **训练权重分文件命名（防互相覆盖）**
   - 航拍检测共用 `models/yolo11n_victim.pt`；
   - 神经局部规划权重按仿真器区分：`models/local_policy_dagger.pt`（Webots）与 `models/local_policy_gazebo_dagger.pt`（Gazebo）。
4. **提交前本地回归检查**
   ```bash
   python3 -m unittest discover -s tests -v
   ```

---

## 文档导航

- **[Gazebo 完整复现手册](gazebo/README.md)** · **[ROS 2 Jazzy 运行时与话题说明](gazebo/ros2_rescue/README.md)**
- [Webots 航测 / 导航文档](docs/UAV_NAVIGATION.md) · [神经网络局部规划说明](docs/LOCAL_POLICY.md) · [工程技术交接文档](AGENTS.md)
- [团队协作说明](TEAM_GUIDE.md) · [素材来源与许可](MATERIAL_SOURCES.md)
