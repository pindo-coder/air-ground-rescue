# D 山洪森林场景 · Gazebo Harmonic

在 Gazebo Sim 8.15.0 上验证。场景源自 Webots `main` 的 `2a9eab7`，包含 84 × 60 m 地形、252 棵树、140 丛新场景灌木、冲沟、废墟、伤员、小车和无人机。地表网格保留 673 × 481 个原始高程点，碰撞网格使用相同的高程和三角剖分。

## 打开场景

在 WSL 的 Ubuntu 终端执行：

```bash
cd ~/bot-rescue/D
./run.sh
```

窗口打开后可同时看 3D 斜视全景、**Rover RGB** 小车前视画面和 **Drone RGB** 无人机俯视画面。相机画面随仿真运行更新。面板里的刷新按钮会列出可用图像话题，可以切换到对应的 `depth`。无人机沿原航线在世界坐标高度 16 m 飞行，漂木和碎石沿原轨迹移动。

| 视角 | 彩色话题 | 深度话题 | 分辨率 |
| --- | --- | --- | --- |
| 小车前视 | `/rover/rgb` | `/rover/depth` | 320 × 240 |
| 无人机下视 | `/drone/rgb` | `/drone/depth` | 640 × 480 |

小车通过 Gazebo Transport 接收 `/rover/cmd_vel` 速度指令，并发布 `/rover/odom`、`/rover/joint_states`、`/rover/lidar`、`/rover/imu`。

```bash
gz topic -t /rover/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.25}'
gz topic -t /rover/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.0}'
```

小车惯量、光照、天空和传感器噪声按 Gazebo 的物理与渲染模型近似。无人机使用预设航线。Webots 搜索控制器、空地融合、GPS 与电台和任务评测尚待接入 Gazebo。

## 从仓库复现

在仓库根目录安装 `requirements-build.txt` 中的构建依赖并执行：

```bash
python3 tools/build_search_worlds.py --case D
python3 tools/export_gazebo_d.py
```

把整个 `gazebo/generated/D` 目录复制到运行机器。运行端需要 Gazebo Harmonic、`python3-gz-sim8`、`python3-gz-math7` 和 `python3-numpy`。在输出目录用 `gz sdf -k search_d.sdf` 校验 SDF，用 `./run.sh -s` 只启动服务端。启动脚本会配置无人机和移动障碍的运动模块路径。多个仿真实例同时运行时，可在启动和读取话题的终端设置相同的 `GZ_PARTITION`。
