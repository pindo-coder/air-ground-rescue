# Gazebo Harmonic · D 山洪森林场景

本分支从 Webots [`2a9eab7`](https://github.com/pindo-coder/webots-vision-rescue/commit/2a9eab7a815b9dc7b467b738b4a1767d92f7664d) 迁移 D 场景。迁移范围包括森林、冲沟、残留水面、废墟、伤员、移动漂木与碎石、小车和无人机。84 × 60 米地表保留 673 × 481 个原始高程点，显示与碰撞网格采用相同的高程和三角剖分。

小车沿用该版本的差速底盘，支持速度控制。无人机在世界坐标高度 16 米沿原航线运动。光照、接触动力学和传感器噪声采用 Gazebo 中的近似配置。最新 Webots 四轮车、无人机外观和自主搜索、航测建图、空地融合及任务评测尚待迁移。相关算法文档见 [Webots 导航与救援说明](https://github.com/pindo-coder/webots-vision-rescue/blob/main/docs/UAV_NAVIGATION.md)。

## 构建

在 `feat/gazebo-models` 分支的仓库根目录，用 Python 安装构建依赖并生成场景。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-build.txt
.venv/bin/python tools/build_search_worlds.py --case D
.venv/bin/python tools/export_gazebo_d.py
```

生成的 `gazebo/generated/D/` 包含 `search_d.sdf`、贴图、网格和运动脚本。换机器运行时复制整个目录。

## 启动

运行端使用 Gazebo Harmonic，已验证版本为 **Gazebo Sim 8.15.0**。运动脚本还依赖 Ubuntu 软件包 `python3-gz-sim8`、`python3-gz-math7` 和 `python3-numpy`。在完整场景包目录中执行以下命令。

```bash
gz sdf -k search_d.sdf
./run.sh
```

WSL 通过 WSLg 打开图形窗口。启动脚本会设置运动模块路径和 WSLg 显示环境。仅运行服务端时用 `./run.sh -s`。

## 视角与控制

Gazebo 界面包含 3D 斜视全景、小车前视和无人机下视。图像面板可从话题列表选择 RGB 或深度，深度以灰度显示距离。小车两个相机位置和朝向相同。

| 平台 | 彩色话题 | 深度话题 | 分辨率 |
| --- | --- | --- | --- |
| 小车前视 | `/rover/rgb` | `/rover/depth` | 320 × 240 |
| 无人机下视 | `/drone/rgb` | `/drone/depth` | 640 × 480 |

小车接收 Gazebo Transport 的 `/rover/cmd_vel`，并发布 `/rover/odom`、`/rover/joint_states`、`/rover/lidar` 和 `/rover/imu`。在另一个终端发送前进指令，再发送零速度停车。

```bash
gz topic -t /rover/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.25}'
gz topic -t /rover/cmd_vel -m gz.msgs.Twist -p 'linear: {x: 0.0}'
```

同机运行多个 Gazebo 世界时，为 D 场景启动终端和读取话题的终端设置相同的 `GZ_PARTITION`，避免话题串到其他实例。WSLg 遇到 `D3D12: Removing Device` 时，先退出同一场景的其他 Gazebo 窗口和残留服务端，再重新启动。

## 验证

在仓库根目录运行 `python3 -m unittest discover -s tests -p 'test_gazebo_*_export.py'`。在 WSLg 上已验证四路 RGB 与深度话题出帧、小车响应速度指令、无人机沿航线移动和两种动态障碍的轨迹。
