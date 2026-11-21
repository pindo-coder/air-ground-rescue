# `gazebo/ros2_rescue` · ROS 2 Jazzy + `ros_gz_bridge` 运行时说明

本目录实现 Gazebo Sim Harmonic（8.15.0）与项目核心算法库之间的标准 ROS 2 Jazzy 桥接与节点编排。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `bridge_config.yaml` | `ros_gz_bridge` (`parameter_bridge`) 完整话题映射，含 `/clock`、车端与无人机全部传感器/控制话题，以及评测隔离话题 `/eval/*` |
| `launch_bridge.sh` | 自动加载 `/opt/ros/jazzy/setup.bash` 并启动 `parameter_bridge`（启用 `use_sim_time:=true`） |
| `sensor_adapters.py` | 纯 Python 传感器与坐标转换层（WGS84 `NavSatToEnu` + 0.04 m 相关噪声、`RGB_INT8`/`32FC1`/`16UC1` 解码、`PointCloud2` 解析、ENU 四元数↔RPY、4 轮编码器均值、静态 TF 表） |
| `uav_mapper_node.py` | ROS 2 航测建图节点：订阅 `/drone/{rgb,depth,gps,imu,fpv_cam}`，调用 `survey_map`（0.5 m + 0.125 m 细地面）、DUCPP `schedule_oblique`（输出 `oblique_survey.json` 触发 6-DoF 斜拍）与 YOLO11n v2 `AerialVictimTracker` |
| `rover_nav_node.py` | ROS 2 无人车节点：发布静态 TF 树，调用 Traversing Mars `plan_dual`、VCTP + IGHA\*、`RouteNavigationController`（规则或 `LocalPolicyExperiment` BC/DAgger 神经网络）、车端 YOLO 近距复核与物资投送事件，以 10 Hz 发布 `/rover/cmd_vel` |
| `evaluator_node.py` | 独立真值评测与场景注入节点：仅订阅 `/eval/world_poses` 与 `/eval/rover_contact`，通过 `gz service` 动态生成 `detour`/`block` 障碍与橙色物资箱，调用 `evaluation/search_metrics.py` 输出 Webots 同格式 `evaluation/summary.json` 与 `trajectory.csv` |

## 坐标系与关键适配点

1. **世界系 & 车体系**：Gazebo `search_d.sdf` 使用 `EARTH_WGS84` 本地 ENU（`X` 东、`Y` 北、`Z` 上）与车体 FLU（`+X` 前、`+Y` 左、`+Z` 上），与 Webots 原算法完全一致。
2. **LiDAR 方向与层序**：Gazebo `gpu_lidar` 仰角按升序 `-0.075, 0, +0.075 rad`、方位角逆时针 `-π..+π`。`rover_nav_node.py` 优先直接消费 `/rover/lidar/points`（`PointCloud2` 3D 传感器 FLU 坐标），避免角度方向歧义；若降级为 `LaserScan`，`ros_laserscan_to_layers()` 会自动传递 `azimuth_order="ccw"` 给 `extract_lidar_obstacles()`。
3. **NavSat 噪声避坑**：为规避 `gz-sensors` 水平噪声度/米单位不一致问题（Issue #325），SDF 保持干净 NavSat 输出，由 `NavSatToEnu` 施加确定性的 `0.04 m`（`noiseCorrelation=0.2`）本地 ENU 高斯相关噪声，与 Webots `GPS` 精度完全一致。
4. **RGB 通道兼容**：`controllers/uav_map_rover/policy_features.py` 的 `encode_rgbd()` 已扩展支持 ROS 2 三通道 `rgb8`/`bgr8` 与 Webots 四通道 `bgra`（默认向后兼容 Webots）。
