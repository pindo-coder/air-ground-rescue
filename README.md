# 援来有路 · Gazebo Harmonic & Webots 空地协同应急救援系统

本分支已基于 Webots 最新 `main` 提交（`2721e96`）完整同步核心算法、微调航拍 YOLO11n 模型、四轮车/无人机工业级外观与全部回归测试，并提供 **Gazebo Sim Harmonic（8.15.0）+ ROS 2 Jazzy（`ros_gz_bridge`）** 完整迁移方案：

- **D 山洪森林场景（84 × 60 m）**：673 × 481 原始高程网格、冲沟、废墟、252 棵树、140 丛灌木、3 名平躺伤员与动态漂木/碎石障碍。
- **全链路核心算法**：无人机粗/细建图（0.5 m + 0.125 m `fine_ground_z_m`）、Traversing Mars 双路线搜索、DUCPP 价值导向斜拍补拍、VCTP 四轮精细通过性检验、IGHA\* 滑移转向混合规划、车载 LiDAR + RGB-D 闭环避障、PyTorch 多模态神经网络局部规划（BC + DAgger）、零泄漏 v2 全色域航拍 YOLO11n 检测与空地融合救援投送。

## 运行 D 场景（Gazebo Harmonic + ROS 2 Jazzy）

已有完整场景包时，进入 `gazebo/generated/D/` 运行 `./run.sh`。首次构建场景与启动 ROS 2 Jazzy 桥接及全链路救援请见：
- [Gazebo 构建与 ROS 2 迁移运行说明](gazebo/README.md)
- [ROS 2 Jazzy 节点与话题说明](gazebo/ros2_rescue/README.md)
- [Webots 航测/导航/局规技术文档](docs/UAV_NAVIGATION.md) · [神经网络局部规划说明](docs/LOCAL_POLICY.md) · [完整工程交接文档](AGENTS.md)

[素材来源与许可](MATERIAL_SOURCES.md)
