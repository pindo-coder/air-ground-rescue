# 援来有路 · Gazebo D 救援场景

本分支提供 D 山洪森林场景的 Gazebo Harmonic 版本。场景包含 84 × 60 米起伏地形、冲沟、废墟、252 棵树、140 丛新增灌木、伤员和两种移动障碍。可控制地面小车行驶，观察无人机沿预设航线飞行，并读取两个平台的 RGB 与深度图像。

迁移基于 Webots 提交 [`2a9eab7`](https://github.com/pindo-coder/webots-vision-rescue/commit/2a9eab7a815b9dc7b467b738b4a1767d92f7664d)，已在 Gazebo Sim **8.15.0（Harmonic）** 上验证。最新 Webots 四轮车、无人机外观与自主救援算法见 [`main` 分支](https://github.com/pindo-coder/webots-vision-rescue/tree/main)。

## 运行 D 场景

已有完整场景包时，进入包目录运行 `./run.sh`。首次使用请按 [Gazebo 构建和使用说明](gazebo/README.md) 生成场景包。说明包含相机视角、消息话题、小车控制和 WSLg 启动排查。

[素材来源与许可](MATERIAL_SOURCES.md)
