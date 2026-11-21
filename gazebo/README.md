# B 山洪场景 · Gazebo Harmonic 静态环境

从仓库根目录执行以下命令。生成器需要 Python 3.9 或以上版本，输出目录可直接复制到装有 Gazebo Harmonic 的机器。

```bash
python3 tools/export_gazebo_b.py
gz sdf -k gazebo/generated/B-full/b_washout.sdf
gz sim -r gazebo/generated/B-full/b_washout.sdf
```

上述启动命令会直接打开 `b_washout`，3D 窗口默认从上方俯视地形。同时打开多个 Gazebo 实例时，可以给命令加环境变量 `GZ_PARTITION=bot-rescue-b`，区分场景各自的消息。

生成器读取已提交的 `worlds/B_washout.wbt` 和 `protos/Pine.proto`，导出地形、远景山体、残留水面、废墟、石块、植被、伤员和背包。地形覆盖 42 × 30 m，显示和碰撞共用保留了 337 × 241 个原始高程点的地表网格。伤员保持原来的模型和碰撞代理。水面、草叶、灌木叶片和远景山体参与显示，树干、茎和硬障碍沿用原碰撞体。

`B-full` 目录包含 SDF、网格和原始颜色、法线、粗糙度贴图。现场换机时复制整目录。Gazebo 中的天空色、雾和材质呈现以近似效果为目标，Webots 的 HDR 天光和雾参数目前只保留在原世界中。当前内容用于静态场景验证，小车、传感器、任务控制和评测属于后续阶段。

离线回归检查 `python3 -m unittest discover -s tests -p 'test_gazebo_b_export.py'`。

如果 VMware 的虚拟显卡无法初始化 Ogre 2，可在 Ubuntu 桌面终端用 `LIBGL_ALWAYS_SOFTWARE=1 gz sim -r gazebo/generated/B-full/b_washout.sdf` 运行。
