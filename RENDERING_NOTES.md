# 山洪灾后搜救渲染版

双击 launch_B.cmd 打开主场景。A、C 是地形与光照不同的对照场景。Webots R2025a、Python 3 为运行环境。

本版加入三处坍塌民房及 51 块散落建筑残骸，房屋、断梁、瓦楞屋顶和主要废墟使用可见几何参与碰撞，并登记到独立评测器。地面使用 4K 泥土、落叶混合贴图及凹凸、粗糙度贴图；树木改为带树皮和针叶纹理的模型。HDR 天空、方向光、接触阴影、薄雾与外围山体用于营造灾后环境。

B 场景冲沟内的浑水是静态视觉效果。外围山体用于画面延伸，任务边界仍由原来的 42×30 米评测区域定义。纯视觉策略的入口为 controllers/vision_agent/policy.py，原包提供的是静止预览策略。

preview 文件夹提供本机 Webots 实际导出的全景、房屋废墟近景和相机图。生成场景可运行 python tools/build_scenarios.py，需要 NumPy 和 Pillow。生成器使用包内素材，运行时无需下载素材。

素材来源和许可见 MATERIAL_SOURCES.md。

## 启动小车试跑

双击 start_rover_B.cmd 启动低速双目避障探索；普通 launch_B.cmd 用于静止场景预览。移动模式使用 OpenCV 立体匹配估算近处障碍与落差，按相机观测转向，最高试跑速度约 0.085 米/秒。需要在 Webots 使用的 Python 中执行 `python -m pip install -r requirements-drive.txt`；该依赖只用于运动演示。

此试跑策略用于验证车辆运动和近距离避障。当前策略没有伤员识别，也不会主动申报到达。点击 Webots 暂停按钮可以暂停行驶。
