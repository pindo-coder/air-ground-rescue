# D：复杂山洪场景与空地多视角验证

本场景在原 B 山洪世界基础上独立生成 `worlds/search_D.wbt`，已扩展为 **84 × 60 m 的物理可探索区域**，面积为原来的四倍。高程网格为 673 × 481，间距 0.125 m；在线地图范围和评测边界同步扩展。D 的配置入口是 `config/scenes/D.json`。本版本用于场景与传感器链路验证，不预设搜索、定位或投送性能提升。

## 实际增加的场景要素

| 要素 | 实现与检查方式 |
|---|---|
| 沟壑 | 新增约 13 m 长、1.8 m 宽、最大下切 1.1 m 的沟槽；修改真实 ElevationGrid，可见地面和碰撞使用同一网格。原 B 的冲沟继续保留。沟壑本身是静态危险地形。 |
| 移动障碍 | 倒木代理和漂移废墟两个 Solid，速度分别为 0.28、0.22 m/s，往返穿越局部通路，带实际碰撞盒；环境控制器记录每秒实际位置。不是画面叠加动画。 |
| 起伏地形 | 六处山丘/土丘、浅谷和原有沟壑，全域高程约 -0.35～7.03 m；另有约 9 × 6 m、幅值 0.055 m 的细小波状起伏。可见地面和碰撞同步更新。 |
| 密集植被 | 22 棵原树替换为有纹理的阔叶树/针叶树，再增加 230 棵，共 252 棵树；另加 140 丛多枝叶片灌木。树冠、低枝和灌木叶片影响真实 RGB/深度画面；树干和灌木茎有碰撞体，叶片不逐片建碰撞。没有用绿色球体作为新增植被外观。 |
| 灾后元素 | 继续使用原场景坍塌房屋、建筑废墟、倒木、泥土地表和外围山地；新增浅色水面仅作沟底外观。 |
| 多目标 | D 共 3 名静止模拟伤员；真实位置保存在私有评测配置中，不传给搜索控制器。 |
| 空中视角 | 一架带 RGB-D、GPS、IMU 和无线发送器的观察平台，绝对高度 16 m，测线覆盖扩展区域；真实拍摄 640 × 480 图像，每秒处理并保存一帧。 |

动态障碍当前用脚本驱动的运动学物体，尚未模拟洪水冲力、漂浮、摩擦变化或泥石流动力学。沟底水面不代表流体仿真。无人机沿固定往返测线运动，转角瞬间改变航向，尚未实现多旋翼动力学、飞控或自主航线规划。它不根据目标真值选择观察位置。

## 暂定传感器与坐标条件

原研究条件为单台无人车、LiDAR＋RGB-D、编码器和 IMU。D **额外增加一架观测无人机及其 RGB-D/GPS/IMU**，并给车端增加启动位置 GPS 和 Receiver；属于需要后续确认的扩展条件。GPS 噪声标准差暂定 0.04 m，姿态误差预算暂定 0.006 rad，不代表真实设备已达到该精度。

地面车仅在启动时使用测得的 GPS 与 IMU 朝向建立公共 ENU 到本地部署坐标的变换；运动仍依赖现有编码器、IMU 和激光扫描匹配，没有使用 Supervisor 实时车位姿。无人机 GPS 安装在相机光心，俯视相机安装旋转为 Ry(π/2)。深度使用光轴深度，不能当作斜距直接乘单位射线。

图像像素 `(u,v)` 与深度 `d` 在相机 FLU 坐标中投影为：

```text
f = width / (2 tan(FOV / 2))
p_camera = [d, -(u-cx)d/f, -(v-cy)d/f]
p_ENU = gps_camera + R_IMU · Ry(pi/2) · p_camera
p_local_xy = R(-initial_heading) · (p_ENU_xy - measured_deployment_GPS_xy)
```

投影协方差包含像素、深度、定位、航向和暂定系统偏差。它是研究原型的误差模型，需要通过标注图像和定位误差实验标定，不能将自估置信区间当作实际准确率。

## 融合与证据流

```text
车载 RGB-D → person 候选与本地位置 ─┐
                                  ├→ 一对一关联 → CI 位置融合 → 观测组 → 报告
无人机 RGB-D → 图像投影 → 无线消息 ─┘                     ↓
                                                主动确认/搜索/物资投送
独立评测器 → 真值计分与日志，不向策略回传目标坐标
```

`aerial_observer` 只能访问普通 Robot 设备；发送带时间戳、序号、相机测量位姿、候选位置/协方差、图像证据路径的包。`AerialInbox` 拒绝重放、超过 2.5 s 的旧消息、异常坐标/协方差/图像框，转换到地面车的部署坐标后进入已有的 Mahalanobis 门控与一对一关联。关联还有 0.9 m 的欧氏距离门限；近距离多人的关联与长期里程计漂移仍是限制。

跨源相关性未知，继续采用协方差交集 CI，不假设两路观测完全独立。同一观察位置的重复帧不能无限增加确认次数；跨视角分组检查基线和夹角。当前目标状态仍是二维，空地观测分组采用近似局部地面高度；大高差场景需进一步做三维地形补偿。

两路均使用同一选定检测器与对应分数门限。默认报告仍要求至少两个几何观测组、存在性信念及定位半径门限；融合开关不改变这些门限。日志额外记录 `sources`、各来源图像以及 `valid_cross_view_reports`，区分单源报告与空地共同支持的有效报告。投送必须有 2 秒内的**车载**目标观测，无人机消息不能刷新车载近距离复核时间。

已有 CI、时间同步、坐标变换和来源管理属于工程实现，不单独声称为算法创新。没有添加全局路径规划，地面车继续使用已有局部避障控制器。动态障碍可能造成局部停滞或碰撞，不能把场景正常加载当作避障成功。

## 启动、保存与复核

Windows 双击 `start_search_D.cmd`，使用本项目虚拟环境启动可见实时仿真，默认 600 个仿真秒结束后保留窗口。启动时自动生成 D 世界和私有评测数据，无需手动把 world 复制到其他工程。

```powershell
.venv\Scripts\python tools/run_search.py --case D --detector yolo --duration 600 --visible --realtime --keep-open
```

快速排查链路可以显式使用颜色/尺寸候选器；这不是通用人员检测模型，不能用于正式识别成绩：

```powershell
.venv\Scripts\python tools/run_search.py --case D --detector surrogate_rgbd --duration 55
```

每轮 `runs/search_D_.../` 包含：

- `aerial/frames.jsonl` 与 JPG：实际无人机 RGB 图像、时间、测量位姿、图像产生的候选。
- `agent/frames.jsonl` 与 JPG：每 2 秒的实际车载图像。
- `agent/fusion.jsonl`：无线接收计数、候选轨迹、协方差、观测组和来源图像。
- `agent/reports.jsonl`：正式报告及各来源证据；报告坐标为部署局部坐标。
- `environment/motion.jsonl`：独立环境侧记录的障碍和无人机实际位置，策略不读取。
- `evaluation/summary.json`：有效报告、漏报、误报、定位误差、物资时间、成功率等独立评价。
- `manifest.json`：模型、场景、控制器、配置、依赖哈希及融合/动态开关。

对一轮日志生成验收记录与同步双视角图：

```powershell
.venv\Scripts\python tools/validate_complex_scene.py runs/本轮目录名
```

输出位于该轮的 `validation/`。`infrastructure_pass` 只表明模型/控制器无已知错误、实际运动、双路图像和无线接收成立。`cross_view_track_ids` 表明关联到双方观测；`valid_cross_view_reports` 还必须通过独立真值匹配。两者都不能替代完整任务成功率。

## 配对对照

保持 D 地形、人物、物资、无人机路径、相机参数、检测模型、运行时长一致，仅切换实验因素：

```powershell
# 同一传感器和场景，启用融合
.venv\Scripts\python tools/run_search.py --case D --detector yolo --duration 600 --fusion on --dynamic on
# 同一场景，两路相机仍运行，但车端忽略空中候选
.venv\Scripts\python tools/run_search.py --case D --detector yolo --duration 600 --fusion off --dynamic on
# 障碍保留同样形状和初始相位，仅冻结障碍运动；无人机照常巡航
.venv\Scripts\python tools/run_search.py --case D --detector yolo --duration 600 --fusion on --dynamic off
```

至少统计首次有效报告时间、固定时长内报告人数、漏报/误报、定位误差、物资到达时间、碰撞与任务成功。目标真值只用于评价。对空地融合另统计双源有效报告数、无效/过期包、配对图像时间差。当前只有一个 D 布局，正式论文实验还需要多组独立布局、遮挡程度、速度与噪声条件；不能从一次预检推断性能提升。

## API 依据

场景渲染和固定机位遮挡检查见 [FOREST_SCENE.md](FOREST_SCENE.md)。

投影与传感器实现依据 Webots R2025a 的 [Camera](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/camera.md)、[RangeFinder](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/rangefinder.md)、[GPS](https://github.com/cyberbotics/webots/blob/R2025a/docs/reference/gps.md) 文档。环境运动遵循 [Supervisor](https://github.com/cyberbotics/webots/blob/R2025a/docs/guide/supervisor-programming.md) 的场景字段更新机制；环境日志读取的实际位置可能比本步写入的目标位置晚一个仿真步。
