# 单车自主搜索、主动确认与物资投送

本模块把两项研究原型接入当前仓库：有效人员报告导向的搜索/确认/投送滚动调度，以及定位不确定性约束的主动多视角确认。单车、LiDAR＋RGB-D、编码器/IMU、载荷2份等都是暂定研究条件。没有证明性能提升，也不宣称完成真实救援能力。

D 场景扩展为单台无人车加一架 RGB-D 观测无人机，额外使用带噪 GPS 完成坐标对齐，并加入实体沟壑、密集植被和移动障碍。该扩展不改变原 A/B/C 单车实验条件；假设、融合边界和运行方式见 [COMPLEX_DISASTER_SCENE.md](COMPLEX_DISASTER_SCENE.md)。

## 启动

已有双目纯视觉场景、控制器和start_rover_B.cmd继续作为原版演示。新模块使用独立ResearchRover、search_agent和mission_referee；原VisionRover保持两台RGB相机。A/B/C派生世界保留原地形、建筑、人物位置与外观，不把原来的方向扇区提示传入搜索控制器。

推荐新建环境（Python 3.12或本机已验证的3.14；其他版本需验证依赖轮子）：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --index-url https://pypi.org/simple -r requirements-search.txt
.venv\Scripts\python tools/run_search.py --case smoke --detector surrogate_rgbd --duration 45 --visible
```

这会使用已有的人物网格，在**独立平地夹具**中验证数据链路。surrogate_rgbd为颜色/尺寸候选器，不是真实人员检测器，不用于正式伤员识别成绩。平地夹具不属于原来的山洪地图或正式测试布局。

接入真实person检测接口：

```powershell
.venv\Scripts\python -m pip install --index-url https://pypi.org/simple -r requirements-search-yolo.txt
.venv\Scripts\python tools/fetch_search_model.py
.venv\Scripts\python tools/run_search.py --case B --detector yolo --duration 600 --visible --realtime --keep-open
```

Windows双击入口：start_search_demo.cmd为平地受控流程；start_search_B.cmd为原B山洪地图上的YOLO搜索。启动器优先使用search.local.cmd中的SEARCH_PYTHON，其次是本仓库.venv，最后是python。可复制search.local.cmd.example配置本机解释器。运行工具自动为两个新增控制器配置被Git忽略的runtime.ini，不更改Webots全局Python设置。

本机测试使用额外的webots-vision-rescue-search-env，并复用系统已有的Torch等大库；队友可按上述命令使用全新独立环境。依赖镜像若找不到Pillow 13.0.0，可使用官方PyPI，不需要降低仓库版本。

## 代码与数据流

```text
传感器 → 轮速/IMU/扫描匹配 → 在线占据图与局部净空
RGB-D → 候选、姿态补偿投影 → 一对一轨迹关联与CI融合
     → 观测分组/报告门限 → 主动观察候选
     → 搜索/确认/投送/补给/返回调度 → 局部速度采样
     → 新观测；合格报告入队；投送前用近期相对观测复核
独立Supervisor → 真值评价、有效投送与失败日志（不回传真假判断）
```

| 文件 | 职责 |
|---|---|
| controllers/search_agent/mapping.py | 轮速、IMU及局部扫描匹配；占据图、前沿、2m滚动局部窗口和速度采样 |
| controllers/search_agent/perception.py | person接口/受控候选；RGB-D姿态补偿；一对一关联、CI、重复报告抑制与主动位姿 |
| controllers/search_agent/mission.py | 两步滚动调度、基线开关、载荷和返回预算、近距离投送复核 |
| controllers/search_agent/search_agent.py | Robot设备适配、闭环执行、证据图与估计状态日志 |
| controllers/mission_referee/mission_referee.py | 独立Supervisor评价，真实位置只在该侧使用 |
| evaluation/search_metrics.py | 一对一有效报告计分、最早报告保留、物理投送检查 |
| tools/build_search_worlds.py | 离线生成派生世界；不作为Robot依赖 |
| tools/run_search.py | 单次运行、模型/素材/代码/依赖哈希、日志、失败检查 |
| tools/run_search_matrix.py / analyze_search_runs.py | 串行配对运行与原始指标CSV，保留失败与删失 |

生成的worlds/search_*.wbt、公开任务和私有评价JSON被Git忽略；启动时由当前版本生成，不重复提交大型地图。相机纹理和人物网格从仓库原有assets、worlds/textures引用。原worlds/A_trail.wbt等不是生成器的输出目标。

## 两项算法的实现范围

调度器枚举两步动作序列，先比较已报告投送任务的预计超时，再比较人员相关探索/确认收益、耗时及距离预算。确认、投送、补给均为显式动作；有期限、载荷、返回储备和重复区域折减。当前行程时间和搜索收益仍是启发式，研究方案中的P_pass和时间机会约束尚未标定，不能称为已实现精确随机优化。

主动模块生成目标周围多个位置和朝向，筛除净空不足、估计遮挡和无法取得新观察组的位姿，再比较可见性、保守定位收益与移动代价。候选位姿额外保留到达误差裕量，避免在观测独立性阈值上停车。执行侧只使用局部窗口和速度采样，**无全局路径搜索**。

RGB-D投影补偿相机安装俯角和IMU横滚/俯仰；平面定位协方差包含深度、像素、车辆位姿和暂定偏差项。多帧采用CI，轨迹关联使用一对一匹配；重复帧不增加观测组。报告暂定需要2个几何观察组、启发式存在性信念≥0.90、95%自估半径≤0.80m；这些并非校准概率。不同位置的真实人员过近时，基于距离的去重仍可能合并目标，需后续外观关联实验。

投送不能仅凭抵达地图停车点完成：还需要2秒内重新观测到同一目标，以相对RGB-D位置进行近距离接近与停稳。独立评价器进一步检查有效报告、真实车到目标距离≤1.8m、停稳≥1.5s和剩余物资。错误投送也消耗库存；补给须实际回到出发点停稳。评价结果不返回控制器。

## 可选检测器的真实限制

YOLO11n权重来自上游，模型来源、SHA256和许可见models/README.md。集成对4种图像方向做检测后合并框，支持横卧目标的部分视角；它不是新训练的救援模型。开发探测中A/C人物细节图在某些方向可检出，B细节图仍漏检，不能由此推断移动相机检测准确率。

检测器缺包或缺模型会明确报错，不会悄悄切换成颜色候选。每个方法对比必须显式使用相同detector。正式实验前需标注多姿态、遮挡和泥污人物，按布局拆分数据、标定并冻结模型；不能把surrogate成绩当人员搜索成绩。

## 基线、验证和指标

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python tools/run_search_matrix.py --cases A B C --methods frontier active_only schedule_only full weighted --detector yolo --duration 600
.venv\Scripts\python tools/analyze_search_runs.py
```

frontier=最近前沿＋固定侧移确认＋FIFO投送；active_only只开主动观测；schedule_only只开滚动调度；full两者均开；weighted为简单加权基线。no_prior、no_covariance、no_view_groups为单独消融。主比较共享报告判据、定位前端、传感器、速度和载荷；只有明确的消融改变对应条件。原纯视觉演示的传感器不同，不作为这个2×2比较中的同条件基线。

输出位于runs/search_*/：agent保存估计状态、候选、报告图、动作、占据图；evaluation保存真实轨迹、有效/无效报告和投送、覆盖人数、误报、漏报、定位误差及置信区域覆盖率、投送时间和任务成功。无报告时首次报告时间为null并标记删失；无报告的FDR为null，不填0。任务成功暂定为全体人员有效报告并按期投送，且无碰撞、翻覆或越界。

A/B/C现有实例只够功能验证；正式结论还需独立布局集和配对统计，不能只换噪声种子。所有运行manifest固定素材、世界、传感PROTO、控制代码、模型和依赖版本；不同配置预检不能混成算法对比表。

## 已知边界

局部里程计没有闭环SLAM，长距离会漂移；协方差和人员信念未标定；局部控制可能陷入死胡同；积水静态，不模拟水动力/陷车；投送是停稳事件和盒子显示，无机械交接。完整研究设计与待验证假设见SEARCH_RESEARCH_PROTOCOL.md。运行成功和单项投送成功都不等于任务全部成功。
