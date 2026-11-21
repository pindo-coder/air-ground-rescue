# Webots 局部规划模仿学习

四轮车使用现有无人机先验通行图和全局路径。局部规划以 10 Hz 从车载感知输出线速度 \(v\) 与角速度 \(\omega\)。实验接在 D 山洪场景的定点导航入口，用于隔离检验局部规划器。

## 网络与采集

| 分支 | 输入 | 编码 |
| --- | --- | --- |
| 视觉 | 对齐的 RGB、深度与深度有效掩码，尺寸 5×24×32 | 两层轻量卷积 |
| 近场障碍 | LiDAR 与 RGB-D 分别提取的障碍距离及命中掩码，各 36 个车体角扇区 | 全连接层 |
| 导航状态 | 车体系前瞻路点、目标距离、轮速估计、上一条控制指令、横滚与俯仰、11 个先验与车载叠加地图的净空查询 | 全连接层 |

三个分支融合后输出 \((v,\omega)\) 和停止、原地转向、行驶三种动作模式。模型共 18,949 个参数。训练对归一化动作使用 Smooth L1 损失，对模式使用类别加权交叉熵，权重乘数为 0.35。两个动作量归一化的分母分别为 0.42 m/s 和 0.85 rad/s。

BC 由现有速度采样规划器运行并标注。DAgger 中同一车载状态同时交给网络和规则专家。以概率 \(\beta\) 执行专家动作，其余拍优先执行网络动作。网络动作先经过共用姿态、车速和 1.2 秒净空预测检查，触发保护时回退到规则专家。每轮把新轨迹和既有训练集聚合后，再从上一份权重微调。到点停稳、局部重规划和卡滞停车均由共用导航状态机负责。

每个采集目录中的 `rover/imitation/manifest.json` 列出样本数、数据文件与动作来源计数。分片在同一行保存传感器特征、专家标签 `expert_vw`、网络建议 `proposal_vw`、局部模块所选动作 `selected_vw`、经外层状态机处理后发给电机的 `executed_vw`，以及时间戳和来源。控制器旁路局部规划的拍不进入训练集。用整条轨迹划分训练与验证，并检查两者的地图、目标和场景组合没有重叠。DAgger 训练入口检查聚合了旧数据，并核实至少有一条来自当前权重且执行过非零网络动作的回采轨迹。

方法来源见 [Ross 等人的 DAgger 论文](https://proceedings.mlr.press/v15/ross11a.html)。模型按照 PyTorch 的 `state_dict` 格式保存，仅以权重加载模式在 CPU 上读入。[PyTorch 保存与加载模型](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

## 从工程根目录运行

本机环境为 Webots R2025a 和项目 `.venv`。新环境可先安装可选训练依赖。需要固定 CPU 版时，从 PyTorch 的 CPU 软件源安装 Torch，然后再安装其余依赖。

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-local-learning.txt
```

使用已有航测目录中的 `uav_map.npz`。以下例子使用 D 场景局部航测图。每次 Webots 结束后，控制台的 `Run` 行给出实际目录，把对应值填入下一个训练命令。训练数据和证据默认写入 `runs/`。

```powershell
$py = '.\.venv\Scripts\python.exe'
$map = 'runs/<已有局部航测目录>'
$out = 'runs/local_learning'
& $py tools/run_uav_navigation.py --map-from $map --record-expert --obstacle none --timeout 45
& $py tools/run_uav_navigation.py --map-from $map --record-expert --obstacle-xy -10.72 -0.10 --obstacle-size .48 .50 .42 --timeout 45
& $py tools/run_uav_navigation.py --map-from $map --record-expert --obstacle-xy -10.15 -1.10 --obstacle-size .45 .47 .36 --timeout 45

$train1 = 'runs/<第一条专家轨迹目录>'
$train2 = 'runs/<第二条专家轨迹目录>'
$val = 'runs/<第三条专家轨迹目录>'
& $py tools/train_local_policy.py --train $train1 $train2 --val $val --output "$out/bc.pt" --epochs 50

& $py tools/run_uav_navigation.py --map-from $map --record-expert --local-policy neural --checkpoint "$out/bc.pt" --expert-mix .15 --obstacle-xy -9.80 .10 --obstacle-size .44 .42 .36 --timeout 45

$dagger = 'runs/<DAgger 回采轨迹目录>'
& $py tools/train_local_policy.py --train $train1 $train2 $dagger --val $val --init "$out/bc.pt" --output "$out/dagger.pt" --epochs 35 --lr .0008

& $py tools/compare_local_policies.py --map-from $map --bc-checkpoint "$out/bc.pt" --dagger-checkpoint "$out/dagger.pt" --obstacle detour --timeout 65 --output "$out/compare_detour"
```

采集达到仿真时间上限后，导航脚本返回非零退出码。若已生成完整的 `rover/imitation/manifest.json` 与运行 `summary.json`，该轨迹仍可供训练使用。只查看一组车辆的运行过程时，可以给 `run_uav_navigation.py` 加 `--visible --realtime`。全场评测可换成 `--profile full`、相应的先验图和目标，并为全部组统一指定 `--planner igha_vctp`。

## 同条件评测

比较器依次运行规则策略、带安全检查的 BC 策略、带同样安全检查的 BC + DAgger 策略。它核对三组的先验图哈希、生成场景哈希、初始全局路线哈希、目标、起点、障碍和时限。三组复用同一个导航控制器、传感器配置、固定的场景随机种子和 100 ms 控制周期。评测时专家混合概率为零。Webots Supervisor 单独读物理接触与真值轨迹并计算停稳、碰撞和到达指标。

`comparison.csv` 与 `comparison.json` 报告到达、停车或超时，及碰撞数、终点误差、到达时间、实际路程、重规划数、网络动作执行数、规则回退比例与 CPU 推理 P95。超时或停车的到达时间为空。JSON 的 `held_out` 字段核对这组场景、地图与目标是否出现在训练或验证数据里。单个方法的 `evaluation/nav_preview.png` 提供可视化证据，`evaluation/trajectory.csv` 保存真值轨迹。

本轮固定了同一片局部地形与同一目标，测试集变化来自碎石布置。网络效果需要结合闭环结果、回退比例与各类动作样本分布判断。

## 首轮 Webots 试跑

本机在 D 局部图上固定全局 A*、同一起终点和 65 秒预算，每种碎石布置各跑三组。以下两种布置均未参与训练或验证。

| 碎石布置 | 规则 | BC | BC + DAgger |
| --- | --- | --- | --- |
| 路径上 0.55 m 碎石 | 失败，1 次碰撞 | 失败，5 次碰撞 | 安全停车，0 次碰撞，距目标 6.22 m |
| 侧方 0.38×0.40 m 碎石 | 失败，2 次接触 | 失败，2 次接触 | 安全停车，0 次碰撞，距目标 6.21 m |

BC 使用 478 帧专家训练数据，独立验证集 400 帧。DAgger 回采 450 帧，其中 363 帧执行了非零网络动作，聚合后共 928 帧训练数据。本轮训练样本以行驶和转向为主，专家停止标签为零。两份权重都复用外层停稳和封堵保护。

DAgger 在两组中均行驶约 1.9 m 后停车，车载地图判为走廊阻塞，绕行会进入未知区。侧方布置中的规则与 BC 在终点附近发生接触，接触点高于地形插值约 5.7–5.8 cm，略超过项目 Supervisor 的 5.5 cm 判定阈值。终点距离、接触点和检测理由均保存在 JSON，可复核。三组各有两次留出测试，成功次数均为零。

本机证据目录为 `runs/local_learning_pilot_20261121/`，含两份权重、训练指标、`compare_detour/`、`compare_flank/` 和真值轨迹总图。复现侧方布置时，把评测命令的 `--obstacle detour` 换为 `--obstacle-xy -8.20 1.47 --obstacle-size .38 .40 .38`，并使用不同的输出目录。
